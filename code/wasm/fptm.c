/* ============================================================================
 * fptm.c  —  Fuzzy-Pattern Tsetlin Machine (FPTM), single-file C port of
 *            /IoT/FuzzyPatternTM/src/Tsetlin.jl, compiled to WASM with emcc.
 *
 *   Multiclass classifier.  TRAINING + INFERENCE in the browser.
 *
 *   INFERENCE is bit-exact to the verified JS/Python references:
 *       clause violations  c   = popcount( (~x & inc) | (x & inc_inv) )
 *       clause output          = max(literals_sum_clamp - c, 0)
 *       class score            = Σ positive-clause outputs − Σ negative-clause outputs
 *       prediction             = argmax_class (score), first max wins (== numpy.argmax)
 *
 *   TRAINING reproduces feedback! from Tsetlin.jl.  The RNG need not match
 *   Julia (it does not); we use a deterministic, well-seeded splitmix64-seeded
 *   xorshift128+ so a given seed always yields the same model.
 *
 *   Hyperparameter convention (Paper_2 driver):
 *       C  == clauses_per_class  == #positive clauses == #negative clauses
 *             (this maps to Julia's ta_clauses_num = floor(clauses_num/2)).
 *
 *   Build:
 *     emcc fptm.c -O2 -s WASM=1 -s SINGLE_FILE=1 -s MODULARIZE=1 \
 *       -s EXPORT_NAME=FPTM -s ALLOW_MEMORY_GROWTH=1 \
 *       -s EXPORTED_FUNCTIONS=... \
 *       -s EXPORTED_RUNTIME_METHODS='["ccall","cwrap","HEAPU8","HEAP32"]'
 *
 *   Dependency-free apart from malloc / free / memset / memcpy (provided by
 *   emscripten's libc; no other library is used).
 * ========================================================================== */

#include <stdint.h>
#include <stddef.h>

/* ---- minimal libc surface (emscripten provides these) -------------------- */
extern void *malloc(size_t);
extern void  free(void *);
extern void *memset(void *, int, size_t);
extern void *memcpy(void *, const void *, size_t);

#define EXPORT __attribute__((used, visibility("default")))

/* ============================================================================
 * Deterministic RNG: splitmix64 seeding -> xorshift128+ core.
 * Independent of Julia's RNG by design (semantics, not bit-stream, must match).
 * ========================================================================== */
static uint64_t g_rng[2];

static uint64_t splitmix64(uint64_t *s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static void rng_seed(uint64_t seed) {
    uint64_t sm = seed ? seed : 0x123456789ABCDEFULL;
    g_rng[0] = splitmix64(&sm);
    g_rng[1] = splitmix64(&sm);
    if (g_rng[0] == 0 && g_rng[1] == 0) g_rng[0] = 0x9E3779B97F4A7C15ULL;
}

static uint64_t rng_next(void) {
    uint64_t x = g_rng[0];
    uint64_t const y = g_rng[1];
    g_rng[0] = y;
    x ^= x << 23;
    g_rng[1] = x ^ y ^ (x >> 17) ^ (y >> 26);
    return g_rng[1] + y;
}

/* uniform double in [0,1) using the top 53 bits */
static double rng_uniform(void) {
    return (double)(rng_next() >> 11) * (1.0 / 9007199254740992.0);
}

/* uniform integer in [0, n)  (n > 0), unbiased rejection sampling */
static int rng_below(int n) {
    uint64_t lim = (uint64_t)n;
    uint64_t thresh = (~lim + 1u) % lim;          /* (2^64 mod n) */
    uint64_t r;
    do { r = rng_next(); } while (r < thresh);
    return (int)(r % lim);
}

/* ============================================================================
 * popcount
 * ========================================================================== */
static inline int popcount64(uint64_t v) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(v);
#else
    v = v - ((v >> 1) & 0x5555555555555555ULL);
    v = (v & 0x3333333333333333ULL) + ((v >> 2) & 0x3333333333333333ULL);
    v = (v + (v >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
    return (int)((v * 0x0101010101010101ULL) >> 56);
#endif
}

/* ============================================================================
 * Model layout.
 *
 * Per class we hold a TATeam = (positive clauses, negative clauses).
 * Each clause owns, for every bit j:
 *     state  c[j]   for literal  x_j
 *     state ci[j]   for literal ¬x_j
 * A literal is INCLUDED iff its state >= include_limit.
 *
 * We keep both the per-bit TA states (for training) and a derived, packed
 * bitmask representation of the included literals (for fast inference), kept
 * in sync after every clause update via clause_refresh().
 *
 * Indexing of TA states for one clause: a flat array of length 2*n_bits,
 *     c[j]  = states[2*j]
 *     ci[j] = states[2*j + 1]
 * (interleaving keeps the two automata of one bit adjacent in memory).
 * ========================================================================== */

typedef struct {
    int n_classes;
    int n_bits;
    int n_words;          /* ceil(n_bits/64) */
    int C;                /* clauses per class per polarity */
    int T;
    int S;
    int s;                /* round(n_bits / S), >= 1 */
    int L;                /* max literals budget for Type-Ia boost */
    int LF;               /* literals_sum clamp ceiling */
    int states_num;
    int include_limit;
    int state_min;        /* 0 */
    int state_max;        /* states_num - 1 */

    /* number of clauses of one polarity per class = C
     * total clauses = n_classes * 2 * C
     * Per class, clause local index 0..C-1 are positive, C..2C-1 are negative.
     */
    int clauses_per_class;        /* = 2*C (both polarities) */
    int total_clauses;            /* = n_classes * 2 * C */

    /* TA states: total_clauses * (2*n_bits) uint16_t.
     * uint16_t covers states_num up to 65536 (>= the UInt8/UInt16 Julia split).
     */
    uint16_t *states;

    /* Derived per-clause packed masks (n_words each):
     *   inc[clause]      bit j set iff literal x_j is included
     *   inc_inv[clause]  bit j set iff literal ¬x_j is included
     */
    uint64_t *inc;
    uint64_t *inc_inv;

    /* Per-clause scalars */
    int32_t *lsum;        /* total #included literals */
    int32_t *lclamp;      /* literals_sum_clamp = (0<lsum<LF)?lsum:LF */

    /* scratch packed sample, n_words */
    uint64_t *xbuf;
} FPTM;

static FPTM *G = NULL;

/* ---- accessors ----------------------------------------------------------- */
static inline uint16_t *clause_states(FPTM *m, int clause) {
    return m->states + (size_t)clause * (2 * m->n_bits);
}
static inline uint64_t *clause_inc(FPTM *m, int clause) {
    return m->inc + (size_t)clause * m->n_words;
}
static inline uint64_t *clause_inc_inv(FPTM *m, int clause) {
    return m->inc_inv + (size_t)clause * m->n_words;
}

/* class `cls`, polarity 0=positive 1=negative, local index k in [0,C) ->
 * global clause id. */
static inline int clause_id(FPTM *m, int cls, int polarity, int k) {
    return (cls * 2 + polarity) * m->C + k;
}

/* ============================================================================
 * Pack a row-major uint8 sample (n_bits values, 0/1) into m->xbuf (n_words).
 * ========================================================================== */
static void pack_sample(FPTM *m, const uint8_t *x) {
    int nw = m->n_words;
    for (int w = 0; w < nw; w++) m->xbuf[w] = 0;
    for (int j = 0; j < m->n_bits; j++) {
        if (x[j]) m->xbuf[j >> 6] |= (uint64_t)1 << (j & 63);
    }
}

/* ============================================================================
 * clause_refresh: recompute inc / inc_inv masks, lsum and lclamp from TA
 * states.  Mirrors aux_update() in Tsetlin.jl.
 * ========================================================================== */
static void clause_refresh(FPTM *m, int clause) {
    uint16_t *st = clause_states(m, clause);
    uint64_t *inc = clause_inc(m, clause);
    uint64_t *inv = clause_inc_inv(m, clause);
    int limit = m->include_limit;
    int nw = m->n_words;
    int nb = m->n_bits;
    int lsum = 0;

    for (int w = 0; w < nw; w++) { inc[w] = 0; inv[w] = 0; }

    for (int j = 0; j < nb; j++) {
        int c  = st[2 * j];
        int ci = st[2 * j + 1];
        if (c >= limit)  { inc[j >> 6] |= (uint64_t)1 << (j & 63); lsum++; }
        if (ci >= limit) { inv[j >> 6] |= (uint64_t)1 << (j & 63); lsum++; }
    }
    m->lsum[clause]   = lsum;
    m->lclamp[clause] = (0 < lsum && lsum < m->LF) ? lsum : m->LF;
}

/* ============================================================================
 * check_clause: clause output for the currently packed sample (m->xbuf).
 *   violations c = popcount( (~x & inc) | (x & inc_inv) )
 *   output       = max(lclamp - c, 0)
 * Bit-exact to the JS/Python reference.
 * ========================================================================== */
static int check_clause(FPTM *m, int clause) {
    uint64_t *inc = clause_inc(m, clause);
    uint64_t *inv = clause_inc_inv(m, clause);
    uint64_t *x   = m->xbuf;
    int clamp = m->lclamp[clause];
    int c = 0;
    int nw = m->n_words;
    for (int w = 0; w < nw; w++) {
        uint64_t xw = x[w];
        uint64_t val = (~xw & inc[w]) | (xw & inv[w]);
        c += popcount64(val);
    }
    int out = clamp - c;
    return out > 0 ? out : 0;
}

/* ============================================================================
 * vote: returns pos and neg sums for one class team on the packed sample.
 * ========================================================================== */
static void vote(FPTM *m, int cls, int *pos_out, int *neg_out) {
    int pos = 0, neg = 0;
    for (int k = 0; k < m->C; k++) {
        pos += check_clause(m, clause_id(m, cls, 0, k));
        neg += check_clause(m, clause_id(m, cls, 1, k));
    }
    *pos_out = pos;
    *neg_out = neg;
}

/* ============================================================================
 * feedback! for one class team's two clause groups.
 *
 *   clauses1 / group1 get Feedback-1 ("recognize")
 *   clauses2 / group2 get Feedback-2 ("reject")
 *
 * For the TRUE class:  group1 = positive clauses, group2 = negative clauses,
 *                      positive = true.
 * For a wrong class:   group1 = negative clauses, group2 = positive clauses,
 *                      positive = false.
 *
 * `pol1`/`pol2` are the polarity codes (0 pos, 1 neg) of those groups.
 *
 *   v       = clamp(pos - neg, -T, T)         (inference sign: pos - neg)
 *   update  = clamp( (positive ? T - v : T + v) / (2T) * class_weight, 0, 1)
 *
 * Note on sign: Julia writes v = clamp(-(vote...)..., -T, T) where -(pos,neg)
 * splats to -(pos - neg) = neg - pos.  The verified inference reference scores
 * a class by (pos - neg), so we adopt v = clamp(pos - neg, -T, T) here and the
 * matching update form (positive -> T - v).  This keeps training pressure and
 * inference on the same convention.
 * ========================================================================== */
static void feedback_group(FPTM *m, int cls,
                           int pol1, int pol2,
                           int positive, double class_weight) {
    int pos, neg;
    vote(m, cls, &pos, &neg);
    int v = pos - neg;
    if (v >  m->T) v =  m->T;
    if (v < -m->T) v = -m->T;

    double raw = (positive ? (double)(m->T - v) : (double)(m->T + v))
                 / (double)(2 * m->T) * class_weight;
    double update = raw < 0.0 ? 0.0 : (raw > 1.0 ? 1.0 : raw);

    int nb = m->n_bits;
    int nw = m->n_words;
    int smin = m->state_min, smax = m->state_max;

    /* ---------------------- Feedback 1 (recognize) ----------------------- */
    for (int k = 0; k < m->C; k++) {
        if (rng_uniform() >= update) continue;
        int cl = clause_id(m, cls, pol1, k);
        uint16_t *st = clause_states(m, cl);

        if (check_clause(m, cl) > 0) {
            /* Type Ia: boost present literals, only while under budget L. */
            if (m->lsum[cl] < m->L) {
                for (int j = 0; j < nb; j++) {
                    if (m->xbuf[j >> 6] & ((uint64_t)1 << (j & 63))) {
                        /* x_j == 1 -> boost x_j automaton */
                        if (st[2 * j] < smax) st[2 * j]++;
                    } else {
                        /* x_j == 0 -> boost ¬x_j automaton */
                        if (st[2 * j + 1] < smax) st[2 * j + 1]++;
                    }
                }
            }
            /* Type Ib (forget), deterministic:
             *   pos = ~x & ~inc      -> decrement x_j automaton
             *   neg =  x & ~inc_inv  -> decrement ¬x_j automaton
             */
            {
                uint64_t *inc = clause_inc(m, cl);
                uint64_t *inv = clause_inc_inv(m, cl);
                for (int w = 0; w < nw; w++) {
                    uint64_t xw = m->xbuf[w];
                    uint64_t dec_c  = (~xw & ~inc[w]);
                    uint64_t dec_ci = ( xw & ~inv[w]);
                    /* mask off bits beyond n_bits in the final word */
                    int base = w << 6;
                    while (dec_c) {
                        int ii = __builtin_ctzll(dec_c);
                        int j = base + ii;
                        if (j < nb && st[2 * j] > smin) st[2 * j]--;
                        dec_c &= dec_c - 1;
                    }
                    while (dec_ci) {
                        int ii = __builtin_ctzll(dec_ci);
                        int j = base + ii;
                        if (j < nb && st[2 * j + 1] > smin) st[2 * j + 1]--;
                        dec_ci &= dec_ci - 1;
                    }
                }
            }
        } else {
            /* clause_output == 0:  do s random single-bit decrements on each
             * automaton family (one random bit for c, one for ci), s times. */
            for (int t = 0; t < m->s; t++) {
                int r1 = rng_below(nb);
                if (st[2 * r1] > smin) st[2 * r1]--;
                int r2 = rng_below(nb);
                if (st[2 * r2 + 1] > smin) st[2 * r2 + 1]--;
            }
        }
        clause_refresh(m, cl);
    }

    /* ---------------------- Feedback 2 (reject) -------------------------- */
    for (int k = 0; k < m->C; k++) {
        if (rng_uniform() >= update) continue;
        int cl = clause_id(m, cls, pol2, k);
        uint16_t *st = clause_states(m, cl);

        if (check_clause(m, cl) > 0) {
            /* pos = ~x & ~inc      -> increment x_j automaton
             * neg =  x & ~inc_inv  -> increment ¬x_j automaton  (no max cap in
             * Julia Feedback-2; states can only matter up to include_limit but
             * we still cap at state_max to keep the type bounded). */
            uint64_t *inc = clause_inc(m, cl);
            uint64_t *inv = clause_inc_inv(m, cl);
            for (int w = 0; w < nw; w++) {
                uint64_t xw = m->xbuf[w];
                uint64_t inc_c  = (~xw & ~inc[w]);
                uint64_t inc_ci = ( xw & ~inv[w]);
                int base = w << 6;
                while (inc_c) {
                    int ii = __builtin_ctzll(inc_c);
                    int j = base + ii;
                    if (j < nb && st[2 * j] < smax) st[2 * j]++;
                    inc_c &= inc_c - 1;
                }
                while (inc_ci) {
                    int ii = __builtin_ctzll(inc_ci);
                    int j = base + ii;
                    if (j < nb && st[2 * j + 1] < smax) st[2 * j + 1]++;
                    inc_ci &= inc_ci - 1;
                }
            }
            clause_refresh(m, cl);
        }
    }
}

/* ============================================================================
 * train one sample (already packed into m->xbuf).
 * ========================================================================== */
static void train_one(FPTM *m, int y, double class_weight) {
    /* TRUE class: F1 on positive (pol 0), F2 on negative (pol 1), positive=true */
    feedback_group(m, y, /*pol1=*/0, /*pol2=*/1, /*positive=*/1, class_weight);

    /* every other class: F1 on negative (pol 1), F2 on positive (pol 0),
     * positive=false */
    for (int cls = 0; cls < m->n_classes; cls++) {
        if (cls == y) continue;
        feedback_group(m, cls, /*pol1=*/1, /*pol2=*/0, /*positive=*/0,
                       class_weight);
    }
}

/* ============================================================================
 * Fisher-Yates shuffle of an int array using the model RNG.
 * ========================================================================== */
static void shuffle_indices(int *idx, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rng_below(i + 1);
        int tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
    }
}

/* ============================================================================
 *                              PUBLIC API
 * ========================================================================== */

EXPORT
int fptm_create(int n_classes, int n_bits, int C, int T, int S,
                int L, int LF, int states_num, int include_limit,
                unsigned seed) {
    if (G) {
        /* free previous model */
        free(G->states); free(G->inc); free(G->inc_inv);
        free(G->lsum);   free(G->lclamp); free(G->xbuf);
        free(G); G = NULL;
    }
    if (n_classes <= 0 || n_bits <= 0 || C <= 0 || T <= 0 ||
        states_num <= 1 || include_limit < 1 || include_limit > states_num - 1)
        return -1;

    FPTM *m = (FPTM *)malloc(sizeof(FPTM));
    if (!m) return -1;
    memset(m, 0, sizeof(FPTM));

    m->n_classes     = n_classes;
    m->n_bits        = n_bits;
    m->n_words       = (n_bits + 63) / 64;
    m->C             = C;
    m->T             = T;
    m->S             = S > 0 ? S : 1;
    /* s = round(n_bits / S), at least 1 */
    {
        double sd = (double)n_bits / (double)m->S;
        int sr = (int)(sd + 0.5);
        m->s = sr < 1 ? 1 : sr;
    }
    m->L             = L;
    m->LF            = LF;
    m->states_num    = states_num;
    m->include_limit = include_limit;
    m->state_min     = 0;
    m->state_max     = states_num - 1;
    m->clauses_per_class = 2 * C;
    m->total_clauses     = n_classes * 2 * C;

    size_t tc = (size_t)m->total_clauses;
    size_t nb2 = (size_t)(2 * n_bits);

    m->states  = (uint16_t *)malloc(tc * nb2 * sizeof(uint16_t));
    m->inc     = (uint64_t *)malloc(tc * m->n_words * sizeof(uint64_t));
    m->inc_inv = (uint64_t *)malloc(tc * m->n_words * sizeof(uint64_t));
    m->lsum    = (int32_t  *)malloc(tc * sizeof(int32_t));
    m->lclamp  = (int32_t  *)malloc(tc * sizeof(int32_t));
    m->xbuf    = (uint64_t *)malloc((size_t)m->n_words * sizeof(uint64_t));

    if (!m->states || !m->inc || !m->inc_inv || !m->lsum ||
        !m->lclamp || !m->xbuf) {
        free(m->states); free(m->inc); free(m->inc_inv);
        free(m->lsum); free(m->lclamp); free(m->xbuf); free(m);
        return -1;
    }

    /* initial state = include_limit - 1 (all literals just-excluded) */
    {
        uint16_t init = (uint16_t)(include_limit - 1);
        size_t total_states = tc * nb2;
        /* memset only works for byte fill; do explicit loop for 16-bit init */
        for (size_t i = 0; i < total_states; i++) m->states[i] = init;
    }
    /* derived masks: with all literals excluded, inc/inc_inv are zero,
     * lsum = 0, lclamp = LF. */
    memset(m->inc,     0, tc * m->n_words * sizeof(uint64_t));
    memset(m->inc_inv, 0, tc * m->n_words * sizeof(uint64_t));
    for (size_t i = 0; i < tc; i++) {
        m->lsum[i]   = 0;
        m->lclamp[i] = LF;   /* (0<lsum<LF)?lsum:LF with lsum=0 -> LF */
    }

    rng_seed((uint64_t)seed);
    G = m;
    return 0;
}

EXPORT
void fptm_free(void) {
    if (!G) return;
    free(G->states); free(G->inc); free(G->inc_inv);
    free(G->lsum);   free(G->lclamp); free(G->xbuf);
    free(G); G = NULL;
}

/* one epoch over n samples; X is n*n_bits row-major uint8; Y is n int labels.
 * Sample order is shuffled each epoch (matching train! shuffle=true). */
EXPORT
void fptm_train_epoch(uint8_t *X, int n, int *Y) {
    FPTM *m = G;
    if (!m || n <= 0) return;

    int *order = (int *)malloc((size_t)n * sizeof(int));
    if (!order) {
        /* fall back to in-order training if allocation fails */
        for (int i = 0; i < n; i++) {
            int y = Y[i];
            if (y < 0 || y >= m->n_classes) continue;
            pack_sample(m, X + (size_t)i * m->n_bits);
            train_one(m, y, 1.0);
        }
        return;
    }
    for (int i = 0; i < n; i++) order[i] = i;
    shuffle_indices(order, n);

    for (int t = 0; t < n; t++) {
        int i = order[t];
        int y = Y[i];
        if (y < 0 || y >= m->n_classes) continue;
        pack_sample(m, X + (size_t)i * m->n_bits);
        train_one(m, y, 1.0);
    }
    free(order);
}

/* Same as fptm_train_epoch but with per-sample class weights (double array of
 * length n_classes).  Pass weights=NULL behaviour is handled by the simpler
 * fptm_train_epoch; this variant always reads CW[Y[i]]. */
EXPORT
void fptm_train_epoch_weighted(uint8_t *X, int n, int *Y, double *CW) {
    FPTM *m = G;
    if (!m || n <= 0) return;
    if (!CW) { fptm_train_epoch(X, n, Y); return; }

    int *order = (int *)malloc((size_t)n * sizeof(int));
    if (!order) return;
    for (int i = 0; i < n; i++) order[i] = i;
    shuffle_indices(order, n);

    for (int t = 0; t < n; t++) {
        int i = order[t];
        int y = Y[i];
        if (y < 0 || y >= m->n_classes) continue;
        pack_sample(m, X + (size_t)i * m->n_bits);
        train_one(m, y, CW[y]);
    }
    free(order);
}

/* predict one sample.  argmax_class(pos - neg), first max wins. */
EXPORT
int fptm_predict(uint8_t *x) {
    FPTM *m = G;
    if (!m) return -1;
    pack_sample(m, x);

    int best_cls = 0;
    int best_vote = -2000000000;
    for (int cls = 0; cls < m->n_classes; cls++) {
        int pos, neg;
        vote(m, cls, &pos, &neg);
        int v = pos - neg;
        if (v > best_vote) {
            best_vote = v;
            best_cls = cls;
        }
    }
    return best_cls;
}

EXPORT
void fptm_predict_batch(uint8_t *X, int n, int *out) {
    FPTM *m = G;
    if (!m) return;
    for (int i = 0; i < n; i++) {
        out[i] = fptm_predict(X + (size_t)i * m->n_bits);
    }
}

/* per-class raw scores (pos - neg) for one sample; out must hold n_classes. */
EXPORT
void fptm_scores(uint8_t *x, int *out) {
    FPTM *m = G;
    if (!m) return;
    pack_sample(m, x);
    for (int cls = 0; cls < m->n_classes; cls++) {
        int pos, neg;
        vote(m, cls, &pos, &neg);
        out[cls] = pos - neg;
    }
}

/* Graded output of EVERY clause for one sample, by global clause id.
 * out must hold fptm_num_clauses() ints; out[id] = max(clamp - violations, 0).
 * Lets the UI reconstruct exactly which clauses fired and by how much (the same
 * per-clause "clauseOut" the JS reference computes) in one WASM call. */
EXPORT
void fptm_clause_outputs(uint8_t *x, int *out) {
    FPTM *m = G;
    if (!m) return;
    pack_sample(m, x);
    for (int id = 0; id < m->total_clauses; id++) {
        out[id] = check_clause(m, id);
    }
}

/* ---------------------------- introspection ------------------------------- */

EXPORT int fptm_num_clauses(void) { return G ? G->total_clauses : 0; }
EXPORT int fptm_n_classes(void)   { return G ? G->n_classes : 0; }
EXPORT int fptm_n_bits(void)      { return G ? G->n_bits : 0; }

/* class of a global clause id */
EXPORT
int fptm_clause_class(int id) {
    FPTM *m = G;
    if (!m || id < 0 || id >= m->total_clauses) return -1;
    return id / (2 * m->C);
}

/* polarity: 1 = positive, -1 = negative (matches JS sign convention). */
EXPORT
int fptm_clause_polarity(int id) {
    FPTM *m = G;
    if (!m || id < 0 || id >= m->total_clauses) return 0;
    int within = id % (2 * m->C);
    return (within < m->C) ? 1 : -1;
}

/* clamp (literals_sum_clamp) of a clause. */
EXPORT
int fptm_clause_clamp(int id) {
    FPTM *m = G;
    if (!m || id < 0 || id >= m->total_clauses) return 0;
    return m->lclamp[id];
}

/* number of included literals in a clause. */
EXPORT
int fptm_clause_lsum(int id) {
    FPTM *m = G;
    if (!m || id < 0 || id >= m->total_clauses) return 0;
    return m->lsum[id];
}

/* Enumerate included literals of a clause.
 *   out_bits[k] = bit index j
 *   out_ops[k]  = 0 for asserted literal  x_j   (operator "≥", bit required ON)
 *                 1 for negated  literal  ¬x_j  (operator "<",  bit required OFF)
 * Returns the number of literals written.  Caller must size out_bits/out_ops
 * to at least fptm_clause_lsum(id) (or 2*n_bits to be safe).
 */
EXPORT
int fptm_clause_literals(int id, int *out_bits, int *out_ops) {
    FPTM *m = G;
    if (!m || id < 0 || id >= m->total_clauses) return 0;
    uint64_t *inc = clause_inc(m, id);
    uint64_t *inv = clause_inc_inv(m, id);
    int nb = m->n_bits;
    int count = 0;
    for (int j = 0; j < nb; j++) {
        int w = j >> 6, b = j & 63;
        if (inc[w] & ((uint64_t)1 << b)) {
            out_bits[count] = j;
            out_ops[count]  = 0;   /* x_j */
            count++;
        }
        if (inv[w] & ((uint64_t)1 << b)) {
            out_bits[count] = j;
            out_ops[count]  = 1;   /* ¬x_j */
            count++;
        }
    }
    return count;
}

/* Raw TA state read/write (for save/load of the full model from JS). */
EXPORT
int fptm_get_state(int clause, int bit, int which) {
    FPTM *m = G;
    if (!m || clause < 0 || clause >= m->total_clauses ||
        bit < 0 || bit >= m->n_bits || which < 0 || which > 1) return -1;
    return clause_states(m, clause)[2 * bit + which];
}

EXPORT
void fptm_set_state(int clause, int bit, int which, int value) {
    FPTM *m = G;
    if (!m || clause < 0 || clause >= m->total_clauses ||
        bit < 0 || bit >= m->n_bits || which < 0 || which > 1) return;
    if (value < m->state_min) value = m->state_min;
    if (value > m->state_max) value = m->state_max;
    clause_states(m, clause)[2 * bit + which] = (uint16_t)value;
}

/* After bulk fptm_set_state calls, refresh all derived masks. */
EXPORT
void fptm_refresh_all(void) {
    FPTM *m = G;
    if (!m) return;
    for (int c = 0; c < m->total_clauses; c++) clause_refresh(m, c);
}

/* Bulk dump of all TA states into a uint16 buffer of length
 * total_clauses * 2 * n_bits (clause-major, then interleaved bit/which).
 * Returns the number of uint16 entries written (0 if buf too small / null). */
EXPORT
int fptm_dump_states(uint16_t *buf, int buf_len) {
    FPTM *m = G;
    if (!m || !buf) return 0;
    size_t total = (size_t)m->total_clauses * (size_t)(2 * m->n_bits);
    if ((size_t)buf_len < total) return 0;
    memcpy(buf, m->states, total * sizeof(uint16_t));
    return (int)total;
}

/* Bulk load of all TA states from a uint16 buffer (inverse of dump).
 * Refreshes derived masks afterwards.  Returns 1 on success, 0 on failure. */
EXPORT
int fptm_load_states(uint16_t *buf, int buf_len) {
    FPTM *m = G;
    if (!m || !buf) return 0;
    size_t total = (size_t)m->total_clauses * (size_t)(2 * m->n_bits);
    if ((size_t)buf_len < total) return 0;
    memcpy(m->states, buf, total * sizeof(uint16_t));
    fptm_refresh_all();
    return 1;
}
