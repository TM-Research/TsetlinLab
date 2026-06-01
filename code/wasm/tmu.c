/* ============================================================================
 * tmu.c — vanilla weighted-crisp Tsetlin Machine inference, single-file C for
 *          emcc / WebAssembly. Companion to fptm.c (which is the graded
 *          Fuzzy-Pattern model); this kernel is the crisp + per-class signed
 *          weight model the TMU StandardBinarizer + TMClassifier produces.
 *
 *   Verified semantics (reproduces TMClassifier.predict(return_class_sums)
 *   bit-exactly):
 *     - PER-CLASS clause banks: every class k has its own C clauses, each with
 *       its own included-literal set and a SIGNED int32 weight (negatives vote
 *       against the class).
 *     - literal layout: bit b in [0,nbits) = asserted literal (input bit b == 1);
 *       the negated literal of bit b requires input bit b == 0.
 *     - crisp fire: a clause fires (1) iff it includes >=1 literal AND every
 *       included asserted literal's bit == 1 AND every included negated
 *       literal's bit == 0. An EMPTY clause (no included literals) fires 0.
 *     - class_sum[k] = sum over that class's clauses of (fire ? weight : 0),
 *       int32, NO clip. prediction = argmax over classes, first max wins.
 *
 *   Memory model: a single global model G. Clauses are flattened class-major:
 *   global id = k*C + j. Each clause stores two bit-masks over n_words 64-bit
 *   words: `inc` (asserted-include) and `inv` (negated-include), plus lsum
 *   (#included literals, for the empty-clause guard) and a signed weight.
 *
 *   Build:
 *     emcc tmu.c -O2 -s WASM=1 -s SINGLE_FILE=1 -s MODULARIZE=1 \
 *       -s EXPORT_NAME=TMU -s ALLOW_MEMORY_GROWTH=1 \
 *       -s EXPORTED_FUNCTIONS=... \
 *       -s EXPORTED_RUNTIME_METHODS='["ccall","cwrap","HEAPU8","HEAP32","HEAPU16"]'
 * ========================================================================== */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef __EMSCRIPTEN__
  #include <emscripten.h>
  #define EXPORT EMSCRIPTEN_KEEPALIVE
#else
  #define EXPORT
#endif

static inline int popcount64(uint64_t v){
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(v);
#else
    int c=0; while(v){ v&=v-1; c++; } return c;
#endif
}

typedef struct {
    int n_classes;
    int n_bits;
    int C;            /* clauses per class */
    int n_words;      /* ceil(n_bits/64) */
    int total;        /* n_classes * C */
    uint64_t *inc;    /* total * n_words : asserted-include masks            */
    uint64_t *inv;    /* total * n_words : negated-include masks             */
    int32_t  *lsum;   /* total           : #included literals (empty->0 fire) */
    int32_t  *weight; /* total           : signed clause weight              */
    uint64_t *xbuf;   /* n_words         : packed current sample             */
} TMU;

static TMU *G = NULL;
void tmu_free(void);   /* fwd decl (used by tmu_create) */

static inline uint64_t *cl_inc(TMU *m, int id){ return m->inc + (size_t)id * m->n_words; }
static inline uint64_t *cl_inv(TMU *m, int id){ return m->inv + (size_t)id * m->n_words; }

/* pack a 0/1 byte sample into 64-bit words (bit b -> word b/64, offset b%64). */
static void pack_sample(TMU *m, const uint8_t *x){
    memset(m->xbuf, 0, (size_t)m->n_words * sizeof(uint64_t));
    for(int b=0; b<m->n_bits; b++){
        if(x[b]) m->xbuf[b>>6] |= ((uint64_t)1 << (b & 63));
    }
}

/* crisp fire: 0 if empty clause, else 1 iff no included literal is violated. */
static int clause_fire(TMU *m, int id){
    if(m->lsum[id] == 0) return 0;                 /* empty clause -> fire 0  */
    uint64_t *inc = cl_inc(m,id), *inv = cl_inv(m,id), *x = m->xbuf;
    int nw = m->n_words;
    for(int w=0; w<nw; w++){
        uint64_t xw = x[w];
        /* violated iff an asserted-include bit is 0, OR a negated-include bit is 1 */
        uint64_t bad = (~xw & inc[w]) | (xw & inv[w]);
        if(bad) return 0;
    }
    return 1;
}

/* ------------------------------- API ------------------------------------- */

EXPORT
int tmu_create(int n_classes, int n_bits, int C){
    if(G){ tmu_free(); }
    TMU *m = (TMU*)calloc(1, sizeof(TMU));
    if(!m) return 1;
    m->n_classes=n_classes; m->n_bits=n_bits; m->C=C;
    m->n_words = (n_bits + 63) / 64;
    m->total   = n_classes * C;
    size_t maskN = (size_t)m->total * m->n_words;
    m->inc    = (uint64_t*)calloc(maskN, sizeof(uint64_t));
    m->inv    = (uint64_t*)calloc(maskN, sizeof(uint64_t));
    m->lsum   = (int32_t*) calloc(m->total, sizeof(int32_t));
    m->weight = (int32_t*) calloc(m->total, sizeof(int32_t));
    m->xbuf   = (uint64_t*)calloc(m->n_words, sizeof(uint64_t));
    if(!m->inc||!m->inv||!m->lsum||!m->weight||!m->xbuf){
        free(m->inc);free(m->inv);free(m->lsum);free(m->weight);free(m->xbuf);free(m);
        return 2;
    }
    G = m;
    return 0;
}

EXPORT
void tmu_free(void){
    if(!G) return;
    free(G->inc); free(G->inv); free(G->lsum); free(G->weight); free(G->xbuf);
    free(G); G=NULL;
}

EXPORT void tmu_set_weight(int id, int w){ if(G && id>=0 && id<G->total) G->weight[id]=(int32_t)w; }
EXPORT int  tmu_clause_weight(int id){ return (G && id>=0 && id<G->total) ? G->weight[id] : 0; }

/* Add one included literal to a clause. bit in [0,n_bits); neg=0 asserted, 1 negated.
 * Increments lsum. Call tmu_create first; literals can be added in any order. */
EXPORT
void tmu_add_literal(int id, int bit, int neg){
    TMU *m=G;
    if(!m || id<0 || id>=m->total || bit<0 || bit>=m->n_bits) return;
    uint64_t mask = (uint64_t)1 << (bit & 63);
    uint64_t *row = neg ? cl_inv(m,id) : cl_inc(m,id);
    int w = bit>>6;
    if(!(row[w] & mask)){ row[w] |= mask; m->lsum[id] += 1; }
}

/* Bulk loader: for clause `id`, set its asserted/negated masks (n_words each)
 * and weight directly. masks are uint64 arrays already packed by the caller. */
EXPORT
void tmu_load_clause(int id, uint64_t *inc, uint64_t *inv, int weight){
    TMU *m=G;
    if(!m || id<0 || id>=m->total) return;
    int nw=m->n_words; int ls=0;
    uint64_t *ri=cl_inc(m,id), *rv=cl_inv(m,id);
    for(int w=0; w<nw; w++){ ri[w]=inc[w]; rv[w]=inv[w]; ls += popcount64(inc[w]) + popcount64(inv[w]); }
    m->lsum[id]=ls; m->weight[id]=(int32_t)weight;
}

/* class_sum[k] = Σ_j (fire ? weight : 0). out must hold n_classes ints. */
EXPORT
void tmu_class_sums(uint8_t *x, int *out){
    TMU *m=G; if(!m) return;
    pack_sample(m, x);
    for(int k=0;k<m->n_classes;k++){
        int32_t s=0; int base=k*m->C;
        for(int j=0;j<m->C;j++){ int id=base+j; if(clause_fire(m,id)) s += m->weight[id]; }
        out[k]=s;
    }
}

/* predict one sample: argmax over class sums, first-max (lowest index) on ties. */
EXPORT
int tmu_predict(uint8_t *x){
    TMU *m=G; if(!m) return -1;
    pack_sample(m, x);
    int best=0; int32_t bestv=-2147483647-1; int first=1;
    for(int k=0;k<m->n_classes;k++){
        int32_t s=0; int base=k*m->C;
        for(int j=0;j<m->C;j++){ int id=base+j; if(clause_fire(m,id)) s += m->weight[id]; }
        if(first || s>bestv){ bestv=s; best=k; first=0; }
    }
    return best;
}

EXPORT
void tmu_predict_batch(uint8_t *X, int n, int *out){
    TMU *m=G; if(!m) return;
    for(int i=0;i<n;i++) out[i]=tmu_predict(X + (size_t)i * m->n_bits);
}

/* per-clause fire flags (0/1) for one sample; out must hold total ints.
 * Lets the explain panel show exactly which clauses fired. */
EXPORT
void tmu_clause_fire_all(uint8_t *x, int *out){
    TMU *m=G; if(!m) return;
    pack_sample(m, x);
    for(int id=0; id<m->total; id++) out[id]=clause_fire(m,id);
}

/* introspection */
EXPORT int tmu_num_clauses(void){ return G?G->total:0; }
EXPORT int tmu_n_classes(void){ return G?G->n_classes:0; }
EXPORT int tmu_n_bits(void){ return G?G->n_bits:0; }
EXPORT int tmu_clause_class(int id){ return (G&&id>=0&&id<G->total)? id/G->C : -1; }
EXPORT int tmu_clause_lsum(int id){ return (G&&id>=0&&id<G->total)? G->lsum[id] : 0; }

/* enumerate included literals of a clause: out_bits[k]=bit, out_ops[k]=0 asserted/1 negated. */
EXPORT
int tmu_clause_literals(int id, int *out_bits, int *out_ops){
    TMU *m=G; if(!m||id<0||id>=m->total) return 0;
    uint64_t *inc=cl_inc(m,id), *inv=cl_inv(m,id);
    int cnt=0;
    for(int b=0;b<m->n_bits;b++){
        int w=b>>6; uint64_t mask=(uint64_t)1<<(b&63);
        if(inc[w]&mask){ out_bits[cnt]=b; out_ops[cnt]=0; cnt++; }
        if(inv[w]&mask){ out_bits[cnt]=b; out_ops[cnt]=1; cnt++; }
    }
    return cnt;
}
