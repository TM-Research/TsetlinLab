# FPTM (base /IoT/Tsetlin.jl) on full 1-bit MNIST — sweep LF to test the hypothesis that LF (the
# fuzzy tolerance/clamp) is what lets self-contradictory clauses survive. Fixed: C=1000,T=750,S=3,
# L=784 (no literal limit), 40 epochs. Reports accuracy + same-bit contradiction count per LF.
using Printf, Random
include("/IoT/Tsetlin.jl/src/Tsetlin.jl")
using .Tsetlin: TMInput, TMClassifier, train!, predict, accuracy
TMP = "/tmp/mnist_full"; NB = 784
x_train = [TMInput([parse(Bool,t) for t in split(l," ")]) for l in readlines("$TMP/X_train.txt")]
x_test  = [TMInput([parse(Bool,t) for t in split(l," ")]) for l in readlines("$TMP/X_test.txt")]
y_train = [parse(Int,l) for l in readlines("$TMP/Y_train.txt")]
y_test  = [parse(Int,l) for l in readlines("$TMP/Y_test.txt")]
getbit(M,j,b) = ((M[((b-1)>>>6)+1,j] >> ((b-1)&63)) & UInt64(1)) == UInt64(1)
function samebit(tm)
    c = 0
    for ta in tm.clauses
        for (A,Nv) in ((ta.positive_included_literals, ta.positive_included_literals_inverted),
                       (ta.negative_included_literals, ta.negative_included_literals_inverted))
            for j in 1:size(A,2)
                any(getbit(A,j,b) && getbit(Nv,j,b) for b in 1:NB) && (c += 1)
            end
        end
    end
    c
end
println("LF\tacc\tsame_bit_clauses")
for LF in [8, 16, 32, 64, 128, 192]
    Random.seed!(42)
    tm = TMClassifier(x_train[1], y_train, 1000, 750, 3, 784, LF, states_num=256, include_limit=128)
    train!(tm, x_train, y_train, x_test, y_test, 40, shuffle=true, index=true, verbose=0)
    acc = accuracy(predict(tm, x_test, index=true), y_test)
    @printf("%d\t%.4f\t%d\n", LF, acc, samebit(tm)); flush(stdout)
end
println("LF SWEEP DONE")
