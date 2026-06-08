# Verification (no new engine): train a given engine on standard-binarised data and COUNT
# both same-bit and thermometer contradictions directly from the trained masks.
#   ENGINE_PATH  : which engine file to include (module Tsetlin)
#   TM_EXCLUSIVE : "true" -> call train! with exclusive=true (perara's built-in guard)
using Printf, Random, JSON
include(get(ENV,"ENGINE_PATH","/IoT/Tsetlin.jl/src/Tsetlin_contradiction_fix.jl"))
using .Tsetlin: TMInput, TMClassifier, train!, predict, accuracy
TMP=ENV["TMP_DIR"]; excl=parse(Bool,get(ENV,"TM_EXCLUSIVE","false")); tag=get(ENV,"TAG","run")
C=parse(Int,ENV["TM_CLAUSES"]);T=parse(Int,ENV["TM_T"]);S=parse(Int,ENV["TM_S"])
L=parse(Int,ENV["TM_L"]);LF=parse(Int,ENV["TM_LF"]);EP=parse(Int,ENV["TM_EPOCHS"])
STATES=parse(Int,get(ENV,"TM_STATES","256"));INCLUDE=parse(Int,get(ENV,"TM_INCLUDE","128"))
x_train=[TMInput([parse(Bool,t) for t in split(l," ")]) for l in readlines("$TMP/X_train.txt")]
x_test =[TMInput([parse(Bool,t) for t in split(l," ")]) for l in readlines("$TMP/X_test.txt")]
y_train=[parse(Int,l) for l in readlines("$TMP/Y_train.txt")]
y_test =[parse(Int,l) for l in readlines("$TMP/Y_test.txt")]
bj=JSON.parsefile("$TMP/bits.json"); fi=Int.(bj["feat_idx"]); th=Float64.(bj["thresh"]); NPX=Int(bj["n_bits"])
Random.seed!(42)
tm=TMClassifier(x_train[1],y_train,C,T,S,L,LF,states_num=STATES,include_limit=INCLUDE)
if excl
    train!(tm,x_train,y_train,x_test,y_test,EP,shuffle=true,index=true,verbose=0,exclusive=true)
else
    train!(tm,x_train,y_train,x_test,y_test,EP,shuffle=true,index=true,verbose=0)
end
acc=accuracy(predict(tm,x_test,index=true),y_test)
getbit(M,j,b)=((M[((b-1)>>>6)+1,j]>>((b-1)&63))&UInt64(1))==UInt64(1)
samebit=0; thermo=0
for ta in tm.clauses
    for (A,Ai) in ((ta.positive_included_literals,ta.positive_included_literals_inverted),
                   (ta.negative_included_literals,ta.negative_included_literals_inverted))
        for j in 1:size(A,2)
            sb=false; lo=Dict{Int,Float64}(); hi=Dict{Int,Float64}()
            for b in 1:NPX
                a=getbit(A,j,b); n=getbit(Ai,j,b)
                a&&n && (sb=true)
                f=fi[b]
                a && (lo[f]=max(get(lo,f,-1e30), th[b]))   # asserted: feature >= th
                n && (hi[f]=min(get(hi,f, 1e30), th[b]))   # negated:  feature <  th
            end
            sb && (global samebit+=1)
            any(haskey(hi,f) && lo[f]>=hi[f] for f in keys(lo)) && (global thermo+=1)
        end
    end
end
@printf("[%s] acc=%.2f%% bits=%d  SAME-BIT clauses=%d  THERMOMETER clauses=%d\n",tag,acc*100,NPX,samebit,thermo)
