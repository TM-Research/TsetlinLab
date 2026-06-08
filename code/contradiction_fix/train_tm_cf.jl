using Printf, Random, JSON
include(get(ENV,"ENGINE_PATH","/IoT/Tsetlin.jl/src/Tsetlin_contradiction_fix.jl"))   # module Tsetlin; default = inline l&=~li fix
using .Tsetlin: TMInput, TMClassifier, train!, predict, accuracy
TMP=ENV["TMP_DIR"]
C=parse(Int,ENV["TM_CLAUSES"]); T=parse(Int,ENV["TM_T"]); S=parse(Int,ENV["TM_S"])
L=parse(Int,ENV["TM_L"]); LF=parse(Int,ENV["TM_LF"]); EP=parse(Int,ENV["TM_EPOCHS"])
STATES=parse(Int,get(ENV,"TM_STATES","256")); INCLUDE=parse(Int,get(ENV,"TM_INCLUDE","128"))
x_train=[TMInput([parse(Bool,t) for t in split(l," ")]) for l in readlines("$TMP/X_train.txt")]
x_test =[TMInput([parse(Bool,t) for t in split(l," ")]) for l in readlines("$TMP/X_test.txt")]
y_train=[parse(Int,l) for l in readlines("$TMP/Y_train.txt")]
y_test =[parse(Int,l) for l in readlines("$TMP/Y_test.txt")]
n_bits=length(split(readlines("$TMP/X_train.txt")[1]," "))
Random.seed!(42)
tm=TMClassifier(x_train[1],y_train,C,T,S,L,LF,states_num=STATES,include_limit=INCLUDE)
if parse(Bool,get(ENV,"TM_EXCLUSIVE","false"))   # perara's built-in same-bit guard
    train!(tm,x_train,y_train,x_test,y_test,EP,shuffle=true,index=true,verbose=0,exclusive=true)
else
    train!(tm,x_train,y_train,x_test,y_test,EP,shuffle=true,index=true,verbose=0)
end
acc=accuracy(predict(tm,x_test,index=true),y_test)
getbit(M,j,b)=((M[((b-1)>>>6)+1,j]>>((b-1)&63))&UInt64(1))==UInt64(1)
imp=0
for ta in tm.clauses, (A,Nv) in ((ta.positive_included_literals,ta.positive_included_literals_inverted),(ta.negative_included_literals,ta.negative_included_literals_inverted))
    for j in 1:size(A,2), b in 1:n_bits; (getbit(A,j,b)&&getbit(Nv,j,b))&&(global imp+=1); end
end
rules=Dict{String,Any}("n_bits"=>n_bits,"n_classes"=>length(tm.classes),"classes"=>[Int(c) for c in tm.classes],
    "config"=>Dict("C"=>C,"T"=>T,"S"=>S,"L"=>L,"LF"=>LF,"EPOCHS"=>EP))
cr=Dict{String,Any}()
for k in eachindex(tm.classes)
    ta=tm.clauses[k]; npc=size(ta.positive_included_literals,2); pos=[]; neg=[]
    for i in 1:npc
        pin=[b-1 for b in 1:n_bits if getbit(ta.positive_included_literals,i,b)]
        pex=[b-1 for b in 1:n_bits if getbit(ta.positive_included_literals_inverted,i,b)]
        push!(pos,Dict("include"=>pin,"exclude"=>pex,"clamp"=>LF,"total"=>length(pin)+length(pex)))
        nin=[b-1 for b in 1:n_bits if getbit(ta.negative_included_literals,i,b)]
        nex=[b-1 for b in 1:n_bits if getbit(ta.negative_included_literals_inverted,i,b)]
        push!(neg,Dict("include"=>nin,"exclude"=>nex,"clamp"=>LF,"total"=>length(nin)+length(nex)))
    end
    cr[string(Int(tm.classes[k]))]=Dict("positive_clauses"=>pos,"negative_clauses"=>neg)
end
rules["class_rules"]=cr
open("$TMP/tm_rules.json","w") do io; JSON.print(io,rules); end
@printf("DONE acc=%.2f%% n_bits=%d classes=%d IMPOSSIBLE=%d\n",acc*100,n_bits,length(tm.classes),imp)
