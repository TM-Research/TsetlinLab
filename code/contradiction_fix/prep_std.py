import sys, os, numpy as np, json
sys.path.insert(0,'/IoT/Paper_2/src')
from paper_2.data_loader import load_and_preprocess
from paper_2.booleanizers.standard import StandardBinarizer
loader, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT,exist_ok=True)
d=load_and_preprocess(loader)
Xtr,Xte,ytr,yte=d["X_train"],d["X_test"],d["y_train"],d["y_test"]
b=StandardBinarizer(max_bits_per_feature=25); b.fit(Xtr)
Xtrb=b.transform(Xtr); Xteb=b.transform(Xte)
np.savetxt(f'{OUT}/X_train.txt',np.asarray(Xtrb,dtype=int),fmt='%d')
np.savetxt(f'{OUT}/X_test.txt', np.asarray(Xteb,dtype=int),fmt='%d')
np.savetxt(f'{OUT}/Y_train.txt',np.asarray(ytr,dtype=int),fmt='%d')
np.savetxt(f'{OUT}/Y_test.txt', np.asarray(yte,dtype=int),fmt='%d')
feat=[]; thr=[]
for f,vals in enumerate(b.unique_values):
    for t in vals: feat.append(int(f)); thr.append(float(t))
json.dump({"feat_idx":feat,"thresh":thr,"n_bits":int(Xtrb.shape[1])}, open(f'{OUT}/bits.json','w'))
print(f"{loader} StandardBinarizer: bits {Xtrb.shape[1]} train {Xtrb.shape[0]} test {Xteb.shape[0]}")
