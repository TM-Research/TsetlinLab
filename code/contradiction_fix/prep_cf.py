import sys, os, numpy as np
sys.path.insert(0,'/IoT/Paper_2/src')
from paper_2.data_loader import load_and_preprocess
from paper_2.booleanizers.glade import GLADEBooleanizer
loader, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT,exist_ok=True)
d=load_and_preprocess(loader)
Xtr,Xte,ytr,yte=d["X_train"],d["X_test"],d["y_train"],d["y_test"]
g=GLADEBooleanizer(n_bins=15); g.fit(Xtr)
Xtrb=g.transform(Xtr); Xteb=g.transform(Xte)
np.savetxt(f'{OUT}/X_train.txt',np.asarray(Xtrb,dtype=int),fmt='%d')
np.savetxt(f'{OUT}/X_test.txt', np.asarray(Xteb,dtype=int),fmt='%d')
np.savetxt(f'{OUT}/Y_train.txt',np.asarray(ytr,dtype=int),fmt='%d')
np.savetxt(f'{OUT}/Y_test.txt', np.asarray(yte,dtype=int),fmt='%d')
g.save_json(f'{OUT}/glade.json')
print(f"{loader}: bits {Xtrb.shape[1]} train {Xtrb.shape[0]} test {Xteb.shape[0]} classes {len(set(int(v) for v in ytr))}")
