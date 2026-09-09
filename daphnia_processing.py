import numpy as np
from functions import *
import pandas as pd

df = pd.read_csv('/home/jason/onedrive/data/daphnia_af.csv', na_values=["X"])

afs=df.loc[:, "P1":"P10"]
afs=[row.dropna().tolist() for _, row in afs.iterrows()]

#%%

pvals_freq=perm_freq(afs)

np.savetxt("daphnia_pvals.csv",pvals_freq,delimiter=",")

#%%

pvals_sign=perm_sign(afs,False)

pvals_freq=np.loadtxt("daphnia_pvals.csv")
np.savetxt("daphnia_pvals.csv", np.stack([pvals_freq,pvals_sign],axis=-1), delimiter=",")
