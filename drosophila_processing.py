import numpy as np
from functions import *
import pandas as pd

pvals_freq=[]
pvals_sign=[]
for rep in range(1,11):
    data=np.transpose(np.loadtxt('/home/jason/git/trajectory_permutation/abby_files/drosophila/drosChr1_R'+str(rep)))
    for chrom in range(2,7):
        data = np.concatenate(
            (data,
            np.transpose(np.loadtxt('/home/jason/git/trajectory_permutation/abby_files/drosophila/drosChr'+str(chrom)+'_R'+str(rep)))
            ))

    #filter out fixation as missing data
    afs=[_[(_>0) & (_<1)] for _ in data]
    pvals_freq.append(perm_freq(afs))
    pvals_sign.append(perm_sign(afs,False))

np.savetxt("drosophila_pvals.csv", np.transpose(np.concatenate([np.array(pvals_freq),np.array(pvals_sign)])), delimiter=",")
