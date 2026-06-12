import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd

data_f = np.loadtxt('daphniaGenome_freqPVals_byTraj_10kPerms.csv', delimiter=',')
data_s = np.loadtxt('daphniaGenome_signPVals_byTraj.csv', delimiter=',')

raw = pd.read_csv('/home/jason/onedrive/data/daphnia_af.csv')
chr_ends=np.array([np.max(raw[raw['Chr']==_]['Position']) for _ in range(1,13)])

for _ in range(2,13):
    raw.loc[raw['Chr']==_,'Position'] += np.cumsum(chr_ends)[_-2] 

#pos_dict={float(_):raw.iloc[_]['Position'] for _ in range(len(raw))}
data_f[:,0] = np.array([raw.iloc[int(_)]['Position'] for _ in data_f[:,0]])
data_f=data_f[np.argsort(data_f[:,0])]

data_s[:,0] = np.array([raw.iloc[int(_)]['Position'] for _ in data_s[:,0]])
data_s=data_s[np.argsort(data_s[:,0])]

#%%

fig, axs=plt.subplots(2,1,figsize=[8,4])

n=100
alpha_tests=0.05/len(ma[::n])

cumsum_f=np.cumsum(data_f[:,1])
ma=(cumsum_f[n:]-cumsum_f[:-n])/n
#axs[0].plot(data_f[n::,0],ma[:])
#axs[0].vlines(np.cumsum(chr_ends),0.4,0.7)

random_1000 = np.array([np.mean(np.random.choice(data_f[:,1],n)) for _ in range(10000)]) 
mu,sig = sp.stats.norm.fit(random_1000)

axs[0].plot(data_f[n::n,0],-np.log10(sp.stats.norm.cdf(ma[::n],mu,sig)),'.')
axs[0].vlines(np.cumsum(chr_ends),0,20,'k')
axs[0].axhline(-np.log10(alpha_tests),color='r')
axs[0].set_title('Frequency permutation test (any detectable evolution)')
axs[0].set_ylabel('$-log(p)$', fontsize=14)

axs[0].set_ylim([0,20])

cumsum_s=np.cumsum(data_s[:,2])
ma=(cumsum_s[n:]-cumsum_s[:-n])/n
#plt.plot(ma[:])

random_1000 = np.array([np.mean(np.random.choice(data_s[:,2],n)) for _ in range(10000)]) 
mu,sig = sp.stats.norm.fit(random_1000)

axs[1].plot(data_s[n::n,0],-np.log10(sp.stats.norm.cdf(ma[::n],mu,sig)),'.')
axs[1].vlines(np.cumsum(chr_ends),0,20,'k')
axs[1].axhline(-np.log10(alpha_tests),color='r')
axs[1].set_title('Sign permutation test (directional selection)')
axs[1].set_xlabel('Position', fontsize=14)
axs[1].set_ylabel('$-log(p)$', fontsize=14)

axs[1].set_ylim([0,20])

plt.tight_layout()

plt.savefig('manhattan.pdf')

#%%

fig, axs=plt.subplots(1,1,figsize=[8,4])

axs.hist(data_f[:,1])

axs.set_xlabel(r'$p$-value', fontsize=14)
axs.set_ylabel(r'Count', fontsize=14)

plt.tight_layout()
plt.savefig('pvalue.pdf')
