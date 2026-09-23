import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd

dros=np.loadtxt('drosophila_pvals.csv',skiprows=1,delimiter=',')
dap=np.loadtxt('daphnia_pvals.csv',skiprows=1,delimiter=',')

#%%

fig, axs=plt.subplots(2,1,figsize=[3,4],constrained_layout=True)

x1, x2, y1, y2 = 0, 5e-4, -1e-4, 5e-4 
axins = axs[0].inset_axes([0.12, 0.6, 0.3, 0.3],xlim=(x1, x2), ylim=(y1, y2))  
axins.set_xticks([5e-4])
axins.set_yticks([5e-4])
axins.set_xticklabels(['5e-4'],fontsize=5)
axins.set_yticklabels(['5e-4'],fontsize=5)

for _ in range(10):
    pvals=dros[~np.isnan(dros[:,_]),_]
    #pvals=-np.log10(dros[~np.isnan(dros[:,_]),_])
    line1, =axs[0].plot(np.arange(len(pvals))/len(pvals),np.sort(pvals),c='C0',lw=0.5)
    axins.plot(np.arange(len(pvals))/len(pvals),np.sort(pvals),c='C0')
    axins.axhline(0.05/len(pvals),c='k',lw=0.5)

pvals=(dap[:,0])
line2, =axs[0].plot(np.arange(len(pvals))/len(pvals),np.sort(pvals),c='C1')
axins.plot(np.arange(len(pvals))/len(pvals),np.sort(pvals),c='C1')
line4, = axs[0].plot([0,1], [0.05/len(pvals),0.05/len(pvals)],c='k',lw=0.5)
axins.axhline(0.05/len(pvals),c='k',lw=0.5)

line3, =axs[0].plot(np.linspace(1e-8,1,1000),(np.linspace(1e-8,1,1000)),'k--',lw=0.5)
axins.plot(np.linspace(1e-8,1,1000),(np.linspace(1e-8,1,1000)),'k--',lw=0.5)

axs[0].indicate_inset_zoom(axins, edgecolor="black")
axs[0].legend([line1,line2,line3,line4],['D. melanogaster', 'D. pulex', 'null','Bonferroni'], loc='lower right',fontsize=5)

axs[0].set_ylabel(r'$p$-value')

for ax, label in zip(axs.flat, 'abcdef'):
    ax.text(-0.1, 1.05, label, transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='bottom')


x1, x2, y1, y2 = -1e-4, 5e-3, -1e-2, 5e-2 
axins = axs[1].inset_axes([0.65, 0.2, 0.3, 0.3],xlim=(x1, x2), ylim=(y1, y2))  
axins.set_xticks([x2])
axins.set_yticks([y2])
axins.set_xticklabels(['5e-3'],fontsize=5)
axins.set_yticklabels(['5e-2'],fontsize=5)

for _ in range(11,20):
    pvals=dros[~np.isnan(dros[:,_]),_]
    #pvals=-np.log10(dros[~np.isnan(dros[:,_]),_])
    line1, =axs[1].plot(np.arange(len(pvals))/len(pvals),np.sort(pvals),c='C0',lw=0.5)
    axins.plot(np.arange(len(pvals))/len(pvals),np.sort(pvals),c='C0')
    axins.axhline(0.05/len(pvals),c='k',lw=0.5)

pvals=(dap[:,1])
axs[1].plot(np.arange(len(pvals))/len(pvals),np.sort(pvals),c='C1')
axins.plot(np.arange(len(pvals))/len(pvals),np.sort(pvals),c='C1')
axs[1].plot([0,1], [0.05/len(pvals),0.05/len(pvals)],c='k',lw=0.5)
axins.axhline(0.05/len(pvals),c='k',lw=0.5)

axs[1].plot(np.linspace(1e-8,1,1000),(np.linspace(1e-8,1,1000)),'k--',lw=0.5)
axins.plot(np.linspace(1e-8,1,1000),(np.linspace(1e-8,1,1000)),'k--',lw=0.5)

axs[1].indicate_inset_zoom(axins, edgecolor="black")
axs[1].set_xlabel(r'Rank (normalized)')
axs[1].set_ylabel(r'$p$-value')

for ax, label in zip(axs.flat, 'abcdef'):
    ax.text(-0.1, 1.05, label, transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='bottom')


plt.savefig('pvalues.pdf', bbox_inches='tight')

#%% daphnia data
"""
#==========#==========#=============

data_f = np.loadtxt('/home/jason/onedrive/data/daphniaGenome_freqPVals_byTraj_10kPerms.csv', delimiter=',')
data_s = np.loadtxt('/home/jason/onedrive/data/daphniaGenome_signPVals_byTraj.csv', delimiter=',')

raw = pd.read_csv('/home/jason/onedrive/data/daphnia_af.csv')
chr_ends=np.array([np.max(raw[raw['Chr']==_]['Position']) for _ in range(1,13)])

for _ in range(2,13):
    raw.loc[raw['Chr']==_,'Position'] += np.cumsum(chr_ends)[_-2] 

#pos_dict={float(_):raw.iloc[_]['Position'] for _ in range(len(raw))}
data_f[:,0] = np.array([raw.iloc[int(_)]['Position'] for _ in data_f[:,0]])
data_f=data_f[np.argsort(data_f[:,0])]

data_s[:,0] = np.array([raw.iloc[int(_)]['Position'] for _ in data_s[:,0]])
data_s=data_s[np.argsort(data_s[:,0])]

#%% daphnia manhattan
#==========#==========#=============

fig, axs=plt.subplots(2,1,figsize=[6,4])
colors=['r','b']

n=100 #window size

cumsum_f=np.cumsum(data_f[:,1])
ma=(cumsum_f[n:]-cumsum_f[:-n])/n

#axs[0].plot(data_f[n::,0],ma[:])
#axs[0].vlines(np.cumsum(chr_ends),0.4,0.7)

random_1000 = np.array([np.mean(np.random.choice(data_f[:,1],n)) for _ in range(10000)]) 
mu,sig = sp.stats.norm.fit(random_1000)

col_array = np.array([colors[np.mod(np.sum(i-np.cumsum(chr_ends)>0),2)] for i in data_f[n::n,0] ])
axs[0].scatter(data_f[n::n,0],-np.log10(sp.stats.norm.cdf(ma[::n],mu,sig)),c=col_array, s=0.5)

alpha_tests=0.05/len(ma[::n])
axs[0].axhline(-np.log10(alpha_tests),linestyle='--', color='k')
axs[0].set_title('Frequency permutation test (any detectable evolution)')
axs[0].set_ylabel('$-log(p)$', fontsize=14)

axs[0].set_ylim([0,20])
axs[0].set_xlim([0,np.cumsum(chr_ends)[-1]])

cumsum_s=np.cumsum(data_s[:,2])
ma=(cumsum_s[n:]-cumsum_s[:-n])/n
#plt.plot(ma[:])

random_1000 = np.array([np.mean(np.random.choice(data_s[:,2],n)) for _ in range(10000)]) 
mu,sig = sp.stats.norm.fit(random_1000)

col_array = np.array([colors[np.mod(np.sum(i-np.cumsum(chr_ends)>0),2)] for i in data_s[n::n,0] ])
axs[1].scatter(data_s[n::n,0],-np.log10(sp.stats.norm.cdf(ma[::n],mu,sig)),c=col_array, s=0.5)
axs[1].axhline(-np.log10(alpha_tests),linestyle='--', color='k')
axs[1].set_title('Sign permutation test (directional selection)')
axs[1].set_xlabel('Position', fontsize=14)
axs[1].set_ylabel('$-log(p)$', fontsize=14)

axs[1].set_ylim([0,20])
axs[1].set_xlim([0,np.cumsum(chr_ends)[-1]])

plt.tight_layout()

plt.savefig('daphnia_manhattan.pdf')

#%% daphnia other
#==========#==========#=============

fig, axs=plt.subplots(1,1,figsize=[8,4])

axs.hist(data_f[:,1])

axs.set_xlabel(r'$p$-value', fontsize=14)
axs.set_ylabel(r'Count', fontsize=14)

plt.tight_layout()
plt.savefig('pvalue.pdf')

#%% drosophila data
#==========#==========#=============

data_f = np.loadtxt('/home/jason/onedrive/data/drosophilaGenome_freqPVals_byTraj_10kPerms.csv', delimiter=',')
data_s = np.loadtxt('/home/jason/onedrive/data/drosophilaGenome_signPVals_byTraj.csv', delimiter=',')

raw = pd.read_table('/home/jason/onedrive/data/frequencies.tab')
chromosomes=raw['chr'].unique()
chr_ends=np.array([np.max(raw[raw['chr']==_]['pos']) for _ in chromosomes])

for _ in range(1,len(chromosomes)):
    raw.loc[raw['chr']==chromosomes[_],'pos'] += np.cumsum(chr_ends)[_-1] 

data_f[:,0] = np.array([raw.iloc[int(_)]['pos'] for _ in data_f[:,0]])
data_f=data_f[np.argsort(data_f[:,0])]

data_s[:,0] = np.array([raw.iloc[int(_)]['pos'] for _ in data_s[:,0]])
data_s=data_s[np.argsort(data_s[:,0])]

#%% drosophila manhattan
#==========#==========#=============

fig, axs=plt.subplots(2,1,figsize=[6,4])
colors=['r','b']

n=100 #window size

cumsum_f=np.cumsum(data_f[:,1])
ma=(cumsum_f[n:]-cumsum_f[:-n])/n

#axs[0].plot(data_f[n::,0],ma[:])
#axs[0].vlines(np.cumsum(chr_ends),0.4,0.7)

random_1000 = np.array([np.mean(np.random.choice(data_f[:,1],n)) for _ in range(10000)]) 
mu,sig = sp.stats.norm.fit(random_1000)

col_array = np.array([colors[np.mod(np.sum(i-np.cumsum(chr_ends)>0),2)] for i in data_f[n::n,0] ])
axs[0].scatter(data_f[n::n,0],-np.log10(sp.stats.norm.cdf(ma[::n],mu,sig)),c=col_array, s=0.5)

alpha_tests=0.05/len(ma[::n])
axs[0].axhline(-np.log10(alpha_tests),linestyle='--', color='k')
axs[0].set_title('Frequency permutation test (any detectable evolution)')
axs[0].set_ylabel('$-log(p)$', fontsize=14)

axs[0].set_ylim([0,20])
axs[0].set_xlim([0,np.cumsum(chr_ends)[-1]])

cumsum_s=np.cumsum(data_s[:,2])
ma=(cumsum_s[n:]-cumsum_s[:-n])/n
#plt.plot(ma[:])

random_1000 = np.array([np.mean(np.random.choice(data_s[:,2],n)) for _ in range(10000)]) 
mu,sig = sp.stats.norm.fit(random_1000)

col_array = np.array([colors[np.mod(np.sum(i-np.cumsum(chr_ends)>0),2)] for i in data_s[n::n,0] ])
axs[1].scatter(data_s[n::n,0],-np.log10(sp.stats.norm.cdf(ma[::n],mu,sig)),c=col_array, s=0.5)
axs[1].axhline(-np.log10(alpha_tests),linestyle='--', color='k')
axs[1].set_title('Sign permutation test (directional selection)')
axs[1].set_xlabel('Position', fontsize=14)
axs[1].set_ylabel('$-log(p)$', fontsize=14)

axs[1].set_ylim([0,20])
axs[1].set_xlim([0,np.cumsum(chr_ends)[-1]])

plt.tight_layout()

plt.savefig('drosophila_manhattan.pdf')
"""
