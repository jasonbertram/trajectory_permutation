import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd

dap=np.loadtxt('/home/jason/onedrive/data/daphnia_pvals.csv',delimiter=',')
dros=np.loadtxt('/home/jason/onedrive/data/drosophila_pvals.csv',delimiter=',')

#%%
####################################

fig, axs=plt.subplots(2,1,figsize=[3,4], constrained_layout=True)

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
#==========#==========#=============

raw = pd.read_csv('/home/jason/onedrive/data/daphnia_af.csv')
chr_ends=np.array([np.max(raw[raw['Chr']==_]['Position']) for _ in range(1,13)])

for _ in range(2,13):
    raw.loc[raw['Chr']==_,'Position'] += np.cumsum(chr_ends)[_-2] 

dap_pos=np.array(raw['Position'])
dap=dap[np.argsort(dap_pos)]
dap_pos=np.sort(dap_pos)


#%% daphnia manhattan
#==========#==========#=============

fig, axs=plt.subplots(2,1,figsize=[6,4], constrained_layout=True)
colors=['r','b']

n=100 #window size
cumsum_f=np.cumsum(dap,0)
ma=(cumsum_f[n:]-cumsum_f[:-n])/n

#plt.plot(dap_pos[:-n],ma[:,0])

random_win = np.array([np.mean(np.random.choice(dap[:,0],n)) for _ in range(10000)]) 
mu,sig = sp.stats.norm.fit(random_win)

col_array = np.array([colors[np.mod(np.sum(i-np.cumsum(chr_ends)>0),2)] for i in dap_pos[n::n]])
axs[0].scatter(dap_pos[n::n],-np.log10(sp.stats.norm.cdf(ma[::n,0],mu,sig)),c=col_array, s=0.5)

alpha_tests=0.05/len(ma[::n])
axs[0].axhline(-np.log10(alpha_tests),linestyle='--', color='k')
axs[0].set_title('Frequency permutation test (any detectable evolution)')
axs[0].set_ylabel('$-log(p)$', fontsize=14)

#axs[0].set_ylim([0,20])
axs[0].set_xlim([0,np.cumsum(chr_ends)[-1]])


random_win = np.array([np.mean(np.random.choice(dap[:,1],n)) for _ in range(10000)]) 
mu,sig = sp.stats.norm.fit(random_win)

axs[1].scatter(dap_pos[n::n],-np.log10(sp.stats.norm.cdf(ma[::n,1],mu,sig)),c=col_array, s=0.5)
axs[1].axhline(-np.log10(alpha_tests),linestyle='--', color='k')
axs[1].set_title('Sign permutation test (directional selection)')
axs[1].set_xlabel('Position', fontsize=14)
axs[1].set_ylabel('$-log(p)$', fontsize=14)

axs[1].set_ylim([0,100])
axs[1].set_xlim([0,np.cumsum(chr_ends)[-1]])

for ax, label in zip(axs.flat, 'ab'):
    ax.text(-0.1, 1.05, label, transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='bottom')

plt.savefig('daphnia_manhattan.pdf')

#%% drosophila data
#==========#==========#=============

raw = pd.read_table('/home/jason/onedrive/data/frequencies.tab')
chromosomes=raw['chr'].unique()
chr_ends=np.array([np.max(raw[raw['chr']==_]['pos']) for _ in chromosomes])

for _ in range(1,len(chromosomes)):
    raw.loc[raw['chr']==chromosomes[_],'pos'] += np.cumsum(chr_ends)[_-1] 

dros_pos=np.array(raw['pos'])

#%% drosophila manhattan
#==========#==========#=============
colors=['r','b']

for rep in range(1,11):
    fig, axs=plt.subplots(2,1,figsize=[6,4], constrained_layout=True)

    pvals_filtered=dros[~np.isnan(dros[:,rep-1])][:,[rep-1,rep-1+10]]
    pos_filtered=dros_pos[~np.isnan(dros[:,rep-1])]

    n=100 #window size
    cumsum_f=np.cumsum(pvals_filtered,0)
    ma=(cumsum_f[n:]-cumsum_f[:-n])/n

    random_win = np.array([np.mean(np.random.choice(pvals_filtered[:,0],n)) for _ in range(10000)]) 
    mu,sig = sp.stats.norm.fit(random_win)

    col_array = np.array([colors[np.mod(np.sum(i-np.cumsum(chr_ends)>0),2)] for i in pos_filtered[n::n]])
    axs[0].scatter(pos_filtered[n::n],-np.log10(sp.stats.norm.cdf(ma[::n,0],mu,sig)),c=col_array, s=0.5)

    alpha_tests=0.05/len(ma[::n])
    axs[0].axhline(-np.log10(alpha_tests),linestyle='--', color='k')
    axs[0].set_title('Frequency permutation test (any detectable evolution)')
    axs[0].set_ylabel('$-log(p)$', fontsize=14)

    #axs[0].set_ylim([0,20])
    axs[0].set_xlim([0,np.cumsum(chr_ends)[-1]])

    random_win = np.array([np.mean(np.random.choice(pvals_filtered[:,1],n)) for _ in range(10000)]) 
    mu,sig = sp.stats.norm.fit(random_win)

    axs[1].scatter(pos_filtered[n::n],-np.log10(sp.stats.norm.cdf(ma[::n,1],mu,sig)),c=col_array, s=0.5)
    axs[1].axhline(-np.log10(alpha_tests),linestyle='--', color='k')
    axs[1].set_title('Sign permutation test (directional selection)')
    axs[1].set_xlabel('Position', fontsize=14)
    axs[1].set_ylabel('$-log(p)$', fontsize=14)

    axs[1].set_ylim([0,20])
    axs[1].set_xlim([0,np.cumsum(chr_ends)[-1]])

    for ax, label in zip(axs.flat, 'ab'):
        ax.text(-0.1, 1.05, label, transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='bottom')


    plt.savefig('drosophila_manhattan'+str(rep)+'.pdf')
