import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.stats import combine_pvalues
#plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.cividis.colors)

#sample size for permuting long trajectories
sample_size=10000

def roc(pvals1,pvals2,n):
    """Receiver operating characteristic"""
    L1=len(pvals1)
    L2=len(pvals2)
    return np.array([[np.sum(pvals1<=alph)/L1,np.sum(pvals2<=alph)/L2] for alph in np.linspace(0,1,n)])


def perm_freq(trajectories):
    p_vals=np.zeros(len(trajectories))
    T=len(trajectories[0])
    for i,p in enumerate(trajectories):
        if T>8: #do exact test for trajectories <= 8 points long
            perm_p=np.array([np.random.permutation(p) for _ in range(sample_size)])
            perm_p[0]=p
        else:
            perm_p=np.array([_ for _ in itertools.permutations(p)])
        dp=np.diff(perm_p)
        d_perm=np.mean(np.abs(dp),1) #average increment magnitude
        d_obs=np.mean(np.abs(np.diff(p)))
        p_vals[i]=np.sum((d_perm-d_obs)<1e-6)/len(d_perm) #unusually small
    return p_vals

def perm_incr(trajectories):
    p_vals=np.zeros(len(trajectories))
    T=len(trajectories[0])-1
    for i,p in enumerate(trajectories):
        dp=np.diff(p)
        if T>8: #do exact test for trajectories <= 9 points long
            perm_dp=np.array([np.random.permutation(dp) for _ in range(sample_size)])
            perm_dp[0]=dp
        else:
            perm_dp=np.array([_ for _ in itertools.permutations(dp)])
        perm_p=np.cumsum(perm_dp,1)

        p_mean=np.mean(perm_p,0)
        abs_devs=np.abs(perm_p-p_mean)
        d_perm=np.mean(abs_devs,1)
        d_obs=np.mean(np.abs(p[1:]-p[0]-p_mean))

        p_vals[i]=np.sum((d_perm-d_obs)<1e-6)/len(d_perm) #unusually small
    return p_vals

def perm_sign(trajectories):
    p_vals=np.zeros(len(trajectories))
    sgn_prm=np.array(list(itertools.product([-1,1], repeat=len(trajectories)))) #sign permutation matrix
    for i,p in enumerate(trajectories):
        dp=np.diff(p)
        d_perm=np.sum(sgn_prm*dp, axis=1)
        p_vals[i]=np.sum((np.abs(d_perm)-np.abs(p[0]-p[-1]))<1e-6)/len(d_perm)
    return p_vals

def gen_traj(N,s,s_std,p0,sig,skip,num_mes,numtraj):
    """ Return numtraj simulated Wright-Fisher trajectories 
    N popsize
    s sel coeff
    s_std standard dev of s
    T time steps
    p0 init
    sig measurement std dev
    skip measurement interval
    """

    T=skip*num_mes
    p=p0*np.ones([numtraj,T])
    p[:,0]=0.5

    for t in range(1,T):
        svec=s*np.ones(numtraj)+np.random.normal(0,s_std,size=numtraj)
        p[:,t]=np.random.binomial(N,p[:,t-1]+svec*p[:,t-1]*(1-p[:,t-1]))/N

    #measurement 
    p=p[:,0:T:skip]+np.random.normal(0,sig,size=[numtraj,num_mes])

    return p

#%%
#frequency permutation
##################################

num_traj=1000
p0=0.5
skip=4
num_mes=10
sig=0.01
#N,s,s_std,sig
scenarios={'Drift':[int(1e3),0,0],
           'Directional weak':[int(1e10),0.001,0],
           'Directional strong':[int(1e10),0.005,0],
           'Fluctuating strong':[int(1e10),0,0.05],
           'Fluctuating weak':[int(1e10),0,0.01]}

fig, axs=plt.subplots(2,1,figsize=[3,6])

N,s,s_std=[int(1e10),0,0]
p_vals_null=perm_freq(gen_traj(N,s,s_std,p0,sig,skip,num_mes,num_traj))
for _ in scenarios:
    N,s,s_std=scenarios[_]
    p_vals=perm_freq(gen_traj(N,s,s_std,p0,sig,skip,num_mes,num_traj))
    roc1=roc(p_vals_null,p_vals,100)
    axs[0].plot(roc1[:,0],roc1[:,1],label=_)

    #alph=0.05
    #L=len(p_vals)
    #fpr,tpr=([np.sum(p_vals_null<alph)/L,np.sum(p_vals<alph)/L])
    #print(_,np.array([[tpr,1-tpr],[fpr,1-fpr]]))

axs[0].legend()
axs[0].plot(np.linspace(0,1),np.linspace(0,1),'k--')
axs[0].set_ylabel('True positive rate')
axs[0].set_title(r'$\sigma=0.01$')
#axs[0].set_xlabel('False positive rate')

sig=0.03
N,s,s_std=[int(1e10),0,0]
p_vals_null=perm_freq(gen_traj(N,s,s_std,p0,sig,skip,num_mes,num_traj))
for _ in scenarios:
    N,s,s_std=scenarios[_]
    p_vals=perm_freq(gen_traj(N,s,s_std,p0,sig,skip,num_mes,num_traj))
    roc1=roc(p_vals_null,p_vals,100)
    axs[1].plot(roc1[:,0],roc1[:,1],label=_)

#axs[1].legend()
axs[1].plot(np.linspace(0,1),np.linspace(0,1),'k--')
axs[1].set_ylabel('True positive rate')
axs[1].set_xlabel('False positive rate')
axs[1].set_title(r'$\sigma=0.03$')

plt.savefig('roc_freq.pdf', bbox_inches='tight')

#print(combine_pvalues(p_vals[fixed_arg_1]))
#print(combine_pvalues(p_vals_complement[fixed_arg_1]))
#print(combine_pvalues(p_vals_twoside[fixed_arg_1]))

#%%
#increment order permutation
#########################

p_vals_null=perm_incr(gen_traj(N,s,s_std,p0,sig,skip,num_mes,num_traj))


#print(combine_pvalues(p_vals[fixed_arg_1]))
#print(combine_pvalues(p_vals_complement[fixed_arg_1]))
#print(combine_pvalues(p_vals_twoside[fixed_arg_1]))


#%%
#increment sign permutation
##################################




#=========================================================================
#%%
#pvalue merge functions with dependence but assuming exchangeability
#Gasparin et al 2025 PNAS.

def harm_mean(x):
    return len(x)/np.sum(1/x)

def harm_mean_ex(x):
    K=len(x)
    #return (np.log(K)+np.log(np.log(K))+2)*np.min([harm_mean(x[:i]) for i in range(1,K)])
    return np.min([harm_mean(x[:i]) for i in range(1,K)])

print('Bonferroni')
print(num_traj*np.min(p_vals))
print(num_traj*np.min(p_vals_complement))

print('Arithmetic')
print(2*np.mean(p_vals))
print(2*np.mean(p_vals_complement))

print('Arithmetic exchangeable')
print(2*np.min([np.mean(p_vals[:i]) for i in range (1,num_traj)]))
print(2*np.min([np.mean(p_vals_complement[:i]) for i in range (1,num_traj)]))

print('Geometric')
print(np.e*np.exp(np.mean(np.log(p_vals))))
print(np.e*np.exp(np.mean(np.log(p_vals_complement))))

print('Geometric exchangeable')
print(np.e*np.min([np.exp(np.mean(np.log(p_vals[:i]))) for i in range (1,num_traj)]))
print(np.e*np.min([np.exp(np.mean(np.log(p_vals_complement[:i]))) for i in range (1,num_traj)]))

print('Harmonic exchangeable')
T=(np.log(num_traj)+np.log(np.log(num_traj))+2)
print(T*harm_mean_ex(p_vals))
print(T*harm_mean_ex(p_vals_complement))
