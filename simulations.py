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


def power(pvals,alph):
    """Receiver operating characteristic"""
    return np.sum(pvals<=alph)/len(pvals)

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

def perm_incr(trajectories,transform):
    p_vals=np.zeros(len(trajectories))
    T=len(trajectories[0])-1
    for i,p in enumerate(trajectories):
        
        dp=np.diff(p)
        if transform==True:
            dp=dp/(p[:-1]*(1-p[:-1]))

        if T>8: #do exact test for trajectories <= 9 points long
            perm_dp=np.array([np.random.permutation(dp) for _ in range(sample_size)])
            perm_dp[0]=dp
        else:
            perm_dp=np.array([_ for _ in itertools.permutations(dp)])

        if transform==True:
            perm_p=reconstruct_transformed(perm_dp,p[0]*np.ones(len(perm_dp)))
        else:
            perm_p=np.cumsum(perm_dp,1)

        p_mean=np.mean(perm_p,0)
        abs_devs=np.abs(perm_p-p_mean)
        d_perm=np.mean(abs_devs,1)
        if transform==True:
            d_obs=np.mean(np.abs(p-p_mean))
        else:
            d_obs=np.mean(np.abs(p[1:]-p[0]-p_mean))

        p_vals[i]=np.sum((d_perm-d_obs)<1e-10)/len(d_perm) #unusually small
    return p_vals

def reconstruct_transformed(dp,p0):
    dims=np.shape(dp)
    T=dims[1]+1
    p=np.zeros([dims[0],T])
    p[:,0]=p0

    for i in range(1,T):
        p[:,i]=p[:,i-1]+dp[:,i-1]*(p[:,i-1]*(1-p[:,i-1]))

    return p

def perm_sign(trajectories):
    p_vals=np.zeros(len(trajectories))
    T=len(trajectories[0])-1
    #sign permutation matrix
    if T>13:
        sgn_prm=np.array([2*np.random.randint(2,size=T)-1 for _ in range(sample_size) ])
        sgn_prm[0]=np.ones(T)
    else:
        sgn_prm=np.array(list(itertools.product([-1,1], repeat=T))) 

    for i,p in enumerate(trajectories):
        dp=np.diff(p)
        d_perm=np.sum(sgn_prm*dp, axis=1)
        #p_vals[i]=np.sum((np.abs(d_perm)-np.abs(p[0]-p[-1]))<1e-10)/len(d_perm)
        p_vals[i]=np.sum((np.abs(d_perm)-np.abs(p[0]-p[-1]))>-1e-10)/len(d_perm)
    return p_vals


def gen_traj(N,s,s_std,p0,n_s,skip,num_mes,numtraj):
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

    for t in range(1,T):
        svec=s*np.ones(numtraj)+np.random.normal(0,s_std,size=numtraj)
        p[:,t]=np.random.binomial(N,p[:,t-1]+svec*p[:,t-1]*(1-p[:,t-1]))/N

    #sampling error 
    p=np.random.binomial(n_s,p[:,0:T:skip])

    return p

#%%
#Parameters
##################################

num_traj=1000
p0=0.5

#N,s,s_std
scenarios={'Drift':[int(1e3),0,0],
           'Directional weak':[int(1e10),0.001,0],
           'Directional strong':[int(1e10),0.005,0],
           'Fluctuating strong':[int(1e10),0,0.05],
           'Fluctuating weak':[int(1e10),0,0.01]}

#%%
#frequency permutation
##################################

#roc

fig, axs=plt.subplots(2,1,figsize=[3,6])

num_mes=10
skip=10

n_s=1000
for _ in scenarios:
    N,s,s_std=scenarios[_]
    p_vals=perm_freq(gen_traj(N,s,s_std,p0,n_s,skip,num_mes,num_traj))
    roc1=roc(np.linspace(0,1,num_traj), p_vals, 100)
    axs[0].plot(roc1[:,0],roc1[:,1],label=_)


axs[0].legend(fontsize=7)
axs[0].plot(np.linspace(0,1),np.linspace(0,1),'k--')
axs[0].set_ylabel('True positive rate')
axs[0].set_title(r'$n=1000$',fontsize=10)
#axs[0].set_xlabel('False positive rate')

n_s=100
for _ in scenarios:
    N,s,s_std=scenarios[_]
    p_vals=perm_freq(gen_traj(N,s,s_std,p0,n_s,skip,num_mes,num_traj))
    roc1=roc(np.linspace(0,1,num_traj), p_vals, 100)
    axs[1].plot(roc1[:,0],roc1[:,1],label=_)

axs[1].plot(np.linspace(0,1),np.linspace(0,1),'k--')
axs[1].set_ylabel('True positive rate')
axs[1].set_xlabel('False positive rate')
axs[1].set_title(r'$n=100$',fontsize=10)

plt.savefig('roc_freq.pdf', bbox_inches='tight')


#%%

num_traj=1000

#Power vs N
fig, axs=plt.subplots(2,3,figsize=[6,4])

s=0
s_std=0
N_vec=np.array([10**6, 10**5, 10**4, 10**3])
num_mes_vec=[10,50]

#No measurement error
n_s=10**10
#short trajectory
for num_mes in num_mes_vec:
    skip=int(100/num_mes)
    axs[0,0].semilogx(N_vec,
                [power(perm_freq(gen_traj(N,s,s_std,p0,n_s,skip,num_mes,num_traj)),0.05) for N in N_vec]
                 ,label=num_mes)

axs[0,0].legend(fontsize=8)
axs[0,0].set_ylabel('Power (short trajectory)')
axs[0,0].set_title('No error',fontsize=10)

#long trajectory
for num_mes in num_mes_vec:
    skip=int(1000/num_mes)
    axs[1,0].semilogx(N_vec,
                [power(perm_freq(gen_traj(N,s,s_std,p0,n_s,skip,num_mes,num_traj)),0.05) for N in N_vec]
                 ,label=num_mes)

axs[1,0].set_ylabel('Power (long trajectory)')

#n=1000
n_s=1000
#short trajectory
for num_mes in num_mes_vec:
    skip=int(100/num_mes)
    axs[0,1].semilogx(N_vec,
                [power(perm_freq(gen_traj(N,s,s_std,p0,n_s,skip,num_mes,num_traj)),0.05) for N in N_vec]
                 ,label='M'+str(num_mes))

axs[0,1].set_title(r'$n=1000$',fontsize=10)
axs[0,1].set_yticklabels('')

#long trajectory
for num_mes in num_mes_vec:
    skip=int(1000/num_mes)
    axs[1,1].semilogx(N_vec,
                [power(perm_freq(gen_traj(N,s,s_std,p0,n_s,skip,num_mes,num_traj)),0.05) for N in N_vec]
                 ,label=num_mes)

axs[1,1].set_xlabel(r'$N$')
axs[1,1].set_yticklabels('')

#n=100
n_s=100
#short trajectory
for num_mes in num_mes_vec:
    skip=int(100/num_mes)
    axs[0,2].semilogx(N_vec,
                [power(perm_freq(gen_traj(N,s,s_std,p0,n_s,skip,num_mes,num_traj)),0.05) for N in N_vec]
                 ,label=num_mes)

axs[0,2].set_title(r'$n=100$',fontsize=10)
axs[0,2].set_yticklabels('')

#long trajectory
for num_mes in num_mes_vec:
    skip=int(1000/num_mes)
    axs[1,2].semilogx(N_vec,
                [power(perm_freq(gen_traj(N,s,s_std,p0,n_s,skip,num_mes,num_traj)),0.05) for N in N_vec]
                 ,label=num_mes)

axs[1,2].set_yticklabels('')

for _ in axs.flatten(): _.set_ylim([0,1.01])
for _ in axs.flatten():_.set_xticks(N_vec)

plt.savefig('power_N_freq.pdf', bbox_inches='tight')

#%%
#increment order permutation
#########################

#transformation check
fig, ax=plt.subplots(1,1,figsize=[3,3])

N,s,s_std=[int(1e3),0,0]
sig=0
num_traj=10000
sim=gen_traj(N,s,s_std,p0,sig,skip,num_mes,num_traj)

p_vals_null=perm_incr(sim,False)
roc1=roc(p_vals_null,np.linspace(0,1,num_traj),100)
ax.plot(roc1[:,0],roc1[:,1],label=r'Untransformed $\sigma_s^2=0$')

p_vals_null=perm_incr(sim,True)
roc1=roc(p_vals_null,np.linspace(0,1,num_traj),100)
ax.plot(roc1[:,0],roc1[:,1],label=r'Transformed $\sigma_s^2=0$')

s_std=0.05
sim=gen_traj(N,s,s_std,p0,sig,skip,num_mes,num_traj)

p_vals_null=perm_incr(sim,False)
roc1=roc(p_vals_null,np.linspace(0,1,num_traj),100)
ax.plot(roc1[:,0],roc1[:,1],label=r'Untransformed $\sigma_s^2=0.05$')

p_vals_null=perm_incr(sim,True)
roc1=roc(p_vals_null,np.linspace(0,1,num_traj),100)
ax.plot(roc1[:,0],roc1[:,1],label=r'Transformed $\sigma_s^2=0.05$')
ax.plot(np.linspace(0,1),np.linspace(0,1),'k--')

ax.set_ylabel('Rate of positives')
ax.set_xlabel('Significance level')
ax.legend(fontsize=7)

plt.savefig('roc_inc_trans.pdf', bbox_inches='tight')

#%%
#increment sign permutation
##################################

#Comparison with Feder et al. 2014 
fig, axs=plt.subplots(2,3,figsize=[6,4])

s_std=0
N=10**4
s_vec=np.array([1,2,5,10,20,50,100])/N

#No measurement error
n_s=10**10
#short trajectory
for num_mes in [10,20,50]:
    skip=int(100/num_mes)
    axs[0,0].plot(N*s_vec,
                [power(perm_sign(gen_traj(N,s,s_std,p0,n_s,skip,num_mes,num_traj)),0.05) for s in s_vec]
                 ,label=num_mes)

axs[0,0].legend(fontsize=8)
axs[0,0].set_ylabel('Power (short trajectory)')
axs[0,0].set_title('No error',fontsize=10)
axs[0,0].set_xticklabels('')

#long trajectory
for num_mes in [10,20,50]:
    skip=int(1000/num_mes)
    axs[1,0].plot(N*s_vec,
                [power(perm_sign(gen_traj(N,s,s_std,p0,n_s,skip,num_mes,num_traj)),0.05) for s in s_vec]
                 ,label=num_mes)

axs[1,0].set_ylabel('Power (long trajectory)')

#n=1000
n_s=1000
#short trajectory
for num_mes in [10,20,50]:
    skip=int(100/num_mes)
    axs[0,1].plot(N*s_vec,
                [power(perm_sign(gen_traj(N,s,s_std,p0,n_s,skip,num_mes,num_traj)),0.05) for s in s_vec]
                 ,label='M'+str(num_mes))

axs[0,1].set_title(r'$n=1000$',fontsize=10)
axs[0,1].set_xticklabels('')
axs[0,1].set_yticklabels('')

#long trajectory
for num_mes in [10,20,50]:
    skip=int(1000/num_mes)
    axs[1,1].plot(N*s_vec,
                [power(perm_sign(gen_traj(N,s,s_std,p0,n_s,skip,num_mes,num_traj)),0.05) for s in s_vec]
                 ,label=num_mes)

axs[1,1].set_xlabel(r'$Ns$')
axs[1,1].set_yticklabels('')

#n=100
n_s=100
#short trajectory
for num_mes in [10,20,50]:
    skip=int(100/num_mes)
    axs[0,2].plot(N*s_vec,
                [power(perm_sign(gen_traj(N,s,s_std,p0,n_s,skip,num_mes,num_traj)),0.05) for s in s_vec]
                 ,label=num_mes)

axs[0,2].set_title(r'$n=100$',fontsize=10)
axs[0,2].set_xticklabels('')
axs[0,2].set_yticklabels('')

#long trajectory
for num_mes in [10,20,50]:
    skip=int(1000/num_mes)
    axs[1,2].plot(N*s_vec,
                [power(perm_sign(gen_traj(N,s,s_std,p0,n_s,skip,num_mes,num_traj)),0.05) for s in s_vec]
                 ,label=num_mes)

axs[1,2].set_yticklabels('')

for _ in axs.flatten(): _.set_ylim([0,1])

plt.savefig('power_NS_sign.pdf', bbox_inches='tight')




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

#%%
#============================================
#N=np.arange(100,1000) 
N=1000
for init in np.arange(1,500): #[1,10,100]:
    p_n=np.sum(binom.pmf(np.arange(init),N,init/N))
    p_p=np.sum(binom.pmf(np.arange(init+1,N+1),N,init/N))
    plt.plot(init,p_n/p_p,'.')

plt.hlines(1,0,500)
