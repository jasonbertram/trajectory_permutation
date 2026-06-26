import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.stats import combine_pvalues
from scipy.stats import binom
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

def perm_incr(trajectories,transform,small):
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

        if small:
            p_vals[i]=np.sum((d_perm-d_obs)<1e-10)/len(d_perm) #unusually small
        else:
            p_vals[i]=np.sum((d_perm-d_obs)>-1e-10)/len(d_perm) #unusually large

    return p_vals

def reconstruct_transformed(dp,p0):
    dims=np.shape(dp)
    T=dims[1]+1
    p=np.zeros([dims[0],T])
    p[:,0]=p0

    for i in range(1,T):
        p[:,i]=p[:,i-1]+dp[:,i-1]*(p[:,i-1]*(1-p[:,i-1]))

    return p

def perm_sign(trajectories,small):
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

        if small:
            p_vals[i]=np.sum((np.abs(d_perm)-np.abs(p[0]-p[-1]))<1e-10)/len(d_perm)
        else:
            p_vals[i]=np.sum((np.abs(d_perm)-np.abs(p[0]-p[-1]))>-1e-10)/len(d_perm)

    return p_vals

def binomial_test(trajectories):
        dp=np.diff(trajectories)
        n=np.sum(np.sign(dp)==1, axis=1)

        #return 2*np.min([binom.cdf(n,len(trajectories[0]),0.5), 1-binom.cdf(n-1,len(trajectories[0]),0.5)],0)
        return 1-binom.cdf(n-1,len(trajectories[0]),0.5)


def gen_traj(N,s,s_std,inhomog_err,p0,n_s,skip,num_mes,numtraj):
    """ Return numtraj simulated Wright-Fisher trajectories 
    N popsize
    s sel coeff
    s_std standard dev of s
    p0 init p 
    sig measurement std dev
    skip measurement interval
    num_mes number of measurements
    n_s measurement sample size (binomial error model)
    """

    T=skip*num_mes
    p=p0*np.ones([numtraj,T])

    if np.isscalar(s):
        s=s*np.ones(T)
    elif not isinstance(s, np.ndarray):
        print("s must be an array or a scalar")

    for t in range(1,T):
        svec=s[t-1]*np.ones(numtraj)+np.random.normal(0,s_std,size=numtraj)
        p[:,t]=np.random.binomial(N,p[:,t-1]+svec*p[:,t-1]*(1-p[:,t-1]))/N

    #sampling error 
    if inhomog_err:
        sample_sizes=np.random.exponential(n_s,np.int64(T/skip))
        p=np.random.binomial(sample_sizes,p[:,0:T:skip])/n_s
    else:
        p=np.random.binomial(n_s,p[:,0:T:skip])/n_s

    return p

#%% frequency permutation roc
##################################

#N,s,s_std,inhomog_err
scenarios={r'Drift $N=10^3$':[int(1e3),0,0,0],
           'Directional $s=10^{-3}$':[int(1e10),0.001,0,0],
           'Directional $s=10^{-2}$':[int(1e10),0.01,0,0],
           'Fluctuating $\sigma^2=10^{-3}$':[int(1e10),0,0.001,0],
           'Fluctuating $\sigma^2=10^{-2}$':[int(1e10),0,0.01,0],
           'Inhomog. Err.':[int(1e10),0,0.01,1]}

fig, axs=plt.subplots(2,1,figsize=[3,6])

num_mes=10
skip=10
num_traj=1000
p0=0.5

n_s=1000
for _ in scenarios:
    N,s,s_std,inhomog_err=scenarios[_]
    p_vals=perm_freq(gen_traj(N,s,s_std,inhomog_err,p0,n_s,skip,num_mes,num_traj))
    roc1=roc(np.linspace(0,1,num_traj), p_vals, 100)
    axs[0].plot(roc1[:,0],roc1[:,1],label=_,linewidth=2)

axs[0].legend(fontsize=6)
axs[0].plot(np.linspace(0,1),np.linspace(0,1),'k--')
axs[0].set_ylabel('True positive rate')
axs[0].set_title(r'$n=1000$',fontsize=10)
#axs[0].set_xlabel('False positive rate')

n_s=100
for _ in scenarios:
    N,s,s_std,inhomog_err=scenarios[_]
    p_vals=perm_freq(gen_traj(N,s,s_std,inhomog_err,p0,n_s,skip,num_mes,num_traj))
    roc1=roc(np.linspace(0,1,num_traj), p_vals, 100)
    axs[1].plot(roc1[:,0],roc1[:,1],label=_,linewidth=2)

axs[1].plot(np.linspace(0,1),np.linspace(0,1),'k--')
axs[1].set_ylabel('True positive rate')
axs[1].set_xlabel('False positive rate')
axs[1].set_title(r'$n=100$',fontsize=10)

plt.savefig('roc_freq.pdf', bbox_inches='tight')

#%% Frequency power vs N,sig,s
##################################

fig, axs=plt.subplots(3,2,figsize=[3,6])

p0=0.5
num_traj=1000
s=0
s_std=0
inhomog_err=0
N_vec=np.array([10**6, 5*10**5, 10**5, 5*10**4, 10**4, 5*10**3, 10**3])
num_mes_vec=[10,50]


#n=1000
n_s=1000
#short trajectory
for num_mes in num_mes_vec:
    skip=int(100/num_mes)
    axs[0,0].semilogx(N_vec,
                [power(perm_freq(gen_traj(N,s,s_std,inhomog_err,p0,n_s,skip,num_mes,num_traj)),0.05) for N in N_vec], 
                    label=str(num_mes)+' pts. Short.')

#long trajectory
for num_mes in num_mes_vec:
    skip=int(1000/num_mes)
    axs[0,0].semilogx(N_vec,
                [power(perm_freq(gen_traj(N,s,s_std,inhomog_err,p0,n_s,skip,num_mes,num_traj)),0.05) for N in N_vec], '--',
                    label=str(num_mes)+' pts. Long.')

axs[0,0].set_title(r'$n=1000$',fontsize=10)
axs[0,0].set_xlabel(r'$N$')
axs[0,0].set_ylabel('Power')
axs[0,0].annotate(r'$A$',[0.85,0.84],xycoords='axes fraction',fontsize=14)

#n=100
n_s=100
#short trajectory
for num_mes in num_mes_vec:
    skip=int(100/num_mes)
    axs[0,1].semilogx(N_vec,
                [power(perm_freq(gen_traj(N,s,s_std,inhomog_err,p0,n_s,skip,num_mes,num_traj)),0.05) for N in N_vec])

axs[0,1].set_title(r'$n=100$',fontsize=10)
axs[0,1].set_yticklabels('')
axs[0,1].set_xlabel(r'$N$')
axs[0,1].annotate(r'$B$',[0.85,0.84],xycoords='axes fraction',fontsize=14)

#long trajectory
for num_mes in num_mes_vec:
    skip=int(1000/num_mes)
    axs[0,1].semilogx(N_vec,
                [power(perm_freq(gen_traj(N,s,s_std,inhomog_err,p0,n_s,skip,num_mes,num_traj)),0.05) for N in N_vec], '--')

axs[0,1].set_yticklabels('')

for _ in axs.flatten(): _.set_ylim([0,1.01])
for _ in axs.flatten():_.set_xticks(N_vec)

#Power vs s 

s=np.array([0,1e-4,5e-4,1e-3,5e-3,1e-2,5e-2,1e-1])
s_std=0
N=10**8
num_mes_vec=[10,50]

#n=1000
n_s=1000
#short trajectory
for num_mes in num_mes_vec:
    skip=int(100/num_mes)
    axs[1,0].semilogx(s,
                [power(perm_freq(gen_traj(N,_,s_std,inhomog_err,p0,n_s,skip,num_mes,num_traj)),0.05) for _ in s])

#long trajectory
for num_mes in num_mes_vec:
    skip=int(1000/num_mes)
    axs[1,0].semilogx(s,
                [power(perm_freq(gen_traj(N,_,s_std,inhomog_err,p0,n_s,skip,num_mes,num_traj)),0.05) for _ in s], '--')

axs[1,0].set_xlabel(r'$s$')
axs[1,0].set_ylabel('Power')
axs[1,0].annotate(r'$C$',[0.85,0.84],xycoords='axes fraction',fontsize=14)

#n=100
n_s=100
#short trajectory
for num_mes in num_mes_vec:
    skip=int(100/num_mes)
    axs[1,1].semilogx(s,
                [power(perm_freq(gen_traj(N,_,s_std,inhomog_err,p0,n_s,skip,num_mes,num_traj)),0.05) for _ in s])

#long trajectory
for num_mes in num_mes_vec:
    skip=int(1000/num_mes)
    axs[1,1].semilogx(s,
                [power(perm_freq(gen_traj(N,_,s_std,inhomog_err,p0,n_s,skip,num_mes,num_traj)),0.05) for _ in s], '--')

axs[1,1].set_yticklabels('')
axs[1,1].set_xlabel(r'$s$')
axs[1,1].annotate(r'$D$',[0.85,0.84],xycoords='axes fraction',fontsize=14)

#power vs s variance
s=0
s_std=np.array([0,1e-3,5e-3,1e-2,5e-2,1e-1])
N=10**8
num_mes_vec=[10,50]

#n=1000
n_s=1000
#short trajectory
for num_mes in num_mes_vec:
    skip=int(100/num_mes)
    axs[2,0].semilogx(s_std,
                [power(perm_freq(gen_traj(N,s,sig,inhomog_err,p0,n_s,skip,num_mes,num_traj)),0.05) for sig in s_std])

#long trajectory
for num_mes in num_mes_vec:
    skip=int(1000/num_mes)
    axs[2,0].semilogx(s_std,
                [power(perm_freq(gen_traj(N,s,sig,inhomog_err,p0,n_s,skip,num_mes,num_traj)),0.05) for sig in s_std], '--')

axs[2,0].set_xlabel(r'$\sigma^2$')
axs[2,0].set_ylabel('Power')
axs[2,0].annotate(r'$E$',[0.85,0.84],xycoords='axes fraction',fontsize=14)

#n=100
n_s=100
#short trajectory
for num_mes in num_mes_vec:
    skip=int(100/num_mes)
    axs[2,1].semilogx(s_std,
                [power(perm_freq(gen_traj(N,s,sig,inhomog_err,p0,n_s,skip,num_mes,num_traj)),0.05) for sig in s_std],
                    label=str(num_mes)+' pts. Short.')

#long trajectory
for num_mes in num_mes_vec:
    skip=int(1000/num_mes)
    axs[2,1].semilogx(s_std,
                [power(perm_freq(gen_traj(N,s,sig,inhomog_err,p0,n_s,skip,num_mes,num_traj)),0.05) for sig in s_std], '--',
                    label=str(num_mes)+' pts. Long.')

axs[2,1].set_yticklabels('')
axs[2,1].set_xlabel(r'$\sigma^2$')
axs[2,1].annotate(r'$F$',[0.85,0.84],xycoords='axes fraction',fontsize=14)

for _ in axs.flatten(): _.set_ylim([0,1.01])

axs[2,1].legend(fontsize=5, loc='upper left')

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.4)

plt.savefig('power_freq.pdf', bbox_inches='tight')


#%% sign permutation roc
##################################

#N,s,s_std,inhomog_err
scenarios={r'Drift $N=10^3$':[int(1e3),0,0,0],
           'Directional $s=10^{-3}$':[int(1e10),0.001,0,0],
           'Directional $s=10^{-2}$':[int(1e10),0.01,0,0],
           'Fluctuating $\sigma^2=10^{-3}$':[int(1e10),0,0.001,0],
           'Fluctuating $\sigma^2=10^{-2}$':[int(1e10),0,0.01,0]}
           #'Inhomog. Err.':[int(1e10),0,0.01,1]}

fig, axs=plt.subplots(3,1,figsize=[3,6])

p0=0.5
num_mes=10
skip=10
num_traj=1000

n_s=10**10
for _ in scenarios:
    N,s,s_std,inhomog_err=scenarios[_]
    p_vals=perm_sign(gen_traj(N,s,s_std,inhomog_err,p0,n_s,skip,num_mes,num_traj),False)
    #p_vals=binomial_test(gen_traj(N,s,s_std,inhomog_err,p0,n_s,skip,num_mes,num_traj))
    roc1=roc(np.linspace(0,1,num_traj), p_vals, 200)
    axs[0].plot(roc1[:,0],roc1[:,1],label=_,linewidth=2)

axs[0].legend(fontsize=5.5)
axs[0].plot(np.linspace(0,1),np.linspace(0,1),'k--')
axs[0].set_title(r'No error',fontsize=10)
axs[0].set_xticklabels('')

n_s=1000
for _ in scenarios:
    N,s,s_std,inhomog_err=scenarios[_]
    p_vals=perm_sign(gen_traj(N,s,s_std,inhomog_err,p0,n_s,skip,num_mes,num_traj),False)
    #p_vals=binomial_test(gen_traj(N,s,s_std,inhomog_err,p0,n_s,skip,num_mes,num_traj))
    roc1=roc(np.linspace(0,1,num_traj), p_vals, 200)
    axs[1].plot(roc1[:,0],roc1[:,1],label=_,linewidth=2)

axs[1].plot(np.linspace(0,1),np.linspace(0,1),'k--')
axs[1].set_ylabel('Rate of positives')
axs[1].set_title(r'$n=1000$',fontsize=10)
axs[1].set_xticklabels('')

n_s=100
for _ in scenarios:
    N,s,s_std,inhomog_err=scenarios[_]
    p_vals=perm_sign(gen_traj(N,s,s_std,inhomog_err,p0,n_s,skip,num_mes,num_traj),False)
    #p_vals=binomial_test(gen_traj(N,s,s_std,inhomog_err,p0,n_s,skip,num_mes,num_traj))
    roc1=roc(np.linspace(0,1,num_traj), p_vals, 200)
    axs[2].plot(roc1[:,0],roc1[:,1],label=_,linewidth=2)

axs[2].plot(np.linspace(0,1),np.linspace(0,1),'k--')
axs[2].set_xlabel('Significance level')
axs[2].set_title(r'$n=100$',fontsize=10)

plt.savefig('roc_sign.pdf', bbox_inches='tight')

#%% sign permutation roc negative 
##################################

#N,s,s_std,inhomog_err
scenarios={r'Drift $N=10^3$':[int(1e3),0,0,0],
           #'Directional $s=10^{-3}$':[int(1e10),0.001,0,0],
           'Directional $s=10^{-2}$':[int(1e10),0.01,0,0],
           #'Fluctuating $\sigma^2=10^{-3}$':[int(1e10),0,0.001,0],
           'Fluctuating $\sigma^2=10^{-2}$':[int(1e10),0,0.01,0],
           'Neg. Corr.':[int(1e10), np.array([0.001*(-1)**np.floor(t/10) for t in range(num_mes*skip)]), 0, 0]}

fig, axs=plt.subplots(3,1,figsize=[3,6])

p0=0.5
num_mes=10
skip=10
num_traj=1000

n_s=10**10
for _ in scenarios:
    N,s,s_std,inhomog_err=scenarios[_]
    p_vals=perm_sign(gen_traj(N,s,s_std,inhomog_err,p0,n_s,skip,num_mes,num_traj),True)
    #p_vals=binomial_test(gen_traj(N,s,s_std,inhomog_err,p0,n_s,skip,num_mes,num_traj))
    roc1=roc(np.linspace(0,1,num_traj), p_vals, 200)
    axs[0].plot(roc1[:,0],roc1[:,1],label=_,linewidth=2)

axs[0].legend(fontsize=5.5)
axs[0].plot(np.linspace(0,1),np.linspace(0,1),'k--')
axs[0].set_title(r'No error',fontsize=10)
axs[0].set_xticklabels('')

n_s=1000
for _ in scenarios:
    N,s,s_std,inhomog_err=scenarios[_]
    p_vals=perm_sign(gen_traj(N,s,s_std,inhomog_err,p0,n_s,skip,num_mes,num_traj),True)
    #p_vals=binomial_test(gen_traj(N,s,s_std,inhomog_err,p0,n_s,skip,num_mes,num_traj))
    roc1=roc(np.linspace(0,1,num_traj), p_vals, 200)
    axs[1].plot(roc1[:,0],roc1[:,1],label=_,linewidth=2)

axs[1].plot(np.linspace(0,1),np.linspace(0,1),'k--')
axs[1].set_ylabel('Rate of positives')
axs[1].set_title(r'$n=1000$',fontsize=10)
axs[1].set_xticklabels('')

n_s=100
for _ in scenarios:
    N,s,s_std,inhomog_err=scenarios[_]
    p_vals=perm_sign(gen_traj(N,s,s_std,inhomog_err,p0,n_s,skip,num_mes,num_traj),True)
    #p_vals=binomial_test(gen_traj(N,s,s_std,inhomog_err,p0,n_s,skip,num_mes,num_traj))
    roc1=roc(np.linspace(0,1,num_traj), p_vals, 200)
    axs[2].plot(roc1[:,0],roc1[:,1],label=_,linewidth=2)

axs[2].plot(np.linspace(0,1),np.linspace(0,1),'k--')
axs[2].set_xlabel('Significance level')
axs[2].set_title(r'$n=100$',fontsize=10)

plt.savefig('roc_sign_negative.pdf', bbox_inches='tight')

#%% sign permutation
##################################
#Comparison with Feder et al. 2014 
fig, axs=plt.subplots(3,1,figsize=[3,6])

s_std=0
N=10**4
s_vec=np.array([1,2,5,10,15,20,25,40,50,75,100])/N
num_mes_vec=[10,50]
num_traj=100
p0=0.5

#No measurement error
n_s=10**10
#short trajectory
for num_mes in num_mes_vec:
    skip=int(100/num_mes)
    axs[0].plot(N*s_vec,
                [power(perm_sign(gen_traj(N,s,s_std,inhomog_err,p0,n_s,skip,num_mes,num_traj),False),0.05) for s in s_vec]
                 ,label=str(num_mes)+' pts. Short.')

axs[0].set_title('No error',fontsize=10)
axs[0].set_xticklabels('')

#long trajectory
for num_mes in num_mes_vec:
    skip=int(1000/num_mes)
    axs[0].plot(N*s_vec,
                [power(perm_sign(gen_traj(N,s,s_std,inhomog_err,p0,n_s,skip,num_mes,num_traj),False),0.05) for s in s_vec]
                 ,'--',label=str(num_mes)+' pts. Long.')

n_s=1000
#short trajectory
for num_mes in num_mes_vec:
    skip=int(100/num_mes)
    axs[1].plot(N*s_vec,
                [power(perm_sign(gen_traj(N,s,s_std,inhomog_err,p0,n_s,skip,num_mes,num_traj),False),0.05) for s in s_vec]
                 ,label='M'+str(num_mes))

axs[1].set_title(r'$n=1000$',fontsize=10)
axs[1].set_xticklabels('')
axs[1].set_ylabel('Power ')

#long trajectory
for num_mes in num_mes_vec:
    skip=int(1000/num_mes)
    axs[1].plot(N*s_vec,
                [power(perm_sign(gen_traj(N,s,s_std,inhomog_err,p0,n_s,skip,num_mes,num_traj),False),0.05) for s in s_vec]
                 ,'--',label=num_mes)

#n=100
n_s=100
#short trajectory
for num_mes in num_mes_vec:
    skip=int(100/num_mes)
    axs[2].plot(N*s_vec,
                [power(binomial_test(gen_traj(N,s,s_std,inhomog_err,p0,n_s,skip,num_mes,num_traj)),0.05) for s in s_vec]
                 ,label=num_mes)

axs[2].set_title(r'$n=100$',fontsize=10)

#long trajectory
for num_mes in num_mes_vec:
    skip=int(1000/num_mes)
    axs[2].plot(N*s_vec,
                [power(binomial_test(gen_traj(N,s,s_std,inhomog_err,p0,n_s,skip,num_mes,num_traj)),0.05) for s in s_vec]
                 ,'--',label=num_mes)

axs[2].set_xlabel(r'$Ns$')

for _ in axs.flatten(): _.set_ylim([0,1])

axs[0].legend(fontsize=6, loc='lower right')

#plt.savefig('power_NS_sign.pdf', bbox_inches='tight')

#%% increment permutation roc
##################################

num_mes=10
skip=10
num_traj=100
p0=0.5
small=False

#N,s,s_std,inhomog_err
scenarios={r'Drift $N=10^3$':[int(1e3),0,0,0],
           'Directional $s=10^{-3}$':[int(1e10),0.001,0,0],
           'Directional $s=10^{-2}$':[int(1e10),0.01,0,0],
           'Fluctuating $\sigma^2=10^{-2}$':[int(1e10),0,0.01,0],
           'Neg. Corr.':[int(1e10), np.array([0.001*(-1)**np.floor(t/10) for t in range(num_mes*skip)]), 0, 0]}

fig, axs=plt.subplots(3,1,figsize=[3,6])

n_s=10**10
for _ in scenarios:
    N,s,s_std,inhomog_err=scenarios[_]
    p_vals=perm_incr(gen_traj(N,s,s_std,inhomog_err,p0,n_s,skip,num_mes,num_traj),False,small)
    roc1=roc(np.linspace(0,1,num_traj), p_vals, 200)
    axs[0].plot(roc1[:,0],roc1[:,1],label=_,linewidth=2)

axs[0].plot(np.linspace(0,1),np.linspace(0,1),'k--')
axs[0].set_title(r'No error',fontsize=10)
axs[0].set_xticklabels('')

n_s=1000
for _ in scenarios:
    N,s,s_std,inhomog_err=scenarios[_]
    p_vals=perm_incr(gen_traj(N,s,s_std,inhomog_err,p0,n_s,skip,num_mes,num_traj),False,small)
    roc1=roc(np.linspace(0,1,num_traj), p_vals, 200)
    axs[1].plot(roc1[:,0],roc1[:,1],label=_,linewidth=2)

axs[1].legend(fontsize=5.5,loc='lower right')
axs[1].plot(np.linspace(0,1),np.linspace(0,1),'k--')
axs[1].set_ylabel('Rate of positives')
axs[1].set_title(r'$n=1000$',fontsize=10)
axs[1].set_xticklabels('')

n_s=100
for _ in scenarios:
    N,s,s_std,inhomog_err=scenarios[_]
    p_vals=perm_incr(gen_traj(N,s,s_std,inhomog_err,p0,n_s,skip,num_mes,num_traj),False,small)
    roc1=roc(np.linspace(0,1,num_traj), p_vals, 200)
    axs[2].plot(roc1[:,0],roc1[:,1],label=_,linewidth=2)

axs[2].plot(np.linspace(0,1),np.linspace(0,1),'k--')
axs[2].set_xlabel('Significance level')
axs[2].set_title(r'$n=100$',fontsize=10)

plt.savefig('roc_inc.pdf', bbox_inches='tight')

#%% increment permutation roc negative
##################################

num_mes=10
skip=10
num_traj=100
p0=0.5

#N,s,s_std,inhomog_err
scenarios={r'Drift $N=10^3$':[int(1e3),0,0,0],
           'Directional $s=10^{-3}$':[int(1e10),0.001,0,0],
           'Directional $s=10^{-2}$':[int(1e10),0.01,0,0],
           'Fluctuating $\sigma^2=10^{-2}$':[int(1e10),0,0.01,0],
           'Neg. Corr.':[int(1e10), np.array([0.001*(-1)**np.floor(t/10) for t in range(num_mes*skip)]), 0, 0]}

fig, axs=plt.subplots(3,1,figsize=[3,6])

n_s=10**10
for _ in scenarios:
    N,s,s_std,inhomog_err=scenarios[_]
    p_vals=perm_incr(gen_traj(N,s,s_std,inhomog_err,p0,n_s,skip,num_mes,num_traj),False,True)
    roc1=roc(np.linspace(0,1,num_traj), p_vals, 200)
    axs[0].plot(roc1[:,0],roc1[:,1],label=_,linewidth=2)

axs[0].legend(fontsize=5.5)
axs[0].plot(np.linspace(0,1),np.linspace(0,1),'k--')
axs[0].set_title(r'No error',fontsize=10)
axs[0].set_xticklabels('')

n_s=1000
for _ in scenarios:
    N,s,s_std,inhomog_err=scenarios[_]
    p_vals=perm_incr(gen_traj(N,s,s_std,inhomog_err,p0,n_s,skip,num_mes,num_traj),False,True)
    roc1=roc(np.linspace(0,1,num_traj), p_vals, 200)
    axs[1].plot(roc1[:,0],roc1[:,1],label=_,linewidth=2)

axs[1].plot(np.linspace(0,1),np.linspace(0,1),'k--')
axs[1].set_ylabel('Rate of positives')
axs[1].set_title(r'$n=1000$',fontsize=10)
axs[1].set_xticklabels('')

n_s=100
for _ in scenarios:
    N,s,s_std,inhomog_err=scenarios[_]
    p_vals=perm_incr(gen_traj(N,s,s_std,inhomog_err,p0,n_s,skip,num_mes,num_traj),False,True)
    roc1=roc(np.linspace(0,1,num_traj), p_vals, 200)
    axs[2].plot(roc1[:,0],roc1[:,1],label=_,linewidth=2)

axs[2].plot(np.linspace(0,1),np.linspace(0,1),'k--')
axs[2].set_xlabel('Significance level')
axs[2].set_title(r'$n=100$',fontsize=10)

plt.savefig('roc_inc_negative.pdf', bbox_inches='tight')

#%% increment permutation
##################################
fig, axs=plt.subplots(3,1,figsize=[3,6])

s_std=0
N=10**4
s_vec=10*np.array([1,2,5,10,15,20,25,40,50,75,100])/N
num_mes_vec=[10,50]
num_traj=1000
p0=0.5
inhomog_err=0
transform=False

#No measurement error
n_s=10**10
#short trajectory
for num_mes in num_mes_vec:
    skip=int(100/num_mes)
    axs[0].plot(N*s_vec,
                [power(
                    perm_incr(
                        gen_traj(
                            N,np.array([s*(-1)**np.floor(t/skip) for t in range(num_mes*skip)]),s_std,inhomog_err,p0,n_s,skip,num_mes,num_traj
                            )
                            ,transform,True)
                            ,0.05) for s in s_vec]
                 ,label=str(num_mes)+' pts. Short')

#axs[0].set_title('No error',fontsize=10)
axs[0].set_xlabel(r'$N|s|$')
axs[0].set_ylabel(r'Power')
axs[0].annotate(r'$A$',[0.85,0.84],xycoords='axes fraction',fontsize=14)

#long trajectory
for num_mes in num_mes_vec:
    skip=int(1000/num_mes)
    axs[0].plot(N*s_vec,
                [power(
                    perm_incr(
                        gen_traj(
                            N,np.array([s*(-1)**np.floor(t/skip) for t in range(num_mes*skip)]),s_std,inhomog_err,p0,n_s,skip,num_mes,num_traj
                            )
                            ,transform,True)
                            ,0.05) for s in s_vec]
                 ,'--',label=str(num_mes)+' pts. Long')


n_s=np.array([100000,10000,1000,100])
s=0.01
#short trajectory
for num_mes in num_mes_vec:
    skip=int(100/num_mes)
    axs[1].semilogx(n_s,
                [power(
                    perm_incr(
                        gen_traj(
                            N,np.array([s*(-1)**np.floor(t/skip) for t in range(num_mes*skip)]),s_std,inhomog_err,p0,_,skip,num_mes,num_traj
                            )
                            ,transform,True)
                            ,0.05) for _ in n_s]
                 )

#long trajectory
for num_mes in num_mes_vec:
    skip=int(1000/num_mes)
    axs[1].semilogx(n_s,
                [power(
                    perm_incr(
                        gen_traj(
                            N,np.array([s*(-1)**np.floor(t/skip) for t in range(num_mes*skip)]),s_std,inhomog_err,p0,_,skip,num_mes,num_traj
                            )
                            ,transform,True)
                            ,0.05) for _ in n_s]
                 ,'--')

#axs[1].set_title(r'$n=1000$',fontsize=10)
axs[1].set_ylabel('Rate of positives')
axs[1].set_xlabel(r'$n$')
axs[1].annotate(r'$B$',[0.85,0.84],xycoords='axes fraction',fontsize=14)

s=0.0
#short trajectory
for num_mes in num_mes_vec:
    skip=int(100/num_mes)
    axs[2].semilogx(n_s,
                [power(
                    perm_incr(
                        gen_traj(
                            N,np.array([s*(-1)**np.floor(t/skip) for t in range(num_mes*skip)]),s_std,inhomog_err,p0,_,skip,num_mes,num_traj
                            )
                            ,transform,True)
                            ,0.05) for _ in n_s]
                 )

#long trajectory
for num_mes in num_mes_vec:
    skip=int(1000/num_mes)
    axs[2].semilogx(n_s,
                [power(
                    perm_incr(
                        gen_traj(
                            N,np.array([s*(-1)**np.floor(t/skip) for t in range(num_mes*skip)]),s_std,inhomog_err,p0,_,skip,num_mes,num_traj
                            )
                            ,transform,True)
                            ,0.05) for _ in n_s]
                 ,'--')

#axs[1].set_title(r'$n=1000$',fontsize=10)
axs[2].set_ylabel('Rate of positives')
axs[2].set_xlabel(r'$n$')
axs[2].annotate(r'$C$',[0.85,0.84],xycoords='axes fraction',fontsize=14)

for _ in axs.flatten(): _.set_ylim([0,1])

axs[0].legend(fontsize=6, loc='lower right')

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.4)
plt.savefig('power_NS_incr.pdf', bbox_inches='tight')

#%% Frequency power Lynch et al.
##################################

fig, axs=plt.subplots(3,2,figsize=[3,6])

p0=0.5
num_traj=100
s=0
s_std=0
inhomog_err=0
N_vec=np.array([10**6, 5*10**5, 10**5, 5*10**4, 10**4, 5*10**3, 10**3])
num_mes_vec=[10]
skip=1


#n=1000
n_s=1000
#short trajectory
for num_mes in num_mes_vec:
    axs[0,0].semilogx(N_vec,
                [power(perm_freq(gen_traj(N,s,s_std,inhomog_err,p0,n_s,skip,num_mes,num_traj)),0.05) for N in N_vec], 
                    label=str(num_mes)+' pts. Short.')


axs[0,0].set_title(r'$n=1000$',fontsize=10)
axs[0,0].set_xlabel(r'$N$')
axs[0,0].set_ylabel('Power')
axs[0,0].annotate(r'$A$',[0.85,0.84],xycoords='axes fraction',fontsize=14)

#n=100
n_s=100
#short trajectory
for num_mes in num_mes_vec:
    axs[0,1].semilogx(N_vec,
                [power(perm_freq(gen_traj(N,s,s_std,inhomog_err,p0,n_s,skip,num_mes,num_traj)),0.05) for N in N_vec])

axs[0,1].set_title(r'$n=100$',fontsize=10)
axs[0,1].set_yticklabels('')
axs[0,1].set_xlabel(r'$N$')
axs[0,1].annotate(r'$B$',[0.85,0.84],xycoords='axes fraction',fontsize=14)
axs[0,1].set_yticklabels('')

for _ in axs.flatten(): _.set_ylim([0,1.01])
for _ in axs.flatten():_.set_xticks(N_vec)

#Power vs s 

s=np.array([0,1e-4,5e-4,1e-3,5e-3,1e-2,5e-2,1e-1])
s_std=0
N=10**8

#n=1000
n_s=1000
#short trajectory
for num_mes in num_mes_vec:
    axs[1,0].semilogx(s,
                [power(perm_freq(gen_traj(N,_,s_std,inhomog_err,p0,n_s,skip,num_mes,num_traj)),0.05) for _ in s])

axs[1,0].set_xlabel(r'$s$')
axs[1,0].set_ylabel('Power')
axs[1,0].annotate(r'$C$',[0.85,0.84],xycoords='axes fraction',fontsize=14)

#n=100
n_s=100
#short trajectory
for num_mes in num_mes_vec:
    skip=int(100/num_mes)
    axs[1,1].semilogx(s,
                [power(perm_freq(gen_traj(N,_,s_std,inhomog_err,p0,n_s,skip,num_mes,num_traj)),0.05) for _ in s])


axs[1,1].set_yticklabels('')
axs[1,1].set_xlabel(r'$s$')
axs[1,1].annotate(r'$D$',[0.85,0.84],xycoords='axes fraction',fontsize=14)

#power vs s variance
s=0
s_std=np.array([0,1e-3,5e-3,1e-2,5e-2,1e-1])
N=10**8

#n=1000
n_s=1000
#short trajectory
for num_mes in num_mes_vec:
    axs[2,0].semilogx(s_std,
                [powerperm_freq(gen_traj(N,s,sig,inhomog_err,p0,n_s,skip,num_mes,num_traj)),0.05) for sig in s_std])

axs[2,0].set_xlabel(r'$\sigma^2$')
axs[2,0].set_ylabel('Power')
axs[2,0].annotate(r'$E$',[0.85,0.84],xycoords='axes fraction',fontsize=14)

#n=100
n_s=100
#short trajectory
for num_mes in num_mes_vec:
    axs[2,1].semilogx(s_std,
                [power(perm_freq(gen_traj(N,s,sig,inhomog_err,p0,n_s,skip,num_mes,num_traj)),0.05) for sig in s_std],
                    label=str(num_mes)+' pts. Short.')


axs[2,1].set_yticklabels('')
axs[2,1].set_xlabel(r'$\sigma^2$')
axs[2,1].annotate(r'$F$',[0.85,0.84],xycoords='axes fraction',fontsize=14)

for _ in axs.flatten(): _.set_ylim([0,1.01])

axs[2,1].legend(fontsize=5, loc='upper left')

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.4)

##=========================================================================
##%%
##pvalue merge functions with dependence but assuming exchangeability
##Gasparin et al 2025 PNAS.
#
#def harm_mean(x):
#    return len(x)/np.sum(1/x)
#
#def harm_mean_ex(x):
#    K=len(x)
#    #return (np.log(K)+np.log(np.log(K))+2)*np.min([harm_mean(x[:i]) for i in range(1,K)])
#    return np.min([harm_mean(x[:i]) for i in range(1,K)])
#
#print('Bonferroni')
#print(num_traj*np.min(p_vals))
#print(num_traj*np.min(p_vals_complement))
#
#print('Arithmetic')
#print(2*np.mean(p_vals))
#print(2*np.mean(p_vals_complement))
#
#print('Arithmetic exchangeable')
#print(2*np.min([np.mean(p_vals[:i]) for i in range (1,num_traj)]))
#print(2*np.min([np.mean(p_vals_complement[:i]) for i in range (1,num_traj)]))
#
#print('Geometric')
#print(np.e*np.exp(np.mean(np.log(p_vals))))
#print(np.e*np.exp(np.mean(np.log(p_vals_complement))))
#
#print('Geometric exchangeable')
#print(np.e*np.min([np.exp(np.mean(np.log(p_vals[:i]))) for i in range (1,num_traj)]))
#print(np.e*np.min([np.exp(np.mean(np.log(p_vals_complement[:i]))) for i in range (1,num_traj)]))
#
#print('Harmonic exchangeable')
#T=(np.log(num_traj)+np.log(np.log(num_traj))+2)
#print(T*harm_mean_ex(p_vals))
#print(T*harm_mean_ex(p_vals_complement))

