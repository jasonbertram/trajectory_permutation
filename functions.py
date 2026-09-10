import numpy as np
import itertools
import multiprocessing as mp

#sample size for permuting long trajectories
sample_size=100000

def roc(pvals1,pvals2,n):
    """Receiver operating characteristic"""
    L1=len(pvals1)
    L2=len(pvals2)
    return np.array([[np.sum(pvals1<=alph)/L1,np.sum(pvals2<=alph)/L2] for alph in np.linspace(0,1,n)])

def power(pvals,alph):
    """Receiver operating characteristic"""
    return np.sum(pvals<=alph)/len(pvals)

def perm_freq_pval_calc(perm_p,p):
        dp=np.diff(perm_p)
        d_perm=np.mean(np.abs(dp),1) #average increment magnitude
        d_obs=np.mean(np.abs(np.diff(p)))
        return np.sum( (d_perm<=d_obs) | np.isclose(d_perm,d_obs,rtol=0,atol=10**-15))/len(d_perm) 

def perm_freq_one(args):
    idx, traj, sample_size, seed = args
    rng = np.random.default_rng(seed)

    T=len(traj)
    if T>8: #do exact test for trajectories <= 8 points long
        rng = np.random.default_rng()
        perm_p = rng.permuted(np.tile(traj, (sample_size, 1)), axis=1)
        #perm_p=np.array([np.random.permutation(traj) for _ in range(sample_size)]) #old version. much slower 
        perm_p[0]=traj
        pval=perm_freq_pval_calc(perm_p,traj)
        #if pvalue is above sample size threshold keep it, otherwise fall through to exact calculation
        if pval>1/sample_size:
            return idx, pval
    
    #safety catch to prevent exact test from hanging if the number of permutations is unmanageable 
    if T<15:
        perm_p=np.array([_ for _ in itertools.permutations(traj)])
    else:
        perm_p = rng.permuted(np.tile(traj, (int(1e6), 1)), axis=1) #simply do a much bigger number of samples instead

    #idx needed to track pvalue identity (multiprocessing with imap_unordered won't wait for slower pvals so ordering must be tracked)
    return idx, perm_freq_pval_calc(perm_p,traj) 

def perm_freq(trajectories):
    seeds = np.random.SeedSequence().spawn(len(trajectories))
    tasks = [(i, t, sample_size, s)
         for i, (t, s) in enumerate(zip(trajectories, seeds))]

    p_vals=np.zeros(len(trajectories))

    with mp.Pool(processes=16) as pool:
        results = pool.imap_unordered(perm_freq_one, tasks)
        from tqdm import tqdm
        results = tqdm(results, total=len(tasks))

        for idx, pval in results:
            p_vals[idx] = pval
        pool.close()
        pool.join()

    return p_vals

#not run on data so not parallelized or able to handle different trajectory lengths
def perm_incr(trajectories,transform,small):
    p_vals=np.zeros(len(trajectories))
    T=len(trajectories[0])-1

    from tqdm import tqdm
    for i,p in tqdm(enumerate(trajectories)):
        
        dp=np.diff(p)
        if transform==True:
            dp=dp/(p[:-1]*(1-p[:-1]))

        if T>8: #do exact test for trajectories <= 9 points long
            rng = np.random.default_rng()
            perm_dp = rng.permuted(np.tile(dp, (sample_size, 1)), axis=1)
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
            #unusually small
            p_vals[i]=np.sum( (d_perm<=d_obs) | np.isclose(d_perm,d_obs,rtol=0,atol=10**-15))/len(d_perm) 
        else:
            #unusually large
            p_vals[i]=np.sum( (d_perm>=d_obs) | np.isclose(d_perm,d_obs,rtol=0,atol=10**-15))/len(d_perm) 

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

    T_old=len(trajectories[0])-1

    from tqdm import tqdm
    for i,p in tqdm(enumerate(trajectories)):
        T=len(p)-1
        #sign permutation matrix. 
        #only rebuild if length changes
        if T != T_old or i==0:
            if T>13: #do exact test for trajectories <= 12 points long
                sgn_prm=np.array([2*np.random.randint(2,size=T)-1 for _ in range(sample_size) ])
                sgn_prm[0]=np.ones(T)
            else:
                sgn_prm=np.array(list(itertools.product([-1,1], repeat=T))) 

            T_old=T

        dp=np.diff(p)
        d_perm=np.abs(np.sum(sgn_prm*dp, axis=1))

        d_obs=np.abs(p[0]-p[-1])
        if small:
            #unusually small
            p_vals[i]=np.sum((d_perm<=d_obs) | np.isclose(d_perm,d_obs,rtol=0,atol=10**-15))/len(d_perm) 
        else:
            #unusually large
            p_vals[i]=np.sum((d_perm>=d_obs) | np.isclose(d_perm,d_obs,rtol=0,atol=10**-15))/len(d_perm) 

    return p_vals

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
        sample_sizes=np.array(np.random.poisson(n_s,[numtraj,np.int64(T/skip)]),dtype=int)
        p=np.random.binomial(sample_sizes,p[:,0:T:skip])/n_s
    else:
        p=np.random.binomial(n_s,p[:,0:T:skip])/n_s

    return p

