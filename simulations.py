import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.stats import combine_pvalues

def roc(pvals1,pvals2,n):
    L1=len(pvals1)
    L2=len(pvals2)
    return np.array([[np.sum(pvals1<alph)/L1,np.sum(pvals2<alph)/L2] for alph in np.linspace(0,1,n)])

#%%
#increment order permutation
#########################

T=71
N=10000
phi=np.array([0.])
s_std=0.00
s_hist=np.random.normal(0,s_std,len(phi))

num_traj=10000
p_vals=np.zeros(num_traj)
p_vals_complement=np.zeros(num_traj)
p_vals_twoside=np.zeros(num_traj)

fixed_arg=(p_vals==p_vals)
fixed_arg_1=(p_vals==p_vals)

#f=open('sim_data_3.txt','a')

for i in range(num_traj):
    p=np.zeros(T)
    p[0]=0.5+0*np.random.rand()/100

    pulse_position=np.random.randint(T)
    for t in range(1,T):
        #s=np.sum((1)**(np.mod(i,2))*phi*s_hist)+np.random.normal(0,s_std)
        #s_hist=np.concatenate([[s],s_hist[1:]])
        #if t==pulse_position:
        #    s=1
        #else:
        s=0.02
        p[t]=np.random.binomial(N,p[t-1]+s*p[t-1]*(1-p[t-1]))/N

    if p[-1]==0:
        fixed_arg[i]=False

    #plt.plot(p)
    p=p[0:71:10]+np.random.normal(0,0.01,8)
    #np.savetxt(f,p.reshape(1,-1),fmt='%1.3f',delimiter=',')

    p=p[p>0]

    if len(p)==1:
        fixed_arg_1[i]=False
    else:
        dp=np.diff(p)

        perm_dp=np.array([_ for _ in itertools.permutations(dp)])
        perm_p=np.cumsum(perm_dp,1)

        #plt.plot(perm_p.transpose(),linewidth=0.5,alpha=0.5)

        p_mean=np.mean(perm_p,0)
        abs_devs=np.abs(perm_p-p_mean)
        d_perm=np.mean(abs_devs,1)
        d_obs=np.mean(np.abs(p[1:]-p[0]-p_mean))

        p_vals[i]=np.sum((d_perm-d_obs)<1e-6)/len(d_perm) #unusually small
        p_vals_complement[i]=np.sum((d_obs-d_perm)<1e-6)/len(d_perm) #unusually large
        p_vals_twoside[i]=2*np.min([p_vals[i],p_vals_complement[i]])
        
#print(np.sum(np.log(p_vals)))

#f.close()

print(combine_pvalues(p_vals[fixed_arg_1]))
print(combine_pvalues(p_vals_complement[fixed_arg_1]))
print(combine_pvalues(p_vals_twoside[fixed_arg_1]))

#%%
plt.hist(p_vals)

plt.figure()
plt.hist(p_vals_complement)

#p_vals_expanded=np.concatenate([p_vals,np.random.rand(1000)])
p_vals_expanded=np.concatenate([p_vals,np.ones(100)])
print(combine_pvalues(p_vals_expanded))

#%%
#increment sign permutation
##################################

T=71
N=1000

num_traj=100
p_vals=np.zeros(num_traj)
p_vals_complement=np.zeros(num_traj)
p_vals_twoside=np.zeros(num_traj)

sgn_prm=np.array(list(itertools.product([-1,1], repeat=int(np.floor(T/10)))))

init=0.3
for i in range(num_traj):
    p=np.zeros(T)
    p[0]=init

    for t in range(1,T):
        #p[t]=0.5*p[t-1]+np.random.normal(0,1)
        p[t]=np.random.binomial(N,p[t-1]+(1)**t*0.02*p[t-1]*(1-p[t-1]))/N

    #p=p+np.random.normal(0,0.05,size=T)
    p=p[0:71:10]+np.random.normal(0,0.00,8)
    dp=np.diff(p)

    d_perm=np.sum(sgn_prm*dp, axis=1)

    p_vals[i]=np.sum((np.abs(d_perm)-np.abs(p[0]-p[-1]))<1e-6)/len(d_perm)
    if p_vals[i]==0: print(p[-1],d_perm)
    p_vals_complement[i]=np.sum((np.abs(p[0]-p[-1])-np.abs(d_perm))<1e-6)/len(d_perm)
    
plt.hist(p_vals)
plt.figure()
plt.hist(p_vals_complement)

print(combine_pvalues(p_vals))
print(combine_pvalues(p_vals_complement))


#%%
#frequency permutation
##################################

T=21+1
N=100000000
phi=np.array([0.])
s_std=0.00
s_hist=np.random.normal(0,s_std,len(phi))

num_traj=1000
p_vals=np.zeros(num_traj)
p_vals_complement=np.zeros(num_traj)
p_vals_twoside=np.zeros(num_traj)

fixed_arg=(p_vals==p_vals)
fixed_arg_1=(p_vals==p_vals)

#f=open('sim_data_3.txt','a')

for i in range(num_traj):
    p=0.5*np.ones(T)
    p[0]=0.5+0*np.random.rand()/100

    pulse_position=np.random.randint(T)
    for t in range(1,T):
        #s=np.sum((1)**(np.mod(i,2))*phi*s_hist)+np.random.normal(0,s_std)
        #s_hist=np.concatenate([[s],s_hist[1:]])
        #if t==pulse_position:
        #    s=1
        #else:
        s=0.00+np.random.normal(0,0.02)
        p[t]=np.random.binomial(N,p[t-1]+s*p[t-1]*(1-p[t-1]))/N

    if p[-1]==0:
        fixed_arg[i]=False

    #plt.plot(p)
    p=p[0:T:3]+np.random.normal(0,0.01,8)
    #np.savetxt(f,p.reshape(1,-1),fmt='%1.3f',delimiter=',')

    p=p[p>0]

    if len(p)==1:
        fixed_arg_1[i]=False
    else:
        perm_p=np.array([_ for _ in itertools.permutations(p)])

        dp=np.diff(perm_p)
        #plt.plot(perm_p.transpose(),linewidth=0.5,alpha=0.5)
    
        #average increment magnitude
        d_perm=np.mean(np.abs(dp),1)
        d_obs=np.mean(np.abs(np.diff(p)))

        p_vals[i]=np.sum((d_perm-d_obs)<1e-6)/len(d_perm) #unusually small
        p_vals_complement[i]=np.sum((d_obs-d_perm)<1e-6)/len(d_perm) #unusually large
        p_vals_twoside[i]=2*np.min([p_vals[i],p_vals_complement[i]])
        
#print(np.sum(np.log(p_vals)))

#f.close()

print(combine_pvalues(p_vals[fixed_arg_1]))
print(combine_pvalues(p_vals_complement[fixed_arg_1]))
print(combine_pvalues(p_vals_twoside[fixed_arg_1]))

#%%
plt.hist(p_vals)

plt.figure()
plt.hist(p_vals_complement)

#p_vals_expanded=np.concatenate([p_vals,np.random.rand(1000)])
p_vals_expanded=np.concatenate([p_vals,np.ones(100)])
print(combine_pvalues(p_vals_expanded))

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
