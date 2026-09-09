import matplotlib.pyplot as plt
from functions import *
import numpy as np
#%% frequency permutation roc
##################################

#N,s,s_std,inhomog_err
scenarios={r'Drift $N=10^3$':[int(1e3),0,0,0],
           'Directional $s=10^{-3}$':[int(1e10),0.001,0,0],
           'Directional $s=10^{-2}$':[int(1e10),0.01,0,0],
           'Fluctuating $\sigma^2=10^{-2}$':[int(1e10),0,0.01,0],
           'Inhomog. Err.':[int(1e10),0.0,0,1]}

fig, axs=plt.subplots(2,1,figsize=[3,6],constrained_layout=True)

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
#axs[0].set_title(r'$n=1000$',fontsize=10)
axs[0].text(-0.1, 1.05, 'a', transform=axs[0].transAxes,
        fontsize=12, fontweight='bold', va='bottom')

n_s=100
for _ in scenarios:
    N,s,s_std,inhomog_err=scenarios[_]
    p_vals=perm_freq(gen_traj(N,s,s_std,inhomog_err,p0,n_s,skip,num_mes,num_traj))
    roc1=roc(np.linspace(0,1,num_traj), p_vals, 100)
    axs[1].plot(roc1[:,0],roc1[:,1],label=_,linewidth=2)

axs[1].plot(np.linspace(0,1),np.linspace(0,1),'k--')
axs[1].set_ylabel('True positive rate')
axs[1].set_xlabel('False positive rate')
#axs[1].set_title(r'$n=100$',fontsize=10)
axs[1].text(-0.1, 1.05, 'b', transform=axs[1].transAxes,
        fontsize=12, fontweight='bold', va='bottom')


plt.savefig('roc_freq.pdf', bbox_inches='tight')

#%% Frequency power vs N,sig,s
##################################

fig, axs=plt.subplots(3,2,figsize=[3.4,6],constrained_layout=True)

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

axs[0,0].set_xlabel(r'$N$')
axs[0,0].set_ylabel('Power')

#n=100
n_s=100
#short trajectory
for num_mes in num_mes_vec:
    skip=int(100/num_mes)
    axs[0,1].semilogx(N_vec,
                [power(perm_freq(gen_traj(N,s,s_std,inhomog_err,p0,n_s,skip,num_mes,num_traj)),0.05) for N in N_vec])

axs[0,1].set_yticklabels('')
axs[0,1].set_xlabel(r'$N$')

#long trajectory
for num_mes in num_mes_vec:
    skip=int(1000/num_mes)
    axs[0,1].semilogx(N_vec,
                [power(perm_freq(gen_traj(N,s,s_std,inhomog_err,p0,n_s,skip,num_mes,num_traj)),0.05) for N in N_vec], '--')

axs[0,1].set_yticklabels('')

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

for _ in axs.flatten(): 
    _.set_ylim([0,1.01])
    _.tick_params(labelsize=7)
    _.xaxis.label.set_size(8)
    _.yaxis.label.set_size(8)

axs[2,1].legend(fontsize=5, loc='upper left')

for ax, label in zip(axs.flat, 'abcdef'):
    ax.text(-0.1, 1.05, label, transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='bottom')

plt.savefig('power_freq.pdf', bbox_inches='tight')


#%% sign permutation roc
##################################

#N,s,s_std,inhomog_err
scenarios={r'Drift $N=10^3$':[int(1e3),0,0,0],
           'Directional $s=10^{-3}$':[int(1e10),0.001,0,0],
           'Directional $s=10^{-2}$':[int(1e10),0.01,0,0],
           'Fluctuating $\sigma^2=10^{-3}$':[int(1e10),0,0.001,0],
           'Fluctuating $\sigma^2=10^{-2}$':[int(1e10),0,0.01,0]}

fig, axs=plt.subplots(3,1,figsize=[3,6],constrained_layout=True)

p0=0.5
num_mes=10
skip=10
num_traj=1000

n_s=10**10
for _ in scenarios:
    N,s,s_std,inhomog_err=scenarios[_]
    p_vals=perm_sign(gen_traj(N,s,s_std,inhomog_err,p0,n_s,skip,num_mes,num_traj),False)
    roc1=roc(np.linspace(0,1,num_traj), p_vals, 200)
    axs[0].plot(roc1[:,0],roc1[:,1],label=_,linewidth=2)

axs[0].legend(fontsize=5.5)
axs[0].plot(np.linspace(0,1),np.linspace(0,1),'k--')

n_s=1000
for _ in scenarios:
    N,s,s_std,inhomog_err=scenarios[_]
    p_vals=perm_sign(gen_traj(N,s,s_std,inhomog_err,p0,n_s,skip,num_mes,num_traj),False)
    roc1=roc(np.linspace(0,1,num_traj), p_vals, 200)
    axs[1].plot(roc1[:,0],roc1[:,1],label=_,linewidth=2)

axs[1].plot(np.linspace(0,1),np.linspace(0,1),'k--')
axs[1].set_ylabel('Rate of positives')

n_s=100
for _ in scenarios:
    N,s,s_std,inhomog_err=scenarios[_]
    p_vals=perm_sign(gen_traj(N,s,s_std,inhomog_err,p0,n_s,skip,num_mes,num_traj),False)
    roc1=roc(np.linspace(0,1,num_traj), p_vals, 200)
    axs[2].plot(roc1[:,0],roc1[:,1],label=_,linewidth=2)

axs[2].plot(np.linspace(0,1),np.linspace(0,1),'k--')
axs[2].set_xlabel('Significance level')

for ax, label in zip(axs.flat, 'abc'):
    ax.text(-0.1, 1.05, label, transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='bottom')

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

fig, axs=plt.subplots(3,1,figsize=[3,6],constrained_layout=True)

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

n_s=1000
for _ in scenarios:
    N,s,s_std,inhomog_err=scenarios[_]
    p_vals=perm_sign(gen_traj(N,s,s_std,inhomog_err,p0,n_s,skip,num_mes,num_traj),True)
    #p_vals=binomial_test(gen_traj(N,s,s_std,inhomog_err,p0,n_s,skip,num_mes,num_traj))
    roc1=roc(np.linspace(0,1,num_traj), p_vals, 200)
    axs[1].plot(roc1[:,0],roc1[:,1],label=_,linewidth=2)

axs[1].plot(np.linspace(0,1),np.linspace(0,1),'k--')
axs[1].set_ylabel('Rate of positives')

n_s=100
for _ in scenarios:
    N,s,s_std,inhomog_err=scenarios[_]
    p_vals=perm_sign(gen_traj(N,s,s_std,inhomog_err,p0,n_s,skip,num_mes,num_traj),True)
    #p_vals=binomial_test(gen_traj(N,s,s_std,inhomog_err,p0,n_s,skip,num_mes,num_traj))
    roc1=roc(np.linspace(0,1,num_traj), p_vals, 200)
    axs[2].plot(roc1[:,0],roc1[:,1],label=_,linewidth=2)

axs[2].plot(np.linspace(0,1),np.linspace(0,1),'k--')
axs[2].set_xlabel('Significance level')

for ax, label in zip(axs.flat, 'abc'):
    ax.text(-0.1, 1.05, label, transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='bottom')

plt.savefig('roc_sign_negative.pdf', bbox_inches='tight')

#%% sign permutation
##################################
#Comparison with Feder et al. 2014 
fig, axs=plt.subplots(3,1,figsize=[3,6], constrained_layout=True)

s_std=0
N=10**4
s_vec=np.array([1,2,5,10,15,20,25,40,50,75,100])/N
num_mes_vec=[10,50]
num_traj=1000
p0=0.5

#No measurement error
n_s=10**10
#short trajectory
for num_mes in num_mes_vec:
    skip=int(100/num_mes)
    axs[0].plot(N*s_vec,
                [power(perm_sign(gen_traj(N,s,s_std,inhomog_err,p0,n_s,skip,num_mes,num_traj),False),0.05) for s in s_vec]
                 ,label=str(num_mes)+' pts. Short.')

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

axs[1].set_ylabel('Power ')

#long trajectory
for num_mes in num_mes_vec:
    skip=int(1000/num_mes)
    axs[1].plot(N*s_vec,
                [power(perm_sign(gen_traj(N,s,s_std,inhomog_err,p0,n_s,skip,num_mes,num_traj),False),0.05) for s in s_vec]
                 ,'--',label=num_mes)

n_s=100
#short trajectory
for num_mes in num_mes_vec:
    skip=int(100/num_mes)
    axs[2].plot(N*s_vec,
                [power(perm_sign(gen_traj(N,s,s_std,inhomog_err,p0,n_s,skip,num_mes,num_traj),False),0.05) for s in s_vec]
                 ,label=num_mes)

#long trajectory
for num_mes in num_mes_vec:
    skip=int(1000/num_mes)
    axs[2].plot(N*s_vec,
                [power(perm_sign(gen_traj(N,s,s_std,inhomog_err,p0,n_s,skip,num_mes,num_traj),False),0.05) for s in s_vec]
                 ,'--',label=num_mes)

axs[2].set_xlabel(r'$Ns$')

for _ in axs.flatten(): _.set_ylim([0,1])

axs[0].legend(fontsize=6, loc='lower right')

for ax, label in zip(axs.flat, 'abc'):
    ax.text(-0.1, 1.05, label, transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='bottom')

plt.savefig('power_NS_sign.pdf', bbox_inches='tight')


#%% increment permutation roc negative
##################################

num_mes=10
skip=10
num_traj=1000
p0=0.5

#N,s,s_std,inhomog_err
scenarios={r'Drift $N=10^3$':[int(1e3),0,0,0],
           'Directional $s=10^{-3}$':[int(1e10),0.001,0,0],
           'Directional $s=10^{-2}$':[int(1e10),0.01,0,0],
           'Fluctuating $\sigma^2=10^{-2}$':[int(1e10),0,0.01,0],
           'Neg. Corr.':[int(1e10), np.array([0.001*(-1)**np.floor(t/10) for t in range(num_mes*skip)]), 0, 0]}

fig, axs=plt.subplots(3,1,figsize=[3,6],constrained_layout=True)

n_s=10**10
for _ in scenarios:
    N,s,s_std,inhomog_err=scenarios[_]
    p_vals=perm_incr(gen_traj(N,s,s_std,inhomog_err,p0,n_s,skip,num_mes,num_traj),False,True)
    roc1=roc(np.linspace(0,1,num_traj), p_vals, 200)
    axs[0].plot(roc1[:,0],roc1[:,1],label=_,linewidth=2)

axs[0].legend(fontsize=5.5)
axs[0].plot(np.linspace(0,1),np.linspace(0,1),'k--')

n_s=1000
for _ in scenarios:
    N,s,s_std,inhomog_err=scenarios[_]
    p_vals=perm_incr(gen_traj(N,s,s_std,inhomog_err,p0,n_s,skip,num_mes,num_traj),False,True)
    roc1=roc(np.linspace(0,1,num_traj), p_vals, 200)
    axs[1].plot(roc1[:,0],roc1[:,1],label=_,linewidth=2)

axs[1].plot(np.linspace(0,1),np.linspace(0,1),'k--')
axs[1].set_ylabel('Rate of positives')

n_s=100
for _ in scenarios:
    N,s,s_std,inhomog_err=scenarios[_]
    p_vals=perm_incr(gen_traj(N,s,s_std,inhomog_err,p0,n_s,skip,num_mes,num_traj),False,True)
    roc1=roc(np.linspace(0,1,num_traj), p_vals, 200)
    axs[2].plot(roc1[:,0],roc1[:,1],label=_,linewidth=2)

axs[2].plot(np.linspace(0,1),np.linspace(0,1),'k--')
axs[2].set_xlabel('Significance level')

for ax, label in zip(axs.flat, 'abc'):
    ax.text(-0.1, 1.05, label, transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='bottom')

plt.savefig('roc_inc_negative.pdf', bbox_inches='tight')

#%% increment permutation transformation test
##################################

num_mes=10
skip=10
num_traj=1000
p0=0.2

#N,s,s_std,inhomog_err
scenarios={r'Drift $N=10^3$':[int(1e3),0,0,0],
           r'Fluctuating $\sigma^2=10^{-2}$':[int(1e10),0,0.01,0],
           r'Directional $s=10^{-2}$':[int(1e10),0.01,0,0]}#,
           #'Neg. Corr.':[int(1e10), np.array([0.001*(-1)**np.floor(t/10) for t in range(num_mes*skip)]), 0, 0]}

fig, ax=plt.subplots(1,1,figsize=[3,3])

n_s=10**3
transformed=False
for _ in scenarios:
    N,s,s_std,inhomog_err=scenarios[_]
    p_vals=perm_incr(gen_traj(N,s,s_std,inhomog_err,p0,n_s,skip,num_mes,num_traj),transformed,True)
    roc1=roc(np.linspace(0,1,num_traj), p_vals, 200)
    ax.plot(roc1[:,0],roc1[:,1],label="Untransformed "+_,linewidth=2)

n_s=10**3
transformed=True
for _ in scenarios:
    N,s,s_std,inhomog_err=scenarios[_]
    p_vals=perm_incr(gen_traj(N,s,s_std,inhomog_err,p0,n_s,skip,num_mes,num_traj),transformed,True)
    roc1=roc(np.linspace(0,1,num_traj), p_vals, 200)
    ax.plot(roc1[:,0],roc1[:,1],'--',label="Transformed "+_,linewidth=2)

ax.set_xlabel('Significance level')
ax.set_ylabel('Rate of positives')

ax.legend(fontsize=5.5)
ax.plot(np.linspace(0,1),np.linspace(0,1),'k--')
 
plt.savefig('roc_inc_trans.pdf', bbox_inches='tight')

#%% increment permutation
##################################
fig, axs=plt.subplots(3,1,figsize=[3,6],constrained_layout=True)

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

for _ in axs.flatten(): _.set_ylim([0,1])

axs[0].legend(fontsize=6, loc='lower right')

for ax, label in zip(axs.flat, 'abc'):
    ax.text(-0.1, 1.05, label, transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='bottom')

plt.savefig('power_NS_incr.pdf', bbox_inches='tight')

#%% Permutation number dependence
##################################

N=10**4
s=0
s_std=0
inhomog_err=False
p0=0.5
n_s=1000
skip=10
num_mes=10
numtraj=1000

traj=gen_traj(N,s,s_std,inhomog_err,p0,n_s,skip,num_mes,numtraj)

sizes=[100000]
min_p=np.zeros(len(sizes))
for i,sample_size in enumerate(sizes):
    #min_p[i]=np.min(perm_freq(traj))
    plt.figure()
    plt.axhline(1/sample_size,c='k')
    plt.plot(np.sort(perm_freq(traj)))

plt.figure()
#plt.plot(sizes,min_p)

##=========================================================================
#%%
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

