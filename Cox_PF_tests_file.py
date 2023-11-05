#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 09:15:24 2023

@author: alvarem
"""

#%%
#import math
import numpy as np
import matplotlib.pyplot as plt 
import progressbar
from scipy import linalg as la
from scipy.sparse import identity
from scipy.sparse import rand
from scipy.sparse import diags
from scipy.sparse import triu
import copy
#from sklearn.linear_model import LinearRegression
from scipy.stats import ortho_group
import time
import PF_functions_def as pff
from Un_cox_PF_functions_def import *
from scipy.stats import multivariate_normal





#%%

# Test and comparison of the M_coup with with M.
"""
np.random.seed(1)
T=20
dim=1
dim_o=dim
xin=np.zeros(dim)+1
l=13
collection_input=[]
I=identity(dim).toarray()

c=2**(-6)
#xin0=np.abs(np.random.normal(1,0,dim))
#xin1=xin0
mu=np.abs(np.random.normal(.5,0,dim))*c
sigs=np.abs(np.random.normal(np.sqrt(2),0,dim))*np.sqrt(c)
#comp_matrix = ortho_group.rvs(dim)
#inv_mat=comp_matrix.T
comp_matrix=np.array([[1]])
inv_mat=comp_matrix.T
S=diags(np.random.normal(1,0,dim),0).toarray()
Sig=inv_mat@S@comp_matrix
fi=[sigs,Sig]
collection_input=[dim, b_gbm,mu,Sig_gbm,fi]
cov=I*1e0
g_pars=[dim,cov]
g_par=cov
x_true=gen_gen_data(T,xin,l,collection_input)
Lamb_par=.5
print(Sig,sigs,mu)
[obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
l=4
N=1000000
d=2**(0)
[x0,x1]=M_coup(xin,xin,b_gbm,mu,Sig_gbm,fi,l,d,N,dim)
x=pff.M(xin,b_gbm,mu,Sig_gbm,fi,l,d,N,dim)
"""

#%%
"""
mean1=np.mean(x1[-1,:,0])
var1=np.var(x1[-1,:,0])
mean=np.mean(x[-1,:,0])
var=np.var(x[-1,:,0])

mean0=np.mean(x0[-1,:,0])
var0=np.var(x0[-1,:,0])
tmean=xin[0]*np.exp(mu*d)
tvar=xin[0]**2*np.exp(2*mu*d)*(np.exp(sigs**2*d)-1)
"""
#%%
"""
print(mean,mean0,mean1,tmean)
print(var,var0,var1,tvar)
"""


#%%
#TEST FOR THE MLPF_COX AND THE COX_PF FOR THE GBM
"""

np.random.seed(1)
T=20
dim=1
dim_o=dim
xin=np.zeros(dim)+1
l=13
collection_input=[]
I=identity(dim).toarray()


#xin0=np.abs(np.random.normal(1,0,dim))
#xin1=xin0
mu=np.abs(np.random.normal(.5,0,dim))
sigs=np.abs(np.random.normal(np.sqrt(2),0,dim))
#comp_matrix = ortho_group.rvs(dim)
#inv_mat=comp_matrix.T
comp_matrix=np.array([[1]])
inv_mat=comp_matrix.T
S=diags(np.random.normal(1,0,dim),0).toarray()
Sig=inv_mat@S@comp_matrix
fi=[sigs,Sig]
collection_input=[dim, b_gbm,mu,Sig_gbm,fi]
cov=I*1e0
g_pars=[dim,cov]
g_par=cov
x_true=gen_gen_data(T,xin,l,collection_input)
Lamb_par=.3
print(Sig,sigs,mu)
[obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
print(obs_time,obs,len(obs_time))
#print(obs_time)
# plots of the test
times=2**(-l)*np.array(range(int(T*2**l+1)))

#plt.plot(times,x_true)
plt.plot(times,x_true,label="True signal")
plt.plot(obs_time,obs,label="Observations")
#plt.plot(times,kbf,label="KBF")
#plt.plot(times,kf,label="KF")
#plt.plot(times,kf-kbf)
#print(len(obs_time))
plt.legend()
"""
#%%
"""
l0=1
L=10
d=2**(1)
N=4
resamp_coef=0.8
l=1
[log_weights,x_pf]=pff.Cox_PF(T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,\
l,d,N,dim,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par)
weights=norm_logweights(log_weights,ax=1)

x_pfmean=np.sum(np.reshape(weights,(int(T/d),N,1))*x_pf,axis=1)
#print(x_pfmean)
plt.plot((np.array(range(int(T/d)))+1)*d,x_pfmean[:,0],label="PF_GBM")
plt.plot(times,x_true,label="True signal")
plt.plot(obs_time,obs,label="Observations")
plt.legend()
"""


#%%

"""
samples=1000
l0=1
L=15
# In this iteration we have eLes and eles, do not confuse. eLes respresents
# the range of maximum number of levels that we take, eles is a direct 
# argument to the MLPF_cox, its the number of levels that we in one ML. 
eles=np.array(range(l0,L+1))
N=200
a=8
aa=0
d=2**(0)
x_pf=np.reshape(np.loadtxt("Observations&data/CCPF_gbm_v6.txt",dtype=float),(2,len(eles),samples,int(T/d),dim))
#x_pf2=np.reshape(np.loadtxt("Observations&data/CCPF_gbm_v5.txt",dtype=float),(2,3,samples,int(T/d),dim))
#x_pf=np.concatenate((x_pf1,x_pf2),axis=1)
#L=15
#eles=np.array(range(l0,L+1))

# CONSTANTS 

Var0=np.var(x_pf[0,0,:,a,0],axis=0)
print(x_pf[0,0,:,a,0])
C=11
C0=3.2
K=15.7
print(Var0*N)
#x_pf_or=np.reshape(np.loadtxt("Observations&data/CCPF_v5.txt",dtype=float),(2,len(eles),samples,int(T/d),dim))
"""
#COUPLING's SECOND MOMENT
"""
sm=np.mean(((x_pf[0]-x_pf[1])**2)[:,:,a],axis=1)
bias
sm_lb=sm-np.sqrt(var_sm)*1.96/np.sqrt(samples)
var=np.var((x_pf[0]-x_pf[1])[:,:,a],axis=1)
[b0,b1]=coef(eles,np.log2(sm[:,0]))
print(b0,b1,(2**b0)*N)
# C=11
#print(x_pf_or.shape,sm.shape,((x_pf_or[0]-x_pf_or[-1])**2)[:,a].shape)
reference_sm=np.array([1/2**(i/2) for i in eles])*2**(eles[0]/2)*sm[0]
reference_var=np.array([1/2**(i/2) for i in eles])*2**(eles[0]/2)*var[0]
#print(sm.shape)
plt.plot(eles,reference_sm,label="Delta_l^(1/2)")
plt.plot(eles,sm_lb,label="LB ")
plt.plot(eles,sm_ub,label="UB")
plt.plot(eles,sm,label="Second Moment")
plt.plot(eles,var,label="Variance")
plt.title("Second moment of the cox observed PF  with N=500 and 100 samples")
"""
#CONCLUSION: The second moment follows the rate (Delta_l^{-1})^2=1/2**(2*l)

#COUPLING'S BIAS

# In the following we plot 3 different ways of computing the bias and 
# additionally compute the constant k from bias_l^2=KDelta_l^2 in three 
# different ways, and then average these values, k=15.7
"""
Cbias=np.abs(np.mean(x_pf[1,:,:,a]-x_pf[0,:,:,a],axis=1))
var_of_bias=np.var(x_pf[1,:,:,a]-x_pf[0,:,:,a],axis=1)
refence_Cbias=np.array([1/2**(i) for i in eles])*2**(eles[3])*Cbias[3]
Cbias_ub=Cbias+np.sqrt(var_of_bias)*1.96/np.sqrt(samples)
Cbias_lb=Cbias-np.sqrt(var_of_bias)*1.96/np.sqrt(samples)
plt.plot(eles[3:],Cbias[3:],label="Sampled Cbias")
plt.plot(eles[3:],Cbias_ub[3:],label="Cbias UB")
plt.plot(eles[3:],Cbias_lb[3:],label="Cbias LB")
print(Cbias.shape)
[b0,b1]=coef(eles[3:],np.log2(Cbias[3:,0]))
print([b0,b1,2**(2*b0)])
k1=2**(2*b0)
#k=0.0250
#plt.plot(eles,refence_Cbias**2,label="Delta_l**2")
plt.plot(eles[3:],refence_Cbias[3:],label="Delta_l")
plt.title("bias cox observed PF  with N=200 and 1000 samples")
"""

"""
aa=0
bias_r=np.abs(np.mean(x_pf[aa,1:]-x_pf[aa,:-1],axis=1))
[b0,b1]=coef(eles[3:-1],np.log2(bias_r[3:,a,0]))
print([b0,b1,2**(2*b0)])
k0=2**(2*b0)
plt.plot(eles[3:-1],2**b0*2**(eles[3:-1]*b1))
plt.plot(eles[:-1],bias_r[:,a,0],label="Sampled bias 0")

aa=1
bias_r=np.abs(np.mean(x_pf[aa,1:]-x_pf[aa,:-1],axis=1))
[b0,b1]=coef(eles[4:-1],np.log2(bias_r[3:-1,a,0]))
print([b0,b1,2**(2*b0)])
k2=2**(2*b0)
reference_bias=np.array([1/2**(i) for i in eles[1:]])*2**(eles[1])*bias_r[0,a,0]
plt.plot(eles[1:],bias_r[:,a,0],label="Sampled bias 1")
print((k1+k2+k0)/3)
#reference_bias_s=np.array([1/2**(2*i) for i in eles[1:]])*2**(2*eles[1])*bias_r[0,a,0]**2
#x_pf_r_ub=x_pf_r+np.sqrt(var_pf[:,1:,a]+var_pf[:,:-1,a])*1.96/np.sqrt(samples)
#x_pf_r_lb=x_pf_r-np.sqrt(var_pf[:,1:,a]+var_pf[:,:-1,a])*1.96/np.sqrt(samples)
#x_pf_r_coup=np.abs(x_pf[1,:,a]-x_pf[0,:,a])
#bias_ub=np.sqrt(var_pf[:,:,a])*1.96/np.sqrt(samples)+bias
#print(bias_ub.shape)
#bias_lb=np.maximum(-np.sqrt(var_pf[:,:,a])*1.96/np.sqrt(samples)+bias,1e-5)
#dtimes=np.array(range(int(T/d)))*d+d
#plt.plot(eles[1:],reference_bias,label="Delta_l")
#plt.title("bias of the cox observed PF  with N=1000 and 1000 samples")
#plt.xlabel("l")

#print(bias_r[0])

#plt.plot(eles[1:],x_pf_r_ub[bi,:,0])
#plt.plot(eles[1:],x_pf_r_lb[bi,:,0])
#plt.plot(eles,reference)
"""

# CONCLUSION: THE BIAS FOLLOWS THE RATE DELTA_l



"""
#plt.plot(dtimes,x_pf[1,-1],label="PF")

#THE MEAN OF THE TWO ks IS 
#print((0.025+0.0204)/2)
#k=0.0227
#plt.plot(times,v,label="True signal")
#plt.plot(obs_time,obs,label="Observations")
plt.xlabel("l")

plt.legend()
plt.yscale("log")
#plt.savefig("Images/bias_CCPF_gbm_v3&2.pdf")
plt.show() 
"""

#%%
# CONSTRUCTION OF THE TRUTH FOR THE MLPF_COX WITH GBM

"""
samples=1
L=8

l0=0
# In this iteration we have eLes and eles, do not confuse. eLes respresents
# the range of maximum number of levels that we take, eles is a direct 
# argument to the MLPF_cox, its the number of levels that we in one ML. 
eles=np.array(np.arange(l0,L+1))
x_l_pf=np.reshape(np.loadtxt("Observations&data/MLPF_cox_gbm_truth_by_parts.txt",\
dtype=float),(2,len(eles),int(T/d),dim))
pf_truth=x_l_pf[0,0,:,:]+np.sum(x_l_pf[1,1:]-x_l_pf[0,1:],axis=0)
"""
#%%

# TEST FOR THE MLPF_COX WITH GBM
"""

samples=50
d=2**(0)
Lmin=0
Lmax=6
l0=0
a=9
eLes=np.array(range(Lmin,Lmax+1))

pf=np.reshape(np.loadtxt("Observations&data/MLPF_cox_gbm_v1.txt",dtype=float),(len(eLes),samples,int(T/d),dim))

"""

"""
pf_var=np.var(pf,axis=1)
plt.plot(eLes,pf_var[:,a],label="Variance")
reference_var=np.array([1/2**(2*i) for i in eLes])*2**(2*eLes[1])*pf_var[1,a] 
print(pf_var[0,a] )
[b0,b1]=coef(eLes,np.log2(np.sqrt(pf_var))[:,-1,0])
#print(b0,b1,N0*2**(2*b0))
#C=3
#C=.15
#N0=C/K=132
plt.plot(eLes,reference_var ,label="Delta_L^2" )
#plt.title("MLPF with Cox observations depending on level L")
#plt.xlabel("L")
"""
#COUPLING'S BIAS
"""
pf_r=pf[1:,:,a]-pf[:-1,:,a]
bias=np.abs(np.mean((pf-pf_truth)[:,:,a],axis=1))
bias_r=np.abs(np.mean(pf_r,axis=1))
reference_bias=np.array([1/2**(i) for i in eLes[1:]])*2**(eLes[1])*bias_r[1]

#x_pf_r_ub=x_pf_r+np.sqrt(var_pf[:,1:,a]+var_pf[:,:-1,a])*1.96/np.sqrt(samples)
#x_pf_r_lb=x_pf_r-np.sqrt(var_pf[:,1:,a]+var_pf[:,:-1,a])*1.96/np.sqrt(samples)
#x_pf_r_coup=np.abs(x_pf[1,:,a]-x_pf[0,:,a])
#bias_ub=np.sqrt(var_pf[:,:,a])*1.96/np.sqrt(samples)+bias
#print(bias_ub.shape)
#bias_lb=np.maximum(-np.sqrt(var_pf[:,:,a])*1.96/np.sqrt(samples)+bias,1e-5)



#print(x_pf_r_ub.shape)
#reference=np.array([1/2**(i) for i in eles])*2**eles[0]*x_pf_r[0,0]
#dtimes=np.array(range(int(T/d)))*d+d
plt.plot(eLes[:-1],bias_r,label="Sampled R_bias")
plt.plot(eLes[1:],reference_bias,label="Delta_l")
print(bias)

plt.plot(eLes,bias,label="Sampled bias")
#plt.title("bias of the cox observed PF  with N=1000 and 1000 samples")
plt.xlabel("l")


#plt.plot(eles[1:],x_pf_r_ub[bi,:,0])
#plt.plot(eles[1:],x_pf_r_lb[bi,:,0])
#plt.plot(eles,reference)
#plt.plot(dtimes,x_pf[1,-1],label="PF")
"""
#MSE
"""
MSE=np.nanmean((pf-pf_truth)**2,axis=1)[:,a,0]
#print(pf[0,:,a])
#plt.plot(eLes,MSE,label="MSE")
#print(MSE.shape)
#BIAS
bias=np.mean(pf-pf_truth,axis=1)[:,a,0]
#plt.plot(eLes,bias**2,label="Bias")

#plt.plot(times,v,label="True signal")
#plt.plot(obs_time,obs,label="Observations")


eLes=np.array(np.arange(Lmin,Lmax+1))

C=11
C0=3.2
K=15.7
Cost=np.zeros(len(eLes))


    

plt.plot(MSE,Cost,label="Cost")
[b0,b1]=coef(np.log(MSE),np.log(Cost))
plt.plot(MSE,np.exp( b0)*MSE**b1,label=r"$\varepsilon^{-2.27}$")
#print(MSE)
#plt.plot(MSE,np.abs(MSE**(-1)*(np.log(np.sqrt(MSE))**2))+10,label=r"$\varepsilon^{-2}\log(\varepsilon)^2$")
print(b0,b1,2*b1)
plt.xlabel(r"$\varepsilon^{2}$")

#Cost=N0*2**(eLes*2.5)*np.array([np.sum()])
"""
"""

plt.yscale("log")
plt.xscale("log")
plt.title("Cost in terms of the MSE with a GBM for 50 samples of the MLPFCox ")
plt.legend()
plt.savefig("Images/MSE_MLPF_cox_GBM_1.pdf")
plt.show()


#%%
"""

# TEST FOR THE SINGLE TERM UNBIASED ESTIMATOR WITH OU PROCESS AND COX OBS
"""

if True==True:
        np.random.seed(1)
        T=20
        dim=1
        dim_o=dim
        xin=np.zeros(dim)+1
        l=15
        collection_input=[]
        I=identity(dim).toarray()
        #comp_matrix = ortho_group.rvs(dim)
        comp_matrix=np.array([[1]])
        inv_mat=la.inv(comp_matrix)
        #S=diags(np.random.normal(1,0.1,dim),0).toarray()
        S=diags(np.random.normal(0.99,0.1,dim),0).toarray()
        #S=np.array([[1.]])
        fi=inv_mat@S@comp_matrix
        #B=diags(np.random.normal(-1,0.1,dim),0).toarray()
        B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
        B=inv_mat@B@comp_matrix
        #B=np.array([[-1.]])
        #print(B)
        #print(B)
        #B=comp_matrix-comp_matrix.T  +B 
        collection_input=[dim, b_ou,B,Sig_ou,fi]
        cov=I*1e0
        g_pars=[dim,cov]
        g_par=cov
        x_true=gen_gen_data(T,xin,l,collection_input)
        Lamb_par=1.1
        [obs_time,obs]=gen_obs(x_true,l,T,Lambda,Lamb_par,g_normal,g_pars)

    start=time.time()
    d=2**(0)
    resamp_coef=0.8    
    dim_out=2
    samples=100000
    
    lmax=7
    l0=0
    pmax=7
    N0=50
    
    eles=np.arange(l0,lmax+1)

    ps=np.arange(0,pmax+1)
    enes=N0*2**ps
    print(enes)
    Pl=(eles+1)*np.log(eles+2)**2/2**(eles)
    Pl[0]=2*Pl[1]
    Pl=Pl/np.sum(Pl)
    Pl_cu=np.cumsum(Pl)
    Pp=(ps+1)*np.log(ps+2)**2/2**(ps)
    Pp[0]=2*Pp[1]
    Pp=Pp/np.sum(Pp)
    Pp_cu=np.cumsum(Pp)
    print(Pl, Pp)

dtimes=np.array(range(int(T/d)))*d+d
N=100
[lw,x]=pff.Cox_PF(T, xin, b_ou, B, Sig_ou, fi, obs, obs_time, l, d,N\
, dim, resamp_coef, g_den, g_par, Lambda, Lamb_par)
weights=norm_logweights(lw,ax=1)           
phi_pfmean=np.sum(np.reshape(weights,(int(T/d),N,1))*x,axis=1)
tel_est=phi_pfmean
#plt.plot(times,x_true)
dtimes=np.array(range(int(T/d)))*d+d
times=2**(-l)*np.array(range(int(T*2**l+1)))
plt.plot(times,x_true,label="True signal")
plt.plot(dtimes,phi_pfmean,label="PF")
plt.plot(obs_time,obs,label="Observations")
#plt.plot(times,kbf,label="KBF")
#plt.plot(times,kf,label="KF")
#plt.plot(times,kf-kbf)
#print(len(obs_time))
plt.legend()
"""
#%%
"""
d=2**(-1)
eles=np.array([5,6])
enes=np.array([50,100])
Pl=np.array([0.5,1])
Pp=np.array([0.5,1])
resamp_coef=0.8




output=SUCPF(T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,eles,Pl,d,enes,Pp,dim,resamp_coef,phi,\
g_den,g_par,Lambda,Lamb_par)

#%%

dtimes=np.array(range(int(T/d)))*d+d
#plt.plot(times,x_true)
plt.plot(times,x_true,label="True signal")
plt.plot(obs_time,obs,label="Observations")
#plt.plot( dtimes,output[:,0],label="PF")
#print(output.shape)

#plt.plot(times,kbf,label="KBF")
#plt.plot(times,kf,label="KF")
#plt.plot(times,kf-kbf)
#print(len(obs_time))
plt.legend()
    

#%%
samples=100000
levels=np.zeros((samples,2),dtype=int)
for sample in range(samples):
    
    np.random.seed(sample+1000)
    l=eles[sampling(Pl_cu)]
    p=sampling(Pp_cu)
    levels[sample]=[l,p]
np.savetxt("Observations&data/SUCPF_levels_v2.txt",levels)
"""
#%%
"""
# THE FOLLOWING CODE IS MADE TO CHECK THE COST IN TERMS OF THE VARIANCE OF THE ESTIMATOR. 
Us=np.reshape(np.loadtxt("Observations&data/SUCPF_v2.txt",dtype=float),(len(eles),samples,int(T/d),dim))
Us=Us[1,:,:,:]
"""
#%%
"""
inv_prob=1/np.reshape(Pl[levels[:,0]]*Pp[levels[:,1]],(-1,1,1))
Us=Us*inv_prob
"""
#%%

"""
plt.plot(times,x_true,label="True signal")
plt.plot(dtimes,Us[38,:,0],label="PF")
plt.plot(obs_time,obs,label="Observations")
plt.legend()
"""
# %%

"""
def CS(l,p):
    return 2**(l)*N0*2**(p)
"""
"""
The following is the code takes a geometrically increasing numpy list that corresponds to the elements of
the unbiased estimator (Us) that we consider and computes the corresponding variance and costs (CS) for those
elements. Finally we plot the variance as funciton of the cost.
"""

"""
variances=np.var(Us,axis=0)/Ms
var_variances=np.zeros((len(Ms),int(T/d)))
for i in range(len(Ms)):
    var_variances[i,:]=np.var(Us[:Ms[i],:],axis=0)[:,0]
print(variances.shape)
costs=np.zeros((len(Ms)))
elem=0
prevM=0
for i in range(len(Ms)):
    for j in range(Ms[i]-prevM):
        costs[i]+=CS(levels[elem,0],levels[elem,1])
        prevM=Ms[i]
        elem+=1
costs=np.cumsum(costs)
plt.plot(variances[-1],costs)

plt.plot(variances[-1],np.abs(variances[-1]**(-1)*(np.log(np.sqrt(variances[-1]))**2)),label=r"$\varepsilon^{-2}\log(\varepsilon)^2$")
plt.xlabel("Cost")
plt.ylabel("Variance")  
plt.title("Variance as function of cost")
plt.xscale("log")
plt.yscale("log")
plt.show()
plt.plot(var_variances[:,:])
"""
#%%
"""
l=15
U_mean=np.mean(Us,axis=0)
times=2**(-l)*np.array(range(int(T*2**l+1)))
print(times.shape,x_true.shape)
plt.plot(times,x_true,label="True signal")
plt.plot(dtimes,U_mean[:,0],label="UPF")
plt.plot(obs_time,obs,label="Observations")
#plt.plot(times,kbf,label="KBF")
#plt.plot(times,kf,label="KF")
#plt.plot(times,kf-kbf)
#print(len(obs_time))
plt.legend()
"""
# %%

# THE FOLLOWING CODE TESTS THE VARIANCE OF THE ESTIMATORS ALONG THE TIME DISCRETIZATION LEVEL FOR OU SIGNAL

"""
samples=1000
    
Lmax=8
l0=0
p=1
N0=250
eLes=np.arange(l0,Lmax+1)
ps=np.arange(0,p+1)
enes=N0*2**ps
print(enes)
tel_est=np.zeros((len(eLes),samples,int(T/d),dim))
tel_est=np.reshape(np.loadtxt("Observations&data/SUCPF_l_levels_v1.txt",dtype=float),(len(eLes),samples,int(T/d),dim))
print(tel_est.shape)
"""
# %%
"""
a=19
var=np.var(tel_est,axis=1)[:,a,0]
var_var=np.var((tel_est-np.mean(tel_est,keepdims=True))**2,axis=1)[:,a,0]
var_lb=np.maximum(-np.sqrt(var_var)*1.96/np.sqrt(samples)+var,1e-6)

reference_var=np.array([1/2**(i) for i in eLes])*2**(eLes[1])*var[1]
plt.plot(eLes[1:],var[1:],label="Variance")
plt.plot(eLes[1:],var_lb[1:],label="CI lower bound")
plt.plot(eLes[1:],reference_var[1:],label="$\Delta_l$")
plt.xlabel("l")
plt.title("Variance of the telescoping estimator for varying l")
plt.legend()
plt.yscale("log")
#plt.savefig("Images/SUCPF_l_levels_var.pdf")
plt.show() 
"""

# %%
# THE FOLLOWING CODE TESTS THE VARIANCE OF THE ESTIMATORS ALONG THE NUMBER OF PARTICLES LEVEL FOR OU SIGNAL
"""
samples=3000
L=5
l0=4
p=8
N0=4
eLes=np.arange(l0,L+1)
ps=np.arange(0,p+1)
eNes=N0*2**ps
print(eNes)
tel_est=np.reshape(np.loadtxt("Observations&data/SUCPF_p_levels_v1.txt",dtype=float),(len(eNes)-1,samples,int(T/d),dim))
print(tel_est.shape)
"""
#%%

"""
a=19
var=np.var(tel_est,axis=1)[:,a,0]
var_var=np.var((tel_est-np.mean(tel_est,keepdims=True))**2,axis=1)[:,a,0]
var_lb=np.maximum(-np.sqrt(var_var)*1.96/np.sqrt(samples)+var,1e-6)

reference_var=np.array([1/2**(i) for i in ps[1:]])*2**(ps[1])*var[0]
plt.plot(ps[1:],var,label="Variance")
plt.plot(ps[1:],var_lb,label="CI lower bound")
plt.plot(ps[1:],reference_var,label="$1/N_p$")
plt.xlabel("p")
plt.title("Variance of the telescoping estimator for varying p")
plt.legend()
plt.yscale("log")
plt.savefig("Images/SUCPF_p_levels_var.pdf")
plt.show() 
"""
# %%
# THIS CODE TEST THE VARIANCE OF THE UNBIASED PARTICLE FILTER WITH OU SIGNAL 
"""

samples=100000
levels= np.loadtxt("Observations&data/SUCPF_levels_v2.txt",dtype=float)
Us=np.reshape(np.loadtxt("Observations&data/SUCPF_v2.txt",dtype=float),(len(eles),samples,int(T/d),dim))
Us=Us[1,:,:,:]
inv_prob=1/np.reshape(Pl[levels[:,0]]*Pp[levels[:,1]],(-1,1,1))
Us=Us*inv_prob
"""
#%%
"""
np.random.seed(0)
Ms=np.array(np.concatenate((np.array([2**i for i in range(3,int(np.floor(np.log2(samples/100)+1)))]),[samples/100])),dtype=int)
print(Us.shape)
tel_est_ind=np.array(range(100000))
#print(np.random.choice(tel_est_ind,replace=True,size=(Ms[0],100)))
#print(np.random.choice(tel_est_ind,replace=True,size=(Ms[0],100)).shape)
U_vars=np.zeros((len(Ms),int(T/d),dim))
for i in range(len(Ms)):
    U_samples=np.random.choice(tel_est_ind,replace=True,size=(Ms[i],100))
    U_vars[i]=np.var(np.mean(Us[U_samples],axis=0), axis=0)

plt.plot(1/Ms,U_vars[:,-1,0],label="Unbiased")
plt.yscale("log")
plt.xscale("log")
reference=(np.array([1/Ms])*Ms[0]*(U_vars[0,-1,0]))[0,:]
print(reference.shape)
plt.plot(1/Ms,reference,label="Reference")
plt.xlabel("1/N")
plt.legend()
"""

# %%
"""
if True==True:
    np.random.seed(1)
    T=20
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=15
    collection_input=[]
    I=identity(dim).toarray()
    
    #comp_matrix = ortho_group.rvs(dim)
    comp_matrix=np.array([[1]])
    inv_mat=la.inv(comp_matrix)
    #S=diags(np.random.normal(1,0.1,dim),0).toarray()
    S=diags(np.random.normal(0.99,0.1,dim),0).toarray()
    #S=np.array([[1.]])
    fi=inv_mat@S@comp_matrix
    #B=diags(np.random.normal(-1,0.1,dim),0).toarray()
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=inv_mat@B@comp_matrix
    #B=np.array([[-1.]])
    #print(B)
    #print(B)
    #B=comp_matrix-comp_matrix.T  +B 
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    cov=I*1e0
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=1.1
    [obs_time,obs]=gen_obs(x_true,l,T,Lambda,Lamb_par,g_normal,g_pars)

    
    d=2**(0)
    resamp_coef=0.8    
    L=6
    l0=5
    """
#%%
"""
l=5
N0=4
N=N0*2**8*40*100
print(N)
[lw,x]=pff.Cox_PF(T, xin, b_ou, B, Sig_ou, fi, obs, obs_time, l, d,N\
, dim, resamp_coef, g_den, g_par, Lambda, Lamb_par)

weights=norm_logweights(lw,ax=1)           
phi_pfmean=np.sum(np.reshape(weights,(int(T/d),N,1))*phi(x,ax=2),axis=1)
#np.savetxt("Observations&data/SUCPF_N20480.txt",phi_pfmean)
"""
# %%
#"""
samples=3000
    
L=6
l0=5
p=8
N0=4
eLes=np.arange(l0,L+1)
ps=np.arange(0,p+1)
eNes=N0*2**ps
a=99
tel_est0=np.reshape(np.loadtxt("Observations&data/SUCPF_p_levels_bias_est_v1.txt",dtype=float),(2,len(eNes)-1,samples,int(T/d),dim))
p=3
N0=4*2**8
eLes=np.arange(l0,L+1)
ps=np.arange(0,p+1)
eNes=N0*2**ps
tel_est1=np.reshape(np.loadtxt("Observations&data/SUCPF_p_levels_bias_est_v2.txt",dtype=float),(2,len(eNes)-1,samples,int(T/d),dim))
p=11
N0=4
eLes=np.arange(l0,L+1)
ps=np.arange(0,p+1)
eNes=N0*2**ps
tel_est=np.concatenate((tel_est0,tel_est1),axis=0)
tel_est_exact=np.reshape(np.loadtxt("Observations&data/SUCPF_N20480.txt",dtype=float),(int(T/d),dim))
bias=np.abs(np.mean(tel_est-tel_est_exact,axis=1)[:,a,0])   
print(ps[1:]/eNes[1:],bias)
reference=(ps[1:]/eNes[1:])*bias[0]*eNes[1]/ps[1]
plt.plot(ps[1:]/eNes[1:],reference,label="Ref: $x=y$")
plt.plot(ps[1:]/eNes[1:],bias,label="bias")
plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"$\frac{p}{N_p}$")
plt.title("Bias in terms of the particle levels")
plt.legend()
plt.savefig("Images/Bias_varying_N.pdf")
plt.show()  
#"""
# %%
# THE FOLLOWING SECTION CORRESPONDS TO THE TESTING, AND SAMPLING OF THE 
# MLPF_COX WITH GBM. 

####!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# IN THE FOLLOWING WE WORK WITH THE GBM IN ORDER TO GET APPLY THE MULTILEVEL
# PARTICLE FILTER WITH COX OBSERVATIONS OR MLPF COX.


# WE USE THE FOLLOWING MODEL

"""

np.random.seed(1)
T=20
dim=1
dim_o=dim
xin=np.zeros(dim)+1
l=13
collection_input=[]
I=identity(dim).toarray()

c=2**(-6)
#xin0=np.abs(np.random.normal(1,0,dim))
#xin1=xin0
mu=np.abs(np.random.normal(.5,0,dim))*c
sigs=np.abs(np.random.normal(np.sqrt(2),0,dim))*np.sqrt(c)
#comp_matrix = ortho_group.rvs(dim)
#inv_mat=comp_matrix.T
comp_matrix=np.array([[1]])
inv_mat=comp_matrix.T
S=diags(np.random.normal(1,0,dim),0).toarray()
Sig=inv_mat@S@comp_matrix
fi=[sigs,Sig]
collection_input=[dim, b_gbm,mu,Sig_gbm,fi]
cov=I*1e0
g_pars=[dim,cov]
g_par=cov
x_true=gen_gen_data(T,xin,l,collection_input)
Lamb_par=.5
print(Sig,sigs,mu)
[obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
print(obs_time,obs,len(obs_time))
#print(obs_time)
# plots of the test
times=2**(-l)*np.array(range(int(T*2**l+1)))

#plt.plot(times,x_true)
plt.plot(times,x_true,label="True signal")
plt.plot(obs_time,obs,label="Observations")
#plt.plot(times,kbf,label="KBF")
#plt.plot(times,kf,label="KF")
#plt.plot(times,kf-kbf)
#print(len(obs_time))
plt.legend()
"""
#%%
"""
l=8
N=100
d=2**(0)
resamp_coef=0.8
[log_weights,x_pf]=pff.Cox_PF(T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l\
,d,N,dim,resamp_coef,g_den,g_par,Lambda,Lamb_par)
weights=norm_logweights(log_weights,ax=1)
print(weights.shape,x_pf.shape)   
phi_pfmean=np.sum(np.reshape(weights,(int(T/d),N,1))*x_pf,axis=1)
phi_pf=phi_pfmean.flatten()
dtimes=np.array(range(int(T/d)))*d+d
plt.plot(dtimes,phi_pf,label="PF")
plt.plot(times,x_true,label="True signal")
plt.plot(obs_time,obs,label="Observations")
#plt.plot(times,kbf,label="KBF")
#plt.plot(times,kf,label="KF")
#plt.plot(times,kf-kbf)
#print(len(obs_time))
plt.legend()
plt.show()

[log_weights0,log_weights1,x0_pf,x1_pf]=CCPF(T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l\
,d,N,dim,resamp_coef,g_den,g_par,Lambda,Lamb_par)
weights0=norm_logweights(log_weights0,ax=1)
print(weights0.shape,x0_pf.shape)   
phi_pfmean=np.sum(np.reshape(weights0,(int(T/d),N,1))*x0_pf,axis=1)
phi_pf=phi_pfmean.flatten()
plt.plot(dtimes,phi_pf,label="PF")
plt.plot(times,x_true,label="True signal")
plt.plot(obs_time,obs,label="Observations")
#plt.plot(times,kbf,label="KBF")
#plt.plot(times,kf,label="KF")
#plt.plot(times,kf-kbf)
#print(len(obs_time))
plt.legend()
plt.show()
"""

#%%
"""
dtimes=np.array(range(int(T/d)))*d+d
plt.plot(dtimes,phi_pf,label="PF")
plt.plot(times,x_true,label="True signal")
plt.plot(obs_time,obs,label="Observations")
#plt.plot(times,kbf,label="KBF")
#plt.plot(times,kf,label="KF")
#plt.plot(times,kf-kbf)
#print(len(obs_time))
plt.legend()
"""
#%%
#%%
# TEST FOR THE COUPLING OF THE COX PF WITH GBM. THIS TEST IS MADE IN ORDER TO 
# CHECK THE VARIANCE AND BIAS RATES THAT ALLOW THE MLPF AND THE UNBIASED 
# ESTIMATOR TO WORK
"""
samples=500
l0=1
L=10
eles=np.array(range(l0,L+1))
N=5000
a=19
aa=0
d=2**(0)
x_pf=np.reshape(np.loadtxt("Observations&data/CCPF_gbm20_p_v5.txt",dtype=float),(2,len(eles),samples,int(T/d),dim))
"""
#%%
#PARAMETERS 
#C=20
#C0=11
#K=0.0072
# BIAS
"""
rbias=np.abs(np.mean(x_pf[1,1:]-x_pf[1,:-1],axis=1)[:,a,0])
plt.plot(eles[1:7],rbias[:6],label="Rbias")
reference=np.array([1/2**(i) for i in range(len(eles[1:]))])*rbias[0]
var=np.var(x_pf[1,1:]-x_pf[1,:-1],axis=1)[:,a,0]
#print(var,np.sqrt(var)*1.96/np.sqrt(samples),np.sqrt(samples),np.sqrt(var))
rbias_ub=rbias+np.sqrt(var)*1.96/np.sqrt(samples)
rbias_lb=rbias-np.sqrt(var)*1.96/np.sqrt(samples)
#plt.plot(eles[1:],rbias_ub)
#plt.plot(eles[1:],rbias_lb)
plt.plot(eles[1:7],reference[:6],label=r"$\Delta_l$")
[b0,b1]=coef(eles[1:7],np.log2(rbias[:6]))
print([b0,b1,2**(2*b0)])

#k=0.0250
"""
    
# VARIANCE
"""
sm=np.mean((x_pf[0]-x_pf[1])**2,axis=1)[:,a,0]
#print((x_pf[0]-x_pf[1])[:2,:,a,0])
reference=np.array([1/2**(i/2) for i in range(len(eles))])*sm[0]
plt.plot(eles,sm,label="Second moment the coupling")
plt.plot(eles,reference,label=r"$\Delta_l^{1/2}$")
var_sm=np.var(((x_pf[0]-x_pf[1])**2)[:,:,a],axis=1)
sm_ub=sm+np.sqrt(var_sm)*1.96/np.sqrt(samples)
sm_lb=sm-np.sqrt(var_sm)*1.96/np.sqrt(samples)
var=np.var((x_pf[0]-x_pf[1])[:,:,a],axis=1)
var0=np.var((x_pf[0])[:,:,a],axis=1)
print(var0*N)
[b0,b1]=coef(eles,np.log2(sm))
print(b0,b1,(2**b0)*N)
"""
"""
plt.xlabel("l")
plt.legend()
plt.yscale("log")
plt.title("Richardson bias in terms of the time discretization levels")
#plt.savefig("Images/Rbias_CCPF_gbm20_p_v5.pdf")
plt.show()
"""
#%%
#TEST FOR THE VARIANCES OF THE MLPF LEVELS

# In the following, depending of the lenght of the MLPF we get the variance 
# of the levels.
"""
samples=100
l0=0
Lmin=0
Lmax=5
eLes=np.array(np.arange(Lmin,Lmax+1)) 

x_pf=np.reshape(np.loadtxt("Observations&data/PMLPF_cox_gbm20_p_levels_v2.txt"\
,dtype=float),(2,len(eLes),len(eLes),samples,int(T/d),dim))
   """ 
#%%
"""
var0=np.var(x_pf[1,:,0],axis=1)[:,a,0]
plt.plot(var0)

plt.legend()
plt.yscale("log")

plt.show()
#%%

L=5
CL=2**(1/4)*(2**(L/4)-1)/(2**(1/4)-1)
NB0=(np.sqrt(C0)+np.sqrt(3*C/2)*CL)*np.sqrt(2*C/3)*2**(-L/4)/K
N0=np.sqrt(C0)*(np.sqrt(C0)+np.sqrt(3*C/2)*CL)*2**(2*L)/K
eles=np.array(np.arange(l0,L+1))
eNes=np.zeros(len(eles),dtype=int)
a=3e-2
eNes[0]=N0*a
eNes[1:]=NB0*2**(-eles[1:]*3/4)*2**(L*(2+1/4))
print(eNes)
eles=np.array(np.arange(l0,L+1))
a=19
pfs_le=x_pf[:,L,:len(eles)]
sm=np.mean((pfs_le[0]-pfs_le[1])**2,axis=1)[:,a,0]
var=np.var((pfs_le[0]-pfs_le[1]),axis=1)[:,a,0]
reference=np.array([2**(i/4) for i in range(len(eles))])*sm[1]/2**(1/4)

print(eles[1:])
plt.plot(eles[1:], sm[1:],label="SM")
plt.plot(eles[1:], reference[1:],label="ref")
plt.plot(eles[1:], var[1:],label="var")

plt.legend()
plt.yscale("log")

plt.show()
"""
#%%

"""
samples=100

l0=0
Lmin=0
Lmax=5
eLes=np.array(np.arange(Lmin,Lmax+1))
x_pf=np.reshape(np.loadtxt("Observations&data/PMLPF_cox_gbm20_p_levels_v2.txt"\
,dtype=float),(2,len(eLes),len(eLes),samples,int(T/d),dim))
a=3e-2
pfs_n=x_pf[:,1:,1,:]
var_n=np.var(pfs_n[1],axis=1)[:,a,0]
ELES=np.array(np.arange(1,6))
print(ELES)
CL=2**(1/4)*(2**(ELES/4)-1)/(2**(1/4)-1)
NB0=(np.sqrt(C0)+np.sqrt(3*C/2)*CL)*np.sqrt(2*C/3)*2**(-ELES/4)/K

ENES=a*NB0*2**(-1*3/4)*2**(ELES*(2+1/4))
print(ENES)
plt.plot(ENES,var_n)


plt.legend()
plt.yscale("log")
plt.xscale("log")

plt.show()
"""
#%%
# IN THIS PART WE TEST THE RELATION VAR=C/N for the variance of the Cox_pf
# and the CCPF, THIS TEST IS CARRIED OUT SINCE WE OBSERVE THAT THIS IS NOT HTE 
# CASE 
"""
samples=200
l=4
Lmin=0
Lmax=5
co=3e-3
a=19 
C=20
C0=11
K=0.0072

ELES=np.array(np.arange(1,6))
print(ELES)
CL=2**(1/4)*(2**(ELES/4)-1)/(2**(1/4)-1)
NB0=(np.sqrt(C0)+np.sqrt(3*C/2)*CL)*np.sqrt(2*C/3)*2**(-ELES/4)/K
enes=np.array(co*NB0*2**(-1*3/4)*2**(ELES*(2+1/4)) ,dtype=int)

pfsC=np.reshape(np.loadtxt("Observations&data/CCPF_gbm_enes_v11.txt"\
,dtype=float),(2,len(enes),samples,int(T/d),dim))
pfsS=np.reshape(np.loadtxt("Observations&data/Cox_pf_gbm_enes_v11.txt"\
,dtype=float),(len(enes),samples,int(T/d),dim))
varC=np.var(pfsC[0]-pfsC[1],axis=1)[:,a,0]
varS=np.var(pfsS,axis=1)[:,a,0]
plt.plot(enes,varC,label="Coupling")
plt.plot(enes,varS,label="Single")
plt.legend()
plt.yscale("log")
plt.xscale("log")
plt.savefig("Images/couplingvssingle.pdf")
plt.show()
"""
#%%
"""
for i in range(len(eLes)):
            L=eLes[i]
            CL=2**(1/4)*(2**(L/4)-1)/(2**(1/4)-1)
            NB0=(np.sqrt(C0)+np.sqrt(3*C/2)*CL)*np.sqrt(2*C/3)*2**(-L/4)/K
            N0=np.sqrt(C0)*(np.sqrt(C0)+np.sqrt(3*C/2)*CL)*2**(2*L)/K
            eles=np.array(np.arange(l0,L+1))
            eNes=np.zeros(len(eles),dtype=int)
            a=3e-2
            eNes[0]=N0*a
            eNes[1:]=NB0*2**(-eles[1:]*3/4)*2**(L*(2+1/4))
            plt.plot(eNes[1:],label="eNes")
            plt.plot(2**(-eles[1:]*3/4)*eNes[0],label="ref")
            plt.legend()
            plt.yscale("log")
            plt.show()
"""
#%%
"""
k=2**(-6)
mu=np.abs(np.random.normal(.5,0,dim))*k
sigs=np.abs(np.random.normal(np.sqrt(2),0,dim))*np.sqrt(k)
print(sigs,mu)
"""
#%%
"""
C=20
C0=11
K=0.0072

print(C0*2**(2*7))
l0=0
Lmin=0
Lmax=5
eLes=np.array(np.arange(Lmin,Lmax+1))
for i in range(len(eLes)):
            L=eLes[i]
            CL=2**(1/4)*(2**(L/4)-1)/(2**(1/4)-1)
            NB0=C*CL*2**(-L/4.)/K
            N0=C0*2**(2*L)/(K)
            eles=np.array(np.arange(l0,L+1))
            eNes=np.zeros(len(eles),dtype=int)
            eNes[0]=N0
            eNes[1:]=NB0*2**(-eles[1:]*3/4)*2**(L*(2+1/4))
    
            print(eNes*5e-2) 
print(eNes[0])
"""
#%%
# IN THIS SECTION  WE CHECK OUT THE MLPF COX WITH GBM FOR T=20
"""

samples=100
l0=0
Lmin=0
Lmax=5
eLes=np.array(np.arange(Lmin,Lmax+1))
a=19
aa=0
d=2**(0)
x_pf=np.reshape(np.loadtxt("Observations&data/PMLPF_cox_gbm20_p_i_v1.txt",dtype=float),(len(eLes),samples,int(T/d),dim))
"""
# BIAS
"""
rbias=np.abs(np.mean(x_pf[1:]-x_pf[:-1],axis=1)[:,a,0])
b_ref=np.array([1/2**i for i in range(len(eLes[1:]))])*rbias[0]
plt.plot(eLes[1:],rbias)
plt.plot(eLes[1:],b_ref)
"""

# VARIANCE
"""
var=np.var(x_pf,axis=1)[:,a,0]
var_var=np.var((x_pf-np.mean(x_pf,axis=1,keepdims=True))**2,axis=1)[:,a,0]
var_ub=var+np.sqrt(var_var)*1.96/np.sqrt(samples)
var_lb=var-np.sqrt(var_var)*1.96/np.sqrt(samples)
reference=np.array([1/2**(2*eLes[i]) for i in range(len(eLes))])*var[1]*2**(2*eLes[1])
print(var[1])
plt.plot(eLes,var)
plt.plot(eLes,var_ub)
plt.plot(eLes,var_lb)
plt.plot(eLes,reference)
"""

"""
plt.xlabel("L")

plt.legend()
plt.yscale("log")
#plt.savefig("Images/bias_CCPF_gbm_v3&2.pdf")
plt.show() 
"""




#%%

    
# IN THIS SECTION WE COMPUTE THE FOLLOWING TRUTH WITH L=7 AND N=54153
"""
l=7
N=541530
d=2**(0)
[log_weights,x_pf]=pff.Cox_PF(T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l\
,d,N,dim,resamp_coef,g_den,g_par,Lambda,Lamb_par)
weights=norm_logweights(log_weights,ax=1)
print(weights.shape,x_pf.shape)   
phi_pfmean=np.sum(np.reshape(weights,(int(T/d),N,1))*x_pf,axis=1)
phi_pf=phi_pfmean.flatten()
#np.savetxt("Observations&data/TruthGBMN541530.txt",phi_pf,fmt="%f")
"""
#%%

# HERE I TRY SEVERAL PARAMETERS IN ORDER TO GET A NICE REALIZATION OF THE 
# GBM FOR T=100

#"""
np.random.seed(0)
T=100
dim=1
dim_o=dim
xin=np.zeros(dim)+0.1
l=13
collection_input=[]
I=identity(dim).toarray()
c=2**(-6)
#xin0=np.abs(np.random.normal(1,0,dim))
#xin1=xin0
mu=np.abs(np.random.normal(1.01,0,dim))
sigs=np.abs(np.random.normal(np.sqrt(2),0,dim))
mu=mu*c
sigs=sigs*np.sqrt(c)
print(mu,sigs)
#comp_matrix = ortho_group.rvs(dim)
#inv_mat=comp_matrix.T
comp_matrix=np.array([[1]])
inv_mat=comp_matrix.T
S=diags(np.random.normal(1,0,dim),0).toarray()
Sig=inv_mat@S@comp_matrix
fi=[sigs,Sig]
collection_input=[dim, b_gbm,mu,Sig_gbm,fi]
cov=I*1e0
g_pars=[dim,cov]
g_par=cov
samples=5

np.random.seed(0)
x_true=gen_gen_data(T,xin,l,collection_input)
Lamb_par=.5
#print(Sig,sigs,mu)
[obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
print(len(obs))
#print(obs_time,obs,len(obs_time))
#print(obs_time)
# plots of the test
times=2**(-l)*np.array(range(int(T*2**l+1)))

#plt.plot(times,x_true)
plt.plot(times,x_true,label="True signal")
plt.plot(obs_time,obs,label="Observations")
#plt.plot(times,kbf,label="KBF")
#plt.plot(times,kf,label="KF")
#plt.plot(times,kf-kbf)
#print(len(obs_time))
plt.legend()
plt.show()
#"""
#%%
#np.savetxt("Observations&data/Realization_true_GBM_T100.txt",x_true,fmt="%f")
#np.savetxt("Observations&data/Observations_time_true_GBM_T100.txt",obs_time,fmt="%f")
#np.savetxt("Observations&data/Observations_true_GBM_T100.txt",obs,fmt="%f")
#%%
x_true=np.reshape(np.loadtxt("Observations&data/Realization_true_GBM_T100.txt",dtype=float),(-1,dim))
obs_time=np.loadtxt("Observations&data/Observations_time_true_GBM_T100.txt",dtype=float)
obs=np.reshape(np.loadtxt("Observations&data/Observations_true_GBM_T100.txt",dtype=float),(-1,dim_o))

#%%
#"""
l=7
d=2**0
N=100
resamp_coef=0.8
[log_weights,x_pf]=pff.Cox_PF(T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,\
l,d,N,dim,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par)
weights=norm_logweights(log_weights,ax=1)
x_pfmean=np.sum(np.reshape(weights,(int(T/d),N,1))*x_pf,axis=1)
print(x_pfmean)
plt.plot((np.array(range(int(T/d)))+1)*d,x_pfmean[:,0],label="PF_GBM")
plt.plot(times,x_true,label="True signal")
#plt.plot(obs_time,obs,label="Observations")
#plt.plot(times,kbf,label="KBF")
#plt.plot(times,kf,label="KF")
#plt.plot(times,kf-kbf)
#print(len(obs_time))
plt.legend()
plt.show()
#"""
#%%
#Cox_PF_gbm100_p_i_v1
#"""
d=2**(0)
resamp_coef=0.8
dim_out=2
g_par=cov
samples=200
N0=10
p=10
enes=N0*np.array([2**i for i in range(p)])
a=99
aa=0
d=2**(0)
x_pf=np.reshape(np.loadtxt("Observations&data/Cox_PF_gbm100_p_i_v1.txt",dtype=float),(len(enes),samples,int(T/d),dim))
var=np.var(x_pf,axis=1)[:,a,0]
plt.plot(enes,var,label="var")
reference=var[0]*enes[0]/enes
plt.plot(enes,reference,label="ref")
plt.legend()
plt.yscale("log")
plt.xscale("log")
C0=np.mean(var*enes)
print(C0,var*enes)
#C0=1.45
#plt.savefig("Images/bias_CCPF_gbm_v3&2.pdf")
plt.show() 
#"""
#%%

#"""
d=2**(0)
resamp_coef=0.8
dim_out=2
g_par=cov
samples=500
l0=1
L=10
eles=np.array(range(l0,L+1))
N=5000
a=99
aa=0
d=2**(0)
x_pf=np.reshape(np.loadtxt("Observations&data/CCPF_gbm100_p_i_v1.txt",dtype=float),(2,len(eles),samples,int(T/d),dim))
#"""
# BIAS
"""
rbias=np.abs(np.mean(x_pf[0]-x_pf[1],axis=1)[:,a,0])

reference_bias=np.array([1/2**(i) for i in range(len(eles))])*rbias[0]
plt.plot(eles,rbias)
plt.plot(eles,reference_bias)
[b0,b1]=coef(eles,np.log2(rbias))
print([b0,b1,2**(2*b0)])
#k= 0.03237
"""

# VARIANCE
#"""
sm=np.mean((x_pf[0]-x_pf[1])**2,axis=1)[:,a,0]
#print((x_pf[0]-x_pf[1])[:2,:,a,0])
reference=np.array([1/2**(i/2) for i in range(len(eles))])*sm[0]
print(sm[2])
plt.plot(eles,sm,label="Second moment")
plt.plot(eles,reference,label="ref")
[b0,b1]=coef(eles,np.log2(sm))
print(b0,b1,(2**b0)*N)
C=(2**b0)*N
#C=10.2
#"""
#"""
plt.xlabel("L")

plt.legend()
plt.yscale("log")
#plt.savefig("Images/bias_CCPF_gbm_v3&2.pdf")
plt.show() 
#"""


#%%
"""
Lmin=0
Lmax=5
l0=0
eLes=np.arange(Lmin,Lmax+1)
K=0.03237
C=10.2
C0=1.45
for i in range(len(eLes)):
            L=eLes[i]
            CL=2**(1/4)*(2**(L/4)-1)/(2**(1/4)-1)
            NB0=(np.sqrt(C0)+np.sqrt(3*C/2)*CL)*np.sqrt(2*C/3)*2**(-L/4)/K
            N0=np.sqrt(C0)*(np.sqrt(C0)+np.sqrt(3*C/2)*CL)*2**(2*L)/K
            eles=np.array(np.arange(l0,L+1))
            eNes=np.zeros(len(eles),dtype=int)
            a=5e-1
            eNes[0]=N0*a
            eNes[1:]=NB0*2**(-eles[1:]*3/4)*2**(L*(2+1/4))*a
            print(eNes)
"""           
#%%


#TEST FOR THE VARIANCES OF THE MLPF LEVELS

# In the following, depending of the lenght of the MLPF we get the variance 
# of the levels.
#"""
samples=100
l0=0
Lmin=0
Lmax=5
eLes=np.array(np.arange(Lmin,Lmax+1)) 

x_pf=np.reshape(np.loadtxt("Observations&data/MLPF_cox_gbm100_p_i_v1.txt"\
,dtype=float),(len(eLes),samples,int(T/d),dim))
    
    
 #   """

#%%

# BIAS
#"""
rbias=np.abs(np.mean(x_pf[1:]-x_pf[:-1],axis=1)[:,a,0])
b_ref=np.array([1/2**i for i in range(len(eLes[1:]))])*rbias[0]
plt.plot(eLes[1:],rbias)
plt.plot(eLes[1:],b_ref)
#"""

# VARIANCE
"""
var=np.var(x_pf,axis=1)[:,a,0]
var_var=np.var((x_pf-np.mean(x_pf,axis=1,keepdims=True)  )**2,axis=1)[:,a,0]
var_ub=var+np.sqrt(var_var)*1.96/np.sqrt(samples)
var_lb=var-np.sqrt(var_var)*1.96/np.sqrt(samples)
reference=np.array([1/2**(2*eLes[i]) for i in range(len(eLes))])*var[1]*2**(2*eLes[1])
print(var[1])
plt.plot(eLes,var)
plt.plot(eLes,var_ub)
plt.plot(eLes,var_lb)
plt.plot(eLes,reference)
"""
#"""

plt.xlabel("L")

plt.legend()
plt.yscale("log")
#plt.savefig("Images/bias_CCPF_gbm_v3&2.pdf")
plt.show() 
#"""
#%%
# TRUE PF
#"""
K=0.03237
C=10.2
C0=1.45
es=1e-6
N=int(2*C0/es)
l=int(np.ceil(-np.log2(es/(2*K))/2))
print(N,l)
resamp_coef=0.8
#%%
[log_weights,x_pf]=pff.Cox_PF(T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,\
l,d,N,dim,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par)
weights=norm_logweights(log_weights,ax=1)
x_pfmean=np.sum(np.reshape(weights,(int(T/d),N,1))*x_pf,axis=1)
phi_pf=x_pfmean.flatten()
#np.savetxt("Observations&data/TruthGBMT100.txt",phi_pf,fmt="%f")

plt.plot((np.array(range(int(T/d)))+1)*d,x_pfmean[:,0],label="PF_GBM")
plt.plot(times,x_true,label="True signal")
#plt.plot(obs_time,obs,label="Observations")
#fplt.plot(times,kbf,label="KBF")
#plt.plot(times,kf,label="KF")
#plt.plot(times,kf-kbf)
#print(len(obs_time))
plt.legend()
plt.show()
#"""
#%%

# IN THE FOLLOWING LINES WE COMPARE THE MSE OF THE MLPF_COX WITH ITS COST
#MSE
#"""
a=99
mpf_true=np.reshape(np.loadtxt("Observations&data/TruthGBMT100.txt"),(int(T/d),dim))
MSE=np.mean((x_pf-mpf_true)**2,axis=1)[:,a,0]

#COST
print(eLes)
Cost=np.zeros(len(eLes))
for i in range(len(eLes)):
            L=eLes[i]
            CL=2**(1/4)*(2**(L/4)-1)/(2**(1/4)-1)
            NB0=(np.sqrt(C0)+np.sqrt(3*C/2)*CL)*np.sqrt(2*C/3)*2**(-L/4)/K
            N0=np.sqrt(C0)*(np.sqrt(C0)+np.sqrt(3*C/2)*CL)*2**(2*L)/K
            eles=np.array(np.arange(l0,L+1))
            eNes=np.zeros(len(eles),dtype=int)
            scale=5e-1
            eNes[0]=N0*scale
            #Cost[i]+=eNes[0]*2**eLes[0]
            eNes[1:]=2**(-eles[1:]*3/4)*2**(L*(2+1/4))*scale
            print(eNes[1:]*2**eles[1:])
            Cost[i]+=np.sum(eNes*2**eles)
k=2        
plt.plot(MSEUN[k:,a,0],costs[k:],label="MSE_unbi")
plt.plot(MSEUN[k:,a,0],4e5*MSEUN[k:,a,0]**(-2.5/2),label=r"$\mathcal{O}(\varepsilon^{-2.5})$")

print(Cost)
plt.plot(MSE,Cost,label="MLPF")
plt.plot(MSE,MSE**(-2.5/2),label=r"$\varepsilon^{-2.5}$")
plt.xlabel(r"$\varepsilon^2$")
plt.title("GBM with T=100")
plt.ylabel("Cost")
plt.yscale("log")
plt.xscale("log")
plt.legend()
#plt.savefig("Images/MSEvsCost_GBM_T=100_MLPF&UnB.pdf")
plt.show()
#"""
#%%


# BIAS
"""
rbias=np.abs(np.mean(x_pf[1:]-x_pf[:-1],axis=1)[:,a,0])
bias=np.abs(np.mean(x_pf-mpf_true,axis=1)[:,a,0])
rb_ref=np.array([1/2**i for i in range(len(eLes[1:]))])*rbias[0]
b_ref=np.array([1/2**i for i in range(len(eLes))])*bias[0]
#plt.plot(eLes[1:],rbias,label="rbias")
plt.plot(eLes,bias**2,label="bias^2")
plt.plot(eLes,b_ref**2,label=r"$\Delta_L^2$")
"""

# VARIANCE
"""
var=np.var(x_pf,axis=1)[:,a,0]
var_var=np.var((x_pf-np.mean(x_pf,axis=1,keepdims=True)  )**2,axis=1)[:,a,0]
var_ub=var+np.sqrt(var_var)*1.96/np.sqrt(samples)
var_lb=var-np.sqrt(var_var)*1.96/np.sqrt(samples)
reference=np.array([1/2**(2*eLes[i]) for i in range(len(eLes))])*var[1]*2**(2*eLes[1])
print(var[1])
plt.plot(eLes,var,label="var")
#plt.plot(eLes,var_ub)
#plt.plot(eLes,var_lb)
#plt.plot(eLes,reference)
"""

"""
plt.xlabel("L")

plt.legend()
plt.yscale("log")
#plt.savefig("Images/bias_CCPF_gbm_v3&2.pdf")
plt.show() 
"""
#%%
"""
L=5
CL=2**(1/4)*(2**(L/4)-1)/(2**(1/4)-1)
NB0=(np.sqrt(C0)+np.sqrt(3*C/2)*CL)*np.sqrt(2*C/3)*2**(-L/4)/K
N0=np.sqrt(C0)*(np.sqrt(C0)+np.sqrt(3*C/2)*CL)*2**(2*L)/K
eles=np.array(np.arange(l0,L+1))
eNes=np.zeros(len(eles),dtype=int)
a=3e-2
eNes[0]=N0*a
eNes[1:]=NB0*2**(-eles[1:]*3/4)*2**(L*(2+1/4))
print(eNes)
eles=np.array(np.arange(l0,L+1))
a=19
pfs_le=x_pf[:,L,:len(eles)]
sm=np.mean((pfs_le[0]-pfs_le[1])**2,axis=1)[:,a,0]
var=np.var((pfs_le[0]-pfs_le[1]),axis=1)[:,a,0]
reference=np.array([2**(i/4) for i in range(len(eles))])*sm[1]/2**(1/4)

print(eles[1:])
plt.plot(eles[1:], sm[1:],label="SM")
plt.plot(eles[1:], reference[1:],label="ref")
plt.plot(eles[1:], var[1:],label="var")

plt.legend()
plt.yscale("log")

plt.show()
"""


#%%

"""
np.random.seed(sample)
x_true=gen_gen_data(T,xin,l,collection_input)
Lamb_par=1
#print(Sig,sigs,mu)
[obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)

times=2**(-l)*np.array(range(int(T*2**l+1)))

plt.plot(times,x_true,label="True signal")
plt.plot(obs_time,obs,label="Observations")
plt.legend()
plt.show()
plt.plot(times[int(T*2**(l)*9/10):],x_true[int(T*2**(l)*9/10):],label="True signal")
plt.plot(obs_time[[i for i in range(len(obs)) if obs_time[i]>90]],\
obs[[i for i in range(len(obs)) if obs_time[i]>90]],label="Observations")
"""


"""
plt.plot(times[int(T*2**(l)*9/10):],x_true[int(T*2**(l)*9/10):],label="True signal")
plt.plot(obs_time[[i for i in range(len(obs)) if obs_time[i]>90]],\
obs[[i for i in range(len(obs)) if obs_time[i]>90]],label="Observations")
print(len(obs))
plt.legend()
plt.show()
print(len(obs))
"""
"""
for sample in range(samples):
    
    print("seed is",sample)
    np.random.seed(sample)
    x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=.3
    #print(Sig,sigs,mu)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    #print(obs_time,obs,len(obs_time))
    #print(obs_time)
    # plots of the test
    times=2**(-l)*np.array(range(int(T*2**l+1)))

    #plt.plot(times,x_true)
    plt.plot(times,x_true,label="True signal")
    plt.plot(obs_time,obs,label="Observations")
    #plt.plot(times,kbf,label="KBF")
    #plt.plot(times,kf,label="KF")
    #plt.plot(times,kf-kbf)
    #print(len(obs_time))
    plt.legend()
    plt.show()


    plt.plot(times[int(T*2**(l)*9/10):],x_true[int(T*2**(l)*9/10):],label="True signal")
    plt.plot(obs_time[[i for i in range(len(obs)) if obs_time[i]>90]],\
    obs[[i for i in range(len(obs)) if obs_time[i]>90]],label="Observations")
    
    #plt.plot(obs_time[],obs,label="Observations")
    #plt.plot(times,kbf,label="KBF")
    #plt.plot(times,kf,label="KF")
    #plt.plot(times,kf-kbf)
    #print(len(obs_time))
    plt.legend()
    plt.show()
    print(len(obs))
"""

#%%
####!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100
# UNBIASED, UNBIASED, # UNBIASED, UNBIASED, # UNBIASED, UNBIASED, # UNBIASED, UNBIASED.
# IN THE FOLLOWING WE CHOOSE A REALIZATION OF THE OU DYNAMICS FOR TIME
# T=100 IN ORDER
# TO APPLY THE PF MACHINERY TO IT

np.random.seed(0)
T=100
dim=1
dim_o=dim
xin=np.zeros(dim)+0.1
l=13
collection_input=[]
I=identity(dim).toarray()
c=2**(-6)
#xin0=np.abs(np.random.normal(1,0,dim))
#xin1=xin0
mu=np.abs(np.random.normal(1.01,0,dim))
sigs=np.abs(np.random.normal(np.sqrt(2),0,dim))
mu=mu*c
sigs=sigs*np.sqrt(c)
print(mu,sigs)
#comp_matrix = ortho_group.rvs(dim)
#inv_mat=comp_matrix.T
comp_matrix=np.array([[1]])
inv_mat=comp_matrix.T
S=diags(np.random.normal(1,0,dim),0).toarray()
Sig=inv_mat@S@comp_matrix
fi=[sigs,Sig]
collection_input=[dim, b_gbm,mu,Sig_gbm,fi]
cov=I*1e0
g_pars=[dim,cov]
g_par=cov
samples=5

np.random.seed(0)
x_true=gen_gen_data(T,xin,l,collection_input)
Lamb_par=.5
#print(Sig,sigs,mu)
[obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
print(len(obs))
#print(obs_time,obs,len(obs_time))
#print(obs_time)
# plots of the test
times=2**(-l)*np.array(range(int(T*2**l+1)))

#plt.plot(times,x_true)
plt.plot(times,x_true,label="True signal")
plt.plot(obs_time,obs,label="Observations")

#plt.plot(times,kbf,label="KBF")
#plt.plot(times,kf,label="KF")
#plt.plot(times,kf-kbf)
#print(len(obs_time))
plt.legend()
plt.show()
#%%

d=2**0
resamp_coef=0.8
l=8
N=1000
[log_weights,x_pf]=pff.Cox_PF(T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,\
l,d,N,dim,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par)
weights=norm_logweights(log_weights,ax=1)
x_pfmean=np.sum(np.reshape(weights,(int(T/d),N,1))*x_pf,axis=1)
#print(x_pfmean)
plt.plot(obs_time,obs,label="Observations")
plt.plot(times,x_true,label="True signal")
plt.plot((np.array(range(int(T/d)))+1)*d,x_pfmean[:,0],label="PF_OU")
#plt.plot((np.array(range(int(T/d)))+1)*d,mpf_true[:,0],label="True filter")
#plt.plot(obs_time,obs,label="Observations")
plt.legend()


#%%
#print(np.sqrt(0.0323))
#cl=9.3
#delta_l=0.18
#cp0=1.45
#cl0=1.45

# COMPUTATION OF CP AND CP0 AND DELTA_P

#dp=3.2
#cp=14.5

# IN THIS CELL WE COMPUTE THE REFERENCE TRUTH IN ORDER TO COMPUTE THE BIAS
# OF THE UNBIASED ESTIMATOR
#"""
l=4
N0=4
N=int(N0*2**8*1000)
print(N)
[lw,x]=pff.Cox_PF(T, xin, b_gbm, mu, Sig_gbm, fi, obs, obs_time, l, d,N\
, dim, resamp_coef, g_den, g_par, Norm_Lambda, Lamb_par)

weights=norm_logweights(lw,ax=1)           
phi_pfmean=np.sum(np.reshape(weights,(int(T/d),N,1))*phi(x,ax=2),axis=1)

#np.savetxt("Observations&data/SUCPF_gbm_N1024000_T100_v1.txt",phi_pfmean)
#"""
#%%
samples=1000
a=99
L=5
l0=4
p=8
N0=4
eLes=np.arange(l0,L+1)
ps=np.arange(0,p+1)
eNes=N0*2**ps
pfs=np.reshape(np.loadtxt("Observations&data/SUCPF_gbm_p_levels_T10o_v1.txt",dtype=float),(2,len(eNes)-1,samples,int(T/d),dim))
tel_est_exact=np.reshape(np.loadtxt("Observations&data/SUCPF_gbm_N1024000_T100_v1.txt",dtype=float),(int(T/d),dim))
#BIAS
#"""
bias=np.abs(np.mean(pfs[1]-tel_est_exact,axis=1)[:,a,0])   
print(ps[1:]/eNes[1:],bias)
[b0,b1]=coef(np.log(ps[2:]/eNes[2:]),np.log(bias[1:]))
print(b0,b1,np.exp(b0))
reference=(ps[1:]/eNes[1:])*bias[1]*eNes[2]/ps[2]
plt.plot(ps[1:]/eNes[1:],reference,label="Ref: $x=y$")
plt.plot(ps[2:]/eNes[2:],bias[1:],label="bias")
plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"$\frac{p}{N_p}$")
plt.title("Bias in terms of the particle levels of the OU for T=100")
plt.legend()
plt.show()  
#dp=3.2
#"""
#VARIANCE
#"""
sm=np.mean((pfs[0]-pfs[1])**2,axis=1)[:,a,0]
var=np.var((pfs[0]-pfs[1]),axis=1)[:,a,0]

ref=np.array([1/2**i for i in range(len(eNes)-1)])*sm[0]
[b0,b1]=coef(np.log(1/eNes[1:]),np.log(sm))
print(b0,b1,np.exp(b0))
#cp=14.5

plt.plot(1/eNes[1:],sm,label="sm")
plt.plot(1/eNes[1:],ref,label="$N^{-1}$")
plt.legend()
plt.xscale("log")
plt.yscale("log")
#plt.savefig("Images/bias_ou_p_levels_T10.pdf")
plt.show()  
#"""

#%%
# IN THIS CELL WE COMPUTE THE REFERENCE TRUTH IN ORDER TO COMPUTE THE BIAS in
# THE DISCRETIZATION LEVEL AND ALSO CL
# OF THE UNBIASED ESTIMATOR
#"""
l=14
N0=250
N=N0
[lw,x]=pff.Cox_PF(T, xin, b_ou, B, Sig_ou, fi, obs, obs_time, l, d,N\
, dim, resamp_coef, g_den, g_par, Norm_Lambda, Lamb_par)

weights=norm_logweights(lw,ax=1)           
phi_pfmean=np.sum(np.reshape(weights,(int(T/d),N,1))*phi(x,ax=2),axis=1)
#np.savetxt("Observations&data/SUCPF_ou_l14_T10_v1.txt",phi_pfmean)
#"""

#%%

#cl=9.3
#delta_l=

samples=1000
a=99
Lmax=8
l0=0
p=1
N0=250
eLes=np.arange(l0,Lmax+1)
ps=np.arange(0,p+1)
eNes=N0*2**ps
#nbot really scaled
pfs=np.reshape(np.loadtxt("Observations&data/SUCPF_gbm_l_levels_T100_v1.txt",dtype=float),(2,len(eLes),samples,int(T/d),dim))
samples_truth=2000
#tel_est_exact=np.mean(np.reshape(np.loadtxt("Observations&data/SUCPF_ou_l14_T100_v1.txt"\
 #                                           ,dtype=float),(samples_truth,int(T/d),dim)),axis=0)

#BIAS
"""
bias=np.abs(np.mean(pfs[0]-tel_est_exact,axis=1)[:,a,0])   

[b0,b1]=coef(eLes,np.log2(bias))
print(b0,b1,2**(b0))
#dp=0.38
reference=np.array([1/2**i for i in range(len(eLes))])*bias[0]
plt.plot(eLes,reference,label="Ref: $x=y$")
plt.plot(eLes,bias,label="bias")
#plt.xscale("log")


plt.xlabel(r"$l$")
plt.title("Bias in terms of the particle levels of the OU for T=10")
"""
#VARIANCE
#"""
sm=np.mean((pfs[0]-pfs[1])**2,axis=1)[:,a,0]
var=np.var(((pfs[0]-pfs[1])**2),axis=1)[:,a,0]
sm_ub=sm+np.sqrt(var)*1.96/np.sqrt(samples)
sm_lb=sm-np.sqrt(var)*1.96/np.sqrt(samples)
plt.plot(eLes,sm_ub)
plt.plot(eLes,sm_lb)
ref=np.array([1/2**(i/2) for i in range(len(eLes))])*sm[-1]*2**(eLes[-1]/2)
print(ref[0]*2.4*250)
#cl=9.3
[b0,b1]=coef(eLes+1,np.log2(sm))
print(b0,b1,2**b0)
plt.plot(eLes,sm,label="sm")
plt.plot(eLes,ref,label=r"$\Delta_l^{1/2}$")
plt.xlabel(r"$l$")
#"""

plt.legend()
#plt.xscale("log")
plt.yscale("log")
#plt.savefig("Images/sm_ou_l_levels_T10.pdf")
plt.show()  


#%%



def Pl(Pl0,lmax):
    ls=np.array(range(1,lmax+1))
    return    (D0**(-beta)-Pl0)/np.sum(np.log2(ls+2)**2*(ls+1)*2**(-beta*ls))

def Pp(Pp0,pmax):
    ps=np.array(range(1,pmax+1))
    return (N0-Pp0)/np.sum(np.log2(ps+2)**2*(ps+1)*2**(-ps))

def Tf(pmax):
    ps=np.array(range(1,pmax+1))
    return np.sum(np.log2(ps+2)**2*(ps+1))

def U(lmax,beta,D0):
    ls=np.array(range(1,lmax+1))
    return np.sum(np.log2(ls+2)**2*(ls+1)*(D0)**(beta-1)/2**((beta-1)*ls))*3/2
def Q(pmax):
    ps=np.array(range(1,pmax+1))
    return np.sum(1/(np.log2(ps+2)**2*(ps+1)))    


#%%

#samples=2000000
samples=500000
a=99
pfs=np.reshape(np.loadtxt("Observations&data/SUCPF_gbm_T100_v1_1.txt",dtype=float),(samples,int(T/d),dim))
 #%%
lps=np.loadtxt("Observations&data/SUCPF_gbm_pls_T100_v1_1.txt")
mpf_true=np.reshape(np.loadtxt("Observations&data/TruthGBMT100.txt"),(int(T/d),dim))
#%%
#%%
lps=np.array(lps,dtype=int)
print(pfs.shape)
pmax=8
Lmax=10

dic=[[[] for j in range(pmax+1)] for i in range(Lmax+1)]


for i in range(len(pfs)):
    dic[lps[i,0]][lps[i,1]].append(i)

Num_samples=np.zeros((Lmax+1,pmax+1),dtype=int)
for i in range(Lmax+1):
    for j in range(pmax+1):
        Num_samples[i,j]=len(dic[i][j])
print(Num_samples)


#%%

# CHECK THE MSE 


#########
#ps=np.array([4,4,4,4,4,5],dtype=int)+1
#ps=np.array([1,2,5,7,9,11],dtype=int)+1
#ls=np.array([1,2,3,4,6,7],dtype=int)+2
#################

ps=np.array([4,4,4,4,4,5],dtype=int)+1
#ps=np.array([1,2,5,7,9,11],dtype=int)+1
ls=np.array([2,3,5,6,8,9],dtype=int)-1
N0=100
D0=1
beta=1/2
cl=9.3
delta_l=0.18
cp0=1.45
cl0=1.45
dp=3.2
cp=14.5
batches=10

costs=np.zeros(len(ps))
var=np.zeros((len(ls),int(T/d),dim))
MSEUN=np.zeros((len(ls),int(T/d),dim))
np.random.seed(3)
for i in range(len(ps)):
    lmax=ls[i]
    pmax=ps[i]
    #print(lmax)
    alphal=np.sqrt(cl0*U(lmax,beta,D0)/(cl*Q(lmax)))
    alphap=np.sqrt(cp0*Tf(pmax)/(cp*Q(pmax)))
    rps=np.array(range(pmax))+1
    Pp=(alphap+np.sum(np.log2(rps+2)**2*(rps+1)/2**(rps)))**(-1)*N0
    Pp0=Pp*alphap
    rls=np.array(range(lmax))+1
    Pl=(alphal*D0**beta+np.sum(np.log2(rls+2)**2*(rls+1)*D0**beta/2**(rls*beta)))**(-1)
    Pl0=Pl*alphal
    dPl=np.concatenate(([Pl0*D0**beta],Pl*np.log2(rls+2)**2*(rls+1)*D0**beta/2**(beta*rls)))
    dPp=np.concatenate(([Pp0/N0],Pp*np.log2(rps+2)**2*(rps+1)/(2**(rps)*N0)))
    rpsc=np.concatenate(([0],rps))
    rlsc=np.concatenate(([0],rls))
    #dPp=np.log2(rpsc+2)**2*(rpsc+1)/2**(rpsc)
    #dPl=np.log2(rlsc+2)**2*(rlsc+1)/2**(beta*rlsc)
    alpha=1/4
    #dPl=2**(-alpha*rlsc)
    #dPp=2**(-alpha*rpsc)
    dPl=dPl/np.sum(dPl)
    dPp=dPp/np.sum(dPp)
    Pp_cum=np.cumsum(dPp)
    Pl_cum=np.cumsum(dPl)
    #print(dPl)
    #print(dPp)
    ml,mp=np.meshgrid(rlsc,rpsc)
    mean_cost=np.sum(dPl[ml]*dPp[mp]*2**(mp+ml))
    #print(mean_cost)
    #var[i]=np.sum(1/(dPl[ml]*dPp[mp]*2**(mp+beta*ml)))/8**i
    #print(var[i])
    #costs[i]=mean_cost*8**i
    print(lmax,pmax)
    #M=int(3e-2*((pmax+1)/(N0*2**(pmax+1)))**(-2))
    M=int(1*(10**(i)))
    print(M)
    Usamp=np.zeros((batches,M,int(T/d),dim))
    samples_count=np.zeros((Lmax+1,pmax+1),dtype=int)
    for b in range(batches):
        for j in range(M):
            l=sampling(Pl_cum)
            p=sampling(Pp_cum)
            if Num_samples[l,p]-samples_count[l,p]>0:
                Usamp[b,j]=pfs[dic[l][p][samples_count[l,p]]]/(dPl[l]*dPp[p])
                samples_count[l,p]+=1
                costs[i]+=2**(l+p)
            else:
                print("not enough samples of",l, " and",p)
                break
    Usamp=np.mean(Usamp,axis=1)
    MSEUN[i]=np.mean((Usamp-mpf_true)**2,axis=0)
    var[i]=np.var(Usamp,axis=0)
costs=costs*N0/batches

a=99
#MSE=1e-1/8**np.array(range(len(ps)))
plt.plot(MSEUN[:,a,0],costs,label="MSE_unbi")
#plt.plot(var[:,a,0],costs,label="Var")
shift1=3e1
shift2=2e3
plt.plot(MSEUN[:,a,0],shift2*np.log(np.sqrt(MSEUN[:,a,0]*shift1))**2/(MSEUN[:,a,0]*shift1),label=r"$\mathcal{O}(\log(\varepsilon)^2\varepsilon^{-2})$")
plt.yscale("log")
plt.xscale("log")
plt.legend()
plt.show()
[b0,b1]=coef(np.log(MSEUN[:,a,0]**(1/2)),np.log(costs[:]))

print([b0,b1])




#%%
N0=100
l0=0
Lmax=10
pmax=8
eLes=np.arange(l0,Lmax+1)
ps=np.array(range(pmax+1))
eNes=N0*2**ps


beta=1/2
Delta0=1/2**eLes[0]
print(eNes,eLes)
Pp=4.33318
Pp0=40.3147
Pl=0.00599866
Pl0=0.589605 
Pld=np.zeros(Lmax+1-l0)
Ppd=np.zeros(pmax+1)
Pld[0]=Delta0**(beta)*Pl0
Pld[1:]=Pl*np.log2(eLes[1:]+2)**2*(eLes[1:]+1)/2**(beta*eLes[1:])
Ppd[0]=Pp0/N0
Ppd[1:]=Pp*np.log2(ps[1:]+2)**2*(ps[1:]+1)/eNes[1:]

Ppd=Ppd/np.sum(Ppd)
Pld=Pld/np.sum(Pld)
print(Ppd,Pld)
Pp_cum=np.cumsum(Ppd)
Pl_cum=np.cumsum(Pld)
#%%


emes=np.array([10**i for i in range(1,5)])
print(emes)
emes=np.concatenate((emes,[50000]))
ps=10
print(emes)
a=99

mse=np.zeros((len(emes),int(T/d),dim))
var=np.zeros((len(emes),int(T/d),dim))
costs=np.zeros(len(emes))
np.random.seed(50)
for i in range(len(emes)):
    m=emes[i]
    plot_samples=np.random.choice(500000,size=ps*m,replace=False)
    costs[i]=np.sum(2**lps[plot_samples,0]*2**lps[plot_samples,1])*N0/ps
    upf=pfs[plot_samples]/(Ppd[lps[plot_samples,1],np.newaxis,\
    np.newaxis]*Pld[lps[plot_samples,0],np.newaxis,np.newaxis])
    upf=np.reshape(upf,(ps,m,int(T/d),dim))
    
    upfs=np.mean(upf,axis=1)
    var[i]=np.var(upfs,axis=0)
    mse[i]=np.mean((upfs-mpf_true)**2,axis=0)

k=-1
k1=0
[b0,b1]=coef(np.log(mse[k1:,a,0]**(1/2)),np.log(costs[k1:]))
print([b0,b1])
print(mse[:,a,0])
print(var[:,a,0])
MSE_un=mse[k1:,a,0]
shift1=2e1
shift2=5e4
k=5
plt.xlabel(r"$\varepsilon^2$")
plt.ylabel("Cost")  
plt.title("GBM, T=100.")
plt.plot(MSE_un,np.exp(b0)*MSE_un**(b1/2),c="dodgerblue",label=r"$\mathcal{O}(\varepsilon^{-2.25})$")
#plt.plot(var[k1:,a,0],costs[k1:],label="var")
plt.plot(MSE_un,costs[k1:],label="Unbiased",c="coral",ls="dashed",marker=".",ms=10)
#plt.plot(MSE,shift2*np.log(np.sqrt(MSE*shift1))**2/(MSE*shift1),label=r"$\mathcal{O}(np.log(\varepsilon)^2\varepsilon^{-2})$")

plt.legend()
plt.yscale("log")
plt.xscale("log")
#plt.savefig("Images/Unbiased_MSEvsCost_GBM_T=100.pdf")

#%%

#%%

# VERSION 2, IN THIS VERSION WE TRY TO GET A COUPLING SMALL ENOUGHT TO MAKE THE MLPF BETTER 
# COMPARED TO THE SINGLE TERM.
# THE VARIANCE CONSTANTS IN THE PREVIOUS ITERATIONS WHERE C=10.2
# AND C0=1.45, THIS MAKES COMPUTING AN EFFICIENTE MLPF IMPOSIBLE FOR 
# LOW LEVELS. 

np.random.seed(0)
T=100
dim=1
dim_o=dim
xin=np.zeros(dim)+4
l=10
collection_input=[]
I=identity(dim).toarray()
c=2**(-18)
cov=I*1e0
mu=np.abs(np.random.normal(1.01,0,dim))*1e3
Lamb_par=0.3
sigs=np.abs(np.random.normal(np.sqrt(2),0,dim))
mu=mu*c
sigs=sigs*np.sqrt(c)
print(mu,sigs)
#comp_matrix = ortho_group.rvs(dim)
#inv_mat=comp_matrix.T
comp_matrix=np.array([[1]])
inv_mat=comp_matrix.T
S=diags(np.random.normal(1,0,dim),0).toarray()
Sig=inv_mat@S@comp_matrix
fi=[sigs,Sig]
collection_input=[dim, b_gbm,mu,Sig_gbm,fi]

g_pars=[dim,cov]
g_par=cov
samples=5

np.random.seed(0)
x_true=gen_gen_data(T,xin,l,collection_input)
#print(Sig,sigs,mu)
[obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
print(len(obs))
#print(obs_time,obs,len(obs_time))
#print(obs_time)
# plots of the test
times=2**(-l)*np.array(range(int(T*2**l+1)))

#plt.plot(times,x_true)
plt.plot(times,x_true,label="True signal")
plt.plot(obs_time,obs,label="Observations")
#plt.plot(times,kbf,label="KBF")
#plt.plot(times,kf,label="KF")
#plt.plot(times,kf-kbf)
#print(len(obs_time))
plt.legend()
plt.show()
# %%
l=7
d=2**0
N=100
resamp_coef=0.8
[log_weights,x_pf]=pff.Cox_PF(T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,\
l,d,N,dim,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par)
weights=norm_logweights(log_weights,ax=1)
x_pfmean=np.sum(np.reshape(weights,(int(T/d),N,1))*x_pf,axis=1)
plt.plot((np.array(range(int(T/d)))+1)*d,x_pfmean[:,0],label="PF_GBM")
plt.plot(times,x_true,label="True signal")
#plt.plot(obs_time,obs,label="Observations")
#plt.plot(times,kbf,label="KBF")
#plt.plot(times,kf,label="KF")
#plt.plot(times,kf-kbf)
#print(len(obs_time))
plt.legend()
plt.show()
#"""
#%%
#Cox_PF_gbm100_p_i_v1
#"""
d=2**(0)
resamp_coef=0.8
dim_out=2
g_par=cov
samples=100
N0=10
p=8
enes=N0*np.array([2**i for i in range(p)])
a=99
aa=0
d=2**(0)
x_pf=np.reshape(np.loadtxt("Observations&data/Cox_PF_gbm100_p_scaled_v16_2.txt",dtype=float),(len(enes),samples,int(T/d),dim))
var=np.var(x_pf,axis=1)[:,a,0]
plt.plot(enes,var,label="var")
reference=var[0]*enes[0]/enes
plt.plot(enes,reference,label="ref")
plt.legend()
plt.yscale("log")
plt.xscale("log")
C0=np.mean(var*enes)
print(C0,var*enes)
#C0=0.03283620263287618
#plt.savefig("Images/bias_CCPF_gbm_v3&2.pdf")
plt.show() 
#"""
#%%

#"""
d=2**(0)
resamp_coef=0.8
dim_out=2
g_par=cov
samples=500
l0=1
L=9
eles=np.array(range(l0,L+1))
N=2000
a=99
aa=0
d=2**(0)
x_pf=np.reshape(np.loadtxt("Observations&data/CCPF_gbm100_p_scaled_v16_2.txt",dtype=float),(2,len(eles),samples,int(T/d),dim))
#"""
# BIAS
"""
rbias=np.abs(np.mean(x_pf[0]-x_pf[1],axis=1)[:,a,0])

reference_bias=np.array([1/2**(i) for i in range(len(eles))])*rbias[0]
plt.plot(eles,rbias)
plt.plot(eles,reference_bias)
[b0,b1]=coef(eles[:5],np.log2(rbias)[:5])
print([b0,b1,2**(2*b0)])
#K= 1.271842812413708e-09
"""

# VARIANCE
#"""
print(N)
sm=np.mean((x_pf[0]-x_pf[1])**2,axis=1)[:,a,0]
#print((x_pf[0]-x_pf[1])[:2,:,a,0])
reference=np.array([1/2**(i/2) for i in range(len(eles))])*sm[0]
plt.plot(eles,sm,label="Second moment")
plt.plot(eles,reference,label="ref")
[b0,b1]=coef(eles,np.log2(sm))
print(b0,b1,(2**b0)*N)
C=(2**b0)*N
#C=0.00025205321130822886
#"""
#"""
plt.xlabel("L")

plt.legend()
plt.yscale("log")
#plt.savefig("Images/bias_CCPF_gbm_v3&2.pdf")
plt.show() 
#"""

# %%


# CCONSTANTS AND PPARAMETERS FOR THE SINGLE COX PARTICLE FILTER
C0=0.035
C=0.00025205321130822886
print(C0/C)
K=1.765425864219179e-09
l0=0
Lmin=l0
Lmax=7
es=5e-9
scale=1e-4
eLes=np.arange(Lmin,Lmax+1,1)
N=int(2*C0/es)
L=int(np.ceil(-np.log2(es/(2*np.sqrt(K)))/2,dtype=float))
print(L,N)
#%%
resamp_coef=0.8
[log_weights,x_pf]=pff.Cox_PF(T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,\
L,d,N,dim,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par)
weights=norm_logweights(log_weights,ax=1)
x_pfmean=np.sum(np.reshape(weights,(int(T/d),N,1))*x_pf,axis=1)
phi_pf=x_pfmean.flatten()
#%%
#np.savetxt("Observations&data/TruthnldtT100.txt",phi_pf,fmt="%f")
plt.plot((np.array(range(int(T/d)))+1)*d,x_pfmean[:,0],label="PF_nldt")
plt.plot(times,x_true,label="True signal")
#plt.plot(obs_time,obs,label="Observations")
#plt.plot(times,kbf,label="KBF")
#plt.plot(times,kf,label="KF")
#plt.plot(times,kf-kbf)
#print(len(obs_time))
plt.legend()
plt.show()
# %%
# AARTIFICIAL TEST FOR THE MSE
C0=0.03540286765887861
C=0.00025205321130822886
K=1.765425864219179e-09
print(C0/C)
l0=1
Lmin=l0
Lmax=6
eLes=np.arange(Lmin,Lmax+1,1)
Cost=np.zeros(len(eLes))
C_sin=0.03283620263287618
k_sin=K
l0_sin=l0
Lmin_sin=l0_sin
Lmax_sin=Lmax
d=1
eLes_sin=np.array(np.arange(Lmin_sin,Lmax_sin+1)) 
Cost_sin=np.float_power(8, eLes_sin)*C_sin/k_sin
particles_sin=np.float_power(4, eLes_sin)*C_sin/k_sin

for i in range(len(eLes)):
        L=eLes[i]
        CL=2**(1/4)*(2**((L-l0)/4)-1)/(2**(1/4)-1)
        #NB0=(np.sqrt(C0)+np.sqrt(3*C/2)*CL)*np.sqrt(2*C/3)*2**(-(L-Lmin)/4)/K
        #print("NB0 is ",NB0)
        #print("is the error",2**(L/4))
        #N0=np.sqrt(C0)*(np.sqrt(C0)+np.sqrt(3*C/2)*CL)*2**(2*(L))/K
        delta_l0=np.float_power(2,-l0)
        N0=np.sqrt(C0*delta_l0)*(np.sqrt(C0/delta_l0)+np.sqrt(3*C/2)*CL*delta_l0**(-1/4))*np.float_power(2,2*L)/K
        #N0=2*C0*np.float_power(2,2*L)/K
        #NB0=2*C*delta_l0**(-1/4)*CL*np.float_power(2,-L/4)/K
        #print("N0 is ",N0)
        eles=np.array(np.arange(l0,L+1))
        eNes=np.zeros(len(eles),dtype=int)
        scale=1e-4
        eNes[0]=N0*scale
        #Cost[i]+=eNes[0]*2**eLes[0]
        eNes[0]=np.maximum(2,eNes[0])
        eNes[1:]=np.sqrt(2*C/3)*(np.sqrt(C0/delta_l0)+np.sqrt(3*C/2)*CL*delta_l0**(-1/4))\
        *np.float_power(2,2*L)*np.float_power(2,-eles[1:]*3/4)/K*scale
        #eNes[1:]=NB0*np.float_power(2,-eles[1:]*3/4)*np.float_power(2,(2+1/4)*L)*scale
        Cost[i]+=3*np.sum(eNes[1:]*np.float_power(2,eles[1:]))/2+eNes[0]*np.float_power(2,eles[0])  
        particles_sin[i]=particles_sin[i]*scale
        Cost_sin[i]=Cost_sin[i]*scale  
        print(eNes)
        print("# of particles of the single term",particles_sin[i])
        print("the MSE is approx", C0/particles_sin[i])
print("particles singles",particles_sin)
plt.plot(eLes,Cost,label="Cost")
plt.plot(eLes_sin,Cost_sin,label="Cost Single")
plt.yscale("log")
#plt.xscale("log")
plt.legend()
plt.show()
# %%

# SSINGLE
# PF, PF, PF, PF, PF, PF, PF, PF, PF, PF, PF, PF, PF, PF, PF, PF, PF, PF, PF
#"""
T=100
samples=100
l0=0
Lmin=0
Lmax=5
d=1
eLes_sin=np.array(np.arange(Lmin,Lmax+1)) 
x_pf1_sin=np.reshape(np.loadtxt("remote/PPF_cox_gbmt100_scaled1_v1.txt"\
,dtype=float),(len(eLes_sin),samples,int(T/d),dim))
#%%
T=100
samples=100
l0=1
d=1
Lmin=l0
Lmax=5
eLes=np.array(np.arange(Lmin,Lmax+1)) 
x_pf=np.reshape(np.loadtxt("remote/PMLPF_cox_gbm100_scaled1_v5.txt"\
,dtype=float),(len(eLes),samples,int(T/d),dim))
#%%

# BIAS
"""
a=-1
#rbias=np.abs(np.mean(x_pf[1:]-x_pf[:-1],axis=1)[:,a,0])
bias=np.abs(np.mean(x_pf-mpf_true,axis=1))[:,a,0]
b_ref=np.array([1/2**i for i in range(len(eLes))])*bias[0]
bias_sin=np.abs(np.mean(x_pf1_sin-mpf_true,axis=1))[:,a,0]
#plt.plot(eLes[:-1],rbias**2,label="Rich bias^2")
plt.plot(eLes,bias**2,label="bias^2")
plt.plot(eLes_sin,bias_sin**2,label="single bias^2")
plt.plot(eLes,b_ref**2,label=r"$\Delta_l^2$")
print(np.mean(x_pf,axis=1)[:,a,0])

#rbias_sin=np.abs(np.mean(x_pf_sin[1:]-x_pf_sin[:-1],axis=1)[:,a,0])
#b_ref=np.array([1/2**i for i in range(len(eLes[1:]))])*rbias_sin[0]
#plt.plot(eLes[1:],rbias_sin**2,label="Single Rich bias^2")

#plt.plot(eLes[1:],b_ref**2,label=r"$\Delta_l^2$")
#print(np.mean(x_pf,axis=1)[:,a,0])
"""

# VARIANCE
#"""
a=-1
var=np.var(x_pf,axis=1)[:,a,0]
print("var is",var)
var_sin=np.var(x_pf1_sin,axis=1)[:,a,0]
#var_sin=np.var(x_pf_sin,axis=1)[:,a,0]
#print("var single is",var_sin)
var_var=np.var((x_pf-np.mean(x_pf,axis=1,keepdims=True)  )**2,axis=1)[:,a,0]
var_ub=var+np.sqrt(var_var)*1.96/np.sqrt(samples)
var_lb=var-np.sqrt(var_var)*1.96/np.sqrt(samples)
reference=np.array([1/np.float_power(2,2*eLes[i]) for i in range(len(eLes))])*var[1]*np.float_power(2,2*eLes[1]) 
print(var[1])
plt.plot(eLes,var,label="var")
plt.plot(eLes_sin,var_sin,label="var single")
#plt.plot(eLes,var_sin,label="var_sin")

#plt.plot(eLes,var_ub)
#plt.plot(eLes,var_lb)
plt.plot(eLes,reference,label=r"$\Delta_l^2$")
#"""
#"""
plt.xlabel("L")
plt.title("Variance and bias of the MLPF cox")
plt.legend()
plt.yscale("log")
#plt.savefig("Images/bias_and_var_CCPF_nldt_v1.pdf")
plt.show()
#"""
#%%

a=99
mpf_true=np.reshape(np.loadtxt("remote/Truth_PF_cox_gbmt100_scaled1_v2.txt"),(int(T/d),dim))
print(mpf_true.shape)
#%%
MSE=np.mean((x_pf-mpf_true)**2,axis=1)[:,a,0]
MSE_sin=np.mean((x_pf1_sin-mpf_true)**2,axis=1)[:,a,0]
#COST
l0=1
Lmin=l0
Lmax=5
eLes=np.array(np.arange(Lmin,Lmax+1,1),dtype=float)
#print(eLes)
Cost=np.zeros(len(eLes))
C0=0.03540286765887861
C=0.00025205321130822886
K=1.765425864219179e-09
C_sin=0.03283620263287618
k_sin=K

#Cost_sin=2**eLes*C_sin*2**(2*eLes)/k_sin
Cost_sin=np.float_power(8, eLes_sin)*C_sin/k_sin
print("eNes single",C_sin*np.float_power(4,eLes)/k_sin)
for i in range(len(eLes)):
            L=eLes[i]
            CL=2**(1/4)*(2**((L-l0)/4)-1)/(2**(1/4)-1)
            delta_l0=np.float_power(2,-l0)
            N0=np.sqrt(C0*delta_l0)*(np.sqrt(C0/delta_l0)+np.sqrt(3*C/2)*CL*delta_l0**(-1/4))*np.float_power(2,2*L)/K
            eles=np.array(np.arange(l0,L+1))
            eNes=np.zeros(len(eles),dtype=int)
            scale=1e-4
            eNes[0]=N0*scale
            eNes[0]=np.maximum(2,eNes[0])
            eNes[1:]=np.sqrt(2*C/3)*(np.sqrt(C0/delta_l0)+np.sqrt(3*C/2)*CL*delta_l0**(-1/4))\
            *np.float_power(2,2*L)*np.float_power(2,-eles[1:]*3/4)/K*scale
            #print(eNes[1:]*2**eles[1:])
            Cost[i]+=np.sum(3*(eNes[1:]*2**eles[1:])/2)+eNes[0]*2**eles[0]
        
Cost_sin=Cost_sin*scale

    
#%%
print("Cost",Cost)
print("Cost_single",Cost_sin)
#plt.plot(MSE_un,costs[k1:],label="Unbiased")
MSE_arti=1/2**(eLes*2)
plt.plot(MSE,Cost,label="MLPF",marker="o",c="coral",alpha=0.8,ls="--")
#plt.plot(MSE_arti,Cost,label="MLPF",marker="o")
plt.plot(MSE_sin,Cost_sin,label="Single PF",marker="+",markersize=10,ls="dashdot",color="dodgerblue")
#plt.plot(MSE_arti[:-1],Cost_sin[:-1],label="Single PF",marker="o")
plt.plot(MSE_sin,1.3e-4*MSE_sin**(-3/2),label=r"$\varepsilon^{-3}$",color="deepskyblue")
plt.plot(MSE,3.2e-3*MSE**(-2.5/2),label=r"$\varepsilon^{-2.5}$",color="salmon")
plt.xlabel(r"$\varepsilon^2$")
plt.title("GBM, T=100.")
plt.ylabel("Cost")
plt.yscale("log")
plt.xscale("log")
plt.legend()
plt.savefig("Images/MSEvsCost_gbm_scaled1_v1.pdf")
plt.show()
#%%