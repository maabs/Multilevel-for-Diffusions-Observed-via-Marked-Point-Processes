{}#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 16:06:28 2023

@author: alvarem

"""
#%%
import numpy as np
import matplotlib.pyplot as plt 
#import progressbar
from scipy import linalg as la
from scipy.sparse import identity
from scipy.sparse import rand
from scipy.sparse import diags
from scipy.sparse import triu
import copy
from scipy.stats import t
#from sklearn.linear_model import LinearRegression
from scipy.stats import ortho_group
import time
import PF_functions_def as pff
from Un_cox_PF_functions_def import *
from scipy.stats import multivariate_normal
#%%

#T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100

# IN THE FOLLOWING WE CHOOSE A REALIZATION OF THE OU DYNAMICS FOR TIME
# T=100 IN ORDER
# TO APPLY THE PF MACHINERY TO IT
if True==True:
        np.random.seed(1)
        T=100
        dim=1
        dim_o=dim
        xin=np.zeros(dim)+1
        l=13
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
        Lamb_par=1.33
        [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
        print(obs_time,obs,len(obs_time))
        
times=2**(-l)*np.array(range(int(T*2**l+1)))

plt.plot(times,x_true,label="True signal")
plt.plot(obs_time,obs,label="Observations")
plt.legend()
d=2**(0)
N=40
resamp_coef=0.8



#%%
#"""
l=6
[log_weights,x_pf]=pff.Cox_PF(T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,\
l,d,N,dim,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par)
weights=norm_logweights(log_weights,ax=1)
x_pfmean=np.sum(np.reshape(weights,(int(T/d),N,1))*x_pf,axis=1)
#print(x_pfmean)
plt.plot(times,x_true,label="True signal")
plt.plot((np.array(range(int(T/d)))+1)*d,x_pfmean[:,0],label="PF_OU")
plt.plot(obs_time,obs,label="Observations")
plt.legend()
#"""
#%%
#"""
samples=1000
l0=1
L=10
eles=np.array(range(l0,L+1))
N=5000
a=99
aa=0
d=2**(0)
x_pf=np.reshape(np.loadtxt("Observations&data/CCPF_ou100_v1.txt",dtype=float),(2,len(eles),samples,int(T/d),dim))
a=99
samples=100
#"""
#%%
#PARAMETERS 
#C=0.8
#C0=0.5
#K=0.0012
# BIAS
#"""
rbias=np.abs(np.mean(x_pf[1,1:]-x_pf[1,:-1],axis=1)[:,a,0])
rbias2=np.abs(np.mean(x_pf[0]-x_pf[1],axis=1)[:,a,0])
plt.plot(eles[1:],rbias,label="Rbias")
plt.plot(eles,rbias2,label="Rbias2")
reference=np.array([1/2**(i) for i in range(len(eles[1:]))])*rbias[0]
var=np.var(x_pf[1,1:]-x_pf[1,:-1],axis=1)[:,a,0]
samples=1000
rbias_ub=rbias+np.sqrt(var)*1.96/np.sqrt(samples)
rbias_lb=rbias-np.sqrt(var)*1.96/np.sqrt(samples)
#plt.plot(eles[1:],rbias_ub)
#plt.plot(eles[1:],rbias_lb)
plt.plot(eles[1:],reference,label=r"$\Delta_l$")
print(eles)
[b0,b1]=coef(eles,np.log2(rbias2))
print([b0,b1,2**(2*b0)])
k=0.0012
#"""
    
# VARIANCE
"""
sm=np.mean((x_pf[0]-x_pf[1])**2,axis=1)[:,a,0]

var=np.var((x_pf[0]-x_pf[1]),axis=1)[:,a,0]
#print((x_pf[0]-x_pf[1])[:2,:,a,0])
reference=np.array([1/2**(i) for i in range(len(eles))])*sm[0]
plt.plot(eles,sm,label="Second moment the coupling")
plt.plot(eles,var,label="variance of the coupling")

plt.plot(eles,reference,label=r"$\Delta_l$")
var_sm=np.var(((x_pf[0]-x_pf[1])**2),axis=1)[:,a,0]
print(np.sqrt(var_sm))

sm_ub=sm+np.sqrt(var_sm)*1.96/np.sqrt(samples)
sm_lb=sm-np.sqrt(var_sm)*1.96/np.sqrt(samples)
plt.plot(eles,sm_ub,label="Upper error bound")
plt.plot(eles,sm_lb,label="Lower error bound")
#var=np.var((x_pf[0]-x_pf[1])[:,:,a],axis=1)
#var0=np.var((x_pf[0])[:,:,a],axis=1)

[b0,b1]=coef(eles,np.log2(sm))
print(b0,b1,(2**b0)*N)
#C=0.84
"""
#"""
plt.xlabel("l")
plt.legend()
plt.yscale("log")
plt.title("Second moment in terms of the time discretization levels")
#plt.savefig("Images/sm_CCPF_lan100_v1&v2.pdf")
plt.show()
#"""
 #%%
#"""
samples=100
p=8
l=5
N0=10
enes=np.array([2**i for i in range(p+1)])*N0
a=99
aa=0
d=2**(0)
x_pf=np.reshape(np.loadtxt("Observations&data/CCPF_ou100_enes_v1.txt",dtype=float),(2,len(enes),samples,int(T/d),dim))
#"""
#%%

variances=np.var(x_pf[0],axis=1)[:,a,0]
plt.plot(enes,variances,label="variance")
plt.plot(enes,enes[0]*variances[0]/enes,label="ref")

#"""
plt.xlabel("l")
plt.legend()
plt.yscale("log")
[b0,b1]=coef(np.log(1/enes),np.log(variances))
print(b0,b1,np.exp(b0))
#C0=0.5
plt.xscale("log")
plt.title("Richardson bias in terms of the time discretization levels")
#plt.savefig("Images/Rbias_CCPF_gbm20_p_v5.pdf")
plt.show()


#%%

# IN THE FOLLOWING WE CHECK THAT THE NUMBER OF PARTICLES TO APPLY THE MLPF IS 
# FEASIBLE


C=0.8
C0=0.5
K=0.0012
l0=0
Lmin=0
Lmax=6
eLes=np.arange(Lmin,Lmax+1,1)

for i in range(len(eLes)):
    
        L=eLes[i]
        NB0=(np.sqrt(C0)+np.sqrt(3*C/2)*L)*np.sqrt(2*C/3)/(K*L)
        N0=np.sqrt(C0)*(np.sqrt(C0)+np.sqrt(3*C/2)*L)*2**(2*L)/K
        eles=np.array(np.arange(l0,L+1))
        eNes=np.zeros(len(eles),dtype=int)
        scale=5e-2
        eNes[0]=N0*scale
        eNes[1:]=NB0*2**(L*2)*L*scale/2**(eles[1:])
        print(eNes)
        
#%%
#%%
#MLPF, MLPF, MLPF, MLPF, MLPF, MLPF, MLPF, MLPF, MLPF, MLPF, MLPF, MLPF, MLPF, MLPF.

#TEST FOR THE VARIANCES OF THE MLPF LEVELS

# In the following, depending of the lenght of the MLPF we get the variance 
# of the levels.
#"""
samples=100
l0=0
LminML=0
LmaxML=6
eLes=np.array(np.arange(LminML,LmaxML+1)) 
x_pf=np.reshape(np.loadtxt("Observations&data/PMLPF_cox_ou100_v1.txt"\
,dtype=float),(len(eLes),samples,int(T/d),dim))
#"""

#%%

# BIAS
#"""
a=99
rbias=np.abs(np.mean(x_pf[1:]-x_pf[:-1],axis=1)[:,a,0])
b_ref=np.array([1/2**i for i in range(len(eLes[1:]))])*rbias[0]
plt.plot(eLes[1:],rbias**2)
plt.plot(eLes[1:],b_ref**2)
#"""

# VARIANCE
#"""
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
#"""
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
C=0.8
C0=0.5
K=0.0012
l0=0
LminML=0
LmaxML=6
es=1e-6
eLes=np.arange(LminML,LmaxML+1,1)
N=int(2*C0/es)
L=int(np.ceil(-np.log2(es/(2*np.sqrt(K)))/2,dtype=float))
print(L,N)

#%%
resamp_coef=0.8
[log_weights,x_pf]=pff.Cox_PF(T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,\
L,d,N,dim,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par)
weights=norm_logweights(log_weights,ax=1)
x_pfmean=np.sum(np.reshape(weights,(int(T/d),N,1))*x_pf,axis=1)
phi_pf=x_pfmean.flatten()
#np.savetxt("Observations&data/Truth_ou_T100.txt",phi_pf,fmt="%f")

plt.plot((np.array(range(int(T/d)))+1)*d,x_pfmean[:,0],label="PF_ou")
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

# IN THE FOLLOWING LINES WE COMPARE THE MSE OF THE MLPF_COX WITH ITS COST
#MSE
#"""lps
a=99
mpf_true=np.reshape(np.loadtxt("Observations&data/Truth_ou_T100.txt"),(int(T/d),dim))
MSE=np.mean((x_pf-mpf_true)**2,axis=1)[:,a,0]
#COST
print(eLes)
Cost=np.zeros(len(eLes))
for i in range(len(eLes)):
            L=eLes[i]
            NB0=(np.sqrt(C0)+np.sqrt(3*C/2)*L)*np.sqrt(2*C/3)/(K*L)
            N0=np.sqrt(C0)*(np.sqrt(C0)+np.sqrt(3*C/2)*L)*2**(2*L)/K
            eles=np.array(np.arange(l0,L+1))
            eNes=np.zeros(len(eles),dtype=int)
            scale=5e-2
            eNes[0]=N0*scale
            eNes[1:]=NB0*2**(L*2)*L*scale/2**(eles[1:])
            print(eNes[1:]*2**eles[1:])
            Cost[i]+=np.sum(eNes*2**eles)

print(Cost) 
plt.plot(MSE,Cost,label="MLPF")

[b0,b1]=coef(-np.log(MSE),np.log(Cost))
print([b0,b1])
#plt.plot(MSE,np.exp(b0-b1*np.log(MSE)),label="refl")
shift1=1e1
shift2=5e0
plt.plot(MSE,shift2*np.log(np.sqrt(MSE*shift1))**2/(MSE*shift1),label=r"$\mathcal{O}(\log(\varepsilon)^2\varepsilon^{-2})$")


plt.plot(MSEUN[:,a,0],costs,label="Unbiased")
#plt.plot(var[:,a,0],costs,label="Var")
shift1=3e1
shift2=2e3
plt.plot(MSEUN[:,a,0],shift2*np.log(np.sqrt(MSEUN[:,a,0]*shift1))**2/(MSEUN[:,a,0]*shift1),label=r"$\mathcal{O}(\log(\varepsilon)^2\varepsilon^{-2})$")
plt.xlabel(r"$\varepsilon^2$")
plt.title("OU process, T=100.")
plt.ylabel("Cost")
plt.yscale("log")
plt.xscale("log")
plt.legend()
#plt.savefig("Images/MSEvsCost_ou_T=100_ML&UnB.pdf")
plt.show()

#plt.plot(eLes,MSE)
#plt.yscale("log")

#plt.show()
#"""
#%%
print(3%1)
#%%
# LEVLES AND PARTICLES TO CREATE THE ERROR TO COST RATE OF THE PFCOX
C=0.8
C0=0.5
K=0.0012
samples=100
es=np.array([1/4**i for i in range(7)])*1e-2
#eLes=np.arange(Lmin,LmaxML+1,1)
enes=np.array( np.ceil(2*C0/es),dtype=int)
a=89
eles=np.array(np.ceil(-np.log2(es/(2*np.sqrt(K)))/2,dtype=float),dtype=int)
print(L,N)
pf1=np.reshape(np.loadtxt("Observations&data/PCox_PF_etc_rate_v1.txt"),(len(eles[:-1]),samples,int(T/d),dim))
pf2=np.reshape(np.loadtxt("Observations&data/PCox_PF_etc_rate_v2.txt"),(1,samples,int(T/d),dim))
pf=np.concatenate((pf1,pf2),axis=0)
mpf_true=np.reshape(np.loadtxt("Observations&data/Truth_ou_T100.txt"),(int(T/d),dim))
MSE=np.mean((x_pf-mpf_true)**2,axis=1)[:,a,0]

costs=enes*2**eles
mse=np.mean((pf-mpf_true)**2,axis=1)[:,a,0]
mean=np.mean((pf-mpf_true),axis=1)[:,a,0]
[b0,b1]=coef(np.log(np.sqrt(mse))[:-2],np.log(costs)[:-2])
print([b0,b1,np.exp(b0)])
plt.plot(mse,costs,label="PF",ls="dashed",marker=".",c="red")
plt.plot(mse,np.exp(b0+b1*np.log(np.sqrt(mse))),label=r"$\mathcal{O}(\varepsilon^{-3.09})$")

plt.plot(MSE,Cost,label="MLPF",ls="dashed",marker="p",c="blue")
[b0,b1]=coef(-np.log(MSE),np.log(Cost))
print([b0,b1])
#plt.plot(MSE,np.exp(b0-b1*np.log(MSE)),label="refl")
shift1=4e0
shift2=1e0
plt.plot(MSE,shift2*np.log(np.sqrt(MSE*shift1))**2/(MSE*shift1),label=r"$\mathcal{O}(log(\varepsilon)^2\varepsilon^{-2})$",c="green")

plt.xlabel(r"$\varepsilon^2$")
plt.title("OU process, T=100.")
plt.ylabel("Cost")
plt.yscale("log")
plt.xscale("log")
plt.legend()
#plt.savefig("Images/MSEvsCost_PF_MLPF_ou_T=100.pdf")
plt.show()
"""
plt.plot(eles,mean**2)
plt.xlabel(r"$l$")
plt.title("OU process, T=100.")
plt.ylabel("Cost")
plt.yscale("log")
#plt.xscale("log")
plt.legend()
#plt.savefig("Images/MSEvsCost_ou_T=100.pdf")
plt.show()
"""


#%%




#UNBIASED, UNBIASED, UNBIASED, UNBIASED, UNBIASED, UNBIASED, UNBIASED, UNBIASED, UNBIASED, UNBIASED, UNBIASED, UNBIASED, 
#UNBIASED, UNBIASED, UNBIASED, UNBIASED, UNBIASED, UNBIASED, UNBIASED, UNBIASED, UNBIASED, UNBIASED, UNBIASED, UNBIASED, 

# IN THE FOLLOWING WE CHOOSE A REALIZATION OF THE OU DYNAMICS FOR TIME
# T=10 IN ORDER
# TO APPLY THE PF MACHINERY TO IT
if True==True:
        np.random.seed(7)
        T=10
        dim=1
        dim_o=dim
        xin=np.zeros(dim)+1
        l=13
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
        Lamb_par=2.2
        [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
        print(obs_time,obs,len(obs_time))
        
times=2**(-l)*np.array(range(int(T*2**l+1)))

plt.plot(times,x_true,label="True signal")
plt.plot(obs_time,obs,label="Observations")
plt.legend()
d=2**(0)
N=40
resamp_coef=0.8

#%%
#"""
l=6
[log_weights,x_pf]=pff.Cox_PF(T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,\
l,d,N,dim,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par)
weights=norm_logweights(log_weights,ax=1)
x_pfmean=np.sum(np.reshape(weights,(int(T/d),N,1))*x_pf,axis=1)
#print(x_pfmean)
upf_mean=np.mean(upf,axis=0)
plt.plot(times,x_true,label="True signal")
plt.plot((np.array(range(int(T/d)))+1)*d,x_pfmean[:,0],label="PF_OU")
plt.plot((np.array(range(int(T/d)))+1)*d,upf_mean[:,0],label="Unbiased_PF_OU")
plt.plot(obs_time,obs,label="Observations")
plt.legend()

#%%

# IN THIS CELL WE COMPUTE THE REFERENCE TRUTH IN ORDER TO COMPUTE THE BIAS
# OF THE UNBIASED ESTIMATOR
"""
l=5
N0=4
N=int(N0*2**8*1000*10)
print(N)
[lw,x]=pff.Cox_PF(T, xin, b_ou, B, Sig_ou, fi, obs, obs_time, l, d,N\
, dim, resamp_coef, g_den, g_par, Norm_Lambda, Lamb_par)

weights=norm_logweights(lw,ax=1)           
phi_pfmean=np.sum(np.reshape(weights,(int(T/d),N,1))*phi(x,ax=2),axis=1)
#np.savetxt("Observations&data/SUCPF_ou_N110240000_T10_v1.txt",phi_pfmean)
"""
#%%



dp=0.38
#cp*cl0=1.27
#cp0*cl/N0=0.007, N0=250
#delta_l=0.1 this quantity was obtained empirically.
#cp*cl/Np=0.003, Np=N0*2^p=125*16 
#cp*cl*Delta_6=0.067
#cl0*cp0*Delta_5=0.98

print(np.log2(1e-8/(2*0.01))/2)


#%%
#"""
#%%

samples=1000
a=9    
L=6
l0=5
p=8

N0=4
eLes=np.arange(l0,L+1)
ps=np.arange(0,p+1)
eNes=N0*2**ps
pfs=np.reshape(np.loadtxt("Observations&data/SUCPF_ou_p_levels_T10_v1.txt",dtype=float),(2,len(eNes)-1,samples,int(T/d),dim))
tel_est_exact=np.reshape(np.loadtxt("Observations&data/SUCPF_ou_N110240000_T10_v1.txt",dtype=float),(int(T/d),dim))
#BIAS
#"""
bias=np.abs(np.mean(pfs[1]-tel_est_exact,axis=1)[:,a,0])   
print(ps[1:]/eNes[1:],bias)
[b0,b1]=coef(np.log(ps[2:]/eNes[2:]),np.log(bias[1:]))
print(b0,b1,np.exp(b0))
#dp=0.38
reference=(ps[1:]/eNes[1:])*bias[1]*eNes[2]/ps[2]
plt.plot(ps[1:]/eNes[1:],reference,label="Ref: $x=y$")
plt.plot(ps[2:]/eNes[2:],bias[1:],label="bias")
plt.xscale("log")
plt.yscale("log")


plt.xlabel(r"$\frac{p}{N_p}$")
plt.title("Bias in terms of the particle levels of the OU for T=10")
plt.show()  

#"""
#VARIANCE

"""
sm=np.mean((pfs[0]-pfs[1])**2,axis=1)[:,a,0]
var=np.var((pfs[0]-pfs[1]),axis=1)[:,a,0]

ref=np.array([1/2**i for i in range(len(eNes)-1)])*sm[0]
[b0,b1]=coef(np.log(1/eNes[1:]),np.log(sm))
print(b0,b1,np.exp(b0))
#cp*cl0=1.27

plt.plot(1/eNes[1:],sm,label="sm")
plt.plot(1/eNes[1:],ref,label="$N^{-1}$")
plt.legend()
plt.xscale("log")
plt.yscale("log")
#plt.savefig("Images/bias_ou_p_levels_T10.pdf")
plt.show()  
"""

#%%


# IN THIS CELL WE COMPUTE THE REFERENCE TRUTH IN ORDER TO COMPUTE THE BIAS
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

samples=1000
a=9    
Lmax=8
l0=0
p=1
N0=250
eLes=np.arange(l0,Lmax+1)
ps=np.arange(0,p+1)
eNes=N0*2**ps
#nbot really scaled
pfs=np.reshape(np.loadtxt("Observations&data/SUCPF_ou_l_levels_T10_v1.txt",dtype=float),(2,len(eLes),samples,int(T/d),dim))
samples_truth=2000
tel_est_exact=np.mean(np.reshape(np.loadtxt("Observations&data/SUCPF_ou_l14_T10_v1.txt"\
                                            ,dtype=float),(samples_truth,int(T/d),dim)),axis=0)

#BIAS
#"""
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
#"""
#VARIANCE
"""
sm=np.mean((pfs[0]-pfs[1])**2,axis=1)[:,a,0]
var=np.var((pfs[0]-pfs[1]),axis=1)[:,a,0]

ref=np.array([1/2**i for i in range(len(eLes))])*sm[0]
[b0,b1]=coef(eLes+1,np.log2(sm))
print(b0,b1,2**b0)


plt.plot(eLes,sm,label="sm")
plt.plot(eLes,ref,label=r"$\Delta_l$")
plt.xlabel(r"$l$")
"""

plt.legend()
#plt.xscale("log")
plt.yscale("log")
plt.savefig("Images/sm_ou_l_levels_T10.pdf")
plt.show()  



#%%
#cp*cl/N0=0.007
#delta_l=0.1 this quantity was obtained empirically.
samples=1000
a=9    
Lmax=8
l0=0
p=1
N0=250
eLes=np.arange(l0,Lmax+1)
ps=np.arange(0,p+1)
eNes=N0*2**ps
#nbot really scaled
pfs=np.reshape(np.loadtxt("Observations&data/SUCPF_ou_lp_levels_T10_v1.txt",dtype=float),(len(eLes),samples,int(T/d),dim))
samples_truth=2000
tel_est_exact=np.mean(np.reshape(np.loadtxt("Observations&data/SUCPF_ou_l14_T10_v1.txt"\
                                            ,dtype=float),(samples_truth,int(T/d),dim)),axis=0)

#BIAS
"""
bias=np.abs(np.mean(pfs-tel_est_exact,axis=1)[:,a,0])   

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
sm=np.mean((pfs)**2,axis=1)[:,a,0]
#var=np.var((pfs[0]-pfs[1]),axis=1)[:,a,0]

ref=np.array([1/2**i for i in range(len(eLes))])*sm[0]
[b0,b1]=coef(eLes,np.log2(sm))
print(b0,b1,2**b0)
plt.plot(eLes,sm,label="sm")
plt.plot(eLes,ref,label=r"$\Delta_l$")
plt.xlabel(r"$l$")
#"""
plt.legend()
#plt.xscale("log")
plt.yscale("log")
#plt.savefig("Images/sm_ou_l_levels_T10.pdf")
plt.show()
#%%
samples=1000
a=9
Lmax=10
l0=0
p=2
N0=250
eLes=np.arange(l0,Lmax+1)
ps=np.arange(0,p+1)
eNes=N0*2**ps
pfs=np.reshape(np.loadtxt("Observations&data/SUCPF_ou_lp_levels_T10_v3.txt",dtype=float),(len(eLes),samples,int(T/d),dim))
#tel_est_exact=np.reshape(np.loadtxt("Observations&data/SUCPF_ou_l14_T10_v1.txt",dtype=float),(int(T/d),dim))
#VARIANCE
#"""
print(np.mean(pfs,axis=1)[:,a,0])
vvar=np.var((pfs-np.mean(pfs,axis=1,keepdims=True))**2,axis=1)[:,a,0]
var=np.var(pfs,axis=1)[:,a,0]
var=np.mean(pfs**2,axis=1)[:,a,0]
var_ub=var+np.sqrt(vvar)*1.96/np.sqrt(samples)
var_lb=var-np.sqrt(vvar)*1.96/np.sqrt(samples)
sm=np.mean(pfs**2,axis=1)[:,a,0]
ref=np.array([1/2**(i/1) for i in range(len(eLes))])*sm[0]
ref2=np.array([1/2**(2*i/3) for i in range(len(eLes))])*sm[0]
ref3=np.array([1/2**(1*i/2) for i in range(len(eLes))])*sm[0]
[b0,b1]=coef(eLes,np.log2(sm))
print(b0,b1,2**b0)
#cp*cl0=1.27
print(2**(-5.5),2**(-7.9))
plt.plot(eLes,var,label="var")
plt.plot(eLes,sm,label="sm")


plt.plot(eLes,var_ub,label="UB")
plt.plot(eLes,var_lb,label="LB")
plt.plot(eLes,ref,label=r"$\Delta_l$")
plt.plot(eLes,ref3,label=r"$\Delta_l^{0.5}$")
plt.plot(eLes,ref2,label=r"$\Delta_l^{2/3}$")

plt.xlabel(r"$l$")
#"""

plt.legend()
#plt.xscale("log")
plt.yscale("log")
plt.savefig("Images/sm_ou_lp_levels_T10.pdf")
plt.show()  

#%%
#cp*cl/Np=0.003, Np=N0*2^p=125*16 

samples=500
a=9
b=0
Lmax=10
l0=0
p=4
N0=125
eLes=np.arange(l0,Lmax+1)
ps=np.arange(0,p+1)
eNes=N0*2**ps
pfs=np.reshape(np.loadtxt("Observations&data/SUCPF_ou_lp_levels_T10_v7.txt",dtype=float),(2,len(eLes),samples,int(T/d),dim))
#tel_est_exact=np.reshape(np.loadtxt("Observations&data/SUCPF_ou_l14_T10_v1.txt",dtype=float),(int(T/d),dim))

 
#VARIANCE
#"""
sm=np.mean((pfs[0]-pfs[1])**2,axis=1)[:,a,0]
sm1=np.mean((pfs[0])**2,axis=1)[:,a,0]

#sm=np.mean((pfs[0]-pfs[1])**2,axis=1)[:,a,0]
ref=np.array([1/2**(i/1) for i in range(len(eLes))])*sm[-1]*2**(len(eLes)-1)
ref2=np.array([1/2**(2*i/3) for i in range(len(eLes))])*sm[0]
ref3=np.array([1/2**(1*i/2) for i in range(len(eLes))])*sm[0]
[b0,b1]=coef(eLes,np.log2(sm))
print(b0,b1,2**b0)
#plt.plot(eLes,var,label="var")
plt.plot(eLes,sm,label="sm")
plt.plot(eLes,sm1,label="sm1")



plt.plot(eLes,ref,label=r"$\Delta_l$")
#plt.plot(eLes,ref2,label=r"$\Delta_l^{0.5}$")
plt.plot(eLes,ref3,label=r"$\Delta_l^{2/3}$")

plt.xlabel(r"$l$")
#"""

plt.legend()
#plt.xscale("log")
plt.yscale("log")
plt.savefig("Images/sm_ou_lp_levels_T10.pdf")
plt.show() 

#%%
#cp*cl*Delta_6=0.067Z

samples=1000
a=9    
L=6
l0=5
p=8
N0=4
eLes=np.arange(l0,L+1)
ps=np.arange(0,p+1)
eNes=N0*2**ps
pfs=np.reshape(np.loadtxt("Observations&data/SUCPF_ou_pl_levels_T10_v1.txt",dtype=float),(len(eNes)-1,samples,int(T/d),dim))
tel_est_exact=np.reshape(np.loadtxt("Observations&data/SUCPF_ou_N110240000_T10_v1.txt",dtype=float),(int(T/d),dim))



sm=np.mean((pfs)**2,axis=1)[:,a,0]
var=np.var((pfs),axis=1)[:,a,0]

ref=np.array([1/2**i for i in range(len(eNes)-1)])*sm[0]
[b0,b1]=coef(np.log(1/eNes[1:]),np.log(sm))
print(b0,b1,np.exp(b0))
#cp*cl0=1.27

plt.plot(1/eNes[1:],var,label="sm")
plt.plot(1/eNes[1:],ref,label="$N^{-1}$")
plt.xlabel(r"$N^{-1}$")
plt.legend()
plt.xscale("log")
plt.yscale("log")
#plt.savefig("Images/bias_ou_p_levels_T10.pdf")
plt.show()  
#%%

# COMPUTATION OF C0
samples=1000
a=9    
L=6
l0=5
p=8
N0=4
eLes=np.arange(l0,L+1)
ps=np.arange(0,p+1)
eNes=N0*2**ps
pfs=np.reshape(np.loadtxt("Observations&data/SUCPF_ou_pl6&5_levels_T10_v1.txt",dtype=float),(len(eNes),samples,int(T/d),dim))



sm=np.mean((pfs)**2,axis=1)[:,a,0]
var=np.var((pfs),axis=1)[:,a,0]

ref=np.array([1/2**i for i in range(len(eNes))])*sm[0]
[b0,b1]=coef(np.log(1/eNes),np.log(sm))
print(b0,b1,np.exp(b0))

plt.plot(1/eNes,var,label="sm")
plt.plot(1/eNes,ref,label="$N^{-1}$")
plt.xlabel(r"$N^{-1}$")
plt.legend()
plt.xscale("log")
plt.yscale("log")
#plt.savefig("Images/bias_ou_p_levels_T10.pdf")
plt.show()  

#%%
#cl0*cp0=0.98

samples=1000
a=9    
L=6
l0=5
p=8
N0=4
eLes=np.arange(l0,L+1)
ps=np.arange(0,p+1)
eNes=N0*2**ps
pfs=np.reshape(np.loadtxt("Observations&data/SUCPF_ou_pl0_levels_T10_v1.txt",dtype=float),(len(eNes),samples,int(T/d),dim))
sm=np.mean((pfs)**2,axis=1)[:,a,0]
var=np.var((pfs),axis=1)[:,a,0]

ref=np.array([1/2**i for i in range(len(eNes))])*var[0]
[b0,b1]=coef(np.log(1/eNes),np.log(var))
print(b0,b1,np.exp(b0))
#cp*cl0=1.27

plt.plot(1/eNes,var,label="var")
plt.plot(1/eNes,ref,label="$N^{-1}$")
plt.xlabel(r"$N^{-1}$")
plt.legend()
plt.xscale("log")
plt.yscale("log")
#plt.savefig("Images/bias_ou_p_levels_T10.pdf")
plt.show()  
#%%

#TEST OF THE DATA FOR THE UNBIASED COXPF

samples=100000
N0=10
l0=0
Lmax=11
pmax=13
eLes=np.arange(l0,Lmax+1)
ps=np.array(range(pmax+1))
eNes=N0*2**ps
    
    
beta=1
Delta0=1/2**eLes[0]
print(eNes,eLes)
Pp=0.166554
Pp0=7.61859
Pl=0.0161376
Pl0=0.769263
Pld=np.zeros(Lmax+1-l0)
Ppd=np.zeros(pmax+1)
Pld[0]=Delta0**(beta)*Pl0
Pld[1:]=Pl*np.log2(eLes[1:]+2)**2*(eLes[1:]+1)/2**(beta*eLes[1:])
Ppd[0]=Pp0/N0
Ppd[1:]=Pp*np.log2(ps[1:]+2)**2*(ps[1:]+1)/eNes[1:]
Ppd=Ppd/np.sum(Ppd)
Pld=Pld/np.sum(Pld)
Pp_cum=np.cumsum(Ppd)
Pl_cum=np.cumsum(Pld)

pfs=np.reshape(np.loadtxt("Observations&data/SUCPF_ou_T10_v1.txt",dtype=float),(samples,int(T/d),dim))
lps=np.loadtxt("Observations&data/SUCPF_ou_pls_T10_v1.txt",dtype=int)
print(lps[4])
#%%

upf=pfs/(Ppd[lps[:,1],np.newaxis,np.newaxis]*Pld[lps[:,0],np.newaxis,np.newaxis])
var=np.var(upf,axis=0)
print(var)

#tieme=456.2824909687042

#%%



samples=100000
N0=10
l0=0
Lmax=11
pmax=13
eLes=np.arange(l0,Lmax+1)
ps=np.array(range(pmax+1))
eNes=N0*2**ps
    
    
beta=1
Delta0=1/2**eLes[0]
print(eNes,eLes)
Pp=1
Pp0=1
Pl=1
Pl0=1
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

pfs=np.reshape(np.loadtxt("Observations&data/SUCPF_ou_T10_v2.txt",dtype=float),(samples,int(T/d),dim))
lps=np.loadtxt("Observations&data/SUCPF_ou_pls_T10_v2.txt",dtype=int)
print(lps[4])
#%%

upf=pfs/(Ppd[lps[:,1],np.newaxis,np.newaxis]*Pld[lps[:,0],np.newaxis,np.newaxis])
var=np.var(upf,axis=0)
print(var)


#%%
#T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100
# UNBIASED, UNBIASED, # UNBIASED, UNBIASED, # UNBIASED, UNBIASED, # UNBIASED, UNBIASED.
# IN THE FOLLOWING WE CHOOSE A REALIZATION OF THE OU DYNAMICS FOR TIME
# T=100 IN ORDER
# TO APPLY THE PF MACHINERY TO IT
if True==True:
        np.random.seed(1)
        T=100
        dim=1
        dim_o=dim
        xin=np.zeros(dim)+1
        l=13
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
        Lamb_par=1.33
        [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
        print(obs_time,obs,len(obs_time))
        
times=2**(-l)*np.array(range(int(T*2**l+1)))

plt.plot(times,x_true,label="True signal")
plt.plot(obs_time,obs,label="Observations")
plt.legend()
d=2**(0)
resamp_coef=0.8

#%%
#"""
l=8
N=1000
[log_weights,x_pf]=pff.Cox_PF(T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,\
l,d,N,dim,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par)
weights=norm_logweights(log_weights,ax=1)
x_pfmean=np.sum(np.reshape(weights,(int(T/d),N,1))*x_pf,axis=1)
#print(x_pfmean)
plt.plot(times,x_true,label="True signal")
plt.plot((np.array(range(int(T/d)))+1)*d,x_pfmean[:,0],label="PF_OU")
plt.plot((np.array(range(int(T/d)))+1)*d,mpf_true[:,0],label="True filter")

plt.plot(obs_time,obs,label="Observations")
plt.legend()
#"""

#%%
print(0.0008*250)
#cl=0.0008*250=0.2
#delta_l=0.03
#cp0=0.52
#cl0=0.52

# COMPUTATION OF CP AND CP0 AND DELTA_P

#dp=0.024
#cp=0.74

# IN THIS CELL WE COMPUTE THE REFERENCE TRUTH IN ORDER TO COMPUTE THE BIAS
# OF THE UNBIASED ESTIMATOR
#"""
l=4
N0=4
N=int(N0*2**8*1000)
print(N)
[lw,x]=pff.Cox_PF(T, xin, b_ou, B, Sig_ou, fi, obs, obs_time, l, d,N\
, dim, resamp_coef, g_den, g_par, Norm_Lambda, Lamb_par)

weights=norm_logweights(lw,ax=1)           
phi_pfmean=np.sum(np.reshape(weights,(int(T/d),N,1))*phi(x,ax=2),axis=1)
#%%
#np.savetxt("Observations&data/SUCPF_ou_N1024000_T100_v1.txt",phi_pfmean)
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
pfs=np.reshape(np.loadtxt("Observations&data/SUCPF_ou_p_levels_T10o_v1.txt",dtype=float),(2,len(eNes)-1,samples,int(T/d),dim))
tel_est_exact=np.reshape(np.loadtxt("Observations&data/SUCPF_ou_N1024000_T100_v1.txt",dtype=float),(int(T/d),dim))
#BIAS
"""
bias=np.abs(np.mean(pfs[1]-tel_est_exact,axis=1)[:,a,0])   
print(ps[1:]/eNes[1:],bias)
[b0,b1]=coef(np.log(ps[2:]/eNes[2:]),np.log(bias[1:]))
print(b0,b1,np.exp(b0))
#dp=0.38
reference=(ps[1:]/eNes[1:])*bias[1]*eNes[2]/ps[2]
plt.plot(ps[1:]/eNes[1:],reference,label="Ref: $x=y$")
plt.plot(ps[2:]/eNes[2:],bias[1:],label="bias")
plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"$\frac{p}{N_p}$")
plt.title("Bias in terms of the particle levels of the OU for T=100")
plt.show()  

"""
#VARIANCE
#"""
sm=np.mean((pfs[0]-pfs[1])**2,axis=1)[:,a,0]
var=np.var((pfs[0]-pfs[1]),axis=1)[:,a,0]

ref=np.array([1/2**i for i in range(len(eNes)-1)])*sm[0]
[b0,b1]=coef(np.log(1/eNes[1:]),np.log(sm))
print(b0,b1,np.exp(b0))
#cp*cl0=1.27

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

#cl=0.0008*250
#delta_l=0.03

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
pfs=np.reshape(np.loadtxt("Observations&data/SUCPF_ou_l_levels_T100_v2.txt",dtype=float),(2,len(eLes),samples,int(T/d),dim))
samples_truth=2000
tel_est_exact=np.mean(np.reshape(np.loadtxt("Observations&data/SUCPF_ou_l14_T100_v1.txt"\
                                            ,dtype=float),(samples_truth,int(T/d),dim)),axis=0)

#BIAS
#"""
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
#"""
#VARIANCE
"""
sm=np.mean((pfs[0]-pfs[1])**2,axis=1)[:,a,0]
var=np.var(((pfs[0]-pfs[1])**2),axis=1)[:,a,0]
sm_ub=sm+np.sqrt(var)*1.96/np.sqrt(samples)
sm_lb=sm-np.sqrt(var)*1.96/np.sqrt(samples)
plt.plot(eLes,sm_ub)
plt.plot(eLes,sm_lb)
ref=np.array([1/2**i for i in range(len(eLes))])*sm[0]
[b0,b1]=coef(eLes+1,np.log2(sm))
print(b0,b1,2**b0)
plt.plot(eLes,sm,label="sm")
plt.plot(eLes,ref,label=r"$\Delta_l$")
plt.xlabel(r"$l$")
"""

plt.legend()
#plt.xscale("log")
plt.yscale("log")
plt.savefig("Images/sm_ou_l_levels_T10.pdf")
plt.show()  

#%%
#cp0=0.52

samples=1000
a=99  
L=6
l0=5
p=8
N0=4
eLes=np.arange(l0,L+1)
ps=np.arange(0,p+1)
eNes=N0*2**ps
pfs=np.reshape(np.loadtxt("Observations&data/SUCPF_ou_pl0_levels_T100_v1.txt",dtype=float),(len(eNes),samples,int(T/d),dim))
sm=np.mean((pfs)**2,axis=1)[:,a,0]
var=np.var((pfs),axis=1)[:,a,0]
ref=np.array([1/2**i for i in range(len(eNes))])*var[0]
[b0,b1]=coef(np.log(1/eNes),np.log(var))
print(b0,b1,np.exp(b0))
#cp*cl0=1.27
plt.plot(1/eNes,var,label="var")
plt.plot(1/eNes,ref,label="$N^{-1}$")
plt.xlabel(r"$N^{-1}$")
plt.legend()
plt.xscale("log")
plt.yscale("log")
#plt.savefig("Images/bias_ou_p_levels_T10.pdf")
plt.show()  
#%%
samples=500000
a=99
pfs=np.reshape(np.loadtxt("remote/SUCPF_ou_T100_v1.txt",dtype=float),(samples,int(T/d),dim))
#%%
lps=np.loadtxt("remote/SUCPF_ou_pls_T100_v1.txt",dtype=float)
mpf_true=np.reshape(np.loadtxt("Observations&data/Truth_ou_T100.txt"),(int(T/d),dim))
#%%
lps=np.array(lps,dtype=int)
print(pfs.shape)
pmax=5
Lmax=10

dic=[[[] for j in range(pmax+1)] for i in range(Lmax+1)]

for i in range(len(pfs)):
    dic[lps[i,0]][lps[i,1]].append(i)

Num_samples=np.zeros((Lmax+1,pmax+1),dtype=int)
for i in range(Lmax+1):
    for j in range(pmax+1):
        Num_samples[i,j]=len(dic[i][j])
print(Num_samples)



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
# IN THE FOLLOWING WE CREATE DIFFERENT PROBABILITIES DISTRIBUTIONS AND 
# CHECK THE MSE 

#seed=0 with ps=np.array([5,5,5,5,5,5],dtype=int)-1, ls=np.array([5,5,4,4,6,7],dtype=int)+1
np.random.seed(2)
ps=np.array([5,5,5,5,5,5],dtype=int)
#ps=np.array([1,2,5,7,9,11],dtype=int)+1
ls=np.array([5,5,6,6,7,7],dtype=int)+3
N0=100
D0=1
beta=1/2
cl=0.2
cp0=0.52
cl0=0.52
cp=0.74
batches=10

costs=np.zeros(len(ps))
var=np.zeros((len(ls),int(T/d),dim))
MSEUN=np.zeros((len(ls),int(T/d),dim))
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
#%%
a=99
#MSE=1e-1/8**np.array(range(len(ps)))
plt.plot(MSEUN[1:,a,0],costs[1:],label="Unbiased",c="coral",ls="dashed",marker=".",ms=10)
#plt.plot(var[:,a,0],costs,label="Var")
shift1=3e1
shift2=2e3
#plt.plot(MSEUN[:,a,0],shift2*np.log(np.sqrt(MSEUN[:,a,0]*shift1))**2\
#/(MSEUN[:,a,0]*shift1),label=r"$\mathcal{O}(\log(\varepsilon)^2\varepsilon^{-2})$")
plt.plot(MSEUN[1:,a,0],np.exp(b0)*MSEUN[1:,a,0]**(b1/2),label=r"$\mathcal{O}(\varepsilon^{-2.4})$",c="dodgerblue")
[b0,b1]=coef(np.log(MSEUN[1:,a,0]**(1/2)),np.log(costs[1:]))
print([b0,b1,np.exp(b1)])
plt.title("Ornstein Uhlenbeck process. T=100")
plt.xlabel(r"$\varepsilon^2$")
plt.ylabel("Cost")
plt.yscale("log")
plt.xscale("log")
plt.legend()
#plt.savefig("Images/Unbiased_MSEvsCost_ou_T=100.pdf")
plt.show()

#%%
lists=np.array([10**(-i) for i in range( 1,6)])
print(-np.log(lists)/np.log(8)+2)
#%%
N0=100
l0=0
Lmax=10
pmax=5
eLes=np.arange(l0,Lmax+1)
ps=np.array(range(pmax+1))
eNes=N0*2**ps


beta=1
Delta0=1/2**eLes[0]
print(eNes,eLes)
Pp=3.75525
Pp0=55.787
Pl=0.0224577
Pl0=0.73559
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
emes=np.concatenate((emes,[50000]))
ps=100
print(emes)

mse=np.zeros((len(emes),int(T/d),dim))
var=np.zeros((len(emes),int(T/d),dim))

costs=np.zeros(len(emes))

for i in range(len(emes)):
    m=emes[i]
    plot_samples=np.random.choice(5000000,size=ps*m,replace=False)
    print(plot_samples[:10])
    costs[i]=np.sum(2**lps[plot_samples,0]*2**lps[plot_samples,1])*N0/ps
    upf=pfs[plot_samples]/(Ppd[lps[plot_samples,1],np.newaxis,\
    np.newaxis]*Pld[lps[plot_samples,0],np.newaxis,np.newaxis])
    upf=np.reshape(upf,(ps,m,int(T/d),dim))
    upfs=np.mean(upf,axis=1)
    var[i]=np.var(upfs,axis=0)
    mse[i]=np.mean((upfs-mpf_true)**2,axis=0)

#%%
k=5
[b0,b1]=coef(np.log(mse[:k,a,0]),np.log(costs[:k]))
print([b0,b1])
print(mse[:,a,0])
a=99
MSE=mse[:,a,0]
shift1=2e1
shift2=5e4
plt.plot(mse[:k,a,0],costs[:k],label="MSE")
plt.plot(MSE,shift2*np.log(np.sqrt(MSE*shift1))**2/(MSE*shift1),label=r"$\mathcal{O}(np.log(\varepsilon)^2\varepsilon^{-2})$")
plt.legend()
plt.yscale("log")
plt.xscale("log")


#%%
# THIS SECTION IS MADE IN ORDER TO TRY NEW PROBABILITY DISTRIBTUION FOR THE 
# LEVELS AND TO DETERMINE THE TRUNCATION IN TERMS OF THE ERROR. 

# IN the following we create a dictionary that assigns the position of the sample 
# in pfs to a set labeled by its levels.



#%%
# this section is made to check the rate whenever we have large
# truncation parameter pmax and lmax
N0=3
l0=0
Lmax=13
pmax=14
eLes=np.arange(l0,Lmax+1)
ps=np.array(range(pmax+1))
eNes=N0*2**ps
beta=1
Delta0=1/2**eLes[0]
print(eNes,eLes)
Pp=3.75525
Pp0=55.787
Pl=0.0224577
Pl0=0.73559
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
emes=np.concatenate((emes,[50000]))
ps=100
emes=np.array(emes*100/(ps),dtype=int)
print(emes)
mse=np.zeros((len(emes),int(T/d),dim))
var=np.zeros((len(emes),int(T/d),dim))

costs=np.zeros(len(emes))
#np.random.seed(5)
for i in range(len(emes)):
    m=emes[i]
    print(m)
    for j in range(m):
        l=sampling(Pl_cum)
        p=sampling(Pp_cum)
        costs[i]+=2**(l+p)
        
    var[i]=1/m

print(costs)
[b0,b1]=coef(-np.log(np.sqrt(var[:,a,0])),np.log(costs))
print([b0,b1])
a=99
MSE=mse[:,a,0]
shift1=2e1
shift2=5e4
plt.plot(var[:,a,0],costs)
#plt.plot(MSE,shift2*np.log(np.sqrt(MSE*shift1))**2/(MSE*shift1),label=r"$\mathcal{O}(np.log(\varepsilon)^2\varepsilon^{-2})$")
plt.yscale("log")
plt.xscale("log")

#%%



# SAMPLING OF THE UNBIASED ESTIMATOR, THIS EXPERIMENT IS MADE SO WE CAN USE SAMPLES ALREADY TAKEN FROM THE OU WITH A DIFFERENT 
# PROBABILITY DISTRIBUTION

T=100
dim=1
dim_o=dim
xin=np.zeros(dim)+1
l=13
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
Lamb_par=1.33
[obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)

start=time.time()
d=2**(0)
resamp_coef=0.8    
dim_out=2
#%%

samples=5000000
N0=3
l0=0
Lmax=13
Lmaxv1=10
pma=8
eNes=np.concatenate(([3,6,12,24,50],100*2**np.array(range(0,pma+1))))
print(len(eNes),eNes)
eLes=np.arange(l0,Lmax+1)

ps=np.arange(len(eNes))

pmax=len(ps)-1
#%%
beta=1
Delta0=1/2**eLes[0]
print(eNes,eLes)
Pp=3.75525
Pp0=55.787
Pl=0.0224577
Pl0=0.73559
Pld=np.zeros(Lmax+1-l0)
Ppd=np.zeros(pmax+1)
Pld[0]=Delta0**(beta)*Pl0
Pld[1:]=Pl*np.log2(eLes[1:]+2)**2*(eLes[1:]+1)/2**(beta*eLes[1:])
print(N0)
Ppd[0]=Pp0/N0
print(ps)
Ppd[1:]=Pp*np.log2(ps[1:]+2)**2*(ps[1:]+1)/eNes[1:]



Ppd=Ppd/np.sum(Ppd)
Pld=Pld/np.sum(Pld)
print(Ppd,Pld)
Pp_cum=np.cumsum(Ppd)
Pl_cum=np.cumsum(Pld)

#%%
pmaxv1=5
lpsv1=np.loadtxt("Observations&data/SUCPF_ou_pls_T100_v1.txt",dtype=float)
lpsv1=np.array(lpsv1,dtype=int)


dic=[[[] for j in range(pmaxv1+1)] for i in range(Lmaxv1+1)]


for i in range(len(pfs)):
    dic[lpsv1[i,0]][lpsv1[i,1]].append(i)

Num_samples=np.zeros((Lmaxv1+1,pmaxv1+1))
for i in range(Lmaxv1+1):
    for j in range(pmaxv1+1):
        Num_samples[i,j]=len(dic[i][j])


lps=np.zeros((samples,4),dtype=int)    
inputs=[]
samples=50
psv1=np.arange(0,pmaxv1+1)
samples_count=np.zeros((Lmaxv1+1,pmaxv1+1),dtype=int)
samplesv2=0
for sample in range(samples):
        np.random.seed(sample)
        l=eLes[sampling(Pl_cum)]
        p=sampling(Pp_cum)
        pv1=p-5
        if (p in psv1+5) and (l in np.arange(l0,Lmaxv1+1)) and (Num_samples[l,pv1]-samples_count[l,pv1]>0):
            
            lps[sample]=np.array([0,l,p,dic[l][pv1][samples_count[l,pv1]]])
            samples_count[l,pv1]+=1
        
        else:
            Pp_cum_det=np.concatenate((np.zeros(p), 1+np.zeros(pmax+1-p)))
            Pl_cum_det=np.concatenate((np.zeros(l), 1+np.zeros(Lmax+1-l)))
            lps[sample]=np.array([1,l,p,samplesv2])
            samplesv2+=1
            collection_input=[T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,eLes,Pl_cum_det,d,eNes\
            ,Pp_cum_det,dim,resamp_coef,phi,g_den,g_par,Norm_Lambda,Lamb_par]

            inputs.append([samplesv2,collection_input]) 
        
pool = multiprocessing.Pool(processes=10)
pool_outputs = pool.map(PSUCPF, inputs)
pool.close()
pool.join()
#blocks_pools.append(pool_outputs)
xend1=time.time()
end=time.time()

        

print("Parallelized processes time:",end-start,"\n")            
pfs=np.zeros((samplesv2,int(T/d),dim))
for sample in range(samplesv2):
        pfs[sample]=pool_outputs[sample][0]

pfs=pfs.flatten()
#np.savetxt("Observations&data/SUCPF_ou_T100_v1.txt",pfs,fmt="%f")
#np.savetxt("Observations&data/SUCPF_ou_pls_T100_v1.txt",lps,fmt="%f")


#%%

# HERE WE CREATE TWO PROPAGATIONS AT TWO SUBSEQUENT LEVELS AND TEST THE INTERPOLATION

#x1=np.array([[[1],[1]],[[1],[2]],[[1],[4]]])
#x0=np.array([[[1],[1]],[[1],[3]]])
obs=np.array([[0.1],[0.7]])
obs_time=np.array([0.3,1.9]) 
T=2
B=np.array([[-1]])   
Sig=diags(np.array([1]),0).toarray()
fi=np.array([[1]])
N=2
l=2
d=1
dim=1
resamp_coef=0.
xin=np.zeros(dim)+1
g_par=np.array([[2]])
Lamb_par=1/3.
#w1=pff.Gox(obs,obs_times,x1,Norm_Lambda,Lamb_par,l,N,dim,g_den,g_par)
#%%
np.random.seed(0)
CCPF(T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par)
#%%
#VERSION 2
#T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100
# IN THE FOLLOWING WE CHOOSE A REALIZATION OF THE OU DYNAMICS FOR TIME
# T=100 IN ORDER
# TO APPLY THE PF MACHINERY TO IT
np.random.seed(2)
T=100
dim=1
dim_o=dim
xin=np.zeros(dim)+1
l=10
collection_input=[]
I=identity(dim).toarray()
comp_matrix=np.array([[1]])
inv_mat=la.inv(comp_matrix)
S=diags(np.random.normal(0.99,0.1,dim),0).toarray()
sca=1e-3
Lamb_par=1.33
fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
cov=I*1e5
B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
B=(inv_mat@B@comp_matrix)*sca*5e1
collection_input=[dim, b_ou,B,Sig_ou,fi]
g_pars=[dim,cov]
g_par=cov
print("B and fi are",B,fi)
x_true=gen_gen_data(T,xin,l,collection_input)
[obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
#obs=np.array([])
#obs_time=np.array([])
print(obs_time,obs,len(obs_time))
times=2**(-l)*np.array(range(int(T*2**l+1)))
plt.plot(times,x_true,label="True signal")
plt.plot(obs_time,obs,label="Observations")
plt.legend()
N=40
resamp_coef=0.8
#%%
#"""
d=5
l=6
#np.random.seed(2)
[log_weights,x_pf]=pff.Cox_PF(T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,\
l,d,N,dim,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par)
weights=norm_logweights(log_weights,ax=1)
x_pfmean=np.sum(np.reshape(weights,(int(T/d),N,1))*x_pf,axis=1)
#print(x_pfmean)
plt.plot(times,x_true,label="True signal")
plt.plot((np.array(range(int(T/d)))+1)*d,x_pfmean[:,0],label="PF_OU")
#plt.plot(obs_time,obs,label="Observations")
plt.legend()
#"""
#%%
samples=10
l0=1
L=9
eles=np.array(range(l0,L+1))
N=10
d=1
T=10
xs=np.reshape(np.loadtxt("Observations&data/CCPF_ou100_scaled1_x_v16_33.txt",dtype=float),(2,len(eles),samples,int(T/d),N,dim))
log_ws=np.reshape(np.loadtxt("Observations&data/CCPF_ou100_scaled1_lw_v16_33.txt",dtype=float),(2,len(eles),samples,int(T/d),N))
#%%
# VARIANCE
#"""
a=-1
print("THE TIME IS T=",a)
sm=np.mean(np.sum((((log_ws[0,:,:,:,:])\
-(log_ws[1,:,:,:,:]))**2),axis=-1),axis=1)[:,a]
#sm=np.mean(np.sum(((np.exp(log_ws[0,:,:,:,:])*xs[0,:,:,:,:,0]\
#-np.exp(log_ws[1,:,:,:,:])*xs[1,:,:,:,:,0])**2),axis=-1),axis=1)[:,a]
#sm=np.mean(np.sum(((np.exp(log_ws[0,:,:,:,:])\
#-np.exp(log_ws[1,:,:,:,:]))**2),axis=-1),axis=1)[:,a]
#sm=np.mean(np.sum(((xs[0,:,:,:,:,0]\
#-xs[1,:,:,:,:,0])**2),axis=-1),axis=1)[:,a]
print(sm)
#sm=np.mean((x_pf[0]-x_pf[1])**2,axis=1)[:,a,0]
#var=np.var((x_pf[0]-x_pf[1]),axis=1)[:,a,0]
#print((x_pf[0]-x_pf[1])[:2,:,a,0])
reference=np.array([1/2**(i) for i in range(len(eles))])*sm[-1]*2**8
reference2=np.array([1/2**(i*2) for i in range(len(eles))])*sm[0]
plt.plot(eles,sm,label="Second moment the coupling")
#plt.plot(eles,var,label="variance of the coupling")
plt.plot(eles,reference,label=r"$\Delta_l$")
plt.plot(eles,reference2,label=r"$\Delta_l^2$")
#var_sm=np.var(((x_pf[0]-x_pf[1])**2),axis=1)[:,a,0]
#print(np.sqrt(var_sm))
#sm_ub=sm+np.sqrt(var_sm)*1.96/np.sqrt(samples)
#sm_lb=sm-np.sqrt(var_sm)*1.96/np.sqrt(samples)
#plt.plot(eles,sm_ub,label="Upper error bound")
#plt.plot(eles,sm_lb,label="Lower error bound")
#var=np.var((x_pf[0]-x_pf[1])[:,:,a],axis=1)
#var0=np.var((x_pf[0])[:,:,a],axis=1)
[b0,b1]=coef(eles,np.log2(sm))
print(b0,b1,(2**b0)*N)
#C=0.84
#"""
#"""
plt.xlabel("l")
plt.legend()
plt.yscale("log")
plt.title("Second moment in terms of the time discretization levels")
#plt.savefig("Images/sm_CCPF_lan100_v1&v2.pdf")
plt.show()
#"""
#%%
#"""
samples=100
l0=1
L=9
eles=np.array(range(l0,L+1))
N=1000
d=25
T=100
x_pf=np.reshape(np.loadtxt("Observations&data/CCPF_ou100_scaled3_v11.txt",dtype=float),(2,len(eles),samples,int(T/d),dim))
#x_pf_1=np.reshape(np.loadtxt("Observations&data/CCPF_ou100_scaled1_v7.txt",dtype=float),(2,1,samples,int(T/d),dim))
#x_pf=np.concatenate((x_pf,x_pf_1),axis=1)
#eles=np.concatenate((eles,[10]))
#"""
#%%
#PARAMETERS 
#C=0.8
#C0=0.5
#K=0.0012
#BIAS
#"""
a=-1
#rbias=np.abs(np.mean(x_pf[1,1:]-x_pf[1,:-1],axis=1)[:,a,0])
rbias=np.abs(np.mean(x_pf[0]-x_pf[1],axis=1)[:,a,0])
plt.plot(eles,rbias,label="Rbias")
#plt.plot(eles,rbias2,label="Rbias2")
reference=np.array([1/2**(i) for i in range(len(eles))])*rbias[0]
var=np.var(x_pf[1,1:]-x_pf[1,:-1],axis=1)[:,a,0]
samples=100
#rbias_ub=rbias+np.sqrt(var)*1.96/np.sqrt(samples)
#rbias_lb=rbias-np.sqrt(var)*1.96/np.sqrt(samples)
#plt.plot(eles[1:],rbias_ub)
#plt.plot(eles[1:],rbias_lb)
plt.plot(eles,reference,label=r"$\Delta_l$")
print(eles)
[b0,b1]=coef(eles,np.log2(rbias))
print([b0,b1,2**(2*b0)])
k=0.0012
#"""
# VARIANCE
#"""
print("THE TIME IS T=",a)
sm=np.mean((x_pf[0]-x_pf[1])**2,axis=1)[:,a,0]
var=np.var((x_pf[0]-x_pf[1]),axis=1)[:,a,0]
print("sm is ",sm)
#print((x_pf[0]-x_pf[1])[:2,:,a,0])
reference=np.array([1/2**(i/2) for i in range(len(eles))])*sm[-1]*2**4
#reference2=np.array([1/2**(i*2) for i in range(len(eles))])*sm[0]
plt.plot(eles,sm,label="Second moment the coupling")
plt.plot(eles,var,label="variance of the coupling")
plt.plot(eles,reference,label=r"$\Delta_l^{-1/2}$")
#plt.plot(eles,reference2,label=r"$\Delta_l^2$")
var_sm=np.var(((x_pf[0]-x_pf[1])**2),axis=1)[:,a,0]
#print(np.sqrt(var_sm))
sm_ub=sm+np.sqrt(var_sm)*1.96/np.sqrt(samples)
sm_lb=sm-np.sqrt(var_sm)*1.96/np.sqrt(samples)
plt.plot(eles,sm_ub,label="Upper error bound")
plt.plot(eles,sm_lb,label="Lower error bound")
#var=np.var((x_pf[0]-x_pf[1])[:,:,a],axis=1)
#var0=np.var((x_pf[0])[:,:,a],axis=1)
[b0,b1]=coef(eles[3:],np.log2(sm)[3:])
print(b0,b1,(2**b0)*N)
#C=0.03
#"""
#"""
plt.xlabel("l")
plt.legend()
plt.yscale("log")
plt.title("Second moment in terms of the time discretization levels")
#plt.savefig("Images/sm_CCPF_lan100_v1&v2.pdf")
plt.show()
#"""
#%%
#"""
T=100
samples=100
p=8
l=5
N0=10
enes=np.array([2**i for i in range(p+1)])*N0
aa=0
d=25
x_pf=np.reshape(np.loadtxt("Observations&data/CCPF_ou100_scaled3_enes_v11.txt",dtype=float),(len(enes),samples,int(T/d),dim))
#"""
#%%
a=-1
variances=np.var(x_pf,axis=1)[:,a,0]
plt.plot(enes,variances,label="variance")
plt.plot(enes,enes[0]*variances[0]/enes,label="ref")

#"""
plt.xlabel("l")
plt.legend()
plt.yscale("log")
[b0,b1]=coef(np.log(1/enes),np.log(variances))
print(b0,b1,np.exp(b0))
#C0=0.2
plt.xscale("log")
plt.title("Richardson bias in terms of the time discretization levels")
#plt.savefig("Images/Rbias_CCPF_gbm20_p_v5.pdf")
plt.show()


# %%



# CCONSTANTS AND PPARAMETERS FOR THE SINGLE COX PARTICLE FILTER
C0=0.00066
C=np.sqrt(2)*5e-6
print(C0/C)
K=2.331505041965383e-08
l0=0
Lmin=l0
Lmax=7
es=1e-9
scale=1e-4
eLes=np.arange(Lmin,Lmax+1,1)
N=int(2*C0/es)
L=int(np.ceil(-np.log2(es/(2*np.sqrt(K)))/2,dtype=float))
print(L,N)
#%%
resamp_coef=0.8
[log_weights,x_pf]=pff.Cox_PF(T,xin,b_ou,B,Sig_nldt,fi,obs,obs_time,\
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
C0=0.00066
C=np.sqrt(2)*5e-6
print(C0/C)
K=2.331505041965383e-08
l0=0
Lmin=l0
Lmax=5
eLes=np.arange(Lmin,Lmax+1,1)
Cost=np.zeros(len(eLes))
C_sin=C0
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
        print("N0 is", N0)
        eles=np.array(np.arange(l0,L+1))
        eNes=np.zeros(len(eles),dtype=int)
        scale=2e-2
        eNes[0]=N0*scale
        #Cost[i]+=eNes[0]*2**eLes[0]
        eNes[0]=np.maximum(2,eNes[0])
        eNes[1:]=np.sqrt(2*C/3)*(np.sqrt(C0/delta_l0)+np.sqrt(3*C/2)*CL*delta_l0**(-1/4))\
        *np.float_power(2,2*L)*np.float_power(2,-eles[1:]*3/4)/K*scale
        #eNes[1:]=NB0*np.float_power(2,-eles[1:]*3/4)*np.float_power(2,(2+1/4)*L)*scale
        Cost[i]+=3*np.sum(eNes[1:]*np.float_power(2,eles[1:]))/2+eNes[0]*np.float_power(2,eles[0])    
        print(eNes)
        particles_sin[i]=scale*particles_sin[i]
        Cost_sin[i]=scale*Cost_sin[i]
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



# IN THE FOLLWOING WE COMPUTE THE PF, MLPF AND THE "TRUE" ESTIMATORS

#%%

# SSINGLE
# PF, PF, PF, PF, PF, PF, PF, PF, PF, PF, PF, PF, PF, PF, PF, PF, PF, PF, PF
#"""
T=100
samples=100
l0=0
Lmin=0
Lmax=5
d=25
eLes_sin=np.array(np.arange(Lmin,Lmax+1)) 
x_pf1_sin=np.reshape(np.loadtxt("remote/PPF_cox_ou_t100_scaled1_v1.txt"\
,dtype=float),(len(eLes_sin),samples,int(T/d),dim))
#x_pf2_sin=np.reshape(np.loadtxt("Observations&data/PPF_cox_nldt100_scaled2_v2.txt"\
#,dtype=float),(1,samples,int(T/d),dim))
#x_pf3_sin=np.reshape(np.loadtxt("remote/PPF_cox_nldt100_scaled2_v3.txt"\
#,dtype=float),(1,samples,int(T/d),dim))
#x_pf1_sin=np.concatenate((x_pf1_sin,x_pf2_sin,x_pf3_sin),axis=0)

#eLes_sin=np.concatenate((eLes_sin,np.array([4,5])),axis=0)
#%%

#MMLPF
T=100
samples=100
l0=0
d=25
Lmin=l0
Lmax=5
eLes=np.array(np.arange(Lmin,Lmax+1)) 
x_pf=np.reshape(np.loadtxt("remote/PMLPF_cox_ou_T100_scaled1_v1.txt"\
,dtype=float),(len(eLes),samples,int(T/d),dim))
#x_pf2=np.reshape(np.loadtxt("Observations&data/PMLPF_cox_nldt_T100_scaled2_v2.txt"\
#,dtype=float),(1,samples,int(T/d),dim))
#x_pf3=np.reshape(np.loadtxt("remote/PMLPF_cox_nldt_T100_scaled2_v3.txt"\
#,dtype=float),(1,samples,int(T/d),dim))

#x_pf=np.concatenate((x_pf,x_pf2,x_pf3),axis=0)
#eLes=np.concatenate((eLes,np.array([4,5])),axis=0)
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
d=1
a=99
mpf_true=np.reshape(np.loadtxt("remote/PPF_cox_ou_Truth_T100_scaled1_v1.txt"),(int(T/d),dim))
print(mpf_true.shape)
#here we adjust mpf_true to have just 4 entries since we d=25 for the PF and MLPF
grid=25*np.array([i for i in range(1,5)],dtype=int)-1
mpf_true=mpf_true[grid]
#%%

#HERE WE COMPUTE THE MSE OF THE PF AND MLPF

MSE=np.mean((x_pf-mpf_true)**2,axis=1)[:,a,0]
MSE_sin=np.mean((x_pf1_sin-mpf_true)**2,axis=1)[:,a,0]
#COST
l0=0
Lmin=l0
Lmax=5
eLes=np.array(np.arange(Lmin,Lmax+1,1),dtype=float)
#print(eLes)
Cost=np.zeros(len(eLes))
C0=0.025
C=0.0003243
K=5.67e-6
C_sin=C0
k_sin=K

#Cost_sin=2**eLes*C_sin*2**(2*eLes)/k_sin

for i in range(len(eLes)):

            L=eLes[i]
            CL=2**(1/4)*(2**((L-l0)/4)-1)/(2**(1/4)-1)
            delta_l0=np.float_power(2,-l0)
            N0=np.sqrt(C0*delta_l0)*(np.sqrt(C0/delta_l0)+np.sqrt(3*C/2)*CL*delta_l0**(-1/4))*np.float_power(2,2*L)/K
            eles=np.array(np.arange(l0,L+1))
            eNes=np.zeros(len(eles),dtype=int)
            scale=2e-2
            eNes[0]=N0*scale
            eNes[0]=np.maximum(2,eNes[0])
            eNes[1:]=np.sqrt(2*C/3)*(np.sqrt(C0/delta_l0)+np.sqrt(3*C/2)*CL*delta_l0**(-1/4))\
            *np.float_power(2,2*L)*np.float_power(2,-eles[1:]*3/4)/K*scale
            print(eNes)
            #print(eNes[1:]*2**eles[1:])
            Cost[i]+=np.sum(3*(eNes[1:]*2**eles[1:])/2)+eNes[0]*2**eles[0]
Cost_sin=scale*np.float_power(8, eLes_sin)*C_sin/k_sin
print("eNes single",np.array(scale*C_sin*np.float_power(4,eLes)/k_sin,dtype=int))
    
#%%
print("Cost",Cost) 
print("Cost_single",Cost_sin) 
#plt.plot(MSE_un,costs[k1:],label="Unbiased")
MSE_arti=1/2**(eLes*2)
plt.plot(MSE,Cost,label="MLPF",marker="o",c="coral",alpha=0.8,ls="--")
#plt.plot(MSE_arti,Cost,label="MLPF",marker="o")
plt.plot(MSE_sin,Cost_sin,label="Single PF",marker="+",markersize=10,ls="dashdot",color="dodgerblue")
#plt.plot(MSE_arti[:-1],Cost_sin[:-1],label="Single PF",marker="o")
plt.plot(MSE_sin,1.6e-7*MSE_sin**(-3/2),label=r"$\varepsilon^{-3}$",color="deepskyblue")
plt.plot(MSE,3.3e-6*MSE**(-2.5/2),label=r"$\varepsilon^{-2.5}$",color="salmon")
plt.xlabel(r"$\varepsilon^2$")
plt.title("Ornstein Uhlenbeck process. T=100.")
plt.ylabel("Cost")
plt.yscale("log")
plt.xscale("log")
plt.legend()
#plt.savefig("Images/name.pdf")
plt.show()
#%%