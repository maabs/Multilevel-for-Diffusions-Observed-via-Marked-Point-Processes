#!/usr/bin/env python3
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
"""
# IN THE FOLLOWING LINES WE TEST THE ASSYMPTOTIC BEHAVIOUR OF THE 
# LANGEVIN DYNAMICS. 
np.random.seed(1)
T=500
dim=1
dim_o=dim
xin=np.zeros(dim)+0
l=5
collection_input=[]
I=identity(dim).toarray()


df=10
fi=np.array([[1]])
collection_input=[dim, b_lan,df,Sig_lan,fi]
cov=I*1e0
g_pars=[dim,cov]
g_par=cov
x_true=gen_gen_data(T,xin,l,collection_input)
Lamb_par=.5
[obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
N=100
d=2**(0)
times=2**(-l)*np.array(range(int(T*2**l+1)))
print(x_true.shape)
plt.plot(times,x_true)

samples=500

rea=np.zeros((samples,dim))

for sample in range(samples):
    rea[sample]=gen_gen_data(T,xin,l,collection_input)[-1]

#%%
pos=np.arange(-10,10,0.4)
plt.hist(rea.flatten(),bins= pos,density=True)
df=10000
x = np.linspace(t.ppf(0.001, df),t.ppf(0.999, df), 100)
plt.plot(x,t.pdf(x,df))

"""

#%%


# IN THE FOLLOWING WE CHOOSE A REALIZATION OF THE LANGEVIN DYNAMICS IN ORDER
# TO APPLY THE PF MACHINERY TO IT.
np.random.seed(4)
T=20
dim=1
dim_o=dim
xin=np.zeros(dim)+1
l=13
collection_input=[]
I=identity(dim).toarray()

df=10
fi=np.array([[1]])
collection_input=[dim, b_lan,df,Sig_lan,fi]
cov=I*1e0
g_pars=[dim,cov]
g_par=cov
x_true=gen_gen_data(T,xin,l,collection_input)
Lamb_par=5/3
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
"""
l=6
[log_weights,x_pf]=pff.Cox_PF(T,xin,b_lan,df,Sig_lan,fi,obs,obs_time,\
l,d,N,dim,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par)
weights=norm_logweights(log_weights,ax=1)

x_pfmean=np.sum(np.reshape(weights,(int(T/d),N,1))*x_pf,axis=1)
#print(x_pfmean)
plt.plot((np.array(range(int(T/d)))+1)*d,x_pfmean[:,0],label="PF_Lan")
plt.plot(times,x_true,label="True signal")
plt.plot(obs_time,obs,label="Observations")
plt.legend()
"""
#%%
#"""
samples=500
l0=1
L=10
eles=np.array(range(l0,L+1))
N=5000
a=19
aa=0
d=2**(0)
x_pf=np.reshape(np.loadtxt("Observations&data/CCPF_lan20_v.txt",dtype=float),(2,len(eles),samples,int(T/d),dim))
#"""
#%%
#PARAMETERS 
#C=
#C0=
#K=
# BIAS
#"""
rbias=np.abs(np.mean(x_pf[1,1:]-x_pf[1,:-1],axis=1)[:,a,0])
rbias2=np.abs(np.mean(x_pf[0]-x_pf[1],axis=1)[:,a,0])
plt.plot(eles[1:],rbias,label="Rbias")

plt.plot(eles,rbias2,label="Rbias2")
reference=np.array([1/2**(i) for i in range(len(eles[1:]))])*rbias[0]
var=np.var(x_pf[1,1:]-x_pf[1,:-1],axis=1)[:,a,0]
#print(var,np.sqrt(var)*1.96/np.sqrt(samples),np.sqrt(samples),np.sqrt(var))
rbias_ub=rbias+np.sqrt(var)*1.96/np.sqrt(samples)
rbias_lb=rbias-np.sqrt(var)*1.96/np.sqrt(samples)
#plt.plot(eles[1:],rbias_ub)
#plt.plot(eles[1:],rbias_lb)
plt.plot(eles[1:],reference,label=r"$\Delta_l$")
#[b0,b1]=coef(eles,np.log2(rbias[:6]))
#print([b0,b1,2**(2*b0)])
k=0.0250
#"""
    
# VARIANCE
#"""
sm=np.mean((x_pf[0]-x_pf[1])**2,axis=1)[:,a,0]
#print((x_pf[0]-x_pf[1])[:2,:,a,0])
reference=np.array([1/2**(i/2) for i in range(len(eles))])*sm[2]
plt.plot(eles[2:],sm[2:],label="Second moment the coupling")
plt.plot(eles[2:],reference[2:],label=r"$\Delta_l^{1/2}$")
var_sm=np.var(((x_pf[0]-x_pf[1])**2)[:,:,a],axis=1)
sm_ub=sm+np.sqrt(var_sm)*1.96/np.sqrt(samples)
sm_lb=sm-np.sqrt(var_sm)*1.96/np.sqrt(samples)
var=np.var((x_pf[0]-x_pf[1])[:,:,a],axis=1)
var0=np.var((x_pf[0])[:,:,a],axis=1)
print(var0*N)
[b0,b1]=coef(eles,np.log2(sm))
print(b0,b1,(2**b0)*N)
#"""
#"""
plt.xlabel("l")
plt.legend()
plt.yscale("log")
plt.title("Richardson bias in terms of the time discretization levels")
#plt.savefig("Images/Rbias_CCPF_gbm20_p_v5.pdf")
plt.show()
#"""
#%%
#"""
samples=100
p=8
l=5
N0=10
enes=np.array([2**i for i in range(p+1)])*N0
a=19
aa=0
d=2**(0)
x_pf=np.reshape(np.loadtxt("Observations&data/Cox_PF_lan20_v1.txt",dtype=float),(len(enes),samples,int(T/d),dim))
#"""
#%%

variances=np.var(x_pf,axis=1)[:,a,0]
plt.plot(enes,variances,label="variance")
plt.plot(enes,enes[0]*variances[0]/enes,label="ref")


#"""
plt.xlabel("l")
plt.legend()
plt.yscale("log")

plt.xscale("log")
plt.title("Richardson bias in terms of the time discretization levels")
#plt.savefig("Images/Rbias_CCPF_gbm20_p_v5.pdf")
plt.show()

#%%
# THE CONCLUSION IS THAT WE OBSERVE THE RATES OF THE PREVIOUS
# FOR THE COUPLING.

#%%








#2,T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100

# IN THE FOLLOWING WE CHOOSE A REALIZATION OF THE LANGEVIN DYNAMICS FOR TIME
# T=100 IN ORDER
# TO APPLY THE PF MACHINERY TO IT
np.random.seed(0)
T=100
dim=1
dim_o=dim
xin=np.zeros(dim)+1
l=13
collection_input=[]
I=identity(dim).toarray()

df=10
fi=np.array([[1]])
collection_input=[dim, b_lan,df,Sig_lan,fi]
cov=I*1e0
g_pars=[dim,cov]
g_par=cov
x_true=gen_gen_data(T,xin,l,collection_input)
Lamb_par=5/3
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
[log_weights,x_pf]=pff.Cox_PF(T,xin,b_lan,df,Sig_lan,fi,obs,obs_time,\
l,d,N,dim,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par)
weights=norm_logweights(log_weights,ax=1)
x_pfmean=np.sum(np.reshape(weights,(int(T/d),N,1))*x_pf,axis=1)
#print(x_pfmean)
plt.plot((np.array(range(int(T/d)))+1)*d,x_pfmean[:,0],label="PF_Lan")
plt.plot(times,x_true,label="True signal")
plt.plot(obs_time,obs,label="Observations")
plt.legend()
#"""
#%%
#"""
samples=1000
l0=1
L=10
eles1=np.array(range(l0,L+1))
N=5000
a=99
aa=0
d=2**(0)
x_pf1=np.reshape(np.loadtxt("Observations&data/CCPF_lan100_v1.txt",dtype=float),(2,len(eles1),samples,int(T/d),dim))
samples=100
l0=11
L=12
eles2=np.array(range(l0,L+1))
x_pf2=np.reshape(np.loadtxt("Observations&data/CCPF_lan100_v2.txt",dtype=float),(2,len(eles2),samples,int(T/d),dim))
#"""
#%%
#PARAMETERS 
#C=1
#C0=0.05
#K=0.021
# BIAS
"""
a=99
x_pf=x_pf1
rbias=np.abs(np.mean(x_pf[1,1:]-x_pf[1,:-1],axis=1)[:,a,0])
rbias21=np.abs(np.mean(x_pf1[0]-x_pf1[1],axis=1)[:,a,0])
rbias22=np.abs(np.mean(x_pf2[0]-x_pf2[1],axis=1)[:,a,0])
rbias2=np.concatenate((rbias21,rbias22),axis=0)
plt.plot(eles1[1:],rbias,label="Rbias")
eles=np.concatenate((eles1,eles2))
plt.plot(eles,rbias2,label="Rbias2")
reference=np.array([1/2**(i) for i in range(len(eles[1:]))])*rbias[0]
var=np.var(x_pf[1,1:]-x_pf[1,:-1],axis=1)[:,a,0]
#print(var,np.sqrt(var)*1.96/np.sqrt(samples),np.sqrt(samples),np.sqrt(var))
samples=1000
rbias_ub=rbias+np.sqrt(var)*1.96/np.sqrt(samples)
rbias_lb=rbias-np.sqrt(var)*1.96/np.sqrt(samples)
#plt.plot(eles[1:],rbias_ub)
#plt.plot(eles[1:],rbias_lb)
plt.plot(eles[1:],reference,label=r"$\Delta_l$")
print(eles)
[b0,b1]=coef(eles[1:],np.log2(rbias2[1:]))
print([b0,b1,2**(2*b0)])
k=0.021
"""
    
# VARIANCE
#"""
a=99
sm1=np.mean((x_pf1[0]-x_pf1[1])**2,axis=1)[:,a,0]
sm2=np.mean((x_pf2[0]-x_pf2[1])**2,axis=1)[:,a,0]
sm=np.concatenate((sm1,sm2))
var1=np.var((x_pf1[0]-x_pf1[1]),axis=1)[:,a,0]
var2=np.var((x_pf2[0]-x_pf2[1]),axis=1)[:,a,0]
var=np.concatenate((var1,var2))
eles=np.array(range(12))+1
#print((x_pf[0]-x_pf[1])[:2,:,a,0])
reference=np.array([1/2**(i) for i in range(len(eles))])*sm[0]
plt.plot(eles,sm,label="Second moment the coupling")
plt.plot(eles,var,label="variance of the coupling")

plt.plot(eles,reference,label=r"$\Delta_l$")
var_sm=np.concatenate((np.var(((x_pf1[0]-x_pf1[1])**2),axis=1)[:,a,0],\
np.var(((x_pf2[0]-x_pf2[1])**2),axis=1)[:,a,0]))
print(np.sqrt(var_sm))
samples=np.where(np.arange(1,13,1)<11,1000,100)
print(samples)
sm_ub=sm+np.sqrt(var_sm)*1.96/np.sqrt(samples)
sm_lb=sm-np.sqrt(var_sm)*1.96/np.sqrt(samples)
plt.plot(eles,sm_ub,label="Upper error bound")
plt.plot(eles,sm_lb,label="Lower error bound")
#var=np.var((x_pf[0]-x_pf[1])[:,:,a],axis=1)
#var0=np.var((x_pf[0])[:,:,a],axis=1)
print(1e-3*N)
print(eles)
[b0,b1]=coef(eles,np.log2(sm))
print(b0,b1,(2**b0)*N)
#C=8.91
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
p=8
l=5
N0=10
enes=np.array([2**i for i in range(p+1)])*N0
a=99
aa=0
d=2**(0)
x_pf=np.reshape(np.loadtxt("Observations&data/CCPF_lan100_enes_v1.txt",dtype=float),(2,len(enes),samples,int(T/d),dim))
#"""
#%%

variances=np.var(x_pf[0],axis=1)[:,a,0]
plt.plot(enes,variances,label="variance")
plt.plot(enes,enes[0]*variances[0]/enes,label="ref")


#"""
plt.xlabel("N")
plt.legend()
plt.yscale("log")
[b0,b1]=coef(np.log(1/enes),np.log(variances))
print(b0,b1,np.exp(b0))
#C0=0.5
plt.xscale("log")
plt.title("Variance in terms of the number of particles")
#plt.savefig("Images/Rbias_CCPF_gbm20_p_v5.pdf")
plt.show()


#%%

# IN THE FOLLOWING WE CHECK THAT THE NUMBER OF PARTICLES TO APPLY THE MLPF IS 
# FEASIBLE

C0=0.5
C=C0*1e-2
K=0.021
l0=0
Lmin=0
Lmax=7
eLes=np.arange(Lmin,Lmax+1,1)
C_sin=C0
k_sin=K
l0_sin=l0
Lmin_sin=l0_sin
Lmax_sin=Lmax
d=1
Cost=np.zeros(len(eLes))
eLes_sin=np.array(np.arange(Lmin_sin,Lmax_sin+1)) 
Cost_sin=np.float_power(8, eLes_sin)*C_sin/k_sin
particles_sin=np.float_power(4, eLes_sin)*C_sin/k_sin

for i in range(len(eLes)):
    
        L=eLes[i]
        Deltal0=np.float_power(2,-l0)
        NB0=(np.sqrt(C0/Deltal0)+np.sqrt(3*C/2)*(L-l0))*np.sqrt(2*C/3)/(K*(L-l0))
        N0=np.sqrt(C0*Deltal0)*(np.sqrt(C0/Deltal0)+np.sqrt(3*C/2)*(L-l0))*2**(2*L)/K
        eles=np.array(np.arange(l0,L+1))
        eNes=np.zeros(len(eles),dtype=int)
        scale=1
        eNes[0]=N0*scale
        eNes[1:]=NB0*2**(L*2)*L*scale/2**(eles[1:])
        print(eNes)
        Cost[i]+=3*np.sum(eNes[1:]*np.float_power(2,eles[1:]))/2+eNes[0]*np.float_power(2,eles[0])  
        particles_sin[i]=particles_sin[i]*scale
        Cost_sin[i]=Cost_sin[i]*scale  
        print("# of particles of the single term",particles_sin[i])
        print("the MSE is approx", C0/particles_sin[i])

plt.plot(eLes,Cost,label="Cost")
plt.plot(eLes_sin,Cost_sin,label="Cost Single")
plt.yscale("log")
#plt.xscale("log")
plt.legend()
plt.show()

        
#%%
#%%
#MLPF, MLPF, MLPF, MLPF, MLPF, MLPF, MLPF, MLPF, MLPF, MLPF, MLPF, MLPF, MLPF, MLPF.

#TEST FOR THE VARIANCES OF THE MLPF LEVELS

# In the following, depending of the lenght of the MLPF we get the variance 
# of the levels.
#"""
samples=100
l0=0
Lmin=0
Lmax=6
eLes=np.array(np.arange(Lmin,Lmax+1)) 
x_pf=np.reshape(np.loadtxt("Observations&data/PMLPF_cox_lan100_v3.txt"\
,dtype=float),(len(eLes),samples,int(T/d),dim))
x_pf1=np.reshape(np.loadtxt("remote/PMLPF_cox_lan100_v5.txt"\
,dtype=float),(1,samples,int(T/d),dim))
x_pf=np.concatenate((x_pf,x_pf1),axis=0)        
Lmax=7
eLes=np.array(np.arange(Lmin,Lmax+1)) 

#"""

#%%

# BIAS
#"""
a=99
print(np.mean(x_pf1[0,:,-1,0],axis=0),np.mean(x_pf[:,:,-1,0],axis=1))
rbias=np.abs(np.mean(x_pf[1:]-x_pf[:-1],axis=1)[:,a,0])
b_ref=np.array([1/2**i for i in range(len(eLes[1:]))])*rbias[0]
plt.plot(eLes[1:],rbias**2)
plt.plot(eLes[1:],b_ref**2)
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
C=2
C0=0.5
K=0.021
l0=0
Lmin=0
Lmax=7
es=1e-6
eLes=np.arange(Lmin,Lmax+1,1)
N=int(2*C0/es)
L=int(np.ceil(-np.log2(es/(2*np.sqrt(K)))/2,dtype=float))
print(L,N)
#%%
#"""
resamp_coef=0.8
[log_weights,x_pf]=pff.Cox_PF(T,xin,b_lan,df,Sig_lan,fi,obs,obs_time,\
L,d,N,dim,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par)
weights=norm_logweights(log_weights,ax=1)
x_pfmean=np.sum(np.reshape(weights,(int(T/d),N,1))*x_pf,axis=1)
phi_pf=x_pfmean.flatten()
#np.savetxt("Observations&data/TruthLanT100.txt",phi_pf,fmt="%f")

plt.plot((np.array(range(int(T/d)))+1)*d,x_pfmean[:,0],label="PF_Lan")
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
#"""
a=99
C=2
C0=0.5
K=0.021
mpf_true=np.reshape(np.loadtxt("Observations&data/TruthLanT100.txt"),(int(T/d),dim))
MSE=np.mean((x_pf-mpf_true)**2,axis=1)[:,a,0]
print(MSE)

#COST
print(eLes)
Cost=np.zeros(len(eLes))
for i in range(len(eLes)):
            L=eLes[i]
            NB0=(np.sqrt(C0)+np.sqrt(3*C/2)*L)*np.sqrt(2*C/3)/(K*L)
            N0=np.sqrt(C0)*(np.sqrt(C0)+np.sqrt(3*C/2)*L)*2**(2*L)/K
            eles=np.array(np.arange(l0,L+1))
            eNes=np.zeros(len(eles),dtype=int)
            scale=3e-1
            eNes[0]=N0*scale
            eNes[1:]=NB0*2**(L*2)*L*scale/2**(eles[1:])
            print(eNes)
            Cost[i]+=np.sum(eNes*2**eles)

plt.plot(MSE,Cost,label="MLPF")
[b0,b1]=coef(-np.log(MSE),np.log(Cost))
print([b0,b1])
#plt.plot(MSE,np.exp(b0-b1*np.log(MSE)),label="refl")
shift1=4e0
shift2=1e1
plt.plot(MSE,shift2*np.log(np.sqrt(MSE*shift1))**2/(MSE*shift1),label=r"$\mathcal{O}(\log(\varepsilon)^2\varepsilon^{-2})$")
print(MSE*b1+b0)

shift1=4e-1
shift2=1e3
#plt.plot(MSEUN[:,a,0],costs,label="MSE_unbi_lan")
#plt.plot(MSEUN[:,a,0],shift2*np.log(np.sqrt(MSEUN[:,a,0]*shift1))**2/(MSEUN[:,a,0]*shift1)\
#,label=r"$\mathcal{O}(\log(\varepsilon)^2\varepsilon^{-2})$")


#plt.plot(MSE,MSE**(-2.5/2),label=r"$\varepsilon^{-2.5}$")
plt.xlabel(r"$\varepsilon^2$")
plt.title("Langevin process, T=100.")
plt.ylabel("Cost")
plt.yscale("log")
plt.xscale("log")
plt.legend()
#plt.savefig("Images/MSEvsCost_lan_T_ML&UnB=100.pdf")
plt.show()
#plt.plot(eLes,MSE) 
#plt.yscale("log")
plt.show()
#"""
#%%


#T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100
# UNBIASED, UNBIASED, # UNBIASED, UNBIASED, # UNBIASED, UNBIASED, # UNBIASED, UNBIASED.
# IN THE FOLLOWING WE CHOOSE A REALIZATION OF THE OU DYNAMICS FOR TIME
# T=100 IN ORDER
# TO APPLY THE PF MACHINERY TO IT

np.random.seed(0)
T=100
dim=1
dim_o=dim
xin=np.zeros(dim)+1
l=13
collection_input=[]
I=identity(dim).toarray()
df=10
fi=np.array([[1]])
collection_input=[dim, b_lan,df,Sig_lan,fi]
cov=I*1e0
g_pars=[dim,cov]
g_par=cov
x_true=gen_gen_data(T,xin,l,collection_input)
Lamb_par=5/3
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
#cp=1.36
#dp=0.2
#cl=2
#dl**2=0.021
# dl=0.14
#cp0=1.27
print(np.sqrt(0.021))
#%%
#"""
l=6
[log_weights,x_pf]=pff.Cox_PF(T,xin,b_lan,df,Sig_lan,fi,obs,obs_time,\
l,d,N,dim,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par)
weights=norm_logweights(log_weights,ax=1)
x_pfmean=np.sum(np.reshape(weights,(int(T/d),N,1))*x_pf,axis=1)
#print(x_pfmean)
plt.plot((np.array(range(int(T/d)))+1)*d,x_pfmean[:,0],label="PF_Lan")
plt.plot(times,x_true,label="True signal")
plt.plot(obs_time,obs,label="Observations")
plt.legend()
#"""
#%%
#"""

# IN THIS CELL WE COMPUTE THE REFERENCE TRUTH IN ORDER TO COMPUTE THE BIAS
# OF THE UNBIASED ESTIMATOR
#"""
"""
l=4
N0=4
N=int(N0*2**8*1000)
print(N)
[lw,x]=pff.Cox_PF(T, xin,b_lan,df,Sig_lan,fi, obs, obs_time, l, d,N\
, dim, resamp_coef, g_den, g_par, Norm_Lambda, Lamb_par)
weights=norm_logweights(lw,ax=1)    
phi_pfmean=np.sum(np.reshape(weights,(int(T/d),N,1))*phi(x,ax=2),axis=1)
"""
#%%
#np.savetxt("Observations&data/SUCPF_lan_N1024000_T100_v1.txt",phi_pfmean)
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
pfs=np.reshape(np.loadtxt("Observations&data/SUCPF_lan_p_levels_T10o_v1.txt",dtype=float),(2,len(eNes)-1,samples,int(T/d),dim))
tel_est_exact=np.reshape(np.loadtxt("Observations&data/SUCPF_lan_N1024000_T100_v1.txt",dtype=float),(int(T/d),dim))
#BIAS
#"""
bias=np.abs(np.mean(pfs[1]-tel_est_exact,axis=1)[:,a,0])   
print(ps[1:]/eNes[1:],bias)
[b0,b1]=coef(np.log(ps[2:]/eNes[2:]),np.log(bias[1:]))
print(b0,b1,np.exp(b0))
#dp=0.2
reference=(ps[1:]/eNes[1:])*bias[1]*eNes[2]/ps[2]
plt.plot(ps[1:]/eNes[1:],reference,label="Ref: $x=y$")
plt.plot(ps[2:]/eNes[2:],bias[1:],label="bias")
plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"$\frac{p}{N_p}$")
plt.title("Bias in terms of the particle levels of the OU for T=100")
plt.show()  

#"""
#VARIANCE
"""
sm=np.mean((pfs[0]-pfs[1])**2,axis=1)[:,a,0]
var=np.var((pfs[0]-pfs[1]),axis=1)[:,a,0]

ref=np.array([1/2**i for i in range(len(eNes)-1)])*sm[0]
[b0,b1]=coef(np.log(1/eNes[1:]),np.log(sm))
print(b0,b1,np.exp(b0))
#cp=1.36

plt.plot(1/eNes[1:],sm,label="sm")
plt.plot(1/eNes[1:],ref,label="$N^{-1}$")
plt.legend()
plt.xscale("log")
plt.yscale("log")
#plt.savefig("Images/bias_ou_p_levels_T10.pdf")
plt.show()  
"""
#%%




# IN THIS CELL WE COMPUTE THE REFERENCE TRUTH IN ORDER TO COMPUTE THE BIAS in
# THE DISCRETIZATION LEVEL AND ALSO CL
# OF THE UNBIASED ESTIMATOR
"""
l=14
N0=250
N=N0
[lw,x]=pff.Cox_PF(T, xin, b_lan, df, Sig_lan, fi, obs, obs_time, l, d,N\
, dim, resamp_coef, g_den, g_par, Norm_Lambda, Lamb_par)
weights=norm_logweights(lw,ax=1)           
phi_pfmean=np.sum(np.reshape(weights,(int(T/d),N,1))*phi(x,ax=2),axis=1)
np.savetxt("Observations&data/SUCPF_lan_l14_T10_v1.txt",phi_pfmean)
"""
#%%
#computation of cl and dl

# Looking at the graph and increasing the height of the reference to match 
# the end of the line (l=8) I regard cl=0.08*250 which is rather high
print(0.04*25)
samples=1000
a=99
Lmax=8
l0=0
p=1
N0=25
eLes=np.arange(l0,Lmax+1)
ps=np.arange(0,p+1)
eNes=N0*2**ps
#nbot really scaled
pfs=np.reshape(np.loadtxt("Observations&data/SUCPF_lan_l_levels_T100_v2.txt",dtype=float),(2,len(eLes),samples,int(T/d),dim))
samples_truth=2000
tel_est_exact=np.mean(np.reshape(np.loadtxt("Observations&data/SUCPF_lan_l14_T100_v1.txt"\
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
ref2=np.array([1/2**(i/2) for i in range(len(eLes))])*sm[0]
[b0,b1]=coef(eLes+1,np.log2(sm))
print(b0,b1,2**b0)


plt.plot(eLes,sm,label="sm")
plt.plot(eLes,ref,label=r"$\Delta_l$")
plt.plot(eLes,ref2,label=r"$\Delta_l^{1/2}$")
plt.xlabel(r"$l$")
"""

plt.legend()
#plt.xscale("log")
plt.yscale("log")
#plt.savefig("Images/sm_ou_l_levels_T10.pdf")
plt.show()  

#%%
# WE CAN OBTAIN CP0 USING THE EXPERIMENTS TO FIND C0, IN THIS CASE 
# CP0=0.6
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

N0=5
l0=0
Lmax=10
pmax=11
eLes=np.arange(l0,Lmax+1)
ps=np.array(range(pmax+1))
eNes=N0*2**ps
beta=1/2
Delta0=1/2**eLes[0]
print(eNes,eLes)
Pp=0.0907152 
Pp0=3.70948 
Pl=0.0248111 
Pl0=0.649027 
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

print(Ppd,Pld,Pp_cum,Pl_cum)
#%%

samples=500000
a=99
pfs=np.reshape(np.loadtxt("remote/SUCPF_lan_T100_v1_1.txt",dtype=float),(samples,int(T/d),dim))
#%%
lps=np.loadtxt("remote/SUCPF_lan_pls_T100_v1_1.txt",dtype=float)
mpf_true=np.reshape(np.loadtxt("Observations&data/TruthLanT100.txt"),(int(T/d),dim))

#%%
lps=np.array(lps,dtype=int)
print(pfs.shape)
pmax=11
N0=5
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
np.random.seed(1)
ps=np.array([4,4,4,4,4,5],dtype=int)+5
#ps=np.array([1,2,5,7,9,11],dtype=int)+1
#ls=np.array([1,2,3,4,6,7],dtype=int)+1
ls=np.array([6,6,7,7,8,8],dtype=int)+1


    
    
#%%
# IN THE FOLLOWING WE CREATE DIFFERENT PROBABILITIES DISTRIBUTIONS AND 
# CHECK THE MSE 
np.random.seed(8)
ps=np.array([2,3,3,4,5],dtype=int)+5
#ps=np.array([1,2,5,7,9,11],dtype=int)+1
#ls=np.array([1,2,3,4,6,7],dtype=int)+1
ls=np.array([5,6,7,7,8],dtype=int)+1


N0=5
D0=1
beta=1/2
cl=4
cp0=1.27
cl0=1.27
cp=1.36
batches=5
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
    M=int(10*(10**(i)))
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
plt.plot(MSEUN[:-1,a,0],costs[:-1],label="Unbiased",c="coral",ls="dashed",marker=".",ms=10)
#plt.plot(var[:,a,0],costs,label="Var")
shift1=3e1
shift2=2e3
#plt.plot(MSEUN[:,a,0],shift2*np.log(np.sqrt(MSEUN[:,a,0]*shift1))**2/(MSEUN[:,a,0]*shift1),label=r"$\mathcal{O}(\log(\varepsilon)^2\varepsilon^{-2})$")
[b0,b1]=coef(np.log(MSEUN[:-1,a,0]**(1/2)),np.log(costs[:-1]))
print([b0,b1])
plt.plot(MSEUN[:-1,a,0],np.exp(b0)*MSEUN[:-1,a,0]**(b1/2),c="dodgerblue",label=r"$\mathcal{O}(\varepsilon^{-2.57})$")
plt.yscale("log")
plt.xscale("log")
plt.legend()
plt.xlabel(r"$\varepsilon^2$")
plt.ylabel("Cost")  
plt.title("Langevin process. T=100.")
plt.savefig("Images/Unbiased_MSEvsCost_lan_T=100.pdf")
plt.show()

#%%
emes=np.array([10**i for i in range(1,5)])
emes=np.concatenate((emes,[20000]))
ps=100
print(emes)
lmax=10
pmax=11
#print(lmax)
alphal=np.sqrt(cl0*U(lmax,beta,D0)/(cl*Q(lmax)))
alphap=np.sqrt(cp0*Tf(pmax)/(cp*Q(pmax)))
rps=np.array(range(pmax))+1
Pp=(alphap+np.sum(np.log2(rps+2)**2*(rps+1)/2**(rps)))**(-1)*N0
Pp0=Pp*alphap
rls=np.array(range(lmax))+1
Pl=(alphal*D0**beta+np.sum(np.log2(rls+2)**2*(rls+1)*D0**beta/2**(rls*beta)))**(-1)
Pl0=Pl*alphal
Pld=np.concatenate(([Pl0*D0**beta],Pl*np.log2(rls+2)**2*(rls+1)*D0**beta/2**(beta*rls)))
Ppd=np.concatenate(([Pp0/N0],Pp*np.log2(rps+2)**2*(rps+1)/(2**(rps)*N0)))
mse=np.zeros((len(emes),int(T/d),dim))
var=np.zeros((len(emes),int(T/d),dim))
costs=np.zeros(len(emes))
for i in range(len(emes)):
    m=emes[i]
    plot_samples=np.random.choice(2000000,size=ps*m,replace=False)
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
[b0,b1]=coef(np.log(mse[:k,a,0]**(1/2)),np.log(costs[:k]))
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

# VERSION 2 MLPF
#2,T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100

# IN THE FOLLOWING WE CHOOSE A REALIZATION OF THE LANGEVIN DYNAMICS FOR TIME
# T=100 IN ORDER
# TO APPLY THE PF MACHINERY TO IT
T=100
dim=1
dim_o=dim
xin=np.zeros(dim)+1
l=10
collection_input=[]
I=identity(dim).toarray()
df=10
sca=(1e-3)/2
pars=[sca*1e1,df]
fi=np.array([[1]])*np.sqrt(sca)
print("sca and fi",sca,fi)
collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
cov=I*1e3
Lamb_par=2/10
print("Lamb_par",Lamb_par)
g_pars=[dim,cov]
g_par=cov
x_true=gen_gen_data(T,xin,l,collection_input)
[obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
print(obs_time,obs,len(obs_time))
times=2**(-l)*np.array(range(int(T*2**l+1)))
plt.plot(times,x_true,label="True signal")
plt.plot(obs_time,obs,label="Observations")
plt.legend()
d=1
N=40
resamp_coef=0.8
# %%
#"""
l=6
np.random.seed(2)
N=100
d=25
[log_weights,x_pf]=pff.Cox_PF(T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,\
l,d,N,dim,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par)
weights=norm_logweights(log_weights,ax=1)
x_pfmean=np.sum(np.reshape(weights,(int(T/d),N,1))*x_pf,axis=1)
#print(x_pfmean)
plt.plot((np.array(range(int(T/d)))+1)*d,x_pfmean[:,0],label="PF_Lan")
plt.plot(times,x_true,label="True signal")
#plt.plot(obs_time,obs,label="Observations")
plt.legend()
#"""
#%%
#"""
samples=100
l0=1
L=9
eLes=np.array(range(l0,L+1))
N=100
a=99
aa=-1
d=25
x_pf=np.reshape(np.loadtxt("Observations&data/CCPF_lan100_scaled1_v62.txt",dtype=float),(2,len(eLes),samples,int(T/d),dim))
#x_pf_1=np.reshape(np.loadtxt("Observations&data/CCPF_lan100_scaled1_v37_4.txt",dtype=float),(2,1,samples,int(T/d),dim))
#x_pf=np.concatenate((x_pf,x_pf_1),axis=1)
#eLes=np.concatenate((eLes,[10]))
#"""

#%%
#PARAMETERS 
#C=1
#C0=0.05
#K=0.021
# BIAS
#"""
a=-1
rbias=np.abs(np.mean(x_pf[0]-x_pf[1],axis=1)[:,a,0])
#rbias21=np.abs(np.mean(x_pf1[0]-x_pf1[1],axis=1)[:,a,0])
#rbias22=np.abs(np.mean(x_pf2[0]-x_pf2[1],axis=1)[:,a,0])
#rbias2=np.concatenate((rbias21,rbias22),axis=0)
plt.plot(eLes,rbias,label="Rbias")
#eles=np.concatenate((eles1,eles2))
#plt.plot(eles,rbias2,label="Rbias2")
reference=np.array([1/2**(i) for i in range(len(eLes))])*rbias[0]
#var=np.var(x_pf[1,1:]-x_pf[1,:-1],axis=1)[:,a,0]
#print(var,np.sqrt(var)*1.96/np.sqrt(samples),np.sqrt(samples),np.sqrt(var))
#rbias_ub=rbias+np.sqrt(var)*1.96/np.sqrt(samples)
#rbias_lb=rbias-np.sqrt(var)*1.96/np.sqrt(samples)
#plt.plot(eles[1:],rbias_ub)
#plt.plot(eles[1:],rbias_lb)
plt.plot(eLes,reference,label=r"$\Delta_l$")
#print(eles)
[b0,b1]=coef(eLes[:-2],np.log2(rbias[:-2]))
print([b0,b1,2**(2*b0)])
#"""

# VARIANCE
#"""
print("THE TIME EVALUATED IS T=",a)
sm=np.mean((x_pf[0]-x_pf[1])**2,axis=1)[:,a,0]
sm_mean=np.mean(np.mean((x_pf[0]-x_pf[1])**2,axis=1),axis=1)[:,0]
print(sm)
var=np.var((x_pf[0]-x_pf[1]),axis=1)[:,a,0]
#print((x_pf[0]-x_pf[1])[:2,:,a,0])
reference=np.array([1/2**(i/2) for i in range(len(eLes))])*sm[-1]*2**4
plt.plot(eLes,sm,label="Second moment the coupling")
#plt.plot(eLes,sm_mean,label="mean of the Second moment the coupling")

#plt.plot(eLes,var,label="variance of the coupling")
plt.plot(eLes,reference,label=r"$\Delta_l^{1/2}$")
var_sm=np.var(((x_pf[0]-x_pf[1])**2),axis=1)[:,a,0]
#print(np.sqrt(var_sm))
#samples=np.where(np.arange(1,13,1)<11,1000,100)
#print(samples)
sm_ub=sm+np.sqrt(var_sm)*1.96/np.sqrt(samples)
sm_lb=sm-np.sqrt(var_sm)*1.96/np.sqrt(samples)
plt.plot(eLes,sm_ub,label="Upper error bound")
plt.plot(eLes,sm_lb,label="Lower error bound")
#var=np.var((x_pf[0]-x_pf[1])[:,:,a],axis=1)
#var0=np.var((x_pf[0])[:,:,a],axis=1)
print(eLes)
[b0,b1]=coef(eLes,np.log2(sm))
print(b0,b1,(2**b0)*N)
#C=8.91
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
p=8
l=5
N0=10
eNes=np.array([2**i for i in range(p+1)])*N0
aa=0
d=25
x_pf=np.reshape(np.loadtxt("Observations&data/CCPF_lan100_scaled1_enes_v62.txt",dtype=float),(len(eNes),samples,int(T/d),dim))
#"""
#%%
a=-1
variances=np.var(x_pf,axis=1)[:,a,0]
plt.plot(eNes,variances,label="variance")
plt.plot(eNes,eNes[0]*variances[0]/eNes,label="ref")


#"""
plt.xlabel("N")
plt.legend()
plt.yscale("log")
[b0,b1]=coef(np.log(1/eNes),np.log(variances))
print(b0,b1,np.exp(b0))
#C0=0.5
plt.xscale("log")
plt.title("Variance in terms of the number of particles")
#plt.savefig("Images/Rbias_CCPF_gbm20_p_v5.pdf")
plt.show()
#%%

# CCONSTANTS AND PPARAMETERS FOR THE SINGLE COX PARTICLE FILTER
C0=0.025
C=0.0003243
print(C0/C)
K=5.67e-6
l0=0
Lmin=l0
Lmax=7
es=1e-4
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
print(C0/4.51499118e+05)
C0=0.025
C=0.0003243
print(C0/C)
K=5.67e-6
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
        scale=1e-1
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
x_pf1_sin=np.reshape(np.loadtxt("remote/PPF_cox_lan_t100_scaled1_v1.txt"\
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
x_pf=np.reshape(np.loadtxt("remote/PMLPF_cox_lan_T100_scaled1_v1_2.txt"\
,dtype=float),(len(eLes),samples,int(T/d),dim))
#x_pf2=np.reshape(np.loadtxt("Observations&data/PMLPF_cox_nldt_T100_scaled2_v2.txt"\
#,dtype=float),(1,samples,int(T/d),dim))
#x_pf3=np.reshape(np.loadtxt("remote/PMLPF_cox_nldt_T100_scaled2_v3.txt"\
#,dtype=float),(1,samples,int(T/d),dim))

#x_pf=np.concatenate((x_pf,x_pf2,x_pf3),axis=0)
#eLes=np.concatenate((eLes,np.array([4,5])),axis=0)
#%%

# BIAS
#"""
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
#"""

# VARIANCE
"""
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
"""
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
mpf_true=np.reshape(np.loadtxt("remote/PPF_cox_lan_Truth_T100_scaled3_v1_2.txt"),(int(T/d),dim))
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
            scale=1e-1
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
plt.plot(MSE_sin,1.6e-4*MSE_sin**(-3/2),label=r"$\varepsilon^{-3}$",color="deepskyblue")
plt.plot(MSE,1.4e-3*MSE**(-2.5/2),label=r"$\varepsilon^{-2.5}$",color="salmon")
plt.xlabel(r"$\varepsilon^2$")
plt.title("Langevin process. T=100.")
plt.ylabel("Cost")
plt.yscale("log")
plt.xscale("log")
plt.legend()
#plt.savefig("Images/MSEvsCost_lan_scaled1_T=100_lan.pdf")
plt.show()
#%%