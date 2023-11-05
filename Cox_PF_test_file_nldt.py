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
import scipy as sp


#%%
#"""
# IN THE FOLLOWING LINES WE TEST THE ASSYMPTOTIC BEHAVIOUR OF THE 
# NLDT DYNAMICS. 
np.random.seed(1)
T=10
dim=1
dim_o=dim
xin=np.zeros(dim)+0
l=3
collection_input=[]
I=identity(dim).toarray()

fi=1
A=np.array([[0.00]])
collection_input=[dim, b_ou,A,Sig_nldt,fi]
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
#%%
samples=50000

rea=np.zeros((samples,dim))

for sample in range(samples):
    xin=np.random.normal(0,1)*0.01
    rea[sample]=gen_gen_data(T,xin,l,collection_input)[-1]

#%%
#%%
pos=np.arange(-10,10,0.1)
plt.hist(rea.flatten(),bins= pos,density=True)
#x = np.linspace(t.ppf(0.001, df),t.ppf(0.999, df), 100)
xs=[-9.9,-9.8,-9.7,-9.6,-9.5,-9.4,-9.3,-9.2,-9.1,-9.,-8.9,-8.8,-8.7,-8.6,-8.5,-8.4,\
-8.3,-8.2,-8.1,-8.,-7.9,-7.8,-7.7,-7.6,-7.5,-7.4,-7.3,-7.2,-7.1,-7.,-6.9,-6.8,-6.7,-6.6\
,-6.5,-6.4,-6.3,-6.2,-6.1,-6.,-5.9,-5.8,-5.7,-5.6,-5.5,-5.4,-5.3,-5.2,-5.1,-5.,-4.9,-4.8,\
-4.7,-4.6,-4.5,-4.4,-4.3,-4.2,-4.1,-4.,-3.9,-3.8,-3.7,-3.6,-3.5,-3.4,-3.3,-3.2,-3.1,-3.,-2.9,-2.8,-2.7,-2.6,-2.5,-2.4,-2.3,-2.2,-2.1,-2.,-1.9,-1.8,-1.7,-1.6,-1.5,-1.4,-1.3,-1.2,-1.1,-1.,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4.,4.1,4.2,4.3,4.4,4.5,4.6,4.7,4.8,4.9,5.,5.1,5.2,5.3,5.4,5.5,5.6,5.7,5.8,5.9,6.,6.1,6.2,6.3,6.4,6.5,6.6,6.7,6.8,6.9,7.,7.1,7.2,7.3,7.4,7.5,7.6,7.7,7.8,7.9,8.,8.1,8.2,8.3,8.4,8.5,8.6,8.7,8.8,8.9,9.,9.1,9.2,9.3,9.4,9.5,9.6,9.7,9.8,9.9,10.]

pdf=[1.45478e-35,1.2e-34,9.53404e-34,7.39422e-33,5.59844e-32,4.13769e-31,2.98482e-30,2.10132e-29,1.44354e-28,9.67533e-28,6.32618e-27,4.03448e-26,2.50917e-25,1.52158e-24,8.9949e-24,5.1826e-23,2.90974e-22,1.59153e-21,8.47866e-21,4.3982e-20,2.22096e-19,1.09144e-18,5.21812e-18,2.42633e-17,1.09688e-16,4.81932e-16,2.05718e-15,8.52815e-15,3.43209e-14,1.34031e-13,5.07711e-13,1.86467e-12,6.63703e-12,2.28843e-11,7.64016e-11,2.46873e-10,7.71721e-10,2.33278e-9,6.81608e-9,1.92427e-8,5.24698e-8,1.38137e-7,3.51028e-7,8.60775e-7,2.03642e-6,4.64743e-6,0.0000102306,0.0000217239,0.0000445013,0.0000879616,0.000167816,0.000309154,0.000550229,0.0009467,0.00157579,0.00253957,0.00396635,0.00600916,0.00884051,0.0126429,0.0175954,0.0238571,0.0315488,0.0407356,0.0514113,0.0634886,0.0767944,0.0910728,0.105995,0.121177,0.136199,0.150628,0.164046,0.17607,0.186372,0.194695,0.20086,0.204772,0.20642,0.205866,0.203241,0.198733,0.19257,0.185012,0.176337,0.166828,0.156764,0.146416,0.136034,0.12585,0.116071,0.106879,0.0984329,0.090866,0.0842896,0.0787939,0.0744496,0.0713098,0.0694116,0.0687765,0.0694116,0.0713098,0.0744496,0.0787939,0.0842896,0.090866,0.0984329,0.106879,0.116071,0.12585,0.136034,0.146416,0.156764,0.166828,0.176337,0.185012,0.19257,0.198733,0.203241,0.205866,0.20642,0.204772,0.20086,0.194695,0.186372,0.17607,0.164046,0.150628,0.136199,0.121177,0.105995,0.0910728,0.0767944,0.0634886,0.0514113,0.0407356,0.0315488,0.0238571,0.0175954,0.0126429,0.00884051,0.00600916,0.00396635,0.00253957,0.00157579,0.0009467,0.000550229,0.000309154,0.000167816,0.0000879616,0.0000445013,0.0000217239,0.0000102306,4.64743e-6,2.03642e-6,8.60775e-7,3.51028e-7,1.38137e-7,5.24698e-8,1.92427e-8,6.81608e-9,2.33278e-9,7.71721e-10,2.46873e-10,7.64016e-11,2.28843e-11,6.63703e-12,1.86467e-12,5.07711e-13,1.34031e-13,3.43209e-14,8.52815e-15,2.05718e-15,4.81932e-16,1.09688e-16,2.42633e-17,5.21812e-18,1.09144e-18,2.22096e-19,4.3982e-20,8.47866e-21,1.59153e-21,2.90974e-22,5.1826e-23,8.9949e-24,1.52158e-24,2.50917e-25,4.03448e-26,6.32618e-27,9.67533e-28,1.44354e-28,2.10132e-29,2.98482e-30,4.13769e-31,5.59844e-32,7.39422e-33,9.53404e-34,1.2e-34,1.45478e-35,0.]    
plt.plot(xs,pdf)
plt.title("Distributions using FP equation and SDE sampling")
#/FPdistSDEsamp.pdf")
print(np.sum(pdf))
print(np.mean(rea))
#"""

#%%


# IN THE FOLLOWING WE CHOOSE A REALIZATION OF THE LANGEVIN DYNAMICS IN ORDER
# TO APPLY THE PF MACHINERY TO IT.
np.random.seed(1)
T=10
dim=1
dim_o=dim
xin=np.zeros(dim)+1
l=13
collection_input=[]
I=identity(dim).toarray()
A=np.array([[0.00]])

df=10
fi=1
collection_input=[dim, b_ou,A,Sig_nldt,fi]
cov=I*1e0
g_pars=[dim,cov]
g_par=cov
x_true=gen_gen_data(T,xin,l,collection_input)
Lamb_par=2/3
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
[log_weights,x_pf]=pff.Cox_PF(T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,\
l,d,N,dim,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par)
weights=norm_logweights(log_weights,ax=1)

x_pfmean=np.sum(np.reshape(weights,(int(T/d),N,1))*x_pf,axis=1)
#print(x_pfmean)
plt.plot((np.array(range(int(T/d)))+1)*d,x_pfmean[:,0],label="PF_nldt")
#plt.plot((np.array(range(int(T/d)))+1)*d,np.mean(pf,axis=1)[-1,:,0],label="PF_nldt_many")
plt.plot(times,x_true,label="True signal")
plt.plot(obs_time,obs,label="Observations")
plt.legend()
#"""
#%%
#"""
samples=500
l0=1
L=10
eles=np.array(range(l0,L+1))
N=5000
a=9
aa=0
d=2**(0)
x_pf=np.reshape(np.loadtxt("Observations&data/CCPF_nldt10_v1.txt",dtype=float),(2,len(eles),samples,int(T/d),dim))
#"""
#%%
#PARAMETERS 
#C=
#C0=
#K=
# BIAS
"""
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
"""
    
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
plt.title("sm in terms of the time discretization levels")
#plt.savefig("Images/sm_CCPF_nldt10_v1.pdf")
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
#T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100

# IN THE FOLLOWING WE CHOOSE A REALIZATION OF THE nldt DYNAMICS FOR TIME
# T=100 IN ORDER
# TO APPLY THE PF MACHINERY TO IT
np.random.seed(0)
T=100
dim=1
dim_o=dim
xin=np.zeros(dim)+0
l=13
collection_input=[]
I=identity(dim).toarray()

A=np.array([[0]])
fi=1
collection_input=[dim, b_ou,A,Sig_nldt,fi]
cov=I*1e0
g_pars=[dim,cov]
g_par=cov
#x_true=gen_gen_data(T,xin,l,collection_input)
Lamb_par=2/9
#[obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
print(obs_time,obs,len(obs_time))
times=2**(-l)*np.array(range(int(T*2**l+1)))
plt.plot(times,x_true,label="True signal")
plt.plot(obs_time,obs,label="Observations")
plt.legend()
d=2**(0)
N=40
resamp_coef=0.8
#%%
#np.savetxt("Observations&data/Truth_realization_nldtT100.txt",x_true,fmt="%f")
#np.savetxt("Observations&data/Observations_time_true_nldt_T100.txt",obs_time,fmt="%f")
#np.savetxt("Observations&data/Observations_true_nldt_T100.txt",obs,fmt="%f")
#%%
x_true=np.loadtxt("Observations&data/Truth_realization_nldtT100.txt",dtype=float)
obs_time=np.loadtxt("Observations&data/Observations_time_true_nldt_T100.txt",dtype=float)
obs=np.loadtxt("Observations&data/Observations_true_nldt_T100.txt",dtype=float)


#%%
#"""
l=6
[log_weights,x_pf]=pff.Cox_PF(T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,\
l,d,N,dim,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par)
weights=norm_logweights(log_weights,ax=1)
x_pfmean=np.sum(np.reshape(weights,(int(T/d),N,1))*x_pf,axis=1)
#print(x_pfmean)
plt.plot((np.array(range(int(T/d)))+1)*d,x_pfmean[:,0],label="PF_nldt")
plt.plot(times,x_true,label="True signal")
plt.plot(obs_time,obs,label="Observations")
plt.legend()
#"""
#%%
#"""
samples=500
l0=1
L=10
eles=np.array(range(l0,L+1))
N=5000
a=99

aa=0
d=2**(0)
x_pf=np.reshape(np.loadtxt("Observations&data/CCPF_nldt100_v1.txt",dtype=float),(2,len(eles),samples,int(T/d),dim))
# IN THE FOLLOWING LINE WE ALSO LOAD THE "TRUTH" TO COMPARE WITH THE SIMULATIONS.
true_f=np.reshape(np.loadtxt("Observations&data/TruthnldtT100.txt"),(int(T/d),dim))


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
bias=np.abs(np.mean(x_pf[0]-true_f,axis=1))[:,a,0]
#rbias22=np.abs(np.mean(x_pf2[0]-x_pf2[1],axis=1)[:,a,0])
#rbias2=np.concatenate((rbias21,rbias22),axis=0)
plt.plot(eles,bias,label="bias")
plt.plot(eles[1:],rbias,label="Rbias")
plt.plot(eles,rbias2,label="Rbias2")
reference=np.array([1/2**(i) for i in range(len(eles))])*bias[0]
var=np.var(x_pf[1,1:]-x_pf[1,:-1],axis=1)[:,a,0]
plt.plot(eles,reference,label="Ref")
#print(var,np.sqrt(var)*1.96/np.sqrt(samples),np.sqrt(samples),np.sqrt(var))
rbias_ub=rbias+np.sqrt(var)*1.96/np.sqrt(samples)
rbias_lb=rbias-np.sqrt(var)*1.96/np.sqrt(samples)
#plt.plot(eles[1:],rbias_ub)
#plt.plot(eles[1:],rbias_lb)
#plt.plot(eles[1:],reference,label=r"$\Delta_l$")
[b0,b1]=coef(eles,np.log2(rbias2))
print([b0,b1,2**(2*b0)])
k=0.00017
# the computation of the two lines bellow is made so we can get 
# teh parameters for the single cox pf.
[b0,b1]=coef(eles[:-3],np.log2(bias[:-3]))
print("for the bias we have the par",[b0,b1,2**(2*b0)])
#k=0.0017094479240330205

#"""
    
# VARIANCE
#"""

sm=np.mean((x_pf[0]-x_pf[1])**2,axis=1)[:,a,0]


#print((x_pf[0]-x_pf[1])[:2,:,a,0])
reference=np.array([1/2**(i/2) for i in range(len(eles))])*sm[2]
#plt.plot(eles,sm,label="Second moment the coupling")
#plt.plot(eles[2:],reference[2:],label=r"$\Delta_l^{1/2}$")
var_sm=np.var(((x_pf[0]-x_pf[1])**2),axis=1)[:,a,0]
print(np.sqrt(var_sm))
print(samples)
sm_ub=sm+np.sqrt(var_sm)*1.96/np.sqrt(samples)
sm_lb=sm-np.sqrt(var_sm)*1.96/np.sqrt(samples)
#plt.plot(eles,sm_ub,label="Upper error bound")
#plt.plot(eles,sm_lb,label="Lower error bound")
#var=np.var((x_pf[0]-x_pf[1])[:,:,a],axis=1)
#var0=np.var((x_pf[0])[:,:,a],axis=1)
print(1e-3*N)
[b0,b1]=coef(eles,np.log2(sm))
print(b0,b1,(2**b0)*N)
#C=0.45463086447
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
x_pf=np.reshape(np.loadtxt("Observations&data/CCPF_nldt100_enes_v1.txt",dtype=float),(2,len(enes),samples,int(T/d),dim))
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
#C0=0.439
plt.xscale("log")
plt.title("")
#plt.savefig("Images/Rbias_CCPF_gbm20_p_v5.pdf")
plt.show()

# FOR THE BIAS OF THE SINGLE COX PARTICLE WE HAVE THE FOLLOWING 
#C=0.3878416411744125


#%%

# IN THE FOLLOWING WE CHECK THAT THE NUMBER OF PARTICLES TO APPLY THE MLPF IS 
# FEASIBLE

C0=0.439
C=0.45463086447
K=0.00017
l0=10
Lmin=l0
Lmax=20
eLes=np.arange(Lmin,Lmax+1,1)
Cost=np.zeros(len(eLes))
C_sin=0.3878416411744125
k_sin=0.0017094479240330205
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
        eNes[1:]=NB0*np.float_power(2,-eles[1:]*3/4)*np.float_power(2,(2+1/4)*L)*scale
        Cost[i]+=3*np.sum(eNes[1:]*np.float_power(2,eles[1:]))/2+eNes[0]*np.float_power(2,eles[0])    
        print(eNes)
print("particles singles",particles_sin)
plt.plot(eLes,Cost,label="Cost")
plt.plot(eLes_sin,Cost_sin,label="Cost Single")
plt.yscale("log")
#plt.xscale("log")
plt.legend()
plt.show()

#%%

# SINGLE
# PF, PF, PF, PF, PF, PF, PF, PF, PF, PF, PF, PF, PF, PF, PF, PF, PF, PF, PF
#"""
T=100
samples=100
l0=0
Lmin=0
Lmax=6
d=1
eLes_sin=np.array(np.arange(Lmin,Lmax+1)) 
x_pf1_sin=np.reshape(np.loadtxt("remote/PPF_cox_nldt100_v1.txt"\
,dtype=float),(len(eLes_sin),samples,int(T/d),dim))
#x_pf1_sin=np.reshape(np.loadtxt("remote/PPF_cox_nldt100_v2.txt"\
#,dtype=float),(1,samples,int(T/d),dim))
#Lmin=0
#Lmax=7
#eLes=np.array(np.arange(Lmin,Lmax+1)) 
#x_pf_sin=np.concatenate((x_pf_sin,x_pf1_sin),axis=0)
#"""


#%%
#%%
#MLPF, MLPF, MLPF, MLPF, MLPF, MLPF, MLPF, MLPF, MLPF, MLPF, MLPF, MLPF, MLPF, MLPF.

#TEST FOR THE VARIANCES OF THE MLPF LEVELS

# In the following, depending of the lenght of the MLPF we get the variance 
# of the levels.
#"""
T=100
samples=100
l0=2
d=1
Lmin=l0
Lmax=5
eLes=np.array(np.arange(Lmin,Lmax+1)) 
x_pf=np.reshape(np.loadtxt("Observations&data/PMLPF_cox_nldt_T100_l03_v2.txt"\
,dtype=float),(len(eLes),samples,int(T/d),dim))
#x_pf=np.reshape(np.loadtxt("Observations&data/PMLPF_cox_nldt100_v1.txt"\
#,dtype=float),(len(eLes),samples,int(T/d),dim))
#x_pf1=np.reshape(np.loadtxt("remote/PMLPF_cox_nldt100_v2.txt"\
#,dtype=float),(1,samples,int(T/d),dim))
#Lmin=0
#Lmax=7
#eLes=np.array(np.arange(Lmin,Lmax+1)) 
#x_pf=np.concatenate((x_pf,x_pf1),axis=0)
#"""
# THIS PART IS MADE WITH A NEGATIVE BASE LEVEL
"""
T=96
samples=100
l0=-5
Lmin=l0
Lmax=2
d=2**(-l0)
eLes=np.array(np.arange(Lmin,Lmax+1)) 
x_pf=np.reshape(np.loadtxt("Observations&data/PMLPF_cox_nldt96_v1.txt"\
,dtype=float),(len(eLes),samples,int(T/d),dim))
#x_pf=np.reshape(np.loadtxt("Observations&data/PMLPF_cox_nldt100_v1.txt"\
#,dtype=float),(len(eLes),samples,int(T/d),dim))
#x_pf1=np.reshape(np.loadtxt("remote/PMLPF_cox_nldt100_v2.txt"\
#,dtype=float),(1,samples,int(T/d),dim))
#Lmin=0
#Lmax=7
#eLes=np.array(np.arange(Lmin,Lmax+1)) 
#x_pf=np.concatenate((x_pf,x_pf1),axis=0)

"""
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
# TRUE PF
#"""

C0=0.439
C=0.45463086447
K=0.00017
l0=-5
Lmin=l0
Lmax=7
es=5e-7
eLes=np.arange(Lmin,Lmax+1,1)
N=int(2*C0/es)
L=int(np.ceil(-np.log2(es/(2*np.sqrt(K)))/2,dtype=float))
print(L,N)
#%%
resamp_coef=0.8
[log_weights,x_pf]=pff.Cox_PF(T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,\
L,d,N,dim,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par)
weights=norm_logweights(log_weights,ax=1)
x_pfmean=np.sum(np.reshape(weights,(int(T/d),N,1))*x_pf,axis=1)
phi_pf=x_pfmean.flatten()
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
#"""
#%%

# IN THE FOLLOWING LINES WE COMPARE THE MSE OF THE MLPF_COX WITH ITS COST
#MSE
#"""
a=99
mpf_true=np.reshape(np.loadtxt("Observations&data/TruthnldtT100.txt"),(int(T/d),dim))
print(mpf_true.shape)
#%%
MSE=np.mean((x_pf-mpf_true)**2,axis=1)[:,a,0]
MSE_sin=np.mean((x_pf1_sin-mpf_true)**2,axis=1)[:,a,0]
#COST
l0=3
Lmin=l0
Lmax=5
eLes=np.array(np.arange(Lmin,Lmax+1,1),dtype=float)
#print(eLes)
Cost=np.zeros(len(eLes))
C0=0.439
C=0.45463086447
K=0.00017

C_sin=0.3878416411744125
k_sin=0.0017094479240330205

#Cost_sin=2**eLes*C_sin*2**(2*eLes)/k_sin
Cost_sin=np.float_power(8, eLes_sin)*C_sin/k_sin
print("eNes single",C_sin*np.float_power(4,eLes)/k_sin)
for i in range(len(eLes)):

            L=eLes[i]
            print("L is",L)
            CL=2**(1/4)*(2**((L-l0)/4)-1)/(2**(1/4)-1)
            #NB0=(np.sqrt(C0)+np.sqrt(3*C/2)*CL)*np.sqrt(2*C/3)*2**(-(L-Lmin)/4)/K
            #print("NB0 is ",NB0)
            #print("is the error",2**(L/4))
            #N0=np.sqrt(C0)*(np.sqrt(C0)+np.sqrt(3*C/2)*CL)*2**(2*(L))/K
            N0=np.sqrt(C0)*(np.sqrt(C0)+np.sqrt(3*C/2)*CL)*np.float_power(2,2*L)*np.float_power(2,-l0/2)/K
            #print("N0 is ",N0)
            eles=np.array(np.arange(l0,L+1))
            print("eles are", eles)
            eNes=np.zeros(len(eles),dtype=int)
            scale=5e-2
            eNes[0]=N0*scale
            #Cost[i]+=eNes[0]*2**eLes[0]
         
            eNes[1:]=np.sqrt(2*C/3)*(np.sqrt(C0)+np.sqrt(3*C/2)*CL)*np.float_power(2,2*L)*np.float_power(2,-l0/4)*np.float_power(2,-eles[1:]*3/4)/K*scale
            print("eNes are",eNes)
            #print(eNes[1:]*2**eles[1:])
            Cost[i]+=np.sum(3*eNes[1:]*2**eles[1:]/2)+eNes[0]*2**eles[0]
      #%%
    
#%%

print("Cost",Cost) 
print("Cost_single",Cost_sin) 
#plt.plot(MSE_un,costs[k1:],label="Unbiased")
MSE_arti=1/2**(eLes*2)
plt.plot(MSE,Cost,label="MLPF",marker="o")
#plt.plot(MSE_arti,Cost,label="MLPF",marker="o")
plt.plot(MSE_sin,Cost_sin,label="Single PF",marker="o")

#plt.plot(MSE_arti[:-1],Cost_sin[:-1],label="Single PF",marker="o")
plt.plot(MSE_sin,1.3e-2*MSE_sin**(-3/2),label=r"$\varepsilon^{-3}$")
plt.plot(MSE,1.2e-1*MSE**(-2.5/2),label=r"$\varepsilon^{-2.5}$")
plt.xlabel(r"$\varepsilon^2$")
plt.title("Nonlinear diffusion term process, T=100.")
plt.ylabel("Cost")
plt.yscale("log")
plt.xscale("log")
plt.legend()
#plt.savefig("Images/MSEvsCost_artificial_nldt.pdf")

plt.show()


#%%
# LEVLES AND PARTICLES TO CREATE THE ERROR TO COST RATE OF THE PFCOX

es=np.array([1/4**i for i in range(-1,6)])*1e-2
samples=100
C0=0.439
C=0.45463086447
K=0.00017
enes=np.array( np.ceil(2*C0/es),dtype=int)
a=99
eles=np.array(np.ceil(-np.log2(es/(2*np.sqrt(K)))/2,dtype=float),dtype=int)
print(eles,enes) 
pf1=np.reshape(np.loadtxt("Observations&data/PCox_PF_nldt_etc_rate_v5.txt"),(len(eles[:1]),samples,int(T/d),dim))

pf2=np.reshape(np.loadtxt("Observations&data/PCox_PF_nldt_etc_rate_v4.txt"),(len(eles[1:]),samples,int(T/d),dim))
#pf2=np.reshape(np.loadtxt("Observations&data/PCox_PF_etc_rate_v2.txt"),(1,samples,int(T/d),dim))
pf=np.concatenate((pf1,pf2),axis=0) 

mpf_true=np.reshape(np.loadtxt("Observations&data/TruthnldtT100.txt"),(int(T/d),dim))

MSE=np.mean((x_pf-mpf_true)**2,axis=1)[:,a,0]

costs=enes*2**eles
mse=np.mean((pf-mpf_true)**2,axis=1)[:,a,0]
var=np.var(pf,axis=1)[:,a,0]*5e-2

mean=np.mean((pf-mpf_true),axis=1)[:,a,0]
[b0,b1]=coef(np.log(np.sqrt(mse)),np.log(costs))

print([b0,b1,np.exp(b0)])
plt.plot(mse,costs,label="PF",ls="dashed",marker=".",c="red")
#plt.plot(mse,np.exp(b0+b1*np.log(np.sqrt(mse))),label=r"$\mathcal{O}(\varepsilon^{-3.09})$")
shift4=3e-2
plt.plot(mse,shift4*mse**(-3/2),label=r"$\mathcal{O}(\varepsilon^{-3})$")

plt.plot(MSE,Cost,label="MLPF",ls="dashed",marker="p",c="blue")
[b0,b1]=coef(-np.log(np.sqrt(MSE)),np.log(Cost))
print([b0,b1])
#plt.plot(MSE,np.exp(b0-b1*np.log(MSE)),label="refl")
shift1=4e0
shift2=1e0
shift3=3.5e-1
#plt.plot(MSE,shift2*np.log(np.sqrt(MSE*shift1))**2/(MSE*shift1),label=r"$\mathcal{O}(log(\varepsilon)^2\varepsilon^{-2})$")
plt.plot(MSE,shift3*MSE**(-2.5/2),label=r"$\mathcal{O}(\varepsilon^{-2.5})$",c="green")
plt.xlabel(r"$\varepsilon^2$")
plt.title("NLDT process, T=100.")
plt.ylabel("Cost")
plt.yscale("log")
plt.xscale("log")
plt.legend()
#plt.savefig("Images/MSEvsCost_PF_MLPF_nldt_T=100.pdf")
plt.show()#"""


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
#T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100
# UNBIASED, UNBIASED, # UNBIASED, UNBIASED, # UNBIASED, UNBIASED, # UNBIASED, UNBIASED.
# IN THE FOLLOWING WE CHOOSE A REALIZATION OF THE OU DYNAMICS FOR TIME
# T=100 IN ORDER
# TO APPLY THE PF MACHINERY TO IT
np.random.seed(0)
T=100
dim=1
dim_o=dim
xin=np.zeros(dim)+0
l=13
collection_input=[]
I=identity(dim).toarray()
A=np.array([[0]])
fi=1
collection_input=[dim,b_ou,A,Sig_nldt,fi]
cov=I*1e0
g_pars=[dim,cov]
g_par=cov
Lamb_par=2/9
times=2**(-l)*np.array(range(int(T*2**l+1)))
"""
x_true=gen_gen_data(T,xin,l,collection_input)
[obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
print(obs_time,obs,len(obs_time))


plt.plot(times,x_true,label="True signal")
plt.plot(obs_time,obs,label="Observations")
plt.legend()
"""
d=2**(0)
N=40
resamp_coef=0.8
#%%
#np.savetxt("Observations&data/Truth_realization_nldt_original_T100.txt",x_true,fmt="%f")
#np.savetxt("Observations&data/Observations_time_true_nldt_original_T100.txt",obs_time,fmt="%f")
#np.savetxt("Observations&data/Observations_true_nldt_original_T100.txt",obs,fmt="%f")
#%%
x_true=np.reshape(np.loadtxt("Observations&data/Truth_realization_nldt_original_T100.txt",dtype=float),(-1,dim))
obs_time=np.loadtxt("Observations&data/Observations_time_true_nldt_original_T100.txt",dtype=float)
obs=np.reshape(np.loadtxt("Observations&data/Observations_true_nldt_original_T100.txt",dtype=float),(-1,dim))


#%%
#"""
l=5
[log_weights,x_pf]=pff.Cox_PF(T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,\
l,d,N,dim,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par)
weights=norm_logweights(log_weights,ax=1)
x_pfmean=np.sum(np.reshape(weights,(int(T/d),N,1))*x_pf,axis=1)
#print(x_pfmean)
plt.plot(times,x_true,label="True signal")
plt.plot((np.array(range(int(T/d)))+1)*d,x_pfmean[:,0],label="PF_nldt")
plt.plot(obs_time,obs,label="Observations")
plt.legend()
#"""
#%%
#cl=1.5
#cp0=0.5 we got it from CCPF_nldt100_enes_v1
#cl0=0.5
#dp=0.21
#cp=1
#delta_l=0.02
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


ps=np.array([1,2,5,7,9,11],dtype=int)+1
#ps=np.array([1,2,5,7,9,11],dtype=int)+1
#ls=np.array([1,2,3,4,6,7],dtype=int)+1
ls=np.array([1,2,3,4,6,7],dtype=int)+1
N0=10
D0=1
beta=0.5
cl=1.5
cp0=0.5
cl0=0.5
dp=0.21
cp=1
costs=np.zeros(len(ps))
var=np.zeros(len(ls))
for i in range(len(ps)):
    lmax=ls[i]
    pmax=ps[i]
    print(lmax)
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
    dPl=dPl/np.sum(dPl)
    dPp=dPp/np.sum(dPp)
    #print(dPl)
    #print(dPp)
    ml,mp=np.meshgrid(rlsc,rpsc)
    mean_cost=np.sum(dPl[ml]*dPp[mp]*2**(mp+ml))
    #print(mean_cost)
    var[i]=np.sum(1/(dPl[ml]*dPp[mp]*2**(mp+beta*ml)))/8**i
    #print(var[i])
    costs[i]=mean_cost*8**i
    M=(pmax/(N0*2**pmax))**(-2)
    print(M)
    

    
MSE=1e-1/8**np.array(range(len(ps)))
plt.plot(var[2:],costs[2:])
plt.yscale("log")
plt.xscale("log")
plt.show()
[b0,b1]=coef(np.log(var**(1/2))[2:],np.log(costs)[2:])
print([b0,b1])
    
#%%

# COMPUTATION OF CP AND CP0 AND DELTA_P


# IN THIS CELL WE COMPUTE THE REFERENCE TRUTH IN ORDER TO COMPUTE THE BIAS
# OF THE UNBIASED ESTIMATOR
#"""
l=4
N0=4
N=int(N0*2**8*10000)
print(N)
[lw,x]=pff.Cox_PF(T, xin, b_ou, A, Sig_nldt, fi, obs, obs_time, l, d,N\
, dim, resamp_coef, g_den, g_par, Norm_Lambda, Lamb_par)

weights=norm_logweights(lw,ax=1)           
phi_pfmean=np.sum(np.reshape(weights,(int(T/d),N,1))*phi(x,ax=2),axis=1)
#np.savetxt("Observations&data/SUCPF_nldt_N10240000_T100_v1.txt",phi_pfmean)
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
pfs=np.reshape(np.loadtxt("Observations&data/SUCPF_nldt_p_levels_T10o_v2.txt",dtype=float),(2,len(eNes)-1,samples,int(T/d),dim))
tel_est_exact=np.reshape(np.loadtxt("Observations&data/SUCPF_nldt_N10240000_T100_v1.txt",dtype=float),(int(T/d),dim))
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
plt.title("Bias in terms of the particle levels of the OU for T=100")
plt.show()  

#"""
#VARIANCE
#"""
sm=np.mean((pfs[0]-pfs[1])**2,axis=1)[:,a,0]
var=np.var((pfs[0]-pfs[1]),axis=1)[:,a,0]
print(sm)
ref=np.array([1/2**i for i in range(len(eNes)-1)])*sm[0]
[b0,b1]=coef(np.log(1/eNes[1:]),np.log(sm))
print(b0,b1,np.exp(b0))
#cp*cl0=0.85

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

#cl=0.003*250
#delta_l=0.02
print(0.003*250)
#%%
samples=1000
Lmax=8
l0=0
p=1
N0=250
eLes=np.arange(l0,Lmax+1)
ps=np.arange(0,p+1)
eNes=N0*2**ps
#nbot really scaled
pfs=np.reshape(np.loadtxt("Observations&data/SUCPF_nldt_l_levels_T100_v2.txt",dtype=float),(2,len(eLes),samples,int(T/d),dim))
samples_truth=2000
tel_est_exact=np.mean(np.reshape(np.loadtxt("Observations&data/SUCPF_nldt_l14_T100_v1.txt"\
                                            ,dtype=float),(samples_truth,int(T/d),dim)),axis=0)
#%%
a=99
#BIAs
"""
bias=np.abs(np.mean(pfs[0]-tel_est_exact,axis=1)[:,a,0])   
varss=np.abs(np.var(pfs[0]-tel_est_exact,axis=1)[:,a,0])   
bias_ub=bias+np.sqrt(varss)*1.96/np.sqrt(samples)
bias_lb=bias-np.sqrt(varss)*1.96/np.sqrt(samples)
plt.plot(eLes,bias_ub)
plt.plot(eLes,bias_lb)
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
var_x=np.var((pfs[0]-pfs[1]),axis=1)[:,a,0]
var=np.var(((pfs[0]-pfs[1])**2),axis=1)[:,a,0]
sm_ub=sm+np.sqrt(var)*1.96/np.sqrt(samples)
sm_lb=sm-np.sqrt(var)*1.96/np.sqrt(samples)
plt.plot(eLes,sm_ub)
plt.plot(eLes,sm_lb)
ref=np.array([1/2**(i/2) for i in range(len(eLes))])*sm[-1]*2**((len(eLes)-1)/2)
[b0,b1]=coef(eLes+1,np.log2(sm))
print(b0,b1,2**b0)


plt.plot(eLes,var_x,label="var")
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
samples=2000000
a=99
pfs=np.reshape(np.loadtxt("/Users/alvarem/Documents/DataunbiasedPF/SUCPF_nldf_T100_v1.txt",dtype=float),(samples,int(T/d),dim))
 #%%
lps=np.loadtxt("Observations&data/SUCPF_nldf_pls_T100_v1.txt")
mpf_true=np.reshape(np.loadtxt("Observations&data/TruthnldtT100.txt"),(int(T/d),dim))

#%%
lps=np.array(lps,dtype=int)

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
Pp=0.100607 
Pp0=3.56875
Pl=0.00266382 
Pl0=0.827428 
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
emes=np.array([10**i for i in range(1,6)])
emes=np.concatenate((emes,[200000]))
ps=10
print(emes)
mse=np.zeros((len(emes),int(T/d),dim))
var=np.zeros((len(emes),int(T/d),dim))
costs=np.zeros(len(emes))
np.random.seed(27)
#27
#18
#23
for i in range(len(emes)): 
    m=emes[i]
    plot_samples=np.random.choice(2000000,size=ps*m,replace=False)
    #print(plot_samples)
    costs[i]=np.sum(2**lps[plot_samples,0]*2**lps[plot_samples,1])*N0/ps
    upf=pfs[plot_samples]/(Ppd[lps[plot_samples,1],np.newaxis,\
    np.newaxis]*Pld[lps[plot_samples,0],np.newaxis,np.newaxis])
    upf=np.reshape(upf,(ps,m,int(T/d),dim))
    upfs=np.mean(upf,axis=1)
    var[i]=np.var(upfs,axis=0)
    mse[i]=np.mean((upfs-mpf_true)**2,axis=0)
#
#%%
k=-1
k1=0
[b0,b1]=coef(np.log(mse[k1:,a,0]**(1/2)),np.log(costs[k1:]))
print([b0,b1])
print(mse[:,a,0])
a=99
MSE_un=mse[k1:,a,0]
plt.plot(MSE_un,np.exp(b0)*MSE_un**(b1/2),c="dodgerblue",label=r"$\mathcal{O}(\varepsilon^{-2.65})$")
shift1=2e1
shift2=5e4
k=5
plt.xlabel(r"$\varepsilon^2$")
plt.ylabel("Cost")  
plt.title("Nonlinear diffusion term process, T=100.")
#plt.plot(var[k1:,a,0],costs[k1:],label="var")
plt.plot(MSE_un,costs[k1:],label="Unbiased",c="coral",ls="dashed",marker=".",ms=10)
#plt.plot(MSE,shift2*np.log(np.sqrt(MSE*shift1))**2/(MSE*shift1),label=r"$\mathcal{O}(np.log(\varepsilon)^2\varepsilon^{-2})$")
plt.legend()
plt.yscale("log")
plt.xscale("log")
#plt.savefig("Images/Unbiased_MSEvsCost_nldt_T=100.pdf")


#%%
#%%
# In this section we compute the parameters and particles in terms of the levels 
# in order to get an approximation with the simgle level particle cox filter.

# the MSE of the pf can be written as MSE=kDelta_l^2+C/N
Lmin=0
Lmax=7
eLes=np.array(range(Lmin,Lmax+1))
C=0.3878416411744125
k=0.0017094479240330205
for i in range(len(eLes)):
    l=eLes[i]
    print(C*2**(2*l)/k)


#%%
#New batch 
#T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100

# IN THE FOLLOWING WE CHOOSE A REALIZATION OF THE LANGEVIN DYNAMICS FOR TIME
# T=100 IN ORDER
# TO APPLY THE PF MACHINERY TO IT
np.random.seed(0)
T=96
dim=1
dim_o=dim
xin=np.zeros(dim)+0
l=13
collection_input=[]
I=identity(dim).toarray()
A=np.array([[0]])
fi=1
collection_input=[dim, b_ou,A,Sig_nldt,fi]
cov=I*1e-1
g_pars=[dim,cov]
g_par=cov
x_true=gen_gen_data(T,xin,l,collection_input)
Lamb_par=2/9
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
l=-5
d=2**(-l)
print(l,d)
print(N)
N=2
[log_weights,x_pf]=pff.Cox_PF(T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,\
l,d,N,dim,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par)
weights=norm_logweights(log_weights,ax=1)
x_pfmean=np.sum(np.reshape(weights,(int(T/d),N,1))*x_pf,axis=1)
#print(x_pfmean)
plt.plot(times,x_true,label="True signal")
plt.plot((np.array(range(int(T/d)))+1)*d,x_pfmean[:,0],label="PF_nldt")
print(x_pfmean)
plt.plot(obs_time,obs,label="Observations")
plt.legend()

#%%
#%%
# THE FOLLOWING ITERATION IS MADE IN ORDER TO FIND PARAMETERS SUCH THAT AN APPRECIABLE 
# IMPROVEMENT IS FOUND USING THE MLPF
# VERSION 2
#T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100

# IN THE FOLLOWING WE CHOOSE A REALIZATION OF THE LANGEVIN DYNAMICS FOR TIME
# T=100 IN ORDER
# TO APPLY THE PF MACHINERY TO IT
np.random.seed(0)
T=100
dim=1
dim_o=dim
xin=np.zeros(dim)+4
l=10
collection_input=[]
I=identity(dim).toarray()

A=np.array([[0]])
sc=np.float_power(2,10)
fi=1/np.sqrt(sc)
print(fi)
collection_input=[dim, b_ou,A,Sig_nldt,fi]
cov=I*1e-1
g_pars=[dim,cov]
g_par=cov
x_true=gen_gen_data(T,xin,l,collection_input)
Lamb_par=2/9
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
#%%
#np.savetxt("Observations&data/Truth_realization_nldt_scaled_T100.txt",x_true,fmt="%f")
#np.savetxt("Observations&data/Observations_time_true_nldt_scaled_T100.txt",obs_time,fmt="%f")
#np.savetxt("Observations&data/Observations_true_nldt_scaled_T100.txt",obs,fmt="%f")
#%%
x_true=np.loadtxt("Observations&data/Truth_realization_nldt_scaled_T100.txt",dtype=float)
obs_time=np.loadtxt("Observations&data/Observations_time_true_nldt_scaled_T100.txt",dtype=float)
obs=np.loadtxt("Observations&data/Observations_true_nldt_scaled_T100.txt",dtype=float)


#%%
#"""
l=6
[log_weights,x_pf]=pff.Cox_PF(T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,\
l,d,N,dim,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par)
weights=norm_logweights(log_weights,ax=1)
x_pfmean=np.sum(np.reshape(weights,(int(T/d),N,1))*x_pf,axis=1)
#print(x_pfmean)
plt.plot((np.array(range(int(T/d)))+1)*d,x_pfmean[:,0],label="PF_nldt")
plt.plot(times,x_true,label="True signal")
plt.plot(obs_time,obs,label="Observations")
plt.legend()
#"""
#%%
#"""
samples=100
l0=1
L=9
eles=np.array(range(l0,L+1))
N=1000
a=99
aa=0
d=2**(0)
x_pf=np.reshape(np.loadtxt("Observations&data/CCPF_nldt100_scaled_v3.txt",dtype=float),(2,len(eles),samples,int(T/d),dim))
# IN THE FOLLOWING LINE WE ALSO LOAD THE "TRUTH" TO COMPARE WITH THE SIMULATIONS.
#true_f=np.reshape(np.loadtxt("Observations&data/TruthnldtT100.txt"),(int(T/d),dim))


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
#bias=np.abs(np.mean(x_pf[0]-true_f,axis=1))[:,a,0]
#rbias22=np.abs(np.mean(x_pf2[0]-x_pf2[1],axis=1)[:,a,0])
#rbias2=np.concatenate((rbias21,rbias22),axis=0)
#plt.plot(eles,bias,label="bias")
#plt.plot(eles[1:],rbias,label="Rbias")
plt.plot(eles,rbias2,label="Rbias2")
reference=np.array([1/2**(i) for i in range(len(eles))])*rbias2[0]
var=np.var(x_pf[1,1:]-x_pf[1,:-1],axis=1)[:,a,0]
plt.plot(eles,reference,label="Ref")
#print(var,np.sqrt(var)*1.96/np.sqrt(samples),np.sqrt(samples),np.sqrt(var))
rbias_ub=rbias+np.sqrt(var)*1.96/np.sqrt(samples)
rbias_lb=rbias-np.sqrt(var)*1.96/np.sqrt(samples)
#plt.plot(eles[1:],rbias_ub)
#plt.plot(eles[1:],rbias_lb)
#plt.plot(eles[1:],reference,label=r"$\Delta_l$")
[b0,b1]=coef(eles,np.log2(rbias2))
print([b0,b1,2**(2*b0)])
plt.title("bias moment in terms of the time discretization levels")
# the computation of the two lines bellow is made so we can get 
# teh parameters for the single cox pf.
#[b0,b1]=coef(eles[:-3],np.log2(bias[:-3]))
#print("for the bias we have the par",[b0,b1,2**(2*b0)])
#K=8.868769502982106e-08

#"""
    
# VARIANCE
"""
a=-1
sm=np.mean((x_pf[0]-x_pf[1])**2,axis=1)[:,a,0]

plt.plot(eles,sm,label="Second moment the coupling")

#print((x_pf[0]-x_pf[1])[:2,:,a,0])
reference=np.array([1/2**(i/2) for i in range(len(eles))])*sm[0]
plt.plot(eles,reference,label=r"$\Delta_l^{1/2}$")
var_sm=np.var(((x_pf[0]-x_pf[1])**2),axis=1)[:,a,0]
print(np.sqrt(var_sm))
print(samples)
sm_ub=sm+np.sqrt(var_sm)*1.96/np.sqrt(samples)
sm_lb=sm-np.sqrt(var_sm)*1.96/np.sqrt(samples)
#plt.plot(eles,sm_ub,label="Upper error bound")
#plt.plot(eles,sm_lb,label="Lower error bound")
#var=np.var((x_pf[0]-x_pf[1])[:,:,a],axis=1)
#var0=np.var((x_pf[0])[:,:,a],axis=1)
print(1e-3*N)
[b0,b1]=coef(eles,np.log2(sm))
print(b0,b1,(2**b0)*N)
#C=6.700571352501338e-05
plt.title("Second moment in terms of the time discretization levels")
"""
#"""
plt.xlabel("l")
plt.legend()
plt.yscale("log")

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
x_pf=np.reshape(np.loadtxt("Observations&data/CCPF_nldt100_enes_scaled_v3.txt",dtype=float),(2,len(enes),samples,int(T/d),dim))
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
#C0=0.439
plt.xscale("log")
plt.title("ds")
#plt.savefig("Images/Rbias_CCPF_gbm20_p_v5.pdf")
plt.show()
# FOR THE BIAS OF THE SINGLE COX PARTICLE WE HAVE THE FOLLOWING 
#C0=0.00082

#%%
C0=0.00082
C=6.700571352501338e-05
K=8.868769502982106e-08
l0=0
Lmin=l0
Lmax=7
es=1e-10
eLes=np.arange(Lmin,Lmax+1,1)
N=int(2*C0/es)
L=int(np.ceil(-np.log2(es/(2*np.sqrt(K)))/2,dtype=float))
print(L,N)
#%%
resamp_coef=0.8
[log_weights,x_pf]=pff.Cox_PF(T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,\
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
C0=0.00082
C=6.700571352501338e-05
K=8.868769502982106e-08
l0=0
Lmin=l0
Lmax=8
eLes=np.arange(Lmin,Lmax+1,1)
Cost=np.zeros(len(eLes))
C_sin=C
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
print("particles singles",particles_sin)
plt.plot(eLes,Cost,label="Cost")
plt.plot(eLes_sin,Cost_sin,label="Cost Single")
plt.yscale("log")
#plt.xscale("log")
plt.legend()
plt.show()
# %%

#%%
# THE FOLLOWING ITERATION IS MADE IN ORDER TO FIND PARAMETERS SUCH THAT AN APPRECIABLE 
# IMPROVEMENT IS FOUND USING THE MLPF
# VERSION 3
#T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100

# IN THE FOLLOWING WE CHOOSE A REALIZATION OF THE LANGEVIN DYNAMICS FOR TIME
# T=100 IN ORDER
# TO APPLY THE PF MACHINERY TO IT

# ORIGINAL SYSTEM
np.random.seed(0)
T=100
dim=1
dim_o=dim
xin=np.zeros(dim)+4
l=10
collection_input=[]
I=identity(dim).toarray()
A=np.array([[0]])
sc=np.float_power(2,10)
fi=1/np.sqrt(sc)
print(fi)
collection_input=[dim, b_ou,A,Sig_nldt,fi]
cov=I*1e-1
g_pars=[dim,cov]
g_par=cov
x_true=gen_gen_data(T,xin,l,collection_input)
Lamb_par=2/9
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
#%%
#np.savetxt("Observations&data/Truth_realization_nldt_scaled_T100.txt",x_true,fmt="%f")
#np.savetxt("Observations&data/Observations_time_true_nldt_scaled_T100.txt",obs_time,fmt="%f")
#np.savetxt("Observations&data/Observations_true_nldt_scaled_T100.txt",obs,fmt="%f")
#%%
x_true=np.loadtxt("Observations&data/Truth_realization_nldt_scaled_T100.txt",dtype=float)
obs_time=np.loadtxt("Observations&data/Observations_time_true_nldt_scaled_T100.txt",dtype=float)
obs=np.loadtxt("Observations&data/Observations_true_nldt_scaled_T100.txt",dtype=float)


#%%
#"""
l=6
[log_weights,x_pf]=pff.Cox_PF(T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,\
l,d,N,dim,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par)
weights=norm_logweights(log_weights,ax=1)
x_pfmean=np.sum(np.reshape(weights,(int(T/d),N,1))*x_pf,axis=1)
#print(x_pfmean)
plt.plot((np.array(range(int(T/d)))+1)*d,x_pfmean[:,0],label="PF_nldt")
plt.plot(times,x_true,label="True signal")
plt.plot(obs_time,obs,label="Observations")
plt.legend()
#"""
#%%
#"""
samples=100
l0=1
L=9
eles=np.array(range(l0,L+1))
N=1000
a=-1
aa=0
d=2**(0)
x_pf=np.reshape(np.loadtxt("Observations&data/CCPF_nldt100_scaled2_v8.txt",dtype=float),(2,len(eles),samples,int(T/d),dim))
# IN THE FOLLOWING LINE WE ALSO LOAD THE "TRUTH" TO COMPARE WITH THE SIMULATIONS.
#true_f=np.reshape(np.loadtxt("Observations&data/TruthnldtT100.txt"),(int(T/d),dim))
#"""
#%%
#PARAMETERS 
#C=
#C0=
#K=
# BIAS
"""
a=-1
rbias=np.abs(np.mean(x_pf[1,1:]-x_pf[1,:-1],axis=1)[:,a,0])
rbias2=np.abs(np.mean(x_pf[0]-x_pf[1],axis=1)[:,a,0])
#bias=np.abs(np.mean(x_pf[0]-true_f,axis=1))[:,a,0]
#rbias22=np.abs(np.mean(x_pf2[0]-x_pf2[1],axis=1)[:,a,0])
#rbias2=np.concatenate((rbias21,rbias22),axis=0)
#plt.plot(eles,bias,label="bias")
#plt.plot(eles[1:],rbias,label="Rbias")
plt.plot(eles,rbias2,label="Rbias2")
reference=np.array([1/2**(i) for i in range(len(eles))])*rbias2[0]
var=np.var(x_pf[1,1:]-x_pf[1,:-1],axis=1)[:,a,0]
plt.plot(eles,reference,label="Ref")
#print(var,np.sqrt(var)*1.96/np.sqrt(samples),np.sqrt(samples),np.sqrt(var))
rbias_ub=rbias+np.sqrt(var)*1.96/np.sqrt(samples)
rbias_lb=rbias-np.sqrt(var)*1.96/np.sqrt(samples)
#plt.plot(eles[1:],rbias_ub)
#plt.plot(eles[1:],rbias_lb)
#plt.plot(eles[1:],reference,label=r"$\Delta_l$")
[b0,b1]=coef(eles[:7],np.log2(rbias2[:7]))
print([b0,b1,2**(2*b0)])
plt.title("bias moment in terms of the time discretization levels")
# the computation of the two lines bellow is made so we can get 
# teh parameters for the single cox pf.
#[b0,b1]=coef(eles[:-3],np.log2(bias[:-3]))
#print("for the bias we have the par",[b0,b1,2**(2*b0)])
#K= 1.9131764860557113e-06

"""
# VARIANCE

#"""
a=-1
sm=np.mean((x_pf[0]-x_pf[1])**2,axis=1)[:,a,0]
plt.plot(eles,sm,label="Second moment the coupling")

#print((x_pf[0]-x_pf[1])[:2,:,a,0])
reference=np.array([1/2**(i/2) for i in range(len(eles))])*sm[0]
plt.plot(eles,reference,label=r"$\Delta_l^{1/2}$")
var_sm=np.var(((x_pf[0]-x_pf[1])**2),axis=1)[:,a,0]
sm_ub=sm+np.sqrt(var_sm)*1.96/np.sqrt(samples)
sm_lb=sm-np.sqrt(var_sm)*1.96/np.sqrt(samples)
#plt.plot(eles,sm_ub,label="Upper error bound")
#plt.plot(eles,sm_lb,label="Lower error bound")
#var=np.var((x_pf[0]-x_pf[1])[:,:,a],axis=1)
#var0=np.var((x_pf[0])[:,:,a],axis=1)
[b0,b1]=coef(eles,np.log2(sm))
print(b0,b1,(2**b0)*N)
#C=0.005161
plt.title("Second moment in terms of the time discretization levels")
#"""
#"""
plt.xlabel("l")
plt.legend()
plt.yscale("log")
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
x_pf=np.reshape(np.loadtxt("Observations&data/CCPF_nldt100_enes_scaled2_v8.txt",dtype=float),(2,len(enes),samples,int(T/d),dim))
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
#C0=5.69882641701004e-05
plt.xscale("log")
plt.title("ds")
#plt.savefig("Images/Rbias_CCPF_gbm20_p_v5.pdf")
plt.show()
# FOR THE BIAS OF THE SINGLE COX PARTICLE WE HAVE THE FOLLOWING 
#C0=

#%%

# CCONSTANTS AND PPARAMETERS FOR THE SINGLE COX PARTICLE FILTER
C0=0.00997
C=0.0004998
K= 1.6e-07
l0=0
Lmin=l0
Lmax=7
es=5e-10
eLes=np.arange(Lmin,Lmax+1,1)
N=int(2*C0/es)
L=int(np.ceil(-np.log2(es/(2*np.sqrt(K)))/2,dtype=float))
print(L,N)
#%%
resamp_coef=0.8
[log_weights,x_pf]=pff.Cox_PF(T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,\
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
C0=0.00997
C=0.0004998
K= 1.6e-07
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
        scale=5e-2
        eNes[0]=N0*scale
        #Cost[i]+=eNes[0]*2**eLes[0]
        eNes[0]=np.maximum(2,eNes[0])
        eNes[1:]=np.sqrt(2*C/3)*(np.sqrt(C0/delta_l0)+np.sqrt(3*C/2)*CL*delta_l0**(-1/4))\
        *np.float_power(2,2*L)*np.float_power(2,-eles[1:]*3/4)/K*scale
        #eNes[1:]=NB0*np.float_power(2,-eles[1:]*3/4)*np.float_power(2,(2+1/4)*L)*scale
        Cost[i]+=3*np.sum(eNes[1:]*np.float_power(2,eles[1:]))/2+eNes[0]*np.float_power(2,eles[0])  
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
#%%

# SSINGLE
# PF, PF, PF, PF, PF, PF, PF, PF, PF, PF, PF, PF, PF, PF, PF, PF, PF, PF, PF
#"""
T=100
samples=100
l0=0
Lmin=0
Lmax=3
d=1
eLes_sin=np.array(np.arange(Lmin,Lmax+1)) 
x_pf1_sin=np.reshape(np.loadtxt("Observations&data/PPF_cox_nldt100_scaled2_v1.txt"\
,dtype=float),(len(eLes_sin),samples,int(T/d),dim))
x_pf2_sin=np.reshape(np.loadtxt("Observations&data/PPF_cox_nldt100_scaled2_v2.txt"\
,dtype=float),(1,samples,int(T/d),dim))
x_pf3_sin=np.reshape(np.loadtxt("remote/PPF_cox_nldt100_scaled2_v3.txt"\
,dtype=float),(1,samples,int(T/d),dim))
x_pf1_sin=np.concatenate((x_pf1_sin,x_pf2_sin,x_pf3_sin),axis=0)

eLes_sin=np.concatenate((eLes_sin,np.array([4,5])),axis=0)
#%%
T=100
samples=100
l0=0
d=1
Lmin=l0
Lmax=3
eLes=np.array(np.arange(Lmin,Lmax+1)) 
x_pf=np.reshape(np.loadtxt("Observations&data/PMLPF_cox_nldt_T100_scaled2_v1.txt"\
,dtype=float),(len(eLes),samples,int(T/d),dim))
x_pf2=np.reshape(np.loadtxt("Observations&data/PMLPF_cox_nldt_T100_scaled2_v2.txt"\
,dtype=float),(1,samples,int(T/d),dim))
x_pf3=np.reshape(np.loadtxt("remote/PMLPF_cox_nldt_T100_scaled2_v3.txt"\
,dtype=float),(1,samples,int(T/d),dim))

x_pf=np.concatenate((x_pf,x_pf2,x_pf3),axis=0)
eLes=np.concatenate((eLes,np.array([4,5])),axis=0)
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
mpf_true=np.reshape(np.loadtxt("remote/PPF_cox_nldt_T100_scaled2_v2.txt"),(int(T/d),dim))
print(mpf_true.shape)
#%%

MSE=np.mean((x_pf-mpf_true)**2,axis=1)[:,a,0]
MSE_sin=np.mean((x_pf1_sin-mpf_true)**2,axis=1)[:,a,0]
#COST
l0=0
Lmin=l0
Lmax=5
eLes=np.array(np.arange(Lmin,Lmax+1,1),dtype=float)
#print(eLes)
Cost=np.zeros(len(eLes))
C0=0.0084   
C=0.00049
K= 1.9131764860557113e-06

C_sin=C0
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
            scale=1
            eNes[0]=N0*scale
            eNes[0]=np.maximum(2,eNes[0])
            eNes[1:]=np.sqrt(2*C/3)*(np.sqrt(C0/delta_l0)+np.sqrt(3*C/2)*CL*delta_l0**(-1/4))\
            *np.float_power(2,2*L)*np.float_power(2,-eles[1:]*3/4)/K*scale
            #print(eNes[1:]*2**eles[1:])
            Cost[i]+=np.sum(3*(eNes[1:]*2**eles[1:])/2)+eNes[0]*2**eles[0]

    
#%%
print("Cost",Cost) 
print("Cost_single",Cost_sin) 
#plt.plot(MSE_un,costs[k1:],label="Unbiased")
MSE_arti=1/2**(eLes*2)
plt.plot(MSE,Cost,label="MLPF",marker="o")
#plt.plot(MSE_arti,Cost,label="MLPF",marker="o")
plt.plot(MSE_sin,Cost_sin,label="Single PF",marker="o")
#plt.plot(MSE_arti[:-1],Cost_sin[:-1],label="Single PF",marker="o")
plt.plot(MSE_sin,1.3e-2*MSE_sin**(-3/2),label=r"$\varepsilon^{-3}$")
plt.plot(MSE,1.2e-1*MSE**(-2.5/2),label=r"$\varepsilon^{-2.5}$")
plt.xlabel(r"$\varepsilon^2$")
plt.title("Nonlinear diffusion term process, T=100.")
plt.ylabel("Cost")
plt.yscale("log")
plt.xscale("log")
plt.legend()
#plt.savefig("Images/MSEvsCost_artificial_nldt.pdf")
plt.show()
#%%


# THE FOLLOWING ITERATION IS MADE IN ORDER TO FIND PARAMETERS SUCH THAT AN APPRECIABLE 
# IMPROVEMENT IS FOUND USING THE MLPF
# VERSION 4
#T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100,T=100

# IN THE FOLLOWING WE CHOOSE A REALIZATION OF THE LANGEVIN DYNAMICS FOR TIME
# T=100 IN ORDER
# TO APPLY THE PF MACHINERY TO IT

# ORIGINAL SYSTEM
np.random.seed(0)
T=100
dim=1
dim_o=dim
xin=np.zeros(dim)+4
l=10
collection_input=[]
I=identity(dim).toarray()
A=np.array([[0]])
sc=np.float_power(2,16)
cov=I*5e-3
Lamb_par=2/20
fi=1/np.sqrt(sc)
print(fi)
collection_input=[dim, b_ou,A,Sig_nldt,fi]
g_pars=[dim,cov]
g_par=cov
x_true=gen_gen_data(T,xin,l,collection_input)
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
#%%
#np.savetxt("Observations&data/Truth_realization_nldt_scaled_T100.txt",x_true,fmt="%f")
#np.savetxt("Observations&data/Observations_time_true_nldt_scaled_T100.txt",obs_time,fmt="%f")
#np.savetxt("Observations&data/Observations_true_nldt_scaled_T100.txt",obs,fmt="%f")
#%%
x_true=np.loadtxt("Observations&data/Truth_realization_nldt_scaled_T100.txt",dtype=float)
obs_time=np.loadtxt("Observations&data/Observations_time_true_nldt_scaled_T100.txt",dtype=float)
obs=np.loadtxt("Observations&data/Observations_true_nldt_scaled_T100.txt",dtype=float)
#%%
#"""
l=6
[log_weights,x_pf]=pff.Cox_PF(T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,\
l,d,N,dim,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par)
weights=norm_logweights(log_weights,ax=1)
x_pfmean=np.sum(np.reshape(weights,(int(T/d),N,1))*x_pf,axis=1)
#print(x_pfmean)
plt.plot((np.array(range(int(T/d)))+1)*d,x_pfmean[:,0],label="PF_nldt")
plt.plot(times,x_true,label="True signal")
#plt.plot(obs_time,obs,label="Observations")
plt.legend()
#"""
#%%
#"""
samples=400
l0=1
L=9
eles=np.array(range(l0,L+1))
N=10000
a=-1
aa=0
d=2**(0)
x_pf=np.reshape(np.loadtxt("Observations&data/CCPF_nldt100_scaled3_v17_3.txt",dtype=float),(2,len(eles),samples,int(T/d),dim))
# IN THE FOLLOWING LINE WE ALSO LOAD THE "TRUTH" TO COMPARE WITH THE SIMULATIONS.
#true_f=np.reshape(np.loadtxt("Observations&data/TruthnldtT100.txt"),(int(T/d),dim))
#"""
#%%
#PARAMETERS 
#C=
#C0=
#K=
# BIAS
"""
a=-1
rbias=np.abs(np.mean(x_pf[1,1:]-x_pf[1,:-1],axis=1)[:,a,0])
rbias2=np.abs(np.mean(x_pf[0]-x_pf[1],axis=1)[:,a,0])
#bias=np.abs(np.mean(x_pf[0]-true_f,axis=1))[:,a,0]
#rbias22=np.abs(np.mean(x_pf2[0]-x_pf2[1],axis=1)[:,a,0])
#rbias2=np.concatenate((rbias21,rbias22),axis=0)
#plt.plot(eles,bias,label="bias")
#plt.plot(eles[1:],rbias,label="Rbias")
plt.plot(eles,rbias2,label="Rbias2")
reference=np.array([1/2**(i) for i in range(len(eles))])*rbias2[0]
var=np.var(x_pf[1,1:]-x_pf[1,:-1],axis=1)[:,a,0]
plt.plot(eles,reference,label="Ref")
#print(var,np.sqrt(var)*1.96/np.sqrt(samples),np.sqrt(samples),np.sqrt(var))
rbias_ub=rbias+np.sqrt(var)*1.96/np.sqrt(samples)
rbias_lb=rbias-np.sqrt(var)*1.96/np.sqrt(samples)
#plt.plot(eles[1:],rbias_ub)
#plt.plot(eles[1:],rbias_lb)
#plt.plot(eles[1:],reference,label=r"$\Delta_l$")
[b0,b1]=coef(eles[:7],np.log2(rbias2[:7]))
print([b0,b1,2**(2*b0)])
plt.title("bias moment in terms of the time discretization levels")
# the computation of the two lines bellow is made so we can get 
# teh parameters for the single cox pf.
#[b0,b1]=coef(eles[:-3],np.log2(bias[:-3]))
#print("for the bias we have the par",[b0,b1,2**(2*b0)])
#K= 8e-11

"""
# VARIANCE

#"""
print("the constant is", np.sqrt(8)*1e-10*1e4)
a=-1
sm=np.mean((x_pf[0]-x_pf[1])**2,axis=1)[:,a,0]
plt.plot(eles,sm,label="Second moment the coupling")

#print((x_pf[0]-x_pf[1])[:2,:,a,0])
reference=np.array([1/2**(i/2) for i in range(len(eles))])*sm[0]
plt.plot(eles,reference,label=r"$\Delta_l^{1/2}$")
var_sm=np.var(((x_pf[0]-x_pf[1])**2),axis=1)[:,a,0]
sm_ub=sm+np.sqrt(var_sm)*1.96/np.sqrt(samples)
sm_lb=sm-np.sqrt(var_sm)*1.96/np.sqrt(samples)
#plt.plot(eles,sm_ub,label="Upper error bound")
#plt.plot(eles,sm_lb,label="Lower error bound")
#var=np.var((x_pf[0]-x_pf[1])[:,:,a],axis=1)
#var0=np.var((x_pf[0])[:,:,a],axis=1)
[b0,b1]=coef(eles,np.log2(sm))
print(b0,b1,(2**b0)*N)
#C=1.882001002951712e-06
plt.title("Second moment in terms of the time discretization levels")
#"""
#"""
plt.xlabel("l")
plt.legend()
plt.yscale("log")
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
x_pf=np.reshape(np.loadtxt("Observations&data/CCPF_nldt100_enes_scaled3_v17.txt",dtype=float),(2,len(enes),samples,int(T/d),dim))
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
#C0=0.0024
plt.xscale("log")
plt.title("ds")
#plt.savefig("Images/Rbias_CCPF_gbm20_p_v5.pdf")
plt.show()
# FOR THE BIAS OF THE SINGLE COX PARTICLE WE HAVE THE FOLLOWING 
#C0=


#%%

# CCONSTANTS AND PPARAMETERS FOR THE SINGLE COX PARTICLE FILTER
C0=0.000117
C=2.8284271247461907e-06
print(C0/C)
K=8e-11
l0=0
Lmin=l0
Lmax=7
es=1e-6
scale=1e-4
eLes=np.arange(Lmin,Lmax+1,1)
N=int(2*C0/es)
L=int(np.ceil(-np.log2(es/(2*np.sqrt(K)))/2,dtype=float))
print(L,N)
#%%
resamp_coef=0.8
[log_weights,x_pf]=pff.Cox_PF(T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,\
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
C0=0.000117
C=2.8284271247461907e-06
print(C0/C)
K= 8e-11
l0=0
Lmin=l0
Lmax=7
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
        scale=1e-4
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

#%%

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
x_pf1_sin=np.reshape(np.loadtxt("remote/PPF_cox_nldt100_scaled3_v1.txt"\
,dtype=float),(len(eLes_sin),samples,int(T/d),dim))
#x_pf2_sin=np.reshape(np.loadtxt("Observations&data/PPF_cox_nldt100_scaled2_v2.txt"\
#,dtype=float),(1,samples,int(T/d),dim))
#x_pf3_sin=np.reshape(np.loadtxt("remote/PPF_cox_nldt100_scaled2_v3.txt"\
#,dtype=float),(1,samples,int(T/d),dim))
#x_pf1_sin=np.concatenate((x_pf1_sin,x_pf2_sin,x_pf3_sin),axis=0)

#eLes_sin=np.concatenate((eLes_sin,np.array([4,5])),axis=0)
#%% 
T=100
samples=100
l0=0
d=1
Lmin=l0
Lmax=5
eLes=np.array(np.arange(Lmin,Lmax+1)) 
x_pf=np.reshape(np.loadtxt("remote/PMLPF_cox_nldt_T100_scaled3_v1.txt"\
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
##plt.plot(eLes,var_sin,label="var_sin")

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
mpf_true=np.reshape(np.loadtxt("remote/PPF_cox_nldt_T100_scaled3_v1.txt"),(int(T/d),dim))
print(mpf_true.shape)
#%%

MSE=np.mean((x_pf-mpf_true)**2,axis=1)[:,a,0]
MSE_sin=np.mean((x_pf1_sin-mpf_true)**2,axis=1)[:,a,0]
#COST
l0=0
Lmin=l0
Lmax=5
eLes=np.array(np.arange(Lmin,Lmax+1,1),dtype=float)
#print(eLes)
Cost=np.zeros(len(eLes))
C0=0.000117
C=2.8284271247461907e-06
K= 8e-11
C_sin=C0
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
plt.plot(MSE_sin,1.5e-7*MSE_sin**(-3/2),label=r"$\varepsilon^{-3}$",color="deepskyblue")
plt.plot(MSE,3.5e-6*MSE**(-2.5/2),label=r"$\varepsilon^{-2.5}$",color="salmon")
plt.xlabel(r"$\varepsilon^2$")
plt.title("Nonlinear diffusion term process. T=100.")
plt.ylabel("Cost")
plt.yscale("log")
plt.xscale("log")
plt.legend()
plt.savefig("Images/MSEvsCost_nldt_scaled3_T100.pdf")
plt.show()

#%%


