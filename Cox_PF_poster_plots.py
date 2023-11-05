#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 16:15:06 2023

@author: alvarem
"""


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


def PosCCPF(T,xin,b,A,Sig,fi,obs,obs_time,l,d,N,dim,resamp_coef,g_den,g_par,Lambda,Lamb_par):
    # Memory friendly version
    # The particle filter function is inspired in 
    # Bain, A., Crisan, D.: Fundamentals of Stochastic Filtering. Springer,
    # New York (2009).
    # ARGUMENTS: T: final time of the propagation, T>0 integer
    # obs: observation process, its a rank two array with discretized observation
    # at random times. with dimension # of observations X dim.
    # obs_time: rank 1 array with the times of observations, which are positive
    # reals. 
    # xin: initial condition of the particle filter, rank 1 array of dimension
    # dim.
    # resamp_coef: coeffient conditions on whether to resample or not, related to 
    # the EES.
    # b_ou: function that represents the drift of the process (its specifications
    # is already in the document. A is the arguments taht takes
    # Sig_out: function that represents the diffusion of the process (its specifications
    # is already in the document. Its arguments are included in fi.
    # l: level parameter of the time discretization.
    # d: time span in which the resampling is computed. d must be a divisor of T.
    # N: number of particles, N \in naturals greater than 1
    # dim: dimension of the problem
    # g_den: likelihood function of the observations given the signal.
    # g_par: parameters of the function g_den.
    # Lambda: Intensity function of the Cox process.
    # Lamb_par: Parameters of the function Lambda.
    # OUTPUTS: x0_pf: is the rank 3 array with the particles at times 
    # i*d, for i \in {1,...,T/d}, its dimentions are (int(T/d),N,dim)
    # log_weights0: logarithm of the weights at times i*d, for i \in {0,1,...,T/d}.
    # it is a rank 2 array with dimensions (int(T/d),N)
    # x1_pf and log_weights1 have the same respective specifications as its counterparts.

    
    x0_pf=np.zeros((int(T/d),N,dim))
    x1_pf=np.zeros((int(T/d),N,dim))
    x0_full=np.zeros((2**(l-1)*T+int(T/d),N,dim))
    x1_full=np.zeros((2**(l)*T+int(T/d),N,dim))
    log_weights0=np.zeros((int(T/d),N))
    log_weights1=np.zeros((int(T/d),N))                                                        
    x0_new=xin
    x1_new=xin
    x0_full[0]=xin[np.newaxis,:]
    x1_full[0]=xin[np.newaxis,:]
    #x0[0]=xin
    #x1[0]=xin
    d_steps0=int(d*2**(l-1))
    d_steps1=int(d*2**l)
    #The following indicees take the bins as (], meaning that the rightmost
    #observation is taken as part of the path of that interval. The motivation
    #of this convention given by the nature of the observations. Since the 
    #time of observation is trigered by a cox process thus the probability of 
    #having an observation at time zero is zero. The probability of getting 
    #having an observation exactly at integers times is also zero, but given the
    # discretization of the cox process the actual change is greater than zero.
    #If we consider a different source of observation, as regularly observed processes,
    # we have no observation at time zero, and thus we set this parameter given 
    # the two previous motivations. 
    c_indices=np.digitize(obs_time,d*np.array(range(int(T/d)+1)),right=True)-1    

    for i in range(int(T/d)):
        obti=obs_time[np.nonzero(c_indices==i)]
        yi=obs[np.nonzero(c_indices==i)]
        #[x0[i*d_steps0:(i+1)*d_steps0+1],x1[i*d_steps1:(i+1)*d_steps1+1]]\
        #=M_coup(x0_new,x1_new,b_ou,A,Sig_ou,fi,l,d,N,dim)
        #xi0=x0[i*d_steps0:(i+1)*d_steps0+1]
        #xi1=x1[i*d_steps1:(i+1)*d_steps1+1]
        [xi0,xi1]=M_coup(x0_new,x1_new,b,A,Sig,fi,l,d,N,dim)
        x0_full[int(i*(d*2**(l-1)+1)):int((i+1)*(d*2**(l-1)+1))]=xi0
        x1_full[int(i*(d*2**(l)+1)):int((i+1)*(d*2**(l)+1))]=xi1
        #here we subtract i*d to the time of observations bcs the 
        #function G works with times starting iwth zero.
        log_weights0[i]=log_weights0[i]\
        +pff.Gox(yi,obti-i*d,xi0,Lambda,Lamb_par,l-1,N,dim,g_den,g_par)
        log_weights1[i]=log_weights1[i]\
        +pff.Gox(yi,obti-i*d,xi1,Lambda,Lamb_par,l,N,dim,g_den,g_par)
        w0=norm_logweights(log_weights0[i],ax=0)
        w1=norm_logweights(log_weights1[i],ax=0)
        #seed_val=i
        #print(xi0,xi1)
        x0_pf[i]=xi0[-1]
        x1_pf[i]=xi1[-1]
        
        #print("the particles are",x1_pf[i])
        #ctimes=d*i+np.array(range(d_steps1+1))/2**(l)
        #plt.plot(ctimes,xi1[:,:,0])
        ESS=np.min(np.array([1/np.sum(w0**2),1/np.sum(w1**2)]))
        
        if ESS<resamp_coef*N:
            print(i*d)
            #x0_new=multi_samp(w0,N,xi0[-1],dim)[1]
            #x1_new=multi_samp(w1,N,xi1[-1],dim)[1]
            #[part0,part1,x0_new,x1_new]=max_coup_sr(w0,w1,N,xi0[-1],xi1[-1],dim)
            [part0,part1,x0_new,x1_new]=max_coup_multi(w0,w1,N,xi0[-1],xi1[-1],dim)
        else:
            
            x0_new=xi0[-1]
            x1_new=xi1[-1]
            
            
            if i<int(T/d)-1:
                log_weights0[i+1]=log_weights0[i]
                log_weights1[i+1]=log_weights1[i]
    
    #weights0=norm_logweights(log_weights0,ax=1)
    #weights1=norm_logweights(log_weights1,ax=1)
    #weights=norm_logweights(log_weights,ax=1)
    
    #x_pfmean0=np.sum(np.reshape(weights0,(int(T/d),N,1))*x0_pf,axis=1)
    #x_pfmean1=np.sum(np.reshape(weights1,(int(T/d),N,1))*x1_pf,axis=1)
    """
    ind=0
    plt.plot(times,x_true[:,ind])
    plt.plot(obs_time,obs[:,ind])    
    d_steps0=int(d*2**(l-1))
    d_steps1=int(d*2**(l))
    spots0=np.arange(d_steps0,2**(l-1)*T+1,d_steps0,dtype=int)
    spots1=np.arange(d_steps1,2**(l)*T+1,d_steps1,dtype=int)    
    weights0=norm_logweights(log_weights0,ax=1)
    weights1=norm_logweights(log_weights1,ax=1)
    weights=norm_logweights(log_weights,ax=1)
    
    x_pfmean0=np.sum(np.reshape(weights0,(int(T/d),N,1))*x0_pf,axis=1)
    x_pfmean1=np.sum(np.reshape(weights1,(int(T/d),N,1))*x1_pf,axis=1)
    #x_pmean=np.sum(np.reshape(weights,(int(T/d),N,1))*x_pf,axis=1)
    plt.plot(times,x_true[:,ind],label="True")
    plt.plot(obs_time,obs[:,ind],label="obs")
    plt.plot(spots0*2**(-l+1),x_pfmean0[:,ind],label="PF0",c="b")
    plt.plot(spots1*2**(-l),x_pfmean1[:,ind],label="PF1")
    plt.legend()
    plt.show() 
    """
        #print(weights.shape)
    #Filter
    #spots0=np.arange(d_steps0,2**(l-1)*T+1,d_steps0,dtype=int)
    #spots1=np.arange(d_steps1,2**(l)*T+1,d_steps1,dtype=int)
    
    #x_pf0=x0[spots0]
    #x_pf1=x1[spots1]
    #weights0=norm_logweights(log_weights0,ax=1)
    #weights1=norm_logweights(log_weights1,ax=1)
    #print(x_pf.shape,weights.shape)
    #suma0=np.sum(x_pf0[:,:,1]*weights0,axis=1)
    #suma1=np.sum(x_pf1[:,:,1]*weights1,axis=1)
    
    
    
    return [log_weights0,log_weights1,x0_pf ,x1_pf,x0_full,x1_full]

#%%
if True==True:
    
        np.random.seed(35)
        T=10
        dim=1
        dim_o=dim
        xin=np.zeros(dim)+1
        l=8
        collection_input=[]
        I=identity(dim).toarray()
        #comp_matrix = ortho_group.rvs(dim)
        comp_matrix=np.array([[1]])
        inv_mat=la.inv(comp_matrix)
        #S=diags(np.random.normal(1,0.1,dim),0).toarray()
        S=diags(np.random.normal(np.sqrt(2),0.1,dim),0).toarray()
        #S=np.array([[1.]])
        fi=inv_mat@S@comp_matrix
        #B=diags(np.random.normal(-1,0.1,dim),0).toarray()
        B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
        B=inv_mat@B@comp_matrix
        print(B)
        #B=np.array([[-1.]])
        #print(B)
        #print(B)
        #B=comp_matrix-comp_matrix.T  +B 
        collection_input=[dim, b_ou,B,Sig_ou,fi]
        cov=I*1e0
        g_pars=[dim,cov]
        g_par=cov
        x_true=gen_gen_data(T,xin,l,collection_input)
        Lamb_par=50
        [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
        #print(obs_time,obs,len(obs_time))
        samples=1000
        
        obss=np.array([])
        for i in range(samples):
            [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
            obss=np.concatenate((obss,obs_time))
            
        d=2**(-2)
#%%

l1=5
plt.hist(obss,bins=2**(-l1)*np.array(range(int(T*2**l1+1))),density=True,label="Conditional Cox process histogram")



times=2**(-l)*np.array(range(int(T*2**l+1)))

plt.plot(times,0.13*np.abs(x_true),label=r"$\lambda(X_t)=|X_t|/\alpha$")
#plt.plot(obs_time,obs,label="Observations")
plt.xlabel(r"$t$")
plt.legend()
#d=2**(1)
#plt.savefig("Images/cox_process.pdf")
N=40
resamp_coef=0.8
#%%

l=4
N=10000

resamp_coef=0.8
np.random.seed(0)
dtimes=np.array(range(int(T/d)))*d+d
[log_weights0,log_weights1,x0_pf,x1_pf,x0_full,x1_full]=PosCCPF(T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,l\
,d,N,dim,resamp_coef,g_den,g_par,Lambda,Lamb_par)
weights0=norm_logweights(log_weights0,ax=1)
#print(weights0.shape,x0_pf.shape)   

phi_pfmean=np.sum(np.reshape(weights0,(int(T/d),N,1))*x0_pf,axis=1)
phi_pf=phi_pfmean.flatten()
plt.plot(dtimes,phi_pf,label="PF",marker=".",c="orange",ls="dashed")
plt.plot(times,x_true,label="True signal",c="gray")
plt.scatter(obs_time,obs,label="Observations",marker="o",c="blue")


i=0
j=int(0)
print(times)
obs_ltimes=np.zeros(len(obs_time),dtype=int)

while i<len(obs_time):
   if times[j]<obs_time[i]:
       j+=1
       #print(j)
       
       obs_ltimes[i]=j
       i+=1
#print(obs_ltimes)
#print(x_true[obs_ltimes],obs, np.minimum(obs,x_true[obs_ltimes]))
#plt.vlines(x = obs_time, ymin = np.minimum(obs,x_true[obs_ltimes]), ymax =np.maximum(obs,x_true[obs_ltimes]),\
#          colors = 'blue')
 

#plt.plot(times,kbf,label="KBF")
#plt.plot(times,kf,label="KF")
#plt.plot(times,kf-kbf)
#print(len(obs_time))
plt.legend()
plt.xlabel(r"$t$")
#plt.savefig("Images/pf_ts_obs_v2.pdf")
plt.show()
"""
colors=["red","blue","green"]
l-=1
times2=2**(-l)*np.array(range(int(T*2**l+1)))
tpoints=dtimes
ppoints1=(np.array(range(int(T/d)))+1)*(int(d*2**l)+1)-1
for j in range(N):
    for i in range(int(T/d)):
        plt.plot(times2[int(i*(d*2**l)):int((i+1)*(d*2**l))+1],x0_full[int(i*(d*2**l+1)):int((i+1)*(d*2**l+1)),j,0],c=colors[j])
    plt.scatter(tpoints,x0_full[ppoints1,j,0],c=colors[j])
   # plt.scatter(tpoints,x0_full[ppoints1+1,j,0])
plt.legend()
plt.xlabel(r"$t$")
#plt.savefig("Images/sp.pdf")
plt.show()
"""
#%%
        
        

t = np.arange(0.01, 10.0, 0.01)
data1 = np.exp(t)
data2 = np.sin(2 * np.pi * t)

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('time (s)')
ax1.set_ylabel('exp', color=color)
ax1.plot(t, data1, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('sin', color=color)  # we already handled the x-label with ax1
ax2.plot(t, data2, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()