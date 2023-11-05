    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  1 16:40:42 2022

@author: alvarem
"""
#Upload the information and the packages 

from Un_cox_PF_functions_def import *
#from PF_functions_def import *
import PF_functions_def as pff
import multiprocessing
import time

import math
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
from scipy.stats import multivariate_normal

#import collections
#%%


def PCox_PF(arg_col):
    [seed_val,collection_input]=arg_col
    [T,xin,b_ou,A,Sig_ou,fi,obs,obs_time,l,d,N,dim\
    ,resamp_coef,g_den,g_par,Lambda,Lamb_par]=collection_input
    np.random.seed(seed_val)
    print(seed_val,l)
    # (T,z,lmax,x0,b_ou,A,Sig_ou,fi,ht,H,l,d,N,dim,resamp_coef,para=True):
    # The particle filter function is inspired in 
    # Bain, A., Crisan, D.: Fundamentals of Stochastic Filtering. Springer,
    # New York (2009).
    # ARGUMENTS: T: final time of the propagation, T>0 and preferably integer
    # z: observation process, its a rank two array with discretized observation
    # at the intervals 2^(-lmax)i, i \in {0,1,...,T2^lmax}. with dimension
    # (T2^{lmax}+1) X dim
    # lmax: level of discretization of the observations
    # x0: initial condition of the particle filter, rank 1 array of dimension
    # dim
    # b_out: function that represents the drift of the process (its specifications
    # is already in the document. A is the arguments taht takes
    # Sig_out: function that represents the diffusion of the process (its specifications
    # is already in the document. Its arguments are included in fi.
    # ht: function in the observation process (its specifications
    # is already in the document). Its arguments are included in H.
    # d: time span in which the resampling is computed. d must be a divisor of T.
    # N: number of particles, N \in naturals greater than 1
    # dim: dimension of the problem
    # para: key to wheter compute the paralelization or not.
    # OUTPUTS: x: is the rank 3 array with the resampled particles at times 
    # 2^{-l}*i, i \in {0,1,..., T*2^l}, its dimentions are (2**l*T+1,N,dim)
    # log_weights: logarithm of the weights at times i*d, for i \in {0,1,...,T/d}.
    # it is a rank 2 array with dimensions (int(T/d),N)
    # suma: its the computation of the particle filter for each dimension of the problem
    # its a rank 2 array with dimensions (int(T/d),dim)
    #x=np.zeros((2**l*T+1,N,dim))
    log_weights=np.zeros((int(T/d),N))
    x_pf=np.zeros((int(T/d),N,dim))                            
    x_new=xin
    #x[0]=xin
    d_steps=int(d*2**l)
    c_indices=np.digitize(obs_time,d*np.array(range(int(T/d)+1)),right=True)-1
    for i in range(int(T/d)):
        obti=obs_time[np.nonzero(c_indices==i)]
        yi=obs[np.nonzero(c_indices==i)]
        #x[i*d_steps:(i+1)*d_steps+1]=M(x_new,b_ou,A,Sig_ou,fi,l,d,N,dim)
        #xi=x[i*d_steps:(i+1)*d_steps+1]
        xi=pff.M(x_new,b_ou,A,Sig_ou,fi,l,d,N,dim)
        #print(xi)
        log_weights[i]=pff.Gox(yi,obti-i*d,xi,Lambda,Lamb_par,l,N,dim,g_den,g_par)
        weights=norm_logweights(log_weights[i],ax=0)
        #print(yi,weights)
        #seed_val=i
        #print(weights.shape)
        x_last=xi[-1]
        x_pf[i]=xi[-1]
        ESS=1/np.sum(weights**2)
        if ESS<resamp_coef*N:
            #[part0,part1,x0_new,x1_new]=max_coup_sr(w0,w1,N,xi0[-1],xi1[-1],dim)
            x_new=multi_samp(weights,N,x_last,dim)[1]
        else:
            x_new=x_last
       #x_new=sr(weights,N,x_pf[i],dim)[1]
    weights=norm_logweights(log_weights,ax=1)
    #weights=norm_logweights(log_weights,ax=1)
    
    x_pfmean=np.sum(np.reshape(weights,(int(T/d),N,1))*x_pf,axis=1)
    #Filter
    #spots=np.arange(d_steps,2**l*T+1,d_steps,dtype=int)
    #x_pf=x[spots]
    #weights=norm_logweights(log_weights,ax=1)
    #print(x_pf.shape,weights.shape)
    #suma=np.sum(x_pf[:,:,1]*weights,axis=1)
    return x_pfmean





def PCCPF(arg_col):
    [seed_val,collection_input]=arg_col
    [T,xin,b_ou,A,Sig_ou,fi,obs,obs_time,l,d,N,dim\
    ,resamp_coef,g_den,g_par,Lambda,Lamb_par]=collection_input
    np.random.seed(seed_val)
    print(seed_val,l,N)
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
    log_weights0=np.zeros((int(T/d),N))
    log_weights1=np.zeros((int(T/d),N))                                                        
    x0_new=xin
    x1_new=xin
    #x0[0]=xin
    #x1[0]=xin
    d_steps0=int(d*2**(l-1))
    d_steps1=int(d*2**l)
    c_indices=np.digitize(obs_time,d*np.array(range(int(T/d)+1)),right=True)-1
    for i in range(int(T/d)):
        obti=obs_time[np.nonzero(c_indices==i)]
        yi=obs[np.nonzero(c_indices==i)]
        #[x0[i*d_steps0:(i+1)*d_steps0+1],x1[i*d_steps1:(i+1)*d_steps1+1]]\
        #=M_coup(x0_new,x1_new,b_ou,A,Sig_ou,fi,l,d,N,dim)
        #xi0=x0[i*d_steps0:(i+1)*d_steps0+1]
        #xi1=x1[i*d_steps1:(i+1)*d_steps1+1]
        [xi0,xi1]=M_coup(x0_new,x1_new,b_ou,A,Sig_ou,fi,l,d,N,dim)
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
            #[part0,part1,x0_new,x1_new]=max_coup_sr(w0,w1,N,xi0[-1],xi1[-1],dim)
            [part0,part1,x0_new,x1_new]=max_coup_multi(w0,w1,N,xi0[-1],xi1[-1],dim)
        else:
            
            x0_new=xi0[-1]
            x1_new=xi1[-1]
            
            if i< int(T/d)-1:
                log_weights0[i+1]=log_weights0[i]
                log_weights1[i+1]=log_weights1[i]
    
    weights0=norm_logweights(log_weights0,ax=1)
    weights1=norm_logweights(log_weights1,ax=1)
    #weights=norm_logweights(log_weights,ax=1)
    
    x_pfmean0=np.sum(np.reshape(weights0,(int(T/d),N,1))*x0_pf,axis=1)
    x_pfmean1=np.sum(np.reshape(weights1,(int(T/d),N,1))*x1_pf,axis=1)
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
    
    
    
    return [x_pfmean0,x_pfmean1]


def PMLPF_cox(arg_col):
    
    [seed_val,collection_input]=arg_col
    [T,xin,b,A,Sig,fi,obs,obs_time,eles,d,Nl,dim,resamp_coef,phi,\
    dim_out,g_den,g_par,Lambda,Lamb_par]=collection_input
    np.random.seed(seed_val)
    print("seed_is",seed_val,"Nl is",Nl)
    # The MLPF stands for Multilevel Particle Filter, and uses the Multilevel
    #methodology to compute the 
    # particle filter. 

    #ARGUMENTS:
    # The arguments are basically the same as those for the PF and CPF
    #functions, with changes in
    # l->eles: is a 1 rank array starting with l_0 and ending with L 
    # N-> Nl: is a 1 rank array starting with the corresponding number of
    #particles to  each level
    # the new parameters are:
    # phi: function that takes as argument a rank M array and computes a function 
    # along the axis_action dimension. the dimensions of the output of phi are the 
    # same as the input except the dimension of the axis axis_action  which is 
    # since the argument of phi here is x_pf, then the axis of action is the axis
    # 2, the one of the dimensions.
    # changed by dim_out
    # g_den: Likelihood function of the observations where g_par 
    #is the parameters of function.
    
    #OUTPUT
    # pf: computation of the particle filter with dimension (int(T/d),dim_out)

    #pf=np.zeros((int(T/d),dim_out))
    [log_weightsl0,x_pfl0]=pff.Cox_PF(T,xin,b,A,Sig,fi,obs,obs_time,eles[0],d\
    ,Nl[0],dim,resamp_coef,g_den,g_par,Lambda,Lamb_par)
    #[log_weightsl0,x_pfl0]=pff.PF(T,z,lmax,xin,b_ou,A,Sig_ou,fi,ht,H,eles[0],
    #d,Nl[0],dim,resamp_coef,para=True)[1:]
    phi_pf=phi(x_pfl0,ax=2)
    weightsl0=np.reshape(norm_logweights(log_weightsl0,ax=1),(int(T/d),Nl[0],1))
    # The reshape of weightsl0 is such that we can multiply easily the weights
    # by all dimensions of the state space model.
    pf= np.sum(phi_pf*weightsl0,axis=1)
    
    #PF(T,z,lmax,x0,b_ou,A,Sig_ou,fi,ht,H,l,d,N,dim,resamp_coef,para=True)
    eles_len=len(eles)
    #x_pf=np.zeros((2,eles_len-1, int(T/d),N,dim))
    #log_weights=np.zeros((2,eles_len-1, int(T/d),N))plt
    
    for i in range(1,eles_len):
        l=eles[i]
        #print(l,Nl[i])
        #(T,xin,b_ou,A,Sig_ou,fi,obs,obs_time,l,d,N,dim,resamp_coef,g_den,g_par,Lambda,Lamb_par)
        [log_weights0,log_weights1,x0_pf,x1_pf]=CCPF(T,xin,b,A,Sig,fi,obs,\
        obs_time,l,d,Nl[i],dim,resamp_coef,g_den,g_par,Lambda,Lamb_par)
        #(T,z,lmax,xin,b_ou,A,Sig_ou,fi,ht,H,eles[i],d,Nl[i],dim,resamp_coef,para=True)[2:]
        #
        #log_weights[:,i-1]=[log_weights0,log_weights1]
        #x_pf[:,i-1]=[x0_pf,x1_pf]
        phi_pf0=phi(x0_pf,ax=2)
        phi_pf1=phi(x1_pf,ax=2)
        weights0=np.reshape(norm_logweights(log_weights0,ax=1),(int(T/d),Nl[i],1))
        weights1=np.reshape(norm_logweights(log_weights1,ax=1),(int(T/d),Nl[i],1))
        pf=pf+np.sum(phi_pf1*weights1,axis=1)-np.sum(phi_pf0*weights0,axis=1)
        
    return pf
#%%


#%%

#%%
#%%


if __name__ == '__main__':
    


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
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8    
    dim_out=2
    samples=1
    L=8

    l0=0
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(np.arange(l0,L+1))

    C=11
    C0=3.2
    K=15.7
    pfs=np.zeros((2,len(eles),int(T/d),dim))
    inputs=[]
    CL=2**(1/4)*(2**(L/4)-1)/(2**(1/4)-1)
    NB0=(1+np.sqrt(3*C/2)*CL)*np.sqrt(2*C/3)*2**(-L/4)/K
    N0=C0*(1+np.sqrt(3*C/2)*CL)*2**(2*L)/K
    N0=np.maximum(N0,3)
    Nl=np.zeros(len(eles))
    Nl[0]=N0
    Nl[1:]=np.array(NB0**2**(-eles[1:]*3/4)*2**(L*(2+1/4)),dtype=int)
    Nl=np.array(Nl,dtype=int)
    print(Nl)
    
    for i in range(len(eles)):
        
        
        l=eles[i]
        if l==0:
            l=1
            
        N=Nl[i]
        for sample in range(samples):
            collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])        
    pool = multiprocessing.Pool(processes=9)
    pool_outputs = pool.map(PCCPF, inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    
          
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i]=pool_outputs[i*samples+sample]
            
    #x_pfs=np.reshape(np.array(pool_outputs),(len(eles),samples,int(T/d),dim))
    
    #log_weightss=log_weightss.flatten()
    pfs=pfs.flatten()
    
    np.savetxt("Observations&data/MLPF_cox_gbm_truth_by_parts.txt",pfs,fmt="%f")
    
    print("Parallelized processes time:",end-start,"\n")      



#%%
"""
np.random.seed(0)
T=100
dim=1
dim_o=dim
xin=np.zeros(dim)+.001
l=13
collection_input=[]
I=identity(dim).toarray()

#xin0=np.abs(np.random.normal(1,0,dim))
#xin1=xin0
mu=np.abs(np.random.normal(1.01,0,dim))
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
samples=10

np.random.seed(27)
x_true=gen_gen_data(T,xin,l,collection_input)
Lamb_par=5
#print(Sig,sigs,mu)
[obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
start=time.time()
    d=2**(0)
    resamp_coef=0.8    
    dim_out=2
    g_par=cov
    samples=200
    l0=1
    L=7
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=100
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            #collection_input=[T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,eles[i],d,N,dim\
            #,resamp_coef,g_den,g_par,Lambda,Lamb_par]
            collection_input=[T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Lambda,Lamb_par]
            
            
            inputs.append([sample+samples*i,collection_input])        
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF, inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    #x_pfs=np.reshape(np.array(pool_outputs),(len(eles),samples,int(T/d),dim))
    
    #log_weightss=log_weightss.flatten()
    pfs=pfs.flatten()
    
    np.savetxt("Observations&data/CCPF_v5.txt",pfs,fmt="%f")

"""
#%%


"""



Observations&data/MLPF_cox_gbm_truth_by_parts.txt corresponds to the 
parallelization of a PCCPF in such a way that we get the necessary levels to obtain 
a MLPF. The parallelization is made in order to speed up the computations since the
number of particles for each level is large.

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
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8    
    dim_out=2
    samples=1
    L=8

    l0=0
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(np.arange(l0,L+1))

    C=11
    C0=3.2
    K=15.7
    pfs=np.zeros((2,len(eles),int(T/d),dim))
    inputs=[]
    CL=2**(1/4)*(2**(L/4)-1)/(2**(1/4)-1)
    NB0=(1+np.sqrt(3*C/2)*CL)*np.sqrt(2*C/3)*2**(-L/4)/K
    N0=C0*(1+np.sqrt(3*C/2)*CL)*2**(2*L)/K
    N0=np.maximum(N0,3)
    Nl=np.zeros(len(eles))
    Nl[0]=N0
    Nl[1:]=np.array(NB0**2**(-eles[1:]*3/4)*2**(L*(2+1/4)),dtype=int)
    Nl=np.array(Nl,dtype=int)
    print(Nl)
    
    for i in range(len(eles)):
        
        
        l=eles[i]
        if l==0:
            l=1
            
        N=Nl[i]
        for sample in range(samples):
            collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])        
    pool = multiprocessing.Pool(processes=16)
    pool_outputs = pool.map(PCCPF, inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    
          
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i]=pool_outputs[i*samples+sample]
            
    #x_pfs=np.reshape(np.array(pool_outputs),(len(eles),samples,int(T/d),dim))
    
    #log_weightss=log_weightss.flatten()
    pfs=pfs.flatten()
    
    np.savetxt("Observations&data/MLPF_cox_gbm_truth_by_parts.txt",pfs,fmt="%f")
    
    print("Parallelized processes time:",end-start,"\n") 


Observations&data/MLPF_cox_gbm_v1.txt corresponds to 
the samples of the MLPF_cox with GBM obtained in orderr 
to get the MSE. 

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
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8    
    dim_out=2
    samples=50
    Lmax=6
    Lmin=0
    l0=0
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eLes=np.array(np.arange(Lmin,Lmax+1))

    C=11
    C0=3.2
    K=15.7
    x_pfs=np.zeros((len(eLes),samples,int(T/d),dim))
    inputs=[]
    
    for i in range(len(eLes)):
        L=eLes[i]

        CL=2**(1/4)*(2**(L/4)-1)/(2**(1/4)-1)
        NB0=(1+np.sqrt(3*C/2)*CL)*np.sqrt(2*C/3)*2**(-L/4)/K
        N0=C0*(1+np.sqrt(3*C/2)*CL)*2**(2*L)/K
        N0=np.maximum(N0,3)
        eles=np.array(np.arange(l0,L+1))
        Nl=np.zeros(len(eles))
        Nl[0]=N0
        Nl[1:]=np.array(NB0**2**(-eles[1:]*3/4)*2**(L*(2+1/4)),dtype=int)
        Nl=np.array(Nl,dtype=int)
        print(Nl)
        

        for sample in range(samples):
            collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,eles,d,Nl,dim\
            ,resamp_coef,phi,dim_out,g_den,g_par,Norm_Lambda,Lamb_par]

            inputs.append([sample+samples*i,collection_input])        
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PMLPF_cox, inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            x_pfs[i,sample]=pool_outputs[i*samples+sample]
            
    #x_pfs=np.reshape(np.array(pool_outputs),(len(eles),samples,int(T/d),dim))
    
    #log_weightss=log_weightss.flatten()
    x_pfs=x_pfs.flatten()
    
    np.savetxt("Observations&data/MLPF_cox_gbm_v1.txt",x_pfs,fmt="%f")



Observations&data/CCPF_gbm_v3.txt is made in order to see 
the variance and the bias of the coupled PF with GMB.

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
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8    
    dim_out=2
    samples=1000
    
    L=10
    l0=1
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    x_pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    
    inputs=[]


    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            #collection_input=[T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,eles[i],d,N,dim\
            #,resamp_coef,g_den,g_par,Lambda,Lamb_par]
            collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]

            
            inputs.append([sample+samples*i+1000,collection_input])        
    pool = multiprocessing.Pool(processes=16)
    pool_outputs = pool.map(PCCPF, inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            x_pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    #x_pfs=np.reshape(np.array(pool_outputs),(len(eles),samples,int(T/d),dim))
    
    #log_weightss=log_weightss.flatten()
    x_pfs=x_pfs.flatten()
    
    np.savetxt("Observations&data/CCPF_gbm_v3.txt",x_pfs,fmt="%f")



Observations&data/CCPF_gbm_v2.txt is made in order to see 
the variance and the bias of the coupled PF with GMB.

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
   [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
   start=time.time()
   d=2**(0)
   resamp_coef=0.8    
   dim_out=2
   samples=50
   
   L=10
   l0=1
   # In this iteration we have eLes and eles, do not confuse. eLes respresents
   # the range of maximum number of levels that we take, eles is a direct 
   # argument to the MLPF_cox, its the number of levels that we in one ML. 
   eles=np.array(range(l0,L+1))
   N=1000
   x_pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
   
   inputs=[]


   for i in range(len(eles)):
       l=eles[i]
       for sample in range(samples):
           #collection_input=[T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,eles[i],d,N,dim\
           #,resamp_coef,g_den,g_par,Lambda,Lamb_par]
           collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,N,dim\
           ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]

           
           inputs.append([sample+samples*i+samples*len(eles),collection_input])        
   pool = multiprocessing.Pool(processes=16)
   pool_outputs = pool.map(PCCPF, inputs)
   pool.close()
   pool.join()
   #blocks_pools.append(pool_outputs)
   xend1=time.time()
   end=time.time()
   
   print("Parallelized processes time:",end-start,"\n")            
   for i in range(len(eles)):
       for sample in range(samples):
           x_pfs[:,i,sample]=pool_outputs[i*samples+sample]
   
   #x_pfs=np.reshape(np.array(pool_outputs),(len(eles),samples,int(T/d),dim))
   
   #log_weightss=log_weightss.flatten()
   x_pfs=x_pfs.flatten()
   
   np.savetxt("Observations&data/CCPF_gbm_v2.txt",x_pfs,fmt="%f")

Observations&data/CCPF_gbm_v1.txt is made in order to see 
the variance and the bias of the coupled PF with GMB.


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
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8    
    dim_out=2
    samples=100
    
    L=7
    l0=1
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    x_pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    
    inputs=[]


    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            #collection_input=[T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,eles[i],d,N,dim\
            #,resamp_coef,g_den,g_par,Lambda,Lamb_par]
            collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]

            
            inputs.append([sample+samples*i,collection_input])        
    pool = multiprocessing.Pool(processes=16)
    pool_outputs = pool.map(PCCPF, inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            x_pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    #x_pfs=np.reshape(np.array(pool_outputs),(len(eles),samples,int(T/d),dim))
    
    #log_weightss=log_weightss.flatten()
    x_pfs=x_pfs.flatten()
    
    np.savetxt("Observations&data/CCPF_gbm_v1.txt",x_pfs,fmt="%f")


Observations&data/MLPF_cox_v12.txt corresponds to the following data and
it was computed in order to estimate the variance constant. 


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
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()*(2/3)
    B=inv_mat@B@comp_matrix
    #B=np.array([[-1.]])
    #print(B)
    #print(B)
    #B=comp_matrix-comp_matrix.T  +B 
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    cov=I*1e0
    g_par=[dim,cov]
    x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=1.1
    [obs_time,obs]=gen_obs(x_true,l,T,Lambda,Lamb_par,g_normal,g_par)
    start=time.time()
    d=2**(1)
    resamp_coef=0.8    
    dim_out=2
    g_par=cov
    samples=100
    Lmin=0
    Lmax=5
    l0=0
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eLes=np.array(range(Lmin,Lmax+1))
    N0=25
    pfs=np.zeros((len(eLes),samples,int(T/d),dim))
    
    inputs=[]

    for i in range(len(eLes)):
        L=eLes[i]
        eles=np.arange(l0,L+1,dtype=int)
        Nl=np.array(N0*2**(-eles+2*L)*(L+1))

        for sample in range(samples):
            #collection_input=[T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,eles[i],d,N,dim\
            #,resamp_coef,g_den,g_par,Lambda,Lamb_par]
            collection_input=[T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,eles,d,Nl,dim,\
            resamp_coef,phi,dim_out,g_den,g_par,Lambda,Lamb_par]
            
            
            inputs.append([sample,collection_input])        
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PMLPF_cox, inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eLes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    
    #x_pfs=np.reshape(np.array(pool_outputs),(len(eles),samples,int(T/d),dim))
    
    #log_weightss=log_weightss.flatten()
    pfs=pfs.flatten()
    
    np.savetxt("Observations&data/MLPF_cox_v12.txt",pfs,fmt="%f")




Observations&data/CCPF_v6.txt corresponds to the following data and
it was computed in order to estimate the variance constant. 

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
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()*(2/3)
    B=inv_mat@B@comp_matrix
    #B=np.array([[-1.]])
    #print(B)
    #print(B)
    #B=comp_matrix-comp_matrix.T  +B 
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    cov=I*1e0
    g_pars=[dim,cov]
    x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=1.1
    [obs_time,obs]=gen_obs(x_true,l,T,Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(1)
    resamp_coef=0.8    
    dim_out=2
    g_par=cov
    samples=800
    l0=1
    L=7
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=100
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            #collection_input=[T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,eles[i],d,N,dim\
            #,resamp_coef,g_den,g_par,Lambda,Lamb_par]
            collection_input=[T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Lambda,Lamb_par]

            
            inputs.append([sample+samples*i+samples*len(eles),collection_input])        
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF, inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    #x_pfs=np.reshape(np.array(pool_outputs),(len(eles),samples,int(T/d),dim))
    
    #log_weightss=log_weightss.flatten()
    pfs=pfs.flatten()
    
    np.savetxt("Observations&data/CCPF_v6.txt",pfs,fmt="%f")



Observations&data/CCPF_v5.txt corresponds to the following data and
it was computed in order to estimate the variance constant. 



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
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()*(2/3)
    B=inv_mat@B@comp_matrix
    #B=np.array([[-1.]])
    #print(B)
    #print(B)
    #B=comp_matrix-comp_matrix.T  +B 
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    cov=I*1e0
    g_pars=[dim,cov]
    x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=1.1
    [obs_time,obs]=gen_obs(x_true,l,T,Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(1)
    resamp_coef=0.8    
    dim_out=2
    g_par=cov
    samples=200
    l0=1
    L=7
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=100
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            #collection_input=[T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,eles[i],d,N,dim\
            #,resamp_coef,g_den,g_par,Lambda,Lamb_par]
            collection_input=[T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Lambda,Lamb_par]
            
            
            inputs.append([sample+samples*i,collection_input])        
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF, inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    #x_pfs=np.reshape(np.array(pool_outputs),(len(eles),samples,int(T/d),dim))
    
    #log_weightss=log_weightss.flatten()
    pfs=pfs.flatten()
    
    np.savetxt("Observations&data/CCPF_v5.txt",pfs,fmt="%f")


Observations&data/MLPF_cox_v8.txt corresponds to 

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
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()*(2/3)
    B=inv_mat@B@comp_matrix
    #B=np.array([[-1.]])
    #print(B)
    #print(B)
    #B=comp_matrix-comp_matrix.T  +B 
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    cov=I*1e0
    g_par=[dim,cov]
    x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=1.1
    [obs_time,obs]=gen_obs(x_true,l,T,Lambda,Lamb_par,g_normal,g_par)
    start=time.time()
    d=2**(1)
    resamp_coef=0.8    
    dim_out=2
    g_par=cov
    samples=100
    Lmin=0
    Lmax=5
    l0=0
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eLes=np.array(range(Lmin,Lmax+1))
    N0=25
    pfs=np.zeros((len(eLes),samples,int(T/d),dim))
    
    inputs=[]

    for i in range(len(eLes)):
        L=eLes[i]
        eles=np.arange(l0,L+1,dtype=int)
        Nl=np.array( N0*2**((-3/4)*eles+5*L/2.),dtype=int)

        for sample in range(samples):
            #collection_input=[T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,eles[i],d,N,dim\
            #,resamp_coef,g_den,g_par,Lambda,Lamb_par]
            collection_input=[T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,eles,d,Nl,dim,\
            resamp_coef,phi,dim_out,g_den,g_par,Lambda,Lamb_par]
            
            
            inputs.append([sample+100,collection_input])        
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PMLPF_cox, inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eLes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    
    #x_pfs=np.reshape(np.array(pool_outputs),(len(eles),samples,int(T/d),dim))
    
    #log_weightss=log_weightss.flatten()
    pfs=pfs.flatten()
    
    np.savetxt("Observations&data/MLPF_cox_v8.txt",pfs,fmt="%f")




Observations&data/MLPF_cox_v5txt corresponds to 

    np.random.seed(1)
    T=20
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
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
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()*(2/3)
    B=inv_mat@B@comp_matrix
    #B=np.array([[-1.]])
    #print(B)
    #print(B)
    #B=comp_matrix-comp_matrix.T  +B 
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    cov=I*1e0
    g_par=[dim,cov]
    x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=1.1
    [obs_time,obs]=gen_obs(x_true,l,T,Lambda,Lamb_par,g_normal,g_par)
    start=time.time()
    d=2**(1)
    resamp_coef=0.8    
    dim_out=2
    g_par=cov
    samples=500
    Lmin=0
    Lmax=5
    l0=0
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eLes=np.array(range(Lmin,Lmax+1))
    N0=5
    pfs=np.zeros((len(eLes),samples,int(T/d),dim))
    
    inputs=[]

    for i in range(len(eLes)):
        L=eLes[i]
        eles=np.arange(l0,L+1,dtype=int)
        Nl=np.array( N0*2**((-3/4)*eles+5*L/2.),dtype=int)

        for sample in range(samples):
            #collection_input=[T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,eles[i],d,N,dim\
            #,resamp_coef,g_den,g_par,Lambda,Lamb_par]
            collection_input=[T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,eles,d,Nl,dim,\
            resamp_coef,phi,dim_out,g_den,g_par,Lambda,Lamb_par]
            
            
            inputs.append([sample+500,collection_input])        
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PMLPF_cox, inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eLes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    
    #x_pfs=np.reshape(np.array(pool_outputs),(len(eles),samples,int(T/d),dim))
    
    #log_weightss=log_weightss.flatten()
    pfs=pfs.flatten()
    
    np.savetxt("Observations&data/MLPF_cox_v5.txt",pfs,fmt="%f")


MLPF_cox_v1.txt corresponds to 

    np.random.seed(1)
    T=10
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
    S=diags(np.random.normal(0.8,0.1,dim),0).toarray()
    #S=np.array([[1.]])
    fi=inv_mat@S@comp_matrix
    #B=diags(np.random.normal(-1,0.1,dim),0).toarray()
    B=diags(np.random.normal(-0.9,0.01,dim),0).toarray()*(2/3)
    B=inv_mat@B@comp_matrix
    #B=np.array([[-1.]])
    #print(B)
    #print(B)
    #B=comp_matrix-comp_matrix.T  +B 
    
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    cov=I*1e-1
    g_par=[dim,cov]
    x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=2
    
    [obs_time,obs]=gen_obs(x_true,l,T,Lambda,Lamb_par,g_normal,g_par)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8    
    dim_out=2
    g_par=cov
    samples=100
    Lmin=0
    Lmax=7
    l0=0
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eLes=np.array(range(Lmin,Lmax+1))
    N0=100
    
    pfs=np.zeros((2,len(eLes),samples,int(T/d),dim))
    
    inputs=[]

    for i in range(len(eLes)):
        L=eLes[i]
        eles=np.arange(l0,L+1,dtype=int)
        Nl=np.array( N0*2**((-3/4+1)*eles+L/4.),dtype=int)

        for sample in range(samples):
            #collection_input=[T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,eles[i],d,N,dim\
            #,resamp_coef,g_den,g_par,Lambda,Lamb_par]
            collection_input=[T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,eles,d,Nl,dim,\
            resamp_coef,phi,dim_out,g_den,g_par,Lambda,Lamb_par]
            
            
            inputs.append([sample,collection_input])        
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PMLPF_cox, inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eLes)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    #x_pfs=np.reshape(np.array(pool_outputs),(len(eles),samples,int(T/d),dim))
    
    #log_weightss=log_weightss.flatten()
    pfs=pfs.flatten()
    
    np.savetxt("Observations&data\MLPF_cox_v1.txt",pfs,fmt="%f")
###########################################################################
Lmin=0
Lmax=6
l0=0
# In this iteration we have eLes and eles, do not confuse. eLes respresents
# the range of maximum number of levels that we take, eles is a direct 
# argument to the MLPF_cox, its the number of levels that we in one ML. 
eLes=np.array(range(Lmin,Lmax+1))
N0=75
pfs=np.zeros((2,len(eLes),samples,int(T/d),dim))

inputs=[]

for i in range(len(eLes)):
    L=eLes[i]
    eles=np.arange(l0,L+1,dtype=int)
    Nl=np.array( N0*2**((-3/4)*eles+5*L/4.),dtype=int)

    for sample in range(samples):
        #collection_input=[T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,eles[i],d,N,dim\
        #,resamp_coef,g_den,g_par,Lambda,Lamb_par]
        collection_input=[T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,eles,d,Nl,dim,\
        resamp_coef,phi,dim_out,g_den,g_par,Lambda,Lamb_par]
        





MLPF_cox.txt corresponds to 

    np.random.seed(1)
    T=10
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
    S=diags(np.random.normal(0.8,0.1,dim),0).toarray()
    #S=np.array([[1.]])
    fi=inv_mat@S@comp_matrix
    #B=diags(np.random.normal(-1,0.1,dim),0).toarray()
    B=diags(np.random.normal(-0.9,0.01,dim),0).toarray()*(2/3)
    B=inv_mat@B@comp_matrix
    #B=np.array([[-1.]])
    #print(B)
    #print(B)
    #B=comp_matrix-comp_matrix.T  +B 
    
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    cov=I*1e-1
    g_par=[dim,cov]
    x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=2
    
    [obs_time,obs]=gen_obs(x_true,l,T,Lambda,Lamb_par,g_normal,g_par)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8    
    dim_out=2
    g_par=cov
    samples=50
    Lmin=1
    Lmax=5
    l0=0
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eLes=np.array(range(Lmin,Lmax+1))
    N0=100
    
    pfs=np.zeros((2,len(eLes),samples,int(T/d),dim))
    
    inputs=[]

    for i in range(len(eLes)):
        L=eLes[i]
        eles=np.arange(l0,L+1,dtype=int)
        Nl=np.array( N0*2**((-3/4+1)*eles+L/4.),dtype=int)

        for sample in range(samples):
            #collection_input=[T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,eles[i],d,N,dim\
            #,resamp_coef,g_den,g_par,Lambda,Lamb_par]
            collection_input=[T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,eles,d,Nl,dim,\
            resamp_coef,phi,dim_out,g_den,g_par,Lambda,Lamb_par]
            
            
            inputs.append([sample,collection_input])        
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PMLPF_cox, inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eLes)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    #x_pfs=np.reshape(np.array(pool_outputs),(len(eles),samples,int(T/d),dim))
    
    #log_weightss=log_weightss.flatten()
    pfs=pfs.flatten()
    
    np.savetxt("Observations&data\MLPF_cox.txt",pfs,fmt="%f")




CCPF_v2.txt corresponds to 

    np.random.seed(1)
    T=10
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
    S=diags(np.random.normal(0.8,0.1,dim),0).toarray()
    #S=np.array([[1.]])
    fi=inv_mat@S@comp_matrix
    #B=diags(np.random.normal(-1,0.1,dim),0).toarray()
    B=diags(np.random.normal(-0.9,0.01,dim),0).toarray()*(2/3)
    B=inv_mat@B@comp_matrix
    #B=np.array([[-1.]])
    #print(B)
    #print(B)
    #B=comp_matrix-comp_matrix.T  +B 
    
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    cov=I*1e-1
    g_par=[dim,cov]
    x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=2
    
    [obs_time,obs]=gen_obs(x_true,l,T,Lambda,Lamb_par,g_normal,g_par)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8    
    samples=1000
    g_par=cov
    l0=1
    L=7
    N=1000
    eles=np.array(range(l0,L+1))
    x_pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    
    inputs=[]



CPF_v1.txt corresponds to 

np.random.seed(1)
    T=10
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=0
    collection_input=[]
    I=identity(dim).toarray()
    
    #comp_matrix = ortho_group.rvs(dim)
    comp_matrix=np.array([[1]])
    inv_mat=la.inv(comp_matrix)
    #S=diags(np.random.normal(1,0.1,dim),0).toarray()
    S=diags(np.random.normal(0.8,0.1,dim),0).toarray()
    
    #S=np.array([[1.]])
    fi=inv_mat@S@comp_matrix
    #B=diags(np.random.normal(-1,0.1,dim),0).toarray()
    B=diags(np.random.normal(-0.9,0.01,dim),0).toarray()*(2/3)
    B=inv_mat@B@comp_matrix
    #B=np.array([[-1.]])
    #print(B)
    #print(B)
    #B=comp_matrix-comp_matrix.T  +B 
    
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    cov=I*5*1e-1
    g_par=[dim,cov]
    #x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=2
    A=B
    R1=fi
    R2=la.sqrtm(2**(l)*g_par[1])
    H=I*2**(l)
    m0=xin
    C0=I*1e-40
    collection_input=[dim,dim,A,R1,R2,H,m0,C0]
    
    [z,v]=pff.gen_data(T,l,collection_input)
    kbf=np.reshape(pff.KBF(T,l,l,z,collection_input)[0],(T*2**l+1,dim))
    #parameters of the KF
    K=la.expm(2**(-l)*B)
    G=la.sqrtm((fi@fi)@la.inv(B+B.T)@(la.expm((B+B.T)*2**(-l))-I))
    #la.expm(2**(-l)*B)@
    H=I
    D=la.sqrtm(cov)
    
    v=np.reshape(v,(T*2**l+1,dim))
    x_true=v
    obs_time=np.array(range(T*2**l))*2**(-l)+2**(-l)
    obs=np.reshape(z[1:]-z[:-1],(T*2**l,dim))
    
    start=time.time()
    d=2**(0)
    resamp_coef=0.8    
    samples=1000
    g_par=cov
    l0=1
    L=7
    N=1000
    eles=np.array(range(l0,L+1))
    x_pfs=np.zeros((2,len(eles),samples,int(T/d),dim))



pf_1d_v4_1pc.txt corresponds to 

np.random.seed(6)
    T=10
    dim=1
    dim_o=dim
    xin=np.random.normal(0,5,dim)
    l=1
    collection_input=[]
    I=identity(dim).toarray()

    #comp_matrix = ortho_group.rvs(dim)
    comp_matrix=np.array([[1]])
    inv_mat=la.inv(comp_matrix)
    S=diags(np.random.normal(0.8,0.1,dim),0).toarray()
    fi=inv_mat@S@comp_matrix
    B=diags(np.random.normal(-0.9,0.1,dim),0).toarray()*(2/3)
    B=inv_mat@B@comp_matrix

    #print(B)
    #B=comp_matrix-comp_matrix.T  +B 

    collection_input=[dim, b_ou,B,Sig_ou,fi]
    cov=I*5*1e-1
    g_par=[dim,cov]
    #x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=2
    A=B
    R1=fi
    R2=la.sqrtm(2**(l)*g_par[1])
    H=I*2**(l)
    m0=xin
    C0=I*1e-20
    collection_input=[dim,dim,A,R1,R2,H,m0,C0]
    [z,v]=pff.gen_data(T,l,collection_input)
    kbf=np.reshape(pff.KBF(T,l,l,z,collection_input)[0],(T*2**l+1,dim))
    #parameters of the KF
    K=la.expm(2**(-l)*B)
    G=la.sqrtm((fi@fi)@la.inv(B+B.T)@(la.expm((B+B.T)*2**(-l))-I))
    

    H=I
    D=la.sqrtm(cov)

    #kf=KF(xin,dim,dim_o,K,G,H,D,obs)
    v=np.reshape(v,(T*2**l+1,dim))
    x_true=v
    obs_time=np.array(range(T*2**l))*2**(-l)+2**(-l)
    obs=np.reshape(z[1:]-z[:-1],(T*2**l,dim))
    start=time.time()
    d=2**(-1)
    resamp_coef=0.8    
    samples=100
    g_par=cov
    l0=1
    L=7
    N=1000
    eles=np.array(range(l0,L+1))
    x_pfs=np.zeros((len(eles),samples,int(T/d),dim))
    inputs=[]

pf_1d_v3_1pc.txt corresponds to 

np.random.seed(6)
    T=10
    dim=1
    dim_o=dim
    xin=np.random.normal(0,5,dim)
    l=1
    collection_input=[]
    I=identity(dim).toarray()

    #comp_matrix = ortho_group.rvs(dim)
    comp_matrix=np.array([[1]])
    inv_mat=la.inv(comp_matrix)
    S=diags(np.random.normal(3,0.1,dim),0).toarray()
    fi=inv_mat@S@comp_matrix
    B=diags(np.random.normal(-10,0.1,dim),0).toarray()*(2/3)
    B=inv_mat@B@comp_matrix

    #print(B)
    #B=comp_matrix-comp_matrix.T  +B 

    collection_input=[dim, b_ou,B,Sig_ou,fi]
    cov=I*5*1e-1
    g_par=[dim,cov]
    #x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=2
    A=B
    R1=fi
    R2=la.sqrtm(2**(l)*g_par[1])
    H=I*2**(l)
    m0=xin
    C0=I*1e-20
    collection_input=[dim,dim,A,R1,R2,H,m0,C0]
    [z,v]=pff.gen_data(T,l,collection_input)
    kbf=np.reshape(pff.KBF(T,l,l,z,collection_input)[0],(T*2**l+1,dim))
    #parameters of the KF
    K=la.expm(2**(-l)*B)
    G=la.sqrtm((fi@fi)@la.inv(B+B.T)@(la.expm((B+B.T)*2**(-l))-I))
    

    H=I
    D=la.sqrtm(cov)

    #kf=KF(xin,dim,dim_o,K,G,H,D,obs)
    v=np.reshape(v,(T*2**l+1,dim))
    x_true=v
    obs_time=np.array(range(T*2**l))*2**(-l)+2**(-l)
    obs=np.reshape(z[1:]-z[:-1],(T*2**l,dim))
    start=time.time()
    d=2**(-1)
    resamp_coef=0.8    
    samples=100
    g_par=cov
    l0=1
    L=7
    N=1000
    eles=np.array(range(l0,L+1))
    x_pfs=np.zeros((len(eles),samples,int(T/d),dim))
    inputs=[]


pf_1d_v2_1pc corresponds to the setting 

np.random.seed(6)
    T=10
    dim=1
    dim_o=dim
    xin=np.random.normal(0,5,dim)
    l=1
    collection_input=[]
    I=identity(dim).toarray()
    
    #comp_matrix = ortho_group.rvs(dim)
    comp_matrix=np.array([[1]])
    inv_mat=la.inv(comp_matrix)
    S=diags(np.random.normal(0.8,0.1,dim),0).toarray()
    fi=inv_mat@S@comp_matrix
    B=diags(np.random.normal(-0.9,0.01,dim),0).toarray()*(2/3)
    B=inv_mat@B@comp_matrix
    print(B)
    #print(B)
    #B=comp_matrix-comp_matrix.T  +B 
    
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    cov=I*5*1e-1
    g_par=[dim,cov]
    #x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=2
    A=B
    R1=fi
    R2=la.sqrtm(2**(l)*g_par[1])
    H=I*2**(l)
    m0=xin
    C0=I*1e-40
    collection_input=[dim,dim,A,R1,R2,H,m0,C0]
    [z,v]=pff.gen_data(T,l,collection_input)
    kbf=np.reshape(pff.KBF(T,l,l,z,collection_input)[0],(T*2**l+1,dim))
    #parameters of the KF
    K=la.expm(2**(-l)*B)
    G=la.sqrtm((fi@fi)@la.inv(B+B.T)@(la.expm((B+B.T)*2**(-l))-I))
    #la.expm(2**(-l)*B)@
    H=I
    D=la.sqrtm(cov)

    v=np.reshape(v,(T*2**l+1,dim))
    x_true=v
    obs_time=np.array(range(T*2**l))*2**(-l)+2**(-l)
    obs=np.reshape(z[1:]-z[:-1],(T*2**l,dim))
    start=time.time()
    d=2**(-1)
    resamp_coef=0.8    
    samples=200
    g_par=cov
    l0=1
    L=7
    N=5000
    eles=np.array(range(l0,L+1))
    x_pfs=np.zeros((len(eles),samples,int(T/d),dim))
    inputs=[] 



pf_1d_1pc.txt corresponds to the cox observations with the setting

 np.random.seed(6)
    T=10
    dim=1
    dim_o=dim
    xin=np.random.normal(0,5,dim)
    l=10
    collection_input=[]
    I=identity(dim).toarray()

    #comp_matrix = ortho_group.rvs(dim)
    comp_matrix=np.array([[1]])
    inv_mat=la.inv(comp_matrix)
    S=diags(np.random.normal(3,0.1,dim),0).toarray()
    fi=inv_mat@S@comp_matrix
    B=diags(np.random.normal(-10,0.1,dim),0).toarray()*(2/3)
    B=inv_mat@B@comp_matrix
    #print(B)
    #B=comp_matrix-comp_matrix.T  +B 
    
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    cov=I*5*1e-1
    
    g_par=[dim,cov]
    x_true=gen_gen_data(T,xin,l,collection_input)
    
    Lamb_par=1
    times=2**(-l)*np.array(range(int(T*2**l+1)))
    [obs_time, obs]=gen_obs(x_true,l,T,Lambda,Lamb_par,g_den,g_par)
    
    start=time.time()
    d=2**(-1)
    resamp_coef=0.8    
    samples=500
    g_par=cov
    l0=1
    L=7
    N=8000
    eles=np.array(range(l0,L+1))
    x_pfs=np.zeros((len(eles),samples,int(T/d),dim))
    inputs=[] 






pfs_shifted_d1.txt corresponds to the settings


np.random.seed(6)
    T=10
    dim=1
    dim_o=dim
    xin=np.random.normal(0,5,dim)
    l=1
    collection_input=[]
    I=identity(dim).toarray()

    #comp_matrix = ortho_group.rvs(dim)
    comp_matrix=np.array([[1]])
    inv_mat=la.inv(comp_matrix)
    S=diags(np.random.normal(3,0.1,dim),0).toarray()
    fi=inv_mat@S@comp_matrix
    B=diags(np.random.normal(-10,0.1,dim),0).toarray()*(2/3)
    B=inv_mat@B@comp_matrix

    #print(B)
    #B=comp_matrix-comp_matrix.T  +B 

    collection_input=[dim, b_ou,B,Sig_ou,fi]
    cov=I*5*1e-1
    g_par=[dim,cov]
    #x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=2
    A=B
    R1=fi
    R2=la.sqrtm(2**(l)*g_par[1])
    H=I*2**(l)
    m0=xin
    C0=I*1e-20
    collection_input=[dim,dim,A,R1,R2,H,m0,C0]
    [z,v]=pff.gen_data(T,l,collection_input)
    kbf=np.reshape(pff.KBF(T,l,l,z,collection_input)[0],(T*2**l+1,dim))
    #parameters of the KF
    K=la.expm(2**(-l)*B)
    G=la.sqrtm((fi@fi)@la.inv(B+B.T)@(la.expm((B+B.T)*2**(-l))-I))
    

    H=I
    D=la.sqrtm(cov)

    #kf=KF(xin,dim,dim_o,K,G,H,D,obs)
    v=np.reshape(v,(T*2**l+1,dim))
    x_true=v
    obs_time=np.array(range(T*2**l))*2**(-l)+2**(-l)
    obs=np.reshape(z[1:]-z[:-1],(T*2**l,dim))
    start=time.time()
    d=2**(-1)
    resamp_coef=0.8    
    samples=200
    g_par=cov
    l0=1
    L=7
    N=6000
    eles=np.array(range(l0,L+1))
    x_pfs=np.zeros((len(eles),samples,int(T/d),dim))
    inputs=[]







pfs_shifted2.txt corresponds to the following setting

 np.random.seed(6)
    T=10
    dim=2
    xin=np.random.normal(0,5,dim)
    l=1
    collection_input=[]
    I=identity(dim).toarray()
    
    comp_matrix = ortho_group.rvs(dim)
    inv_mat=la.inv(comp_matrix)
    S=diags(np.random.normal(3,0.1,dim),0).toarray()
    fi=inv_mat@S@comp_matrix
    B=diags(np.random.normal(-.4,0.1,dim),0).toarray()*(2/3)
    B=inv_mat@B@comp_matrix
    
    print(B)
    #B=comp_matrix-comp_matrix.T  +B 
        
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    cov=I*5*1e-1
    g_par=[dim,cov]
    #x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=2
    A=B
    R1=fi
    R2=2**(l)*la.sqrtm(g_par[1])
    H=I*2**(l)
    m0=xin
    C0=I*1e-20
    collection_input=[dim,dim,A,R1,R2,H,m0,C0]
    [z,v]=pff.gen_data(T,l,collection_input)
    kbf=np.reshape(pff.KBF(T,l,l,z,collection_input)[0],(T*2**l+1,dim))
    print(kbf.shape)
    v=np.reshape(v,(T*2**l+1,dim))
    x_true=v
    obs_time=np.array(range(T*2**l))*2**(-l)+2**(-l)
    obs=np.reshape(z[1:]-z[:-1],(T*2**l,dim))
    start=time.time()
    d=2**(-1)
    resamp_coef=0.8    
    samples=1000
    g_par=cov
    l0=1
    L=10
    N=4000
    eles=np.array(range(l0,L+1))
    x_pfs=np.zeros((len(eles),samples,int(T/d),dim))
    inputs=[]

pfs_shifted1.txt corresponds to the configuration

    np.random.seed(6)
    T=10    
    dim=2
    xin=np.random.normal(0,5,dim)
    l=1
    collection_input=[]
    I=identity(dim).toarray()
    comp_matrix = ortho_group.rvs(dim)
    inv_mat=la.inv(comp_matrix)
    S=diags(np.random.normal(.9,0.1,dim),0).toarray()
    fi=inv_mat@S@comp_matrix
    B=diags(np.random.normal(-.1,0.1,dim),0).toarray()*(2/3)
    B=inv_mat@B@comp_matrix
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    cov=I*2*1e-0
    g_par=[dim,cov]
    #x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=2
    A=B
    R1=fi
    R2=np.sqrt(2**(l))*g_par[1]
    H=I*2**(l)
    m0=xin
    C0=I*1e-20
    collection_input=[dim,dim,A,R1,R2,H,m0,C0]
    [z,v]=pff.gen_data(T,l,collection_input)
    kbf=np.reshape(pff.KBF(T,l,l,z,collection_input)[0],(T*2**l+1,dim))
    v=np.reshape(v,(T*2**l+1,dim))
    x_true=v
    obs_time=np.array(range(T*2**l))*2**(-l)+2**(-l)
    obs=np.reshape(z[1:]-z[:-1],(T*2**l,dim))
    start=time.time()
    d=2**(-1)
    resamp_coef=0.8    
    samples=100
    g_par=cov 
    l0=1
    L=10
    N0=3
    pmax=2
    N=4000
    eles=np.array(range(l0,L+1))
    x_pfs=np.zeros((len(eles),samples,int(T/d),dim))
    inputs=[]
    
    
    

pfs_shifted.txt corresponds to the configuration

    np.random.seed(6)
    T=10    
    dim=2
    xin=np.random.normal(0,5,dim)
    l=1
    collection_input=[]
    I=identity(dim).toarray()
    comp_matrix = ortho_group.rvs(dim)
    inv_mat=la.inv(comp_matrix)
    S=diags(np.random.normal(.9,0.1,dim),0).toarray()
    fi=inv_mat@S@comp_matrix
    B=diags(np.random.normal(-.1,0.1,dim),0).toarray()*(2/3)
    B=inv_mat@B@comp_matrix
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    cov=I*2*1e-0
    g_par=[dim,cov]
    #x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=2
    A=B
    R1=fi
    R2=np.sqrt(2**(l))*la.sqrtm( g_par[1])
    H=I*2**(l)
    m0=xin
    C0=I*1e-20
    collection_input=[dim,dim,A,R1,R2,H,m0,C0]
    [z,v]=pff.gen_data(T,l,collection_input)
    kbf=np.reshape(pff.KBF(T,l,l,z,collection_input)[0],(T*2**l+1,dim))
    v=np.reshape(v,(T*2**l+1,dim))
    x_true=v
    obs_time=np.array(range(T*2**l))*2**(-l)+2**(-l)
    obs=np.reshape(z[1:]-z[:-1],(T*2**l,dim))
    start=time.time()
    d=2**(-1)
    resamp_coef=0.8    
    samples=50
    g_par=cov 
    l0=1
    L=10
    N0=3
    pmax=2
    N=2000
    eles=np.array(range(l0,L+1))
    x_pfs=np.zeros((len(eles),samples,int(T/d),dim))
    inputs=[]


pfs_beta.txt corresponds to the setting 

start=time.time()
    samples=150
    np.random.seed(6)
    T=10
    dim=2
    xin=np.random.normal(0,5,dim)
    I=identity(dim).toarray()
    comp_matrix = ortho_group.rvs(dim)
    inv_mat=la.inv(comp_matrix)
    S=diags(np.random.normal(.9,0.1,dim),0).toarray()
    fi=inv_mat@S@comp_matrix
    B=diags(np.random.normal(-.1,0.1,dim),0).toarray()*(2/3)
    B=inv_mat@B@comp_matrix
    #collection_input=[dim, b_ou,B,Sig_ou,fi]
    cov=I*1e-0
    g_par=cov
    #x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=1/4
    
    start1=time.time()
    obs=np.loadtxt("obs_T10dim2l10s6.txt", dtype="float")
    obs_time=np.loadtxt("obs_time_T10dim2l10s6.txt", dtype="float")
    #l=7
    d=1./2.**2
    #N=8000
    resamp_coef=0.8
    dim_out=dim
    axis_action=0
    #g_par=cov 
    l0=3
    L=8
    epsilon=2*1e-1
    N0=3
    pmax=8
    enes=N0*np.array([2**i for i in range(pmax+1)])
    eles=np.array(range(l0,L+1))
    Nl=np.array(epsilon**(-2)*2**(-eles*(1/4))*2**(eles[-1]*(3/4)),dtype=int)
    print(eles,Nl)
    #log_weightss=np.zeros((len(eles),p[i],1max+1,samples,int(T/d),N))
    pfs=np.zeros((2,samples,int(T/d),dim))
    #x_pfs=np.zeros((2,pmax+1,len(eles),samples,int(T/d),dim))



 x_pfs_grid1.txt corresponds to the system 
 We compute this cojointly with x_pfs_grid.txt in order to create 
 samples with low variance.


samples=50
    np.random.seed(6)
    T=10
    dim=2
    xin=np.random.normal(0,5,dim)
    I=identity(dim).toarray()
    comp_matrix = ortho_group.rvs(dim)
    inv_mat=la.inv(comp_matrix)
    S=diags(np.random.normal(.9,0.1,dim),0).toarray()
    fi=inv_mat@S@comp_matrix
    B=diags(np.random.normal(-.1,0.1,dim),0).toarray()*(2/3)
    B=inv_mat@B@comp_matrix
    #collection_input=[dim, b_ou,B,Sig_ou,fi]
    cov=I*1e-0
    g_par=cov
    #x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=1/4
    
    start1=time.time()
    obs=np.loadtxt("obs_T10dim2l10s6.txt", dtype="float")
    obs_time=np.loadtxt("obs_time_T10dim2l10s6.txt", dtype="float")
    #l=7
    d=1./2.**2
    #N=8000
    resamp_coef=0.8
    dim_out=dim
    axis_action=0
    #g_par=cov 
    l0=3
    L=11
    N0=3
    pmax=8
    enes=N0*np.array([2**i for i in range(pmax+1)])
    eles=np.array(range(l0,L+1))
    #log_weightss=np.zeros((len(eles),pmax+1,samples,int(T/d),N))
    x_pfs=np.zeros((2,pmax+1,len(eles),samples,int(T/d),dim))



 x_pfs_grid.txt corresponds to the system 


samples=50
    np.random.seed(6)
    T=10
    dim=2
    xin=np.random.normal(0,5,dim)
    I=identity(dim).toarray()
    comp_matrix = ortho_group.rvs(dim)
    inv_mat=la.inv(comp_matrix)
    S=diags(np.random.normal(.9,0.1,dim),0).toarray()
    fi=inv_mat@S@comp_matrix
    B=diags(np.random.normal(-.1,0.1,dim),0).toarray()*(2/3)
    B=inv_mat@B@comp_matrix
    #collection_input=[dim, b_ou,B,Sig_ou,fi]
    cov=I*1e-0
    g_par=cov
    #x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=1/4
    
    start1=time.time()
    obs=np.loadtxt("obs_T10dim2l10s6.txt", dtype="float")
    obs_time=np.loadtxt("obs_time_T10dim2l10s6.txt", dtype="float")
    #l=7
    d=1./2.**2
    #N=8000
    resamp_coef=0.8
    dim_out=dim
    axis_action=0
    #g_par=cov 
    l0=3
    L=11
    N0=3
    pmax=8
    enes=N0*np.array([2**i for i in range(pmax+1)])
    eles=np.array(range(l0,L+1))
    #log_weightss=np.zeros((len(eles),pmax+1,samples,int(T/d),N))
    x_pfs=np.zeros((2,pmax+1,len(eles),samples,int(T/d),dim))

log_weightss_bias.txt and x_pfs_bias.txt corresponds to the system 
start=time.time()
    samples=20
    np.random.seed(6)
    T=10
    dim=2
    xin=np.random.normal(0,5,dim)
    I=identity(dim).toarray()
    comp_matrix = ortho_group.rvs(dim)
    inv_mat=la.inv(comp_matrix)
    S=diags(np.random.normal(.9,0.1,dim),0).toarray()
    fi=inv_mat@S@comp_matrix
    B=diags(np.random.normal(-.1,0.1,dim),0).toarray()*(2/3)
    B=inv_mat@B@comp_matrix
    #collection_input=[dim, b_ou,B,Sig_ou,fi]
    cov=I*1e-0
    g_par=cov
    #x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=1/4
    
    start1=time.time()
    obs=np.loadtxt("obs_T10dim2l10s6.txt", dtype="float")
    obs_time=np.loadtxt("obs_time_T10dim2l10s6.txt", dtype="float")
    l=7
    d=1./2.**2
    N=8000
    resamp_coef=0.8
    dim_out=dim
    axis_action=0
    g_par=cov 
    l0=3
    L=11
    eles=np.array(range(l0,L+1))
    log_weightss=np.zeros((len(eles),samples,int(T/d),N))
    x_pfs=np.zeros((len(eles),samples,int(T/d),N,dim))


    
log_weightss_bias1.txt and x_pfs_bias1.txt corresponds to the system 
start=time.time()
    samples=50
    np.random.seed(6)
    T=10
    dim=2
    xin=np.random.normal(0,5,dim)
    I=identity(dim).toarray()
    comp_matrix = ortho_group.rvs(dim)
    inv_mat=la.inv(comp_matrix)
    S=diags(np.random.normal(.9,0.1,dim),0).toarray()
    fi=inv_mat@S@comp_matrix
    B=diags(np.random.normal(-.1,0.1,dim),0).toarray()*(2/3)
    B=inv_mat@B@comp_matrix
    #collection_input=[dim, b_ou,B,Sig_ou,fi]
    cov=I*1e-0
    g_par=cov
    #x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=1/4
    
    start1=time.time()
    obs=np.loadtxt("obs_T10dim2l10s6.txt", dtype="float")
    obs_time=np.loadtxt("obs_time_T10dim2l10s6.txt", dtype="float")
    l=7
    d=1./2.**2
    N=8000
    resamp_coef=0.8
    dim_out=dim
    axis_action=0
    g_par=cov 
    l0=3
    L=13
    eles=np.array(range(l0,L+1))
    log_weightss=np.zeros((len(eles),samples,int(T/d),N))
    x_pfs=np.zeros((len(eles),samples,int(T/d),N,dim))    

    
"""