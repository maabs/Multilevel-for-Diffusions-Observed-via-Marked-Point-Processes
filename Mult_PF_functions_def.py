#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 17:29:35 2022

@author: alvarem
"""


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
from sklearn.linear_model import LinearRegression
from scipy.stats import ortho_group
import time
import PF_functions_def as pff

#%%

def M_coup(xin0,xin1,b,A,Sig,fi,l,d,N,dim):
    # This is the function for the transition Kernel M(x,du): R^{d_x}->P(E_l)
    # ARGUMENTS: the argument of the Kernel xin0 and xin1 corresponding
    # to the initial conditions for the process 0 and 1 respectively
    # where the process 0 is the one with step size 2^{-l+1} and the process 1 
    # is the one with time discretization 2^{-l}, both xin0 and xin1 are rank two
            #  arrays with \in NxR^{d_x}, although they can be rank 1 with dimension
    # d_x=d(for the initial condition of the whole process)
    # the drift and diffusion b, and Sig respectively (rank 1 and 2 numpy arrays of dim=d_x respectively)
    # the level of discretization l, in this case l is the 
    # larger level of discretization, i.e., the time step of the other 
    # process is 2^{l-1}, the distance of resampling, the number of
    # particles N, and the dimension of the problem dim=d_x
    # OUTCOMES: x0 and x1 are arrays of rank 3 with dimension 2**(l-1)*d,N,dim 
    # and  2**l*d,N,dim respectively, these arrays represents the paths simulated
    # along the discretized time for a number of particles N.
    steps0=int(2**(l-1)*d)
    steps1=int(2**(l)*d)
    dt1=1./2**l
    dt0=2./2**l
    x1=np.zeros((steps1+1,N,dim))
    x0=np.zeros((steps0+1,N,dim))
    x0[0]=xin0
    x1[0]=xin1
    
    I=identity(dim).toarray()
    dW=np.zeros((2,N,dim))

    for t0 in range(steps0):
        for s in range(2):
            dW[s]=np.random.multivariate_normal(np.zeros(dim),I,N)*np.sqrt(dt1)
            
            #diff=np.einsum("nd,njd->nj",dW[s],Sig(x1[2*t0+s],fi))
            #x1[2*t0+s+1]=x1[2*t0+s]+b(x1[2*t0+s],A)*dt1+diff
            x1[2*t0+s+1]=x1[2*t0+s]+b(x1[2*t0+s],A)*dt1+ dW[s]@(Sig(x1[2*t0+s],fi).T)
            
        #diff=np.einsum("nd,njd->nj",dW[0]+dW[1],Sig(x0[t0],fi))
        x0[t0+1]=x0[t0]+b(x0[t0],A)*dt0+ (dW[0]+dW[1])@(Sig(x0[t0],fi).T)
        #x0[t0+1]=x0[t0]+b(x0[t0],A)*dt0+ diff
    return [x0,x1]


def b_ou(x,A):
    # Returns the drift "vector" evaluated at x
    # ARGUMENTS: x is a rank two array with dimensions where the first dimension 
    # corresponds to the number of particles and the second to the dimension
    # of the probelm. The second argument is A, which is a squared rank 2 array 
    # with dimension of the dimesion of the problem
    
    # OUTPUTS: A rank 2 array where the first dimension corresponds to the number
    # of particles and the second to the dimension of the system.
        
        mult=x@(A.T)
        #mult=np.array(mult)*10
        return mult

#dim=len(x.T)

def Sig_ou(x,fi):
    # Returns the Ornstein-Oulenbeck diffusion matrix 
        
        return fi

#x=np.array([[1,0],[0,1],[10,10]])




def b_gbm(x,mu):
    # Returns the drift "vector" evaluated at x
    # ARGUMENTS: x is a rank two array with dimensions where the first dimension 
    # corresponds to the number of particles and the second to the dimension
    # of the probelm. The second argument is mu, which is arank 1 array 
    # with dimension of the dimesion of the problem
    
    # OUTPUTS: A rank 2 array where the first dimension corresponds to the number
    # of particles and the second to the dimension of the system.
        #mu=reshape(mu,(-1,1))
        mult=x*mu
        #mult=np.array(mult)*10
        return mult
    
    
    
def Sig_gbm(x,fi):
    # Returns the drift "vector" evaluated at x
    # ARGUMENTS: x is a rank two array with dimensions where the first dimension 
    # corresponds to the number of particles and the second to the dimension
    # of the probelm. The second argument is fi, which is composed of a vector
    # sigs with dimension dim and a square matrix Sig with the dimension of the 
    # system ank 1 array 
    
    # OUTPUTS: A rank 2 array where the first dimension corresponds to the number
    # of particles and the second to the dimension of the system.
    [sigs,Sig]=fi
    if x.ndim==1:
        Sig_m=((sigs*x)*Sig.T).T
    else:
        Sig_m=np.einsum("ij,ni->nij",Sig,sigs*x)
    return Sig_m

#%%
"""
x=np.array([[0,2],[4,3],[5,7]])
mu=np.array([1,2])
sigs=mu
Sig=np.array([[1,0],[0,3]])
fi=[sigs,Sig]
print(Sig_gbm(x,fi))
print(np.reshape(mu,(-1,1)))
"""
    
#%%
"""
#test of M_coup
l=0
d=20

N=10
dim=10

xin0=np.random.normal(1,1,dim)
xin1=xin0
np.random.seed(3)
comp_matrix = ortho_group.rvs(dim)
print(comp_matrix)
inv_mat=la.inv(comp_matrix)
S=diags(np.random.normal(1,0.1,dim),0).toarray()
fi=inv_mat@S@comp_matrix

B=diags(np.random.normal(-1,0.1,dim),0).toarray()*(2/3)
B=inv_mat@B@comp_matrix
#B=comp_matrix-comp_matrix.T  +B 
np.random.seed(3)

x3=M_coup(xin0,xin1,b_ou,B,Sig_ou,fi,l,d,N,dim)

"""
#%%
"""
# plot of the test of M_coup
print(x3[0][-1])
steps0=int(2**(l-1)*d)
steps1=int(2**(l)*d)
time0=np.array(range(steps0+1))/2**(l-1)
time1=np.array(range(steps1+1))/2**l

a=0
plt.plot(time0,x3[0][:,0,a])

plt.plot(time1,x3[1][:,0,a])
"""
#%%


def gen_gen_data(T,x0,l,collection_input):
    # parameters [dim,dim_o, b_ou,A,Sig_ou,fi,ht,H]
    
    [dim,dim_o, b,A,Sig,fi,ht,H]=collection_input
    
    J=T*(2**l)
    I=identity(dim).toarray()
    I_o=identity(dim_o).toarray()

    tau=2**(-l)
    
    v=np.zeros((J+1,dim))
    z=np.zeros((J+1,dim_o))
    #v[0]=np.random.multivariate_normal(m0,C0,(1)).T
    v[0]=x0
    z[0]=np.zeros(dim_o)


    for j in range(J):
        ## truth
        #print(np.shape(Sig(v[j],fi)),np.shape(b(v[j],A)))
        v[j+1] = v[j]+b(v[j],A)*tau + np.sqrt(tau)*(np.random.multivariate_normal(np.zeros(dim),I))@(Sig(v[j],fi).T)
        ## observation
        z[j+1] = z[j] +(v[j+1]@H.T)*tau + np.sqrt(tau)*np.random.multivariate_normal(np.zeros(dim_o),I_o)
        
    return [z,v]


#%%

def gen_data(T,l,collection_input):
    [dim,dim_o,A,R1,R2,H,m0,C0]=collection_input
    J=T*(2**l)
    I=identity(dim).toarray()
    tau=2**(-l)
    L=la.expm(A*tau)
    ## We are going to need W to be symmetric! 
    W=(R1@R1)@(la.inv(A+A.T)@(L@(L.T)-I))
    W=(W+W.T)/2.
    C=tau*H
    V=(R2@R2)*tau

    v=np.zeros((J+1,dim,1))
    z=np.zeros((J+1,dim_o,1))
    #v[0]=np.random.multivariate_normal(m0,C0,(1)).T
    v[0]=np.random.multivariate_normal(m0,C0,(1)).T
    z[0]=np.zeros((dim_o,1))


    for j in range(J):
        ## truth
        v[j+1] = L@v[j] + np.random.multivariate_normal(np.zeros(dim),W,(1)).T
        ## observation
        z[j+1] = z[j] + C@v[j+1] + np.random.multivariate_normal(np.zeros(dim_o),V,(1)).T
        
    return([z,v])



def cut(T,lmax,l,v):
    ind = np.arange(T*2**l+1)
    rtau = 2**(lmax-l)
    w = v[ind*rtau]
    return(w)

def KBF(T,l,lmax,z,collection_input):
    
    [dim,dim_o,A,R1,R2,H,m0,C0]=collection_input
    J=T*(2**l)
    I=identity(dim).toarray()
    tau=2**(-l)
    L=la.expm(A*tau)
    W=(R1@R1)@(la.inv(A+A.T)@(L@(L.T)-I))
    W=(W+W.T)/2.
    
    ## C: dim_o*dim matrix
    C=tau*H
    V=(R2@R2)*tau
    
    z=cut(T,lmax,l,z)
    m=np.zeros((J+1,dim,1))
    c=np.zeros((J+1,dim,dim))
    m[0]=np.array([m0]).T
    c[0]=C0
    
    for j in range(J):
       
        ## prediction mean-dim*1 vector
        mhat=L@m[j]
        ## prediction covariance-dim*dim matrix
        chat=L@c[j]@(L.T)+W
        ## innovation-dim_o*1 vector
        d=(z[j+1]-z[j])-C@mhat
        ## Kalman gain-dim*dim_o vector
        K=(chat@(C.T))@la.inv(C@chat@(C.T)+V)
        ## update mean-dim*1 vector
        
        m[j+1]=mhat+K@d
        ## update covariance-dim*dim matrix
        c[j+1]=(I-K@C)@chat
    return([m,c])


#%%


def ht(x,H, para=True):
    #This function takes as argument a rank 3 array where for each element(2 rank array)
    # x[i]  the function applies x[i]@H.T
    #ARGUMENTS: rank 3 array x, para=True computes the code with the einsum 
    #function from numpy. Otherwise the function is computed with a for
    #in the time discretization
    #OUTPUTS: rank 3 array h
    
    
    if para==True:
        h=np.einsum("ij,tkj->tki",H,x)
    else:
        h=np.zeros(x.shape)
        for i in range(len(x)):
            h[i]=x[i]@(H.T)
    return h
            
            
    

def G(z,x,ht,H,l,para=True):
    #This function emulates the Radon-Nykodim derivative of the Girsanov formula
    #ARGUMENTS: z are the observations(2 rank array) , x is the array of the 
    #particles (rank 3 array, with 1 dimension less in the time discretization), 
    #ht is the function that computes the h(x) and d is the distance in which we
    #compute the paths.
    #OUTPUT: logarithm of the weights    
    h=ht(x,H,para=para)
    delta_z=z[1:]-z[:-1]
    delta_l=1./2**l
    suma1=np.einsum("tnd,td->n",h,delta_z)

    suma2=-(1/2.)*delta_l*np.einsum("tnj,tnj->n",h,h)
    #print(suma1,suma2)
    log_w=suma1+suma2
   
    return log_w


#%%



def sr(W,N,x,dim):
    # This function does 2 things, given probability weights (normalized)
    # it uses systematic resampling, and constructs the set of resampled 
    # particles.
    # ARGUMENTS: W: Normalized weights with dimension N (number of particles)
    # x: rank 2 array of the positions of the N particles, its dimesion is
    # Nxdim, where dim is the dimension of the problem.
    # OUTPUTs: part: is a N dimentional array where its value in the ith position
    # represents the number of times that particle was sampled.
    # x_new: is the new set of resampled particles.
    
    Wsum=np.cumsum(W)
    
    U=np.random.uniform(0,1./N)
    part=np.zeros(N,dtype=int)
    part[0]=np.floor(N*(Wsum[0]-U)+1)
    k=part[0]
    #print(U,part[0])
    for i in range(1,N):
        j=np.floor(N*(Wsum[i]-U)+1)
        part[i]=j-k
        k=j
    x_new=np.zeros((N,dim))
    k=0
    part=np.array(part,dtype=int)
    originals=np.nonzero(part)[0]
    #the resampled particles go to its original distribution
    x_new[originals]=x[originals]

    new_part=np.maximum(np.zeros(N,dtype=int),part-1,dtype=int)
    
    for i in range(N):
        Ni=new_part[i]
        while Ni>0:
            if k in originals:
                k+=1
        
            else:
                x_new[k]=x[i]
                k+=1
                Ni-=1
    
    return [part, x_new]


def sr_coup(W,N,x0,x1,dim):
    # This function does 2 things, given probability weights (normalized)
    # it uses systematic resampling, and constructs the set of resampled 
    # particles with the same particles for .
    # ARGUMENTS: W: Normalized weights with dimension N (number of particles)
    # x0,x1: rank 2 arrays of the positions of the N particles, its dimesion is
    # Nxdim, where dim is the dimension of the problem.
    # OUTPUTs: part: is a N dimentional array where its value in the ith position
    # represents the number of times that particle was sampled.
    # x_new: is the new set of resampled particles.
    
    Wsum=np.cumsum(W)
    #np.random.seed()
    U=np.random.uniform(0,1./N)
    part=np.zeros(N,dtype=int)
    part[0]=np.floor(N*(Wsum[0]-U)+1)
    k=part[0]
    #print(U,part[0])
    for i in range(1,N):
        j=np.floor(N*(Wsum[i]-U)+1)
        part[i]=j-k
        k=j
    x0_new=np.zeros((N,dim))
    x1_new=np.zeros((N,dim))
    k=0 
    for i in range(N):
        for j in range(part[i]):
            x0_new[k]=x0[i]
            x1_new[k]=x1[i]
            k+=1
    return [part, x0_new,x1_new]
        


def sr_coup2(w0,w1,N,seed_val,x0,x1,dim):
    # This function does 2 things, given probability weights (normalized)
    # it uses systematic resampling, and constructs the set of resampled 
    # particles with the same particles for .
    # ARGUMENTS: W: Normalized weights with dimension N (number of particles)
    # x0,x1: rank 2 arrays of the positions of the N particles, its dimesion is
    # Nxdim, where dim is the dimension of the problem.
    # OUTPUTs: part: is a N dimentional array where its value in the ith position
    # represents the number of times that particle was sampled.
    # x_new: is the new set of resampled particles.
    
    wmin=np.minimum(w0, w1)
    r=np.sum(wmin)
    wmin_den=wmin/r
    

    w4_den=(w0-wmin)/(1-r)
    w5_den=(w1-wmin)/(1-r)
    w4sum=np.cumsum(w4_den)
    w5sum=np.cumsum(w5_den)
    np.random.seed(seed_val)
    U4=np.random.uniform(0,1./N)
    U5=np.random.uniform(0,1./N)
    part4=np.zeros(N,dtype=int)
    part4[0]=np.floor(N*(w4sum[0]-U4)+1)
    k4=part4[0]
    part5=np.zeros(N,dtype=int)
    part5[0]=np.floor(N*(w5sum[0]-U5)+1)
    k5=part5[0]
    for i in range(1,N):
        j4=np.floor(N*(w4sum[i]-U4)+1)
        part4[i]=j4-k4
        k4=j4
        j5=np.floor(N*(w5sum[i]-U5)+1)
        part5[i]=j5-k5
        k5=j5
        
    x0_new=np.zeros((N,dim))
    x1_new=np.zeros((N,dim))
        
    min_part_common=np.minimum(part4,part5,dtype=int)
    originals=np.nonzero(min_part_common)[0]
    k=0
    for j in originals:
        for i in min_part_common[j]:
            x0_new[k]=x0[j]
            x1_new[k]=x1[j]
            k+=1
    rem_4=part4-min_part_common

    rem_5=part5-min_part_common            
    #print(part4,part5)    
    rem_4_pos=np.nonzero(rem_4)[0]
    rem_5_pos=np.nonzero(rem_5)[0]
    
    k4=k
    #print(k4)
    for j in rem_4_pos:
        for i in rem_4[j]:
            x0_new[k4]=x0[j]
            
            k4+=1
    k5=k     
    for j in rem_5_pos:
        for i in rem_5[j]:
            x1_new[k5]=x1[j]
            
            k5+=1
    
    
    return [part4,part5, x0_new,x1_new]

def max_coup_sr(w0,w1,N,x0,x1,dim):
    # This function does 2 things, given probability weights (normalized)
    # it uses systematic resampling, and  constructs the set of resampled 
    # particles.
    # ARGUMENTS: W: Normalized weights with dimension N (number of particles)
    # x: rank 2 array of the positions of the N particles, its dimesion is
    # Nxdim, where dim is the dimension of the problem.
    # OUTPUTs: part: is a N dimentional array where its value in the ith position
    # represents the number of times that particle was sampled.
    # x_new: is the new set of resampled particles.
    wmin=np.minimum(w0, w1)
    r=np.sum(wmin)
    wmin_den=wmin/r
    #np.random.seed(seed_val)
    U=np.random.uniform(0,1)
    
    if U<r:
        [part,x0_new,x1_new]=sr_coup(wmin_den,N,x0,x1,dim)
        part0=part
        part1=part
    else:
        w4_den=(w0-wmin)/(1-r)
        [part0,x0_new]=sr(w4_den,N,x0,dim)
        w5_den=(w1-wmin)/(1-r)
        [part1,x1_new]=sr(w5_den,N,x1,dim)
    return [part0,part1,x0_new,x1_new]


def max_coup_multi(w0,w1,N,x0,x1,dim):
    # This function does 2 things, given probability weights (normalized)
    # it uses systematic resampling, and  constructs the set of resampled 
    # particles.
    # ARGUMENTS: W: Normalized weights with dimension N (number of particles)
    # x: rank 2 array of the positions of the N particles, its dimesion is
    # Nxdim, where dim is the dimension of the problem.
    # OUTPUTs: part: is a N dimentional array where its value in the ith position
    # represents the number of times that particle was sampled.
    # x_new: is the new set of resampled particles.
    wmin=np.minimum(w0, w1)
    r=np.sum(wmin)
    wmin_den=wmin/r
    #np.random.seed(seed_val)
    U=np.random.uniform(0,1)
    
    if U<r:
        [part,x0_new,x1_new]=multi_samp_coup(wmin_den,N,x0,x1,dim)
        part0=part
        part1=part
    else:
        w4_den=(w0-wmin)/(1-r)
        [part0,x0_new]=multi_samp(w4_den,N,x0,dim)
        w5_den=(w1-wmin)/(1-r)
        [part1,x1_new]=multi_samp(w5_den,N,x1,dim)
    return [part0,part1,x0_new,x1_new]

        
def multi_samp(W,N,x,dim): #from multinomial sampling
    # This function does 2 things, given probability weights (normalized)
    # it uses multinomial resampling, and constructs the set of resampled 
    # particles.
    # ARGUMENTS: W: Normalized weights with dimension N (number of particles)
    # x: rank 2 array of the positions of the N particles, its dimesion is
    # Nxdim, where dim is the dimension of the problem.
    # OUTPUTs: part: is a N dimentional array where its value in the ith position
    # represents the number of times that particle was sampled.
    # x_new: is the new set of resampled particles.
    
    part_samp=np.random.choice(N,size=N,p=W,replace=True) #particles resampled 
    #print(part_samp)
    x_resamp=x[part_samp]
    return [part_samp+1,x_resamp] #here we add 1 bc it is par_lab are thought 
    # as python labels, meaning that they start with 0.
 
def multi_samp_coup(W,N,x0,x1,dim): #from multinomial sampling
    # This function does 2 things, given probability weights (normalized)
    # it uses multinomial resampling, and constructs the set of resampled 
    # particles.
    # ARGUMENTS: W: Normalized weights with dimension N (number of particles)
    # x: rank 2 array of the positions of the N particles, its dimesion is
    # Nxdim, where dim is the dimension of the problem.
    # OUTPUTs: part: is a N dimentional array where its value in the ith position
    # represents the number of times that particle was sampled.
    # x_new: is the new set of resampled particles.
    
    part_samp=np.random.choice(N,size=N,p=W) #particles resampled 
    #print(part_samp)
    x0_new=x0[part_samp]
    x1_new=x1[part_samp]
    #return [part, x0_new,x1_new]
    return [part_samp+1,x0_new,x1_new] #here we add 1 bc it is par_lab are thought 
    # as python labels, meaning that they start with 0.
 
             
        
    
   
def norm_logweights(lw,ax=0):
    # returns the normalized weights given the log of the normalized weights 
    #ARGUMENTS: lw is an arbitrary=ar rank array with weigts along the the axis ax 
    #OUTPUT: w a rank ar array of the same dimesion of lw 
    m=np.max(lw,axis=ax,keepdims=True)
    wsum=np.sum(np.exp(lw-m),axis=ax,keepdims=True)
    w=np.exp(lw-m)/wsum
    return w




#%%
#test for the max_coup
"""
B=5
N=5
dim=2

enes0=np.zeros((B,N))
enes1=np.zeros((B,N))
enes2=np.zeros((B,N))
xs0=np.zeros((B,N,dim))
xs1=np.zeros((B,N,dim))
xs2=np.zeros((B,N,dim))


x0=np.array([[1,0],[2,0],[3,0],[4,0],[5,0]])
x1=np.array([[1,1],[2,1],[3,1],[4,1],[5,1]])
w0=np.array([0.1,0.2,0.3,0.3,0.1])
w1=np.array([0.2,0.25,0.15,0.25,0.15])

for i in range(B):
    [enes0[i],enes1[i],xs0[i],xs1[i]]=max_coup_sr(w0,w1,N,i,x0,x1,dim)
    #[enes2[i],xs2[i]]=sr_coup2(w0,w1,N,i+10,x0,x1,dim)
    
print([enes0,enes1,xs0,xs1])    
#print([enes0,xs0,enes1,xs1])
print(np.mean(enes0,axis=0)/3,np.mean(enes1,axis=0)/3,np.mean(enes2,axis=0)/3)
    
#print(np.concatenate((enes0,enes1,enes2),axis=1))
#%%
B=np.zeros((3,2))
A=np.array([[0,1],[0,2]])
B[[2,1,0]]=A[[1,1,0]]
print(B)

A=np.array([0,2,3,0,0])
B=np.array([1,1,1,1,1])
#%%
print(np.array(np.nonzero(A)))
print(B)
"""
#%%

def CPF(T,z,lmax,xin,b_ou,A,Sig_ou,fi,ht,H,l,d,N,dim,resamp_coef,para=True):
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
    # resamp_coef: coeffient conditions on whether to resample or not.
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
    # x_pf: positions of the particles after resampling, it is a rank 3 array 
    # with dimensions (int(T/d),N,dim)
    
    x0=np.zeros((2**(l-1)*T+1,N,dim))
    x1=np.zeros((2**l*T+1,N,dim))
    x0_pf=np.zeros((int(T/d),N,dim))
    x1_pf=np.zeros((int(T/d),N,dim))
    z1=cut(T,lmax,l,z)
    z0=cut(T,lmax,l-1,z)
    log_weights0=np.zeros((int(T/d),N))
    log_weights1=np.zeros((int(T/d),N))                                                        
    x0_new=xin
    x1_new=xin
    x0[0]=xin
    x1[0]=xin
    d_steps0=int(d*2**(l-1))
    d_steps1=int(d*2**l)
    for i in range(int(T/d)):
        [x0[i*d_steps0:(i+1)*d_steps0+1],x1[i*d_steps1:(i+1)*d_steps1+1]]\
        =M_coup(x0_new,x1_new,b_ou,A,Sig_ou,fi,l,d,N,dim)
        xi0=x0[i*d_steps0:(i+1)*d_steps0]
        xi1=x1[i*d_steps1:(i+1)*d_steps1]
        zi0=z0[i*d_steps0:(i+1)*d_steps0+1]
        zi1=z1[i*d_steps1:(i+1)*d_steps1+1]
        log_weights0[i]=G(zi0,xi0,ht,H,l-1,para=True)
        log_weights1[i]=G(zi1,xi1,ht,H,l,para=True)
        w0=norm_logweights(log_weights0[i],ax=0)
        w1=norm_logweights(log_weights1[i],ax=0)
        #seed_val=i
        #print(xi0,xi1)
        x0_pf[i]=xi0[-1]
        x1_pf[i]=xi1[-1]
        
   
        ESS=np.min(np.array([1/np.sum(w0**2),1/np.sum(w1**2)]))
        if ESS<resamp_coef*N:
            #[part0,part1,x0_new,x1_new]=max_coup_sr(w0,w1,N,xi0[-1],xi1[-1],dim)
            [part0,part1,x0_new,x1_new]=max_coup_multi(w0,w1,N,xi0[-1],xi1[-1],dim)
        else:
            x0_new=xi0[-1]
            x1_new=xi1[-1]
        
        
        #print(weights.shape)
    #Filter
    spots0=np.arange(d_steps0,2**(l-1)*T+1,d_steps0,dtype=int)
    spots1=np.arange(d_steps1,2**(l)*T+1,d_steps1,dtype=int)
    
    x_pf0=x0[spots0]
    x_pf1=x1[spots1]
    weights0=norm_logweights(log_weights0,ax=1)
    weights1=norm_logweights(log_weights1,ax=1)
    #print(x_pf.shape,weights.shape)
    suma0=np.sum(x_pf0[:,:,1]*weights0,axis=1)
    suma1=np.sum(x_pf1[:,:,1]*weights1,axis=1)
    
    
    
    return [x0,x1,log_weights0,log_weights1,x0_pf,x1_pf]

#%%
"""
l=9
lmax=l
d=1./2**7
N=500
T=10
dim=3
dim_o=dim
resamp_coef=0.8
#x=M(x0,b_ou,Sig_ou,l,d,N,dim)
I=identity(dim).toarray()
I_o=identity(dim_o).toarray()
#R2=(identity(dim_o).toarray() + np.tri(dim_o,dim_o,1) - np.tri(dim_o,dim_o,-2))/20
R2=I
np.random.seed(1)
H=rand(dim_o,dim,density=0.75).toarray()/1e-2

xin=np.random.normal(1,0,dim)
np.random.seed(3)
comp_matrix = ortho_group.rvs(dim)
inv_mat=la.inv(comp_matrix)
S=diags(np.random.normal(1,0.001,dim),0).toarray()
fi=inv_mat@S@comp_matrix

B=diags(np.random.normal(-.1,0.001,dim),0).toarray()
A=inv_mat@B@comp_matrix

#A=b_ou(I,B).T
R1=Sig_ou(np.zeros(dim),fi)

np.random.seed(3)
C0=I*1e-6
m0=np.random.multivariate_normal(xin,C0)
collection_input=[dim,dim_o,A,R1,R2,H,m0,C0]

np.random.seed(2)

[z,x_true]=gen_data(T,l,collection_input)
z=np.reshape(z,z.shape[:2])
np.random.seed(2)


[x0,x1,log_weights0,log_weights1,suma0,suma1]= CPF(T,z,lmax,xin,b_ou,A,Sig_ou,fi,ht,H,l,d,N,dim,resamp_coef,para=True)
"""
#%%


"""
l=10
lmax=l
d=1./2**7
N=50
T=1
dim=2
dim_o=dim
#x=M(x0,b_ou,Sig_ou,l,d,N,dim)
I=identity(dim).toarray()
I_o=identity(dim_o).toarray()
#R2=(identity(dim_o).toarray() + np.tri(dim_o,dim_o,1) - np.tri(dim_o,dim_o,-2))/20
R2=I
np.random.seed(2)
H=rand(dim_o,dim,density=0.75).toarray()/1e-1

#np.random.seed(3)
xin=np.abs(np.random.normal(1,1,dim))
mu=np.abs(np.random.normal(0.001,10,dim))
sigs=np.abs(np.random.normal(30,1,dim))
comp_matrix = ortho_group.rvs(dim)
inv_mat=la.inv(comp_matrix)
S=diags(np.random.normal(0.1,0.001,dim),0).toarray()
Sig=inv_mat@S@comp_matrix
fi=[sigs,Sig]
np.random.seed(3)
collection_input=[dim,dim_o, b_gbm,mu,Sig_gbm,fi,ht,H]

[z,x_true]=gen_gen_data(T,xin,l,collection_input)
#z=np.reshape(z,z.shape[:2])


[x0,x1,log_weights0,log_weights1,suma0,suma1]= CPF(T,z,lmax,xin,b_gbm,mu,Sig_gbm,fi,ht,H,l,d,N,dim,para=True)
"""
#%%
"""
d_steps0=int(d*2**(l-1))
d_steps1=int(d*2**(l))
spots0=np.arange(d_steps0,2**(l-1)*T+1,d_steps0,dtype=int)
spots1=np.arange(d_steps1,2**l*T+1,d_steps1,dtype=int)
#z=np.reshape(z,(2**l*T+1,dim,1))

a=1
times0=np.array(range(int(2**(l-1)*T+1)))/2**(l-1)
times1=np.array(range(int(2**l*T+1)))/2**l

#kbf=KBF(T,l,lmax,z,collection_input)
xmean0=np.mean(x0[:,:,a],axis=1)
xmean1=np.mean(x1[:,:,a],axis=1)
#plt.plot(times,z[:,a,0])
#plt.plot(times1,x_true[:,a,0],label="True signal")
#plt.plot(spots1/2**l,kbf[0][spots1,a,0],label="KBF")

#plt.plot(2**(-l)*np.array(range(T*2**l+1)),kbf[0][:,0,0])
plt.plot(spots0/2**(l-1),xmean0[spots0],label="PF0")
plt.plot(times1,x_true[:,a],label="True")
plt.plot(spots1/2**(l),xmean1[spots1],label="PF1")
#plt.plot(times,x[:,:,a])
#plt.plot(times0,xmean0,label="mean of the propagation 0")
#plt.plot(times1,xmean1,label="mean of the propagation 1")


plt.legend()

"""

#%%

"""
#Test for the coupling of the PF
#We generare a coupling and two independent PF with subsequent levels of 
#discretization 


#For the c3upling we have 
l=13
lmax=13

T=10
dim=2
dim_o=dim
#x=M(x0,b_ou,Sig_ou,l,d,N,dim)
I=identity(dim).toarray()
I_o=identity(dim_o).toarray()
#R2=(identity(dim_o).toarray() + np.tri(dim_o,dim_o,1) - np.tri(dim_o,dim_o,-2))/20
R2=I
np.random.seed(0)
#H=rand(dim_o,dim,density=0.75).toarray()/1e-0
H=I
xin=np.random.normal(1,0,dim)
np.random.seed(3)
#comp_matrix = ortho_group.rvs(dim)
comp_matrix=I
inv_mat=la.inv(comp_matrix)
S=diags(np.random.normal(.1,0.001,dim),0).toarray()
fi=inv_mat@S@comp_matrix

B=diags(np.random.normal(-.1,0.001,dim),0).toarray()
A=inv_mat@B@comp_matrix

#A=b_ou(I,B).T
R1=Sig_ou(np.zeros(dim),fi)

np.random.seed(3)
C0=I*1e-6
m0=np.random.multivariate_normal(xin,C0)
collection_input=[dim,dim_o,A,R1,R2,H,m0,C0]


[z,x_true]=gen_data(T,l,collection_input)
z=np.reshape(z,z.shape[:2])
"""
#%%
"""
B=100
d=1./2.**1
N=300
#N0=200

resamp_coef=0.8
levels=np.array(range(2,10))
n_levels=len(levels)
xc_samp=np.zeros((n_levels,B,dim))
xi_samp=np.zeros((n_levels,B,dim))

for i in range(n_levels):
    l=levels[i]
    
    print(l)
    d_steps0=int(d*2**(l-1))
    d_steps1=int(d*2**(l))
    spots0=np.arange(d_steps0,2**(l-1)*T+1,d_steps0,dtype=int)
    spots1=np.arange(d_steps1,2**l*T+1,d_steps1,dtype=int)
    #z=np.reshape(z,(2**l*T+1,dim,1))
    
    a=1
    times0=np.array(range(int(2**(l-1)*T+1)))/2**(l-1)
    times1=np.array(range(int(2**l*T+1)))/2**l
        
    #N=int(N0/2**((-levels[0]+l)/4))
    print(N)
    for samp in range(B):
        np.random.seed(samp)
        print(l)
    
        [x0,x1,log_weights0,log_weights1,x0_pf,x1_pf]= CPF(T,z,lmax,xin,b_ou,A,Sig_ou,fi,ht,H,l,d,N,dim,resamp_coef,para=True)
        
        #CPF(T,z,lmax,xin,b_ou,A,Sig_ou,fi,ht,H,l,d,N,dim,para=True)
        #print(norm_logweights(log_weights1,ax=1).shape,x1_pf.shape)
        
        w1=np.reshape(norm_logweights(log_weights1,ax=1),(int(T/d),N,1))
        w0=np.reshape(norm_logweights(log_weights0,ax=1),(int(T/d),N,1))
        mean0=np.mean((w0*x0_pf),axis=1)
        mean1=np.mean((w1*x1_pf),axis=1)
        #tel_samp=np.mean(x1[-1]-x0[-1],axis=0) #From telescopic sample
        xc_samp[i,samp]=np.sum((mean1-mean0)**2,axis=0)
        
        
        
                
        np.random.seed(samp)
        [x0i,log_weights,x0_pfi]= pff.PF(T,z,lmax,xin,b_ou,A,Sig_ou,fi,ht,H,l-1,d,N,dim,para=True)
        np.random.seed(samp+1)
        [x1i,log_weights,x1_pfi]= pff.PF(T,z,lmax,xin,b_ou,A,Sig_ou,fi,ht,H,l,d,N,dim,para=True)
        w1i=np.reshape(norm_logweights(log_weights1,ax=1),(int(T/d),N,1))
        w0i=np.reshape(norm_logweights(log_weights0,ax=1),(int(T/d),N,1))
        mean0i=np.sum(np.mean((w0i*x0_pfi),axis=0))
        mean1i=np.sum(np.mean((w1i*x1_pfi),axis=0))
        #tel_samp=np.mean(x1i[-1]-x0i[-1],axis=0) #From telescopic sample
        xi_samp[i,samp]=np.sum((mean1i-mean0i)**2,axis=0)
        
        
        
        #kbf=KBF(T,l,lmax,z,collection_input)
        
        
        xmean0=np.sum(x0_pf*w0,axis=(1,2))
        xmean1=np.sum(x1_pf*w1,axis=(1,2))
        xmean0i=np.sum(x0_pfi*w0i,axis=(1,2))
        xmean1i=np.sum(x1_pfi*w1i,axis=(1,2))
        
        
        #xmean0=np.mean(x0[:,:,a],axis=1)
        #xmean1=np.mean(x1[:,:,a],axis=1)
        #xmean0i=np.mean(x0i[:,:,a],axis=1)
        #xmean1i=np.mean(x1i[:,:,a],axis=1)
        #plt.plot(times,z[:,a,0])
        #plt.plot(times1,x_true[:,a,0],label="True signal")
        #plt.plot(spots1/2**l,kbf[0][spots1,a,0],label="KBF")

        #plt.plot(2**(-l)*np.array(range(T*2**l+1)),kbf[0][:,0,0])
        
        
        plt.plot(spots0/2**(l-1),xmean0,label="PF0C")
        #plt.plot(times1,x_true[:,a],label="True")
        plt.plot(spots0/2**(l-1),xmean1,label="PF1C")
        plt.legend()
        plt.show()
                plt.plot(spots0/2**(l-1),xmean0i,label="PF0I")
        #plt.plot(times1,x_true[:,a],label="True")
        plt.plot(spots0/2**(l-1),xmean1i,label="PF1I")
        """
        #plt.plot(times,x[:,:,a])
        #plt.plot(times0,xmean0,label="mean of the propagation 0")
        #plt.plot(times1,xmean1,label="mean of the propagation 1")
        #"""
        
      

        

       
        
#[x0,x1,log_weights0,log_weights1,suma0,suma1]= CPF(T,z,lmax,xin,b_ou,A,Sig_ou,fi,ht,H,l,d,N,dim,para=True)

#[x,log_weights,mean]= PF(T,z,l,xin,b_ou,A,Sig_ou,fi,ht,H,l,d,N,dim,para=False)


#%%
"""
print(xc_samp.shape)

secc_tel=np.sum(np.mean(xc_samp,axis=1),axis=1)
seci_tel=np.sum(np.mean(xi_samp,axis=1),axis=1)
plt.plot(levels,np.log(secc_tel),label="coupled")
#plt.plot(levels,seci_tel,label="independent")
print(secc_tel)
plt.legend()
#print(varc_tel,vari_tel)
"""

#%%
"""
a=1
secc_tel=np.mean(xc_samp**2,axis=1)[:,a]
seci_tel=np.mean(xi_samp**2,axis=1)[:,a]
plt.plot(levels[1:],secc_tel[1:],label="coupled")
plt.plot(levels[1:],seci_tel[1:],label="independent")
plt.legend()
print(varc_tel,vari_tel)
"""

#%%
"""
d_steps0=int(d*2**(l-1))
d_steps1=int(d*2**(l))
spots0=np.arange(d_steps0,2**(l-1)*T+1,d_steps0,dtype=int)
spots1=np.arange(d_steps1,2**l*T+1,d_steps1,dtype=int)
#z=np.reshape(z,(2**l*T+1,dim,1))

a=1
times0=np.array(range(int(2**(l-1)*T+1)))/2**(l-1)
times1=np.array(range(int(2**l*T+1)))/2**l

#kbf=KBF(T,l,lmax,z,collection_input)
xmean0=np.mean(x0[:,:,a],axis=1)
xmean1=np.mean(x1[:,:,a],axis=1)
xmean=np.mean(x[:,:,a],axis=1)
#plt.plot(times,z[:,a,0])
#plt.plot(times1,x_true[:,a,0],label="True signal")
#plt.plot(spots1/2**l,kbf[0][spots1,a,0],label="KBF")

#plt.plot(2**(-l)*np.array(range(T*2**l+1)),kbf[0][:,0,0])
plt.plot(spots1/2**(l),xmean[spots1],label="PFI")

plt.plot(spots0/2**(l-1),xmean0[spots0],label="PF0")
plt.plot(times1,x_true[:,a],label="True")
plt.plot(spots1/2**(l),xmean1[spots1],label="PF1")
#plt.plot(times,x[:,:,a])
#plt.plot(times0,xmean0,label="mean of the propagation 0")
#plt.plot(times1,xmean1,label="mean of the propagation 1")


plt.legend()


"""

#%%

def MLPF(T,z,lmax,xin,b_ou,A,Sig_ou,fi,ht,H,eles,d,Nl,dim,resamp_coef,phi,dim_out,axis_action,para=True):
    # The MLPF stands for Multilevel Particle Filter, and uses the Multilevel methodology to compute the 
    # particle filter. 

    #ARGUMENTS:
    # The arguments are basically the same as those for the PF and CPF functions, with changes in
    # l->eles: is a 1 rank array starting with l_0 and ending with L 
    # N-> Nl: is a 1 rank array starting with the corresponding number of particles to 
    # each level
    # the new parameters are:
    # phi: function that takes as argument a rank M array and computes a function 
    # along the axis_action dimension. the dimension of the output of phi is the 
    # same as the input changing the dimension of the axis axis_action by dim_out

    #OUTPUT
    # pf: computation of the particle filter with dimension (int(T/d),dim_out)

    pf=np.zeros((int(T/d),dim_out))
    [log_weightsl0,x_pfl0]=pff.PF(T,z,lmax,xin,b_ou,A,Sig_ou,fi,ht,H,eles[0],d,Nl[0],dim,resamp_coef,para=True)[1:]
    phi_pf=phi(x_pfl0,axis_action)
    weightsl0=np.reshape(norm_logweights(log_weightsl0,ax=1),(int(T/d),Nl[0],1))
    pf= np.sum(phi_pf*weightsl0,axis=1)
    
    #PF(T,z,lmax,x0,b_ou,A,Sig_ou,fi,ht,H,l,d,N,dim,resamp_coef,para=True)
    eles_len=len(eles)
    #x_pf=np.zeros((2,eles_len-1, int(T/d),N,dim))
    #log_weights=np.zeros((2,eles_len-1, int(T/d),N))
    
    for i in range(1,eles_len):
        l=eles[i]
        [log_weights0,log_weights1,x0_pf,x1_pf]=CPF(T,z,lmax,xin,b_ou,A,Sig_ou,fi,ht,H,eles[i],d,Nl[i],dim,resamp_coef,para=True)[2:]
        #log_weights[:,i-1]=[log_weights0,log_weights1]
        #x_pf[:,i-1]=[x0_pf,x1_pf]
        phi_pf0=phi(x0_pf,axis_action)
        phi_pf1=phi(x1_pf,axis_action)
        weights0=np.reshape(norm_logweights(log_weights0,ax=1),(int(T/d),Nl[i],1))
        weights1=np.reshape(norm_logweights(log_weights1,ax=1),(int(T/d),Nl[i],1))
        pf= pf+np.sum(phi_pf1*weights1,axis=1)-np.sum(phi_pf0*weights0,axis=1)
        
    
        
    return pf

def phi(x,axis=0):
    #phi has to keep the dimensions! i.e. keepdims=True
    return x


#%%

#Test for the MLPF
"""
l=13
lmax=13

T=10
dim=2
dim_o=dim
#x=M(x0,b_ou,Sig_ou,l,d,N,dim)
I=identity(dim).toarray()
I_o=identity(dim_o).toarray()
#R2=(identity(dim_o).toarray() + np.tri(dim_o,dim_o,1) - np.tri(dim_o,dim_o,-2))/20
R2=I
np.random.seed(0)
#H=rand(dim_o,dim,density=0.75).toarray()/1e-0
H=I*1e2
xin=np.random.normal(1,0,dim)
np.random.seed(3)
#comp_matrix = ortho_group.rvs(dim)
comp_matrix=I
inv_mat=la.inv(comp_matrix)
S=diags(np.random.normal(.1,0.001,dim),0).toarray()
fi=inv_mat@S@comp_matrix

B=diags(np.random.normal(-.1,0.001,dim),0).toarray()
A=inv_mat@B@comp_matrix

#A=b_ou(I,B).T
R1=Sig_ou(np.zeros(dim),fi)

np.random.seed(3)
C0=I*1e-6
m0=np.random.multivariate_normal(xin,C0)
collection_input=[dim,dim_o,A,R1,R2,H,m0,C0]


[z,x_true]=gen_data(T,l,collection_input)
z=np.reshape(z,z.shape[:2])
"""
#%%

"""
d=1./2.**4
l0=6
L=13
eles=np.array([i for i in range(l0,L+1)])
N0=2000
Nl=np.array((N0/2**(eles/2)),dtype=int)
resamp_coef=0.8
z=np.reshape(z,z.shape[:2])
print(Nl,eles)
    #MLPF(T,z,lmax,xin,b_ou,A,Sig_ou,fi,ht,H,eles,d,Nl,dim,resamp_coef,phi,dim_out,axis_action,para=True)
pf=MLPF(T,z,lmax,xin,b_ou,A,Sig_ou,fi,ht,H,eles,d,Nl,dim,resamp_coef,phi,dim,axis_action=2,para=True)
"""

#%%

#kbf=KBF(T,l,lmax,z,collection_input)
#%%
"""
d_steps0=int(d*2**(l-1))
d_steps1=int(d*2**(l))
spots0=np.arange(d_steps0,2**(l-1)*T+1,d_steps0,dtype=int)
spots1=np.arange(d_steps1,2**l*T+1,d_steps1,dtype=int)
z=np.reshape(z,(2**l*T+1,dim,1))

a=1
times0=np.array(range(int(2**(l-1)*T+1)))/2**(l-1)
times1=np.array(range(int(2**l*T+1)))/2**l

#xmean0=np.mean(x0[:,:,a],axis=1)
#xmean1=np.mean(x1[:,:,a],axis=1)
#plt.plot(times,z[:,a,0])
plt.plot(times1,x_true[:,a,0],label="True signal")
plt.plot(spots1/2**l,kbf[0][spots1,a,0],label="KBF")

#plt.plot(2**(-l)*np.array(range(T*2**l+1)),kbf[0][:,0,0])
plt.plot(spots0/2**(l-1),pf[:,a],label="MLPF")
#plt.plot(times1,x_true[:,a],label="True")
#plt.plot(spots1/2**(l),xmean1[spots1],label="PF1")
#plt.plot(times,x[:,:,a])
#plt.plot(times0,xmean0,label="mean of the propagation 0")
#plt.plot(times1,xmean1,label="mean of the propagation 1")


plt.legend()    
"""
