#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 17:29:35 2022



General description: This file is specifically made for the PF implementation
using Cox observations. Includes a way to coupled PFs with subsequent discretization


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
#from sklearn.linear_model import LinearRegression
from scipy.stats import ortho_group
import time
import PF_functions_def as pff
from scipy.stats import multivariate_normal
#%%


def coef(x, y): 
    # number of observations/points 
    n = np.size(x) 
  
    # mean of x and y vector 
    m_x, m_y = np.mean(x), np.mean(y) 
  
    # calculating cross-deviation and deviation about x 
    SS_xy = np.sum(y*x) - n*m_y*m_x 
    SS_xx = np.sum(x*x) - n*m_x*m_x 
  
    # calculating regression coefficients 
    b_1 = SS_xy / SS_xx 
    b_0 = m_y - b_1*m_x 
    
    return np.asarray((b_0, b_1)) 


# STATE SPACE MODEL
def gen_gen_data(T,x0,l,collection_input): #After generator of general data
    # parameters [dim,dim_o, b_ou,A,Sig_ou,fi,ht,H]
    # Function that generates euler maruyama samples of a difussion process
    # ARGUMENTS: T: final time of the discretization
    # x0: Initial position of the difussion, rank 1 array of dimension dim
    # l: level of discretization, the time step of this discretization is 2^{-l}
    # collection input: dim is the dimension of the difussion, b is a drift 
    # function that takes on two arguments, the diffusion array at an specific time
    # and the parameters of the functions A. The diffusion function of the process is 
    # Sig, which takes on the same first argument as b and the second argument is 
    # fi.
    # OUTPUTS: A rank two array of dimensions (T*2**l,dim) with the Euler-Maruyama
    # discretization.
    
    [dim, b,A,Sig,fi]=collection_input
    
    J=T*(2**l)
    I=identity(dim).toarray()
    #I_o=identity(dim_o).toarray()

    tau=2**(-l)
    
    v=np.zeros((J+1,dim))
    
    #v[0]=np.random.multivariate_normal(m0,C0,(1)).T
    v[0]=x0



    for j in range(J):
        ## truth
        #print(np.shape(Sig(v[j],fi)),np.shape(b(v[j],A)))
        #print(b(v[j],A).shape,Sig(v[j],fi).shape)
        v[j+1] = v[j]+b(v[j],A)*tau + np.sqrt(tau)*(np.random.multivariate_normal(np.zeros(dim),I))@(Sig(v[j],fi).T)
        ## observation
        
    return v

    
#%%
"""
#Test of gen_gen_data

l=7
d=20
T=2
N=10
dim=2


xin0=np.random.normal(1,1,dim)
xin1=xin0

#for the OU process
np.random.seed(3)
comp_matrix = ortho_group.rvs(dim)
inv_mat=la.inv(comp_matrix)
S=diags(np.random.normal(.1,0.1,dim),0).toarray()
fi=inv_mat@S@comp_matrix
B=diags(np.random.normal(-.5,0.1,dim),0).toarray()*(2/3)
B=inv_mat@B@comp_matrix
#B=comp_matrix-comp_matrix.T  +B 

#For the GBM

mu=np.abs(np.random.normal(-1,1,dim))
sigs=np.abs(np.random.normal(.1,1,dim))
Sig=comp_matrix+comp_matrix.T
fi=[sigs,Sig]
#print(Sig_gbm(x,fi))
#print(np.reshape(mu,(-1,1)))

np.random.seed(3)
collection_input=[dim, b_gbm,mu,Sig_gbm,fi]

x_true=gen_gen_data(T,np.abs(xin0),l,collection_input)
# plots of the test
times=2**(-l)*np.array(range(int(T*2**l+1)))
a=8
plt.plot(times,x_true)
"""


"""
#Test of gen_gen_data in 1 dimension

l=7
d=1
T=2
N=3
dim=1


xin0=np.random.normal(1,1,dim)
xin1=xin0

#for the OU process
np.random.seed(3)
#comp_matrix = ortho_group.rvs(dim)
comp_matrix=[[1]]
inv_mat=la.inv(comp_matrix)
S=diags(np.random.normal(.1,0.1,dim),0).toarray()
fi=inv_mat@S@comp_matrix
B=diags(np.random.normal(-.5,0.1,dim),0).toarray()*(2/3)
B=inv_mat@B@comp_matrix
#B=comp_matrix-comp_matrix.T  +B 
"""
#For the GBM
"""
mu=np.abs(np.random.normal(-1,1,dim))
sigs=np.abs(np.random.normal(.1,1,dim))
Sig=comp_matrix+comp_matrix.T
fi=[sigs,Sig]
#print(Sig_gbm(x,fi))
#print(np.reshape(mu,(-1,1)))

np.random.seed(3)
collection_input=[dim,pff.b_ou,S,pff.Sig_ou,fi]

x_true=gen_gen_data(T,np.abs(xin0),l,collection_input)
# plots of the test
times=2**(-l)*np.array(range(int(T*2**l+1)))
a=8
plt.plot(times,x_true)

"""
#%%


#Given a sample of the difussion we sample the observations 

def Lambda(x,Lamb_par,ax=0):
    
    # Intensity function of the Cox process. Appies a function on
    # x with the parameters Lamb_par along the axis ax
    # ARGUEMNTS:
    # x: Multirank array, ex: (# particles,dim)
    # Lamb_par: arguments feeded to the function
    # ax: is the axis of x to which we apply the function
    #Lamb=Lamb_par*np.linalg.norm(np.sin(x),axis=axis)**2
    # Lamb=Lamb_par*np.linalg.norm(np.sin(x),axis=axis)
    Lamb=Lamb_par*np.linalg.norm(x,axis=ax)
    
    return Lamb


def Cte_Lambda(x,Lamb_par,ax=0):
    
    #Lamb=Lamb_par*np.linalg.norm(np.sin(x),axis=axis)**2
    #Lamb=Lamb_par*np.linalg.norm(np.sin(x),axis=axis)
    #Lamb=Lamb_par*np.linalg.norm(x,axis=ax)
    #print(x.shape)
    return np.zeros(x.shape[:2])+1


def g_normal(x,g_pars):
    # Samples from a normal multivariate with cov=g_pars[1] and mean zero
    #and ads it to x.
    # ARGUMENTS:
    # x is a rank 1 or 2 array with dimensions (dim) or (# of particles, dim) 
    # respectively
    # OUTPUT:
    #  rank 1 or 2 array with dimensions (dim) or (# of particles, dim) 
    # respectively
    [dim,cov]=g_pars
    return x+np.random.multivariate_normal(np.zeros(dim),cov,len(x))



def gen_obs(x,l,T,Lambda,Lamb_par,g,g_par):
    # Function that generates poisson arrival times 
    # with intensity function Lambda(x)
    # ARGUMENTS: x: is a sample path of the difussion process, it is a rank 2
    # array with dimensions (T*2**l+1,dim), where dim is the dimension of the 
    # diff process.
    # l, T: are the time discretization level and T is the total time of the samples
    # Lamb: is a function that takes on two arguments, x and Lamb_par, which correspond
    # to the principal argument of the function and the hyperparameters of the 
    # function. 
    # g: is a function that generates the samples accordingly to a given kernel,
    # its arguments are the true position of the signal and the hyperparameters of 
    # the function. 
    # OUTPUTS: We have 3 outputs 
    # cox_times: which is 
    # a rank 1 array with undetermined lenght.
    # ks: the nearest incexes of the cox_times according to the array x
    # obs: the observations generated, wich are a two rank array with dimension
    # (number of times sampled, dim)
    
    Gamma=np.cumsum(Lambda(x,Lamb_par,ax=1))/2**(l)
    #generate the times
    #print(Gamma)
    cox_times=[]
    c=1
    while c==1:
        U=np.random.uniform()
        s=-np.log(1-U)
        cou=Gamma-s
        #print(cou)
        sign=cou[0]
        k=0
   
        while sign<0 and c==1:
            k+=1
            if k>len(cou)-1:
                c=0
                k-=1
            sign=cou[k]
            
        #k is the first position such that Gamma[k]>s
        cox_times.append(k*2**(-l))
        #print(len(Gamma))
        Gamma=Gamma[k:]-Gamma[k]
        #print(t)
    cox_times= np.cumsum(np.array(cox_times)[:-1])
    ks=np.array(cox_times*2**(l),dtype=int)
    obs=g_normal(x[ks],g_par)
    return [cox_times,obs]
        
        


#%%

#TEST FOR THE COX_PROCESS

"""
l=8
T=10
dim=2
x_1=np.array(range(T*2**l+1))/2**(l)
x=np.zeros((2,T*2**l+1))
x+=x_1
I=identity(dim).toarray()
g_par=[dim,I]

Lamb_par=1
cox_times=gen_obs(x.T,l,T,Lambda,Lamb_par,g_normal,g_par)[0]
B=5000
for i in range(B):
    cox_times=np.append(cox_times,gen_obs(x.T,l,T,Lambda,Lamb_par,g_normal,g_par)[0])
    
    
#%%
plt.hist(cox_times,bins=T*2**(l-2),density=True)
plt.plot(x[0],Lambda(x,1,axis=0)/(np.sum(Lambda(x,1,axis=0))/(len(x[0])/T)))
plt.show
print(len(cox_times)/B)
"""
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

# In the following function I check the Kalman Filter for the state space model


def KF(xin,dim,dim_o,K,G,H,D,obs):
    
    #This function computes the Kalman Filter observations at arbitrary times
    # and a time indendent linear setting, i.e.
    #X_t=KX_{t-1}+ G W_t
    #Y_t=HX_t+DV_t
    
    #ARGUMENTS: 
    #obs: is a rank 2 array with dimensions (number of observations, dim_o)
    #xin: rank 1 array with dimensions (dim)
    #OUTPUTS:
    #x: rank two array with dimensions (T+1,dim)
    T=obs.shape[0]
    x=np.zeros((T+1,dim))
    P=np.zeros((T+1,dim,dim))
    x[0]=xin
    I=identity(dim).toarray()
    P[0]=0
    for i in range(T):
        xh=K@x[i]
        #print("xh",xh)
        Ph=K@P[i]@(K.T)+(G@(G.T))
        #print("Ph",Ph)
        y=obs[i]-H@xh
        #print("y",y)
        S=H@Ph@(H.T)+(D@(D.T))
        
        Kg=Ph@(H.T)@la.inv(S)
        x[i+1]=xh+Kg@y
        P[i+1]=(I-Kg@H)@Ph
    return x
        
        
#%%
"""
xin=np.array([1,2])
dim=2
dim_o=dim
K=np.array([[-1,0],[-1,2]])
G=np.array([[1,0],[0,1]])
D=np.array([[1,0],[0,1]])
H=np.array([[1,1.20],[1,0.03]])
obs=np.array([[2,3],[3,2.3],[3,3]])
kf=KF(xin,dim,dim_o,K,G,H,D,obs)
print(kf)
"""
#%%
"""
xin=np.array([1])
dim=1
dim_o=dim
K=np.array([[-1]])
G=np.array([[1]])
D=np.array([[1]])
H=np.array([[1]])
obs=np.array([[2],[3],[3]])
kf=KF(xin,dim,dim_o,K,G,H,D,obs)
print(kf)
"""
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
            # Uncomment the following two lines for the GBM and comment
            # the third line
            #diff=np.einsum("nd,njd->nj",dW[s],Sig(x1[2*t0+s],fi))
            #x1[2*t0+s+1]=x1[2*t0+s]+b(x1[2*t0+s],A)*dt1+diff
            # For the OU process comment the previous two lines and uncomment 
            # the following line
            x1[2*t0+s+1]=x1[2*t0+s]+b(x1[2*t0+s],A)*dt1+ dW[s]@(Sig(x1[2*t0+s],fi).T)
        
        # Uncomment the following two lines for the GBM and comment
        # the third line
        #diff=np.einsum("nd,njd->nj",dW[0]+dW[1],Sig(x0[t0],fi))
        #x0[t0+1]=x0[t0]+b(x0[t0],A)*dt0+ diff
        # For the OU process comment the previous two lines and uncomment 
        # the following line
        x0[t0+1]=x0[t0]+b(x0[t0],A)*dt0+ (dW[0]+dW[1])@(Sig(x0[t0],fi).T)
    return [x0,x1]

#%%
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
    
#We take the design of the brownian motion to the as the one in wikipedia:
#https://en.wikipedia.org/wiki/Geometric_Brownian_motion
    
def Sig_gbm(x,fi):
    # Returns the drift "vector" evaluated at x
    # ARGUMENTS: x is a rank two array with dimensions where the last dimension 
    # corresponds to the dimesion of the problem and the previous dimension 
    # (in case to exist) corresponds to the number of particles.
    # The second argument is fi, which is composed of a vector
    # sigs with dimension dim (of exclusively positive components)
    # and a square matrix Sig with the dimension of the 
    # system. The matrix Sig corresponds to the covariance matrix
    
    # OUTPUTS: A rank 3 array where the first dimension corresponds to the number
    # of particles and the second to the dimension of the system. Or a rank 3 array
    # 
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
l=8
d=2

N=3
dim=2

#OU
xin0=np.random.normal(1,1,dim)
xin1=xin0
#np.random.seed(3)
comp_matrix = ortho_group.rvs(dim)
#print(comp_matrix)
inv_mat=la.inv(comp_matrix)
S=diags(np.random.normal(1,0.1,dim),0).toarray()
#S=np.zeros(dim)+1
fi=inv_mat@S@comp_matrix

B=diags(np.random.normal(-1,0.1,dim),0).toarray()*(2/3)
B=inv_mat@B@comp_matrix
#B=comp_matrix-comp_matrix.T  +B 
#np.random.seed(3)


#GBM

#np.random.seed(3)
xin0=np.abs(np.random.normal(1,1,dim))
xin1=xin0
mu=np.abs(np.random.normal(1,0,dim))
sigs=np.abs(np.random.normal(np.sqrt(2),0,dim))
comp_matrix = ortho_group.rvs(dim)
inv_mat=comp_matrix.T
S=diags(np.random.normal(1,1,dim),0).toarray()
Sig=inv_mat@S@comp_matrix
fi=[sigs,Sig]

#z=np.reshape(z,z.shape[:2])




x3=M_coup(xin0,xin1,b_gbm,mu,Sig_gbm,fi,l,d,N,dim)

"""
#%%
"""
# plot of the test of M_coup
print(x3[0][-1])
steps0=int(2**(l-1)*d)
steps1=int(2**(l)*d)
time0=np.array(range(steps0+1))/2**(l-1)
time1=np.array(range(steps1+1))/2**l

a=1
plt.plot(time0,x3[0][:,0,a])

plt.plot(time1,x3[1][:,0,a])
"""
#%%

def g_den(y,x,g_par):
    # Likelihood function
    # ARGUMENTS: 
    # y: as the observations, the function computes the conditional density of
    # y given x. y is a rank 2 array with dimensions (# of observations, dim).
    # x: position of the particles depending on the time, rank 3 array with 
    # dimensions (# of observations, particles, dim)
    # g_par: parameters of the density.
    # OUTPUT: 
    # gs: rank 2 array of the probabilities, with dimensions (# of observations,
    # # of particles)
    shape=x.shape
    #print(g_par)
    gs=np.zeros(shape[:2])
    for i in range(shape[0]):
        for n in range(shape[1]):
            var=multivariate_normal(mean=x[i,n],cov=g_par)
            gs[i,n]=var.pdf(y[i])
    return gs
        

    
    

def G(obs,obs_times,x,Lambda,Lamb_par,l,N,dim,g_den,g_par):
    #Function that computes the weights of the system depending on the observations
    
    #ARGUMENTS: obs: are the observations(2 rank array) with dimensions (# of observations,dim)
    # x: is the array of the 
    # particles (rank 3 array) with dimensions (2**l+1,N,dim), where N is the number
    # of prticles.
    # obs_times: are the times of observation (cox process).
    # Lambda: intensity function of the poisson process, takes on two arguments,
    # x as the principal argumetn and Lamb_par as hyperparameters.
    # g_den: likelihood function of the observation once the observation time
    # has been computed, it has three parameters, 
    #OUTPUT: logarithm of the weights 
    log_int=-np.sum(Lambda(x[:-1],Lamb_par,ax=2),axis=0)/2**(l)
    #log_int=-np.sum(Lambda(x[:-1],Lamb_par,axis=2))/2**(l)
    x_inter=np.zeros((len(obs),N,dim))
    
    if len(obs_times)>0:
        
        for i in range(len(obs)):
            t_obs=obs_times[i]
            k=int(np.floor(t_obs*2**l))
            x_inter[i]=(x[k+1]-x[k])*(t_obs-(k+1)/2**(l))*2**(l)+x[k+1]
            
        log_prod1=np.sum(np.log(Lambda(x_inter,Lamb_par,axis=2)),axis=0)
        log_prod2=np.sum(np.log(g_den(obs,x_inter,g_par)),axis=0)
    else:
        log_prod1=0
        log_prod2=0
        
    #print(suma1,suma2)
    log_w=log_int+log_prod1+log_prod2
   
    return log_w

#%%
"""
obs=[1]
obs_times=[0.24]
d=2
dim=2
l=2
x0=np.array(range(d*2**l+1))*2**(-l)
x0=np.reshape(x0,(d*2**l+1,1,1))
N=5
x=np.zeros((d*2**l+1,N,dim))
x=x+x0
g_par=identity(dim).toarray()
Lamb_par=1
#print(np.sum(Lambda(x,Lamb_par,axis=2)[:-1])*)

print(G(obs,obs_times,x,Lambda,Lamb_par,l,N,dim,g_den,g_par))


"""

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
    
    
def multi_samp1(W,N,x,dim): #from multinomial sampling
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
    x_resamp=x[part_samp]
    return [part_samp+1,x_resamp] #here we add 1 bc it is par
 
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
    #print("lw is:",lw)
    w=np.exp(lw-m)/wsum
    #print("w is:",w)
    return w


#%%
#Test for the multisampling algorithm
"""
samples=1000
N=4
#W=np.array([np.abs(np.sin(i/10)) for i in range(N)])
W=np.array([0.7,8,4,4])
W=W/np.sum(W)
x=np.zeros((samples,N))
x1=np.zeros((samples,N))
dim=1

pos=np.array([3,4,5,6])
print(multi_samp(W, N, pos, dim))


for i in range(samples):
    x[i]=multi_samp(W, N, pos, dim)[1]
    x1[i]=multi_samp1(W, N,  pos, dim)[1]
"""
#%%
"""
plt.hist(x.flatten(),bins= pos,density=True)
plt.hist(x1.flatten(),bins= pos,density=True)

plt.show()
"""
"""
plt.hist(x1.flatten(),bins= np.array(range(N)),density=True)
plt.hist(x.flatten(),bins= np.array(range(N)),density=True)
plt.show()
"""

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



#COMPARATATION BETWEEN THE Cox_PF AND THE KBF FOR REGULAR OBSERVATIONS
#FOR TIME T=1

"""
np.random.seed(3)
T=10
dim=1
dim_o=dim
xin=np.zeros(dim)+1
l=10
collection_input=[]
I=identity(dim).toarray()

#comp_matrix = ortho_group.rvs(dim)
comp_matrix=np.array([[1]])
inv_mat=la.inv(comp_matrix)
S=diags(np.random.normal(1,0.1,dim),0).toarray()
#S=np.array([[1.]])
fi=inv_mat@S@comp_matrix
B=diags(np.random.normal(-1,0.1,dim),0).toarray()
B=inv_mat@B@comp_matrix
#B=np.array([[-1.]])
print(B)
#print(B)
#B=comp_matrix-comp_matrix.T  +B 

collection_input=[dim, b_ou,B,Sig_ou,fi]
cov=I*1e-1
g_par=[dim,cov]
#x_true=gen_gen_data(T,xin,l,collection_input)
Lamb_par=1
A=B
R1=fi
R2=la.sqrtm(2**(l)*g_par[1])
H=I*2**(l)
m0=xin
C0=I*1e-40
collection_input=[dim,dim,A,R1,R2,H,m0,C0]

[z,v]=pff.gen_data(T,l,collection_input)
#kbf=np.reshape(pff.KBF(T,l,l,z,collection_input)[0],(T*2**l+1,dim))
#parameters of the KF
#K=la.expm(2**(-l)*B)
#G=la.sqrtm((fi@fi)@la.inv(B+B.T)@(la.expm((B+B.T)*2**(-l))-I))
#la.expm(2**(-l)*B)@
#H=I
#D=la.sqrtm(cov)

v=np.reshape(v,(T*2**l+1,dim))
x_true=v
#obs_time=np.array(range(T*2**l))*2**(-l)+2**(-l)
#obs=np.reshape(z[1:]-z[:-1],(T*2**l,dim))
#kf=KF(xin,dim,dim_o,K,G,H,D,obs)
[obs_time,obs]=gen_obs(x_true,l,T,Lambda,Lamb_par,g_normal,g_par)
print(obs.shape)
print(obs_time)
#so what is probably happening with the observations is that 
#somehow we are approximation to the time on the grid, ofr so reason.
#print(obs_time)
# plots of the test
times=2**(-l)*np.array(range(int(T*2**l+1)))

#plt.plot(times,x_true)
plt.plot(times,v,label="True signal")
plt.plot(obs_time,obs,label="Observations")
#plt.plot(times,kbf,label="KBF")
#plt.plot(times,kf,label="KF")
#plt.plot(times,kf-kbf)
#print(len(obs_time))
plt.legend()
#print(obs_time)
#"""
#%%
"""
d=2**(0)
print(obs)
samples=2000
resamp_coef=0.8
l=0
N=5000
g_par=cov
x_pfmean=np.zeros((samples,dim))
[log_weights,x_pf]=pff.Cox_PF(T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,l,d,N,dim,resamp_coef,g_den,g_par,Lambda,Lamb_par)
xs=np.zeros((samples,dim))

#print(T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,l,d,N,dim,resamp_coef,g_den,g_par,Lambda,Lamb_par)
theta=-B
alphas=((1-theta/2**(l))**(2**(l)))
l0=0
L=6
eles=np.array(range(l0,L+1))
x_pfmean=np.zeros((len(eles),samples,int(T/d),dim))
x_paths=np.zeros((len(eles),samples,int(T/d),dim))


for i in range(len(eles)):  
    print(eles[i])
    for sample in range(samples):
        np.random.seed(samples*i+sample)
        [log_weights,x_pf]=pff.Cox_PF(T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,eles[i],d,N,dim,resamp_coef,g_den,g_par,Lambda,Lamb_par)
        #weights=norm_logweights(np.cumsum(log_weights,axis=0),ax=1)
        weights=norm_logweights(log_weights,ax=1)
        #x_pfmean[i,sample]=np.sum(np.reshape(weights,(int(T/d),N,1))*x_pf,axis=1)
        x_pfmean[i,sample]=np.sum(np.reshape(weights,(int(T/d),N,1))*x_pf,axis=1)
        x_paths[i,sample]=np.mean(x_pf,axis=1)
#This experiment is performed in order to find the bias of the Cox_PF with a variable intensity.
#%%
theta=-B
d=2**(-0)
#samples=100
l0=0
L=6
eles=np.array(range(l0,L+1))
l=0
times=2**(-l)*np.array(range(int(T*2**l+1)))
a=0


x_pf_av=np.mean(x_pfmean,axis=1)
rich_bias=np.abs(x_pf_av[1:,a,0]-x_pf_av[:-1,a,0])
print(x_pf_av.shape)
x_var=np.var(x_pfmean,axis=1)
print(np.sqrt(x_var[:,-1,0]/samples)*1.96)
x_paths_av=np.mean(x_paths,axis=1)
#print((np.abs(x_pf_av[:,0]-kbf[-1])+np.sqrt(x_var[:,0]/samples)*1.96).shape)
#print(np.sqrt(x_var/samples)*1.96)
plt.plot(eles[:-1],rich_bias)
#plt.plot(eles,np.abs(x_pf_av[:,a,0]-kbf[a+1,0]),label="Bias")
#plt.plot(eles,np.abs(x_pf_av[:,a]-kf[a+1])+np.sqrt(x_var[:,a]/samples)*1.96,label="CI Upper bound")
#plt.plot(eles,np.abs(x_pf_av[:,a]-kf[a+1])-np.sqrt(x_var[:,a]/samples)*1.96,label="CI Lower bound")

alphas=((1-theta/2**(eles))**(2**(eles)))[0]


log_alphas=2**eles*np.log(np.abs(1-theta/2**(eles)))
#print(alphas,np.exp(log_alphas))
sou=fi

sigmas=np.sqrt(sou**2*(1-alphas**2)/(2*theta-theta**2/2**eles))

#print(sigmas)
alphas=np.reshape(alphas,(len(eles),dim,dim))
sigmas=np.reshape(sigmas,(len(eles),dim,dim))
kfs=np.zeros((len(eles),int(T/d)+1, dim))
#print(x_paths_av[:,0,0],alphas)

#print(sigmas**2)
"""
"""
for i in range(len(eles)):
    K=alphas[i]
    G=sigmas[i]
    #la.expm(2**(-l)*B)@
    H=I
    D=la.sqrtm(cov)
    kfs[i]=KF(xin,dim,dim_o,K,G,H,D,obs)
   
#print(kfs.shape)
#plt.plot(eles,np.abs(kfs[:,a+1]-kf[a+1]),label="Discretized KF bias")

#plt.plot(eles,1/2**(eles)*(np.abs(x_pf_av[0,a,0]-kf[a+1,0]))*2**(eles[0]),label="$\Delta_l$")
plt.legend()
plt.xlabel("$l$")
#plt.title("$T=10$, $N=5000$, samples=200")
plt.yscale("log")
#plt.savefig("finally_biases.pdf")


# CONCLUSIONS: The PF with constant intensity funciton follow the bias of the 
#particle filter given that we compute exactly the particle filter. Now, 
#one question remains and it is why the code didn't work when I parallelized the 
#samples. 
#"""
#%%

#Question, do we still have problems with the last iteration? do we? why? 




#%%

#COMPARATATION BETWEEN THE Cox_PF AND THE KBF FOR REGULAR OBSERVATIONS
"""
if True==True:
    np.random.seed(6)
    T=10
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
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
    C0=I*1e-40
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

"""
#%%
"""
kf=KF(xin,dim,dim_o,K,G,H,D,obs)
#[obs_time,obs]=gen_obs(x_true,l,T,Lambda,Lamb_par,g_normal,g_par)
#print(obs_time)
# plots of the test
times=2**(-l)*np.array(range(int(T*2**l+1)))

#plt.plot(times,x_true)
plt.plot(times,v,label="True signal")
plt.plot(obs_time,obs,label="Observations")
#plt.plot(times,kbf,label="KBF")
plt.plot(times,kf,label="KF")
#plt.plot(times,kf-kbf)
#print(len(obs_time))
plt.legend()
#print(obs_time)
"""

#%%
# TEST FOR Gox WITH OBSERVATION ON THE DISCRETIZED MESH
"""
d=1/2**(1)
N=2
print(obs[:1],obs_time[:1])
g_par=cov
x=pff.M(xin,b_ou,B,Sig_ou,fi,l,d,N,dim)
#x=np.zeros((2,2,1))
x[0,0]=obs[0]
x[0,1]=1
print(x)
log_w=pff.Gox(obs[:1],obs_time[:1]-d,x,Cte_Lambda,Lamb_par,l,N,dim,g_den,g_par)
print(obs[0])
print(norm_logweights(log_w, ax=0))
"""


	#%%
"""
N0=3
d=2**(-1)
pmax=11
enes=np.array([2**i for i in range(pmax+1)])*N0
samples=200
resamp_coef=0.8

x_pfmean=np.zeros((pmax+1,samples,int(T/d),dim))

for i in range(pmax+1):
    print(enes[i],i)
    for sample in range(samples):
        np.random.seed(i*samples+sample)
        [log_weights,x_pf]=pff.Cox_PF(T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,l,d,enes[i],dim,resamp_coef,g_den,g_par,Cte_Lambda,Lamb_par)
        weights=norm_logweights(log_weights,ax=1)
        x_pfmean[i,sample]=np.sum(np.reshape(weights,(int(T/d),enes[i],1))*x_pf,axis=1)
"""

#%%
#COMPUTATION OF THE KALMAN FILTER OF THE DISCRETIZED ORNSTEIN UHLENBECK PROCESS
"""
theta=-B
d=2**(-1)
samples=2400
l0=1
L=7

eles=np.array(range(l0,L+1))
#print((1-theta/2**(eles)))
alphas=((1-theta/2**(eles))**(2**(eles-1)))[0]

log_alphas=2**eles*np.log(np.abs(1-theta/2**(eles)))
#print(alphas,np.exp(log_alphas))
sou=fi

sigmas=np.sqrt(sou**2*(1-alphas**2)/(2*theta-theta**2/2**eles))
#print(sigmas)
alphas=np.reshape(alphas,(len(eles),dim,dim))
sigmas=np.reshape(sigmas,(len(eles),dim,dim))
kfs=np.zeros((len(eles),dim))

for i in range(len(eles)):
    K=alphas[i]
    G=sigmas[i]
    #la.expm(2**(-l)*B)@
    H=I
    D=la.sqrtm(cov)
    kfs[i]=KF(xin,dim,dim_o,K,G,H,D,obs)[-1]
"""
#%%
"""
#what is going on? The bias it's not exactly bad, just not as good as we expected.
#It's the pf wrong? are some of the conditions wrong? 
#what is it? 

d=2**(-1)
l0=1
L=7
N=1000

eles=np.array(range(l0,L+1))
#x_pf=np.reshape(np.loadtxt("pfs_shifted_d1_pc2.txt",dtype=float),(len(eles),samples,int(T/d),dim))
samples=100
x_pf=np.reshape(np.loadtxt("pf_1d_v4_1pc.txt",dtype=float),(len(eles),samples,int(T/d),dim))
#x_pf=np.concatenate((x_pf,x_pf2),axis=1)
var_pf=np.var(x_pf[:,:,-1],axis=1)
x_pf=np.mean(x_pf,axis=1)
x_pf_r=x_pf[1:,-1]-x_pf[:-1,-1]
bias=np.abs(x_pf[:,-1]-kbf[-1])
print(x_pf[:,-1],kfs)
bias_ub=np.sqrt(var_pf)*1.96/np.sqrt(samples)+bias
#print(bias_ub.shape)
bias_lb=np.maximum(-np.sqrt(var_pf)*1.96/np.sqrt(samples)+bias,1e-5)

bias_dis_kf=np.abs(kfs[:,0]-kbf[-1])
#print(kfs.shape)
print(bias_dis_kf)
reference=np.array([1/2**(i) for i in eles])*2**eles[0]*bias[0]
#print(bias)
#bias_ub=np.sqrt(var_pf)*1.96/np.sqrt(samples)
#print(bias,np.abs(x_pf_r))
#print(x_pf.shape)
dtimes=np.array(range(int(T/d)))*d+d
#plt.plot(eles[1:],np.abs(x_pf_r))
#plt.plot(eles[1:],x_pf_r_ub)
plt.plot(dtimes,x_pf[-1],label="PF")
plt.plot(times,kf,label="KF")

#plt.plot(times,v,label="True signal")
#plt.plot(obs_time,obs,label="Observations")
#plt.plot(times,kbf,label="KBF")
#print(kbf.shape)
#plt.plot(eles,bias,label="PF bias")
#plt.plot(eles,bias_ub-bias,label="ci")
#plt.plot(eles,bias_ub,label="PF bias confidence interval UB")
#plt.plot(eles,bias_lb,label="PF bias confidence interval interval LB")


#l0=1
#L=12
#eles=np.array(range(l0,L+1))
#plt.plot(eles,bias_dis_kf,label="bias discretized kf")
#reference=np.array([1/2**(i) for i in eles])*2**4*bias[4]
#print(bias_dis_kf-reference)
#plt.plot(eles,reference,label="Reference")
#plt.plot(eles,np.abs(bias_dis_kf-reference),label="Reference")
#plt.plot(eles,np.sqrt(var_pf)*1.28/np.sqrt(samples),label="bias_confidence_interval")
#plt.plot(eles[1:],np.abs(x_pf_r),label="Richardson bias")
#plt.plot(eles[1:],x_pf_r_ub,label="Richardson confidence interval") 
#plt.plot(eles,var_pf)
#x_rich = x_pf[1:]
#print(x_pf.shape, kbf.shape)

#bias=np.abs(np.mean(x_pf-kbf[:-1],axis=1))
#print(bias.shape)
#plt.plot(enes,bias)
#plt.xscale("log")
#plt.yscale("log")
plt.xlabel("Levels")
plt.legend()
#plt.savefig("DKF_bias.pdf")
#plt.savefig("biases.pdf")
"""




#%%
"""


d=2**(-1)
N=9000
resamp_coef=0.8
g_par=cov
l=3
start=time.time()

[log_weights,x_pf]=pff.Cox_PF(T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,l,d,N,dim,resamp_coef,g_den,g_par,Cte_Lambda,Lamb_par)
end=time.time()
#print(log_weights)
print(end-start)
#[x0,x1,log_weights0,log_weights1,x0_pf,x1_pf]=\
#CCPF(T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,l,d,N,dim,resamp_coef,g_den,g_par,Lambda,Lamb_par)
"""
#%%
"""
weights=norm_logweights(log_weights,ax=1)
x_pfmean=np.sum(np.reshape(weights,(int(T/d),N,1))*x_pf,axis=1)
#print(x_pfmean)
dtimes=np.array(range(int(T/d)))*d+d
plt.plot(dtimes,x_pfmean,label="PF")
#plt.plot(times,v,label="True signal")
#plt.plot(obs_time,obs,label="Observations")
#plt.plot(times,kbf,label="KBF")

plt.plot(times,kf,label="KF")
#plt.plot(times,v,label="True signal")

 


plt.legend()
"""
#%%
#COMPARATATION BETWEEN THE Cox_PF AND THE KBF FOR IRREGULAR TIME OBSERVATIONS

"""    
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
B=diags(np.random.normal(-6,0.1,dim),0).toarray()*(2/3)
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

print(len(obs_time))
plt.plot(times,x_true,label="True signal")
#plt.plot(times,v,label="True signal")
plt.plot(obs_time,obs,label="Observations")
#plt.plot(times,kbf,label="KBF")
#plt.plot(times,kf,label="KF")
#plt.plot(times,kf-kbf)
#print(len(obs_time))
plt.legend()
#print(obs_time)
"""
#%%
"""
np.savetxt("obs_1d.txt",obs,fmt='%f')
np.savetxt("obs_time_1d.txt",obs,fmt='%f')
np.savetxt("true_signal_1d.txt",x_true,fmt='%f')
"""

#%%
"""
sd=np.loadtxt("true_signal_1d.txt",dtype=float)
print(x_true,sd)
"""
#%%


def CCPF(T,xin,b_ou,A,Sig_ou,fi,obs,obs_time,l,d,N,dim,resamp_coef,g_den,g_par,Lambda,Lamb_par):
    # Memory friendly version
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
    #x0=np.zeros((2**(l-1)*T+1,N,dim))
    #x1=np.zeros((2**l*T+1,N,dim))
    x0_pf=np.zeros((int(T/d),N,dim))
    x1_pf=np.zeros((int(T/d),N,dim))
    log_weights0=np.zeros((int(T/d),N))
    log_weights1=np.zeros((int(T/d),N))                                                        
    x0_new=xin
    x1_new=xin
    #x0[0]=xin
    #x1[0]=xin
    #d_steps0=int(d*2**(l-1))
    #d_steps1=int(d*2**l)
    c_indices=np.digitize(obs_time,d*np.array(range(int(T/d)+1)))-1

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
        log_weights0[i]=G(yi,obti-i*d,xi0,Lambda,Lamb_par,l-1,N,dim,g_den,g_par)
        log_weights1[i]=G(yi,obti-i*d,xi1,Lambda,Lamb_par,l,N,dim,g_den,g_par)
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
    
    
    
    return [log_weights0,log_weights1,x0_pf ,x1_pf]
#%%
#BEGINNING OF THE TEST


#First test on the CCPF and the Cox_PF for the GBM
"""
np.random.seed(24)
T=10
dim=2
xin=np.abs(np.random.normal(0,5,dim))
xin[1]=0
print(xin)
#print(x0)
l=10
#collection_input=[]
I=identity(dim).toarray()
#xin0=np.abs(np.random.normal(1,1,dim))
#xin1=xin0
mu=np.abs(np.random.normal(1,0,dim))
sigs=np.abs(np.random.normal(np.sqrt(2),0,dim))
#comp_matrix = ortho_group.rvs(dim)
comp_matrix =I
inv_mat=comp_matrix.T
S=diags(np.random.normal(1,0,dim),0).toarray()
Sig=(inv_mat@S@comp_matrix)
Sig=Sig/np.diag(Sig)
print(Sig)
fi=[sigs,Sig]


collection_input=[dim, b_gbm,mu,Sig_gbm,fi]
cov=I*1e0
g_par=[dim,cov]
B=5
x_trues=np.zeros((B,int(T*2**(l)+1),dim))
for i in range(B):
    x_trues[i]=gen_gen_data(T,xin,l,collection_input)
x_true_mean=np.mean(np.log(x_trues),axis=0)
x_true=gen_gen_data(T,xin,l,collection_input)
Lamb_par=10

[obs_time,obs]=gen_obs(x_true,l,T,Lambda,Lamb_par,g_normal,g_par)
print(obs_time)
# plots of the test
times=2**(-l)*np.array(range(int(T*2**l+1)))
plt.plot(times,x_true[:,0])
plt.plot(obs_time,obs[:,0])
"""
#%%
"""
a=1
print(x_true)
#print(np.log(xin))
plt.plot(times,x_true_mean[:,a])
#plt.plot(obs_time,obs[:,a])
print(len(obs_time))
#print(obs_time)
"""

#%%
"""

l=6
d=2**(-2)
N=3
resamp_coef=0.8
g_par=cov


[x,log_weights,x_pf]=pff.Cox_PF(T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,N,dim,resamp_coef,g_den,g_par,Lambda,Lamb_par)
[x0,x1,log_weights0,log_weights1,x0_pf,x1_pf]=\
CCPF(T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,N,dim,resamp_coef,g_den,g_par,Lambda,Lamb_par)
"""

#%%
"""
d_steps0=int(d*2**(l-1))
d_steps1=int(d*2**(l))
spots0=np.arange(d_steps0,2**(l-1)*T+1,d_steps0,dtype=int)
spots1=np.arange(d_steps1,2**(l)*T+1,d_steps1,dtype=int)    
weights0=norm_logweights(log_weights0,ax=1)
weights1=norm_logweights(log_weights1,ax=1)
weights=norm_logweights(log_weights,ax=1)

x_pfmean0=np.sum(np.reshape(weights0,(int(T/d),N,1))*x0_pf,axis=1)
x_pfmean1=np.sum(np.reshape(weights1,(int(T/d),N,1))*x1_pf,axis=1)
x_pmean=np.sum(np.reshape(weights,(int(T/d),N,1))*x_pf,axis=1)
"""
#%%
"""
ind=0
print(x_pfmean0[:,ind])
plt.plot(times,x_true[:,ind])
plt.plot(obs_time,obs[:,ind])
plt.plot(spots0*2**(-l+1),x_pfmean0[:,ind])
plt.plot(spots1*2**(-l),x_pfmean1[:,ind])
plt.plot(spots1*2**(-l),x_pfmean[:,ind])

    #print(x_pf.shape,weights.shape)
    #suma0=np.sum(x_pf0[:,:,1]*weights0,axis=1)
    #suma1=np.sum(x_pf1[:,:,1]*weights1,axis=1)
    

"""
#END OF THE TEST
# Conclusions: There is an allegdely abnormal thing regarding the 
# control of the particles whenever we don't have observation, in this case 
# the particles tend to increase its position, even if the measurement is rather 
# small for one of the dimensions, but, since it might not be small for all of them 
# we don't have necessarely a control whenever one of the dimensions in small.
# bug found in the function G() that made differences between particles in an i
# interval without observations equal in weights-wise.


#%%


#First test on the CCPF and the Cox_PF
"""
np.random.seed(6)
T=10
dim=2
x0=np.random.normal(0,5,dim)
print(x0)
l=3
collection_input=[]
I=identity(dim).toarray()

comp_matrix = ortho_group.rvs(dim)
inv_mat=la.inv(comp_matrix)
S=diags(np.random.normal(.9,0.1,dim),0).toarray()
fi=inv_mat@S@comp_matrix
B=diags(np.random.normal(-.1,0.1,dim),0).toarray()*(2/3)
B=inv_mat@B@comp_matrix

print(B)
#B=comp_matrix-comp_matrix.T  +B 

collection_input=[dim, b_ou,B,Sig_ou,fi]
cov=I*2*1e-1
g_par=[dim,cov]
x_true=gen_gen_data(T,x0,l,collection_input)
Lamb_par=2

[obs_time,obs]=gen_obs(x_true,l,T,Lambda,Lamb_par,g_normal,g_par)
print(obs_time)
# plots of the test
times=2**(-l)*np.array(range(int(T*2**l+1)))
a=8
plt.plot(times,x_true)
plt.plot(obs_time,obs)
print(len(obs_time))
#print(obs_time)
#"""

#%%
"""


d=2**(-1)
N=100
resamp_coef=0.8
xin=x0
g_par=cov


[x,log_weights,x_pf]=pff.Cox_PF(T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,l,d,N,dim,resamp_coef,g_den,g_par,Lambda,Lamb_par)
[x0,x1,log_weights0,log_weights1,x0_pf,x1_pf]=\
CCPF(T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,l,d,N,dim,resamp_coef,g_den,g_par,Lambda,Lamb_par)
"""

#%%

"""
#For the coupling we have 
np.random.seed(6)
T=10
dim=2
xin=np.random.normal(0,5,dim)
print(xin)
l=3
collection_input=[]
I=identity(dim).toarray()

comp_matrix = ortho_group.rvs(dim)
inv_mat=la.inv(comp_matrix)
S=diags(np.random.normal(.9,0.1,dim),0).toarray()
fi=inv_mat@S@comp_matrix
B=diags(np.random.normal(-.1,0.1,dim),0).toarray()*(2/3)
B=inv_mat@B@comp_matrix

print(B)
#B=comp_matrix-comp_matrix.T  +B 

collection_input=[dim, b_ou,B,Sig_ou,fi]
cov=I*2*1e-1
g_par=[dim,cov]
x_true=gen_gen_data(T,xin,l,collection_input)
Lamb_par=2

[obs_time,obs]=gen_obs(x_true,l,T,Lambda,Lamb_par,g_normal,g_par)
print(obs_time)
# plots of the test
times=2**(-l)*np.array(range(int(T*2**l+1)))
a=8
plt.plot(times,x_true)
plt.plot(obs_time,obs)
#print(obs_time)
"""
#%%


"""
Samp=400
d=1./2.**1
N=100
#N0=200
print(xin.shape)
resamp_coef=0.8
levels=np.array(range(2,10))
n_levels=len(levels)
xc_samp=np.zeros((n_levels,Samp,dim))
xi_samp=np.zeros((n_levels,Samp,dim))


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
    for samp in range(Samp):
        np.random.seed(samp)
        [x0,x1,log_weights0,log_weights1,x0_pf,x1_pf]= CCPF(T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,l,d,N,dim,resamp_coef,g_den,g_par[1],Lambda,Lamb_par)
        # (T,z,lmax,xin,b_ou,A,Sig_ou,fi,ht,H,l,d,N,dim,resamp_coef,para=True)
        
        
        #CPF(T,z,lmax,xin,b_ou,A,Sig_ou,fi,ht,H,l,d,N,dim,para=True)
        #print(norm_logweights(log_weights1,ax=1).shape,x1_pf.shape)
        
        w1=np.reshape(norm_logweights(log_weights1,ax=1),(int(T/d),N,1))
        w0=np.reshape(norm_logweights(log_weights0,ax=1),(int(T/d),N,1))
        mean0=np.sum((w0*x0_pf),axis=1)
        mean1=np.sum((w1*x1_pf),axis=1)
        #tel_samp=np.mean(x1[-1]-x0[-1],axis=0) #From telescopic sample
        xc_samp[i,samp]=np.sum((mean1-mean0)**2,axis=0)
      
        np.random.seed(samp)
        [x0i,log_weights,x0_pfi]= pff.Cox_PF(T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,l-1,d,N,dim,resamp_coef,g_den,g_par[1],Lambda,Lamb_par)
        np.random.seed(samp+1)
        [x1i,log_weights,x1_pfi]= pff.Cox_PF(T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,l,d,N,dim,resamp_coef,g_den,g_par[1],Lambda,Lamb_par)
        w1i=np.reshape(norm_logweights(log_weights1,ax=1),(int(T/d),N,1))
        w0i=np.reshape(norm_logweights(log_weights0,ax=1),(int(T/d),N,1))
        mean0i=np.sum(np.mean((w0i*x0_pfi),axis=0))
        mean1i=np.sum(np.mean((w1i*x1_pfi),axis=0))
        
        xi_samp[i,samp]=np.sum((mean1i-mean0i)**2,axis=0)
        
        
       
        xmean0i=np.sum(x0_pfi*w0i,axis=(1,2))
        xmean1i=np.sum(x1_pfi*w1i,axis=(1,2))
        plt.plot(spots0/2**(l-1),xmean0i,label="PF0I")
        #plt.plot(times1,x_true[:,a],label="True")
        plt.plot(spots0/2**(l-1),xmean1i,label="PF1I")
        
        
        ind=0
        plt.plot(spots0/2**(l-1),mean0[:,ind],label="PF0C")
        #plt.plot(times1,x_true[:,a],label="True")
        plt.plot(spots0/2**(l-1),mean1[:,ind],label="PF1C")
        
        plt.legend()
        plt.show()
        
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
#seci_tel=np.sum(np.mean(xi_samp,axis=1),axis=1)
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
#Test for the coupling of the PF
#We generare a coupling and two independent PF with subsequent levels of 
#discretization 
"""
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
#seci_tel=np.sum(np.mean(xi_samp,axis=1),axis=1)
plt.plot(levels,secc_tel,label="OU particle filter")
[ba,ma]=coef(levels,np.log2(secc_tel))
plt.plot(levels,2**(ba+ma*levels),label="rate 1: 2.69")
print(ma)
[ba,ma]=coef(levels[:-2],np.log2(secc_tel)[:-2])
print(ma)
plt.plot(levels,2**(ba+ma*levels),label="rate 2: 1.38")
#plt.plot(levels,seci_tel,label="independent")
plt.yscale("log")
plt.title("Second moment of the difference of levels for 1000 samples")
print(secc_tel)
plt.legend()
#plt.savefig("sm.pdf")
#print(varc_tel,vari_tel)
#"""

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

def MLPF_cox(T,xin,b,A,Sig,fi,obs,obs_time,eles,d,Nl,dim,resamp_coef,phi,dim_out,axis_action,g_den,g_par,Lambda,Lamb_par):
    #(T,xin,b_ou,A,Sig_ou,fi,l,d,N,dim,resamp_coef,g_den,g_par,Lambda,Lamb_par)
    #(T,z,lmax,xin,b_ou,A,Sig_ou,fi,ht,H,eles,d,Nl,dim,resamp_coef,phi,dim_out,axis_action,para=True)

    # The MLPF stands for Multilevel Particle Filter, and uses the Multilevel methodology to compute the 
    # particle filter. 

    #ARGUMENTS:
    # The arguments are basically the same as those for the PF and CPF functions, with changes in
    # l->eles: is a 1 rank array starting with l_0 and ending with L 
    # N-> Nl: is a 1 rank array starting with the corresponding number of particles to 
    # each level
    # the new parameters are:
    # phi: function that takes as argument a rank M array and computes a function 
    # along the axis_action dimension. the dimensions of the output of phi are the 
    # same as the input except the dimension of the axis axis_action  which is 
    # changed by dim_out

    #OUTPUT
    # pf: computation of the particle filter with dimension (int(T/d),dim_out)

    pf=np.zeros((int(T/d),dim_out))
    [log_weightsl0,x_pfl0]=pff.Cox_PF(T,xin,b,A,Sig,fi,obs,obs_time,eles[0],d,Nl[0],dim,resamp_coef,g_den,g_par,Lambda,Lamb_par)
    #[log_weightsl0,x_pfl0]=pff.PF(T,z,lmax,xin,b_ou,A,Sig_ou,fi,ht,H,eles[0],d,Nl[0],dim,resamp_coef,para=True)[1:]
    phi_pf=phi(x_pfl0,axis=axis_action)
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
        print(l,Nl[i])
        [log_weights0,log_weights1,x0_pf,x1_pf]=CCPF(T,xin,b,A,Sig,fi,obs,obs_time,eles[i],d,Nl[i],dim,resamp_coef,g_den,g_par,Lambda,Lamb_par)
        #(T,z,lmax,xin,b_ou,A,Sig_ou,fi,ht,H,eles[i],d,Nl[i],dim,resamp_coef,para=True)[2:]
        #
        #log_weights[:,i-1]=[log_weights0,log_weights1]
        #x_pf[:,i-1]=[x0_pf,x1_pf]
        phi_pf0=phi(x0_pf,axis=axis_action)
        phi_pf1=phi(x1_pf,axis=axis_action)
        weights0=np.reshape(norm_logweights(log_weights0,ax=1),(int(T/d),Nl[i],1))
        weights1=np.reshape(norm_logweights(log_weights1,ax=1),(int(T/d),Nl[i],1))
        pf=pf+np.sum(phi_pf1*weights1,axis=1)-np.sum(phi_pf0*weights0,axis=1)
        
    
        
    return pf

def phi(x,axis=0):
    #phi has to keep the dimensions! i.e. keepdims=True
    return x



#%%

#FIRST TEST FOR THE MLPF_cox
"""
#Test for the MLPF_cox
np.random.seed(6)
T=10
dim=2
xin=np.random.normal(0,5,dim)
print(xin)
l=10
collection_input=[]
I=identity(dim).toarray()

comp_matrix = ortho_group.rvs(dim)
inv_mat=la.inv(comp_matrix)
S=diags(np.random.normal(.9,0.1,dim),0).toarray()
fi=inv_mat@S@comp_matrix
B=diags(np.random.normal(-.1,0.1,dim),0).toarray()*(2/3)
B=inv_mat@B@comp_matrix

print(B)
#B=comp_matrix-comp_matrix.T  +B 

collection_input=[dim, b_ou,B,Sig_ou,fi]
cov=I*1e-0
g_par=[dim,cov]
x_true=gen_gen_data(T,xin,l,collection_input)
Lamb_par=2

[obs_time,obs]=gen_obs(x_true,l,T,Lambda,Lamb_par,g_normal,g_par)
print(obs_time)
# plots of the test
times=2**(-l)*np.array(range(int(T*2**l+1)))
a=8
plt.plot(times,x_true)
plt.plot(obs_time,obs)
print(len(obs_time))

#print(obs_time)
"""
#%%
"""
d=1./2.**2
l0=6
L=9
eles=np.array([i for i in range(l0,L+1)])
N0=2000
Nl=np.array((N0/2**(eles/2)),dtype=int)
resamp_coef=0.8
dim_out=dim
axis_action=0
g_par=cov
print(Nl,eles)
#MLPF(T,z,lmax,xin,b_ou,A,Sig_ou,fi,ht,H,eles,d,Nl,dim,resamp_coef,phi,dim_out,axis_action,para=True)
pf=MLPF_cox(T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,eles,d,Nl,dim,resamp_coef,phi,dim_out,axis_action,g_den,g_par,Lambda,Lamb_par)

"""
#%%
"""
a=1
times1=np.array(range(int(T*2**l+1)))/2**l
plt.plot(obs_time,obs[:,a])
plt.plot(times1,x_true[:,a],label="True signal")
dtimes=np.array(range(int(T/d)))*d+d


plt.plot(dtimes,pf[:,a],label="MLPF")


plt.legend()    
"""
#Conclusion: the test seems to be working well. Even if the observations are not accurate.
#END OF THE TEST
#%%

#Second TEST FOR THE MLPF_cox
"""
np.random.seed(6)
T=10
dim=2
xin=np.random.normal(0,5,dim)
print(xin)
l=10
collection_input=[]
I=identity(dim).toarray()

comp_matrix = ortho_group.rvs(dim)
inv_mat=la.inv(comp_matrix)
S=diags(np.random.normal(.9,0.1,dim),0).toarray()
fi=inv_mat@S@comp_matrix
B=diags(np.random.normal(-.1,0.1,dim),0).toarray()*(2/3)
B=inv_mat@B@comp_matrix

print(B)
#B=comp_matrix-comp_matrix.T  +B 

collection_input=[dim, b_ou,B,Sig_ou,fi]
cov=I*1e-0
g_par=[dim,cov]
x_true=gen_gen_data(T,xin,l,collection_input)
Lamb_par=1/4

[obs_time,obs]=gen_obs(x_true,l,T,Lambda,Lamb_par,g_normal,g_par)
print(obs_time)
# plots of the test
times=2**(-l)*np.array(range(int(T*2**l+1)))
a=0
np.savetxt("obs_T10dim2l10s6.txt", obs, fmt='%f')
np.savetxt("obs_time_T10dim2l10s6.txt", obs_time, fmt='%f')
plt.plot(times,x_true)
plt.plot(obs_time,obs)
print(len(obs_time))

#print(obs_time)
"""
#%%
"""
d=1./2.**2
l0=3
L=6
eles=np.array([i for i in range(l0,L+1)])
N0=2000
Nl=np.array((N0/2**(eles/2)),dtype=int)
resamp_coef=0.8
dim_out=dim
axis_action=0
g_par=cov
"""
#%%
"""
l=15
N=8000
[log_weights,x_pf]=pff.Cox_PF(T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,l,d,N,dim,resamp_coef,g_den,g_par,Lambda,Lamb_par)[1:]
weights=np.reshape(norm_logweights(log_weights,ax=1),(int(T/d),N,1))
sol_pf= np.sum(x_pf*weights,axis=1)
"""
#%%
"""
#sol_reshaped=sol_pf.reshape(sol_pf.shape[0],-1)
#np.savetxt("sol_T10dim2l15N20000s6.txt", sol_reshaped, fmt='%f')
sol_reshaped=np.loadtxt("sol_T10dim2l15N20000s6.txt", dtype="float")
sol_pf=np.reshape(sol_reshaped,(int(T/d),dim))
a=0
l=10
times1=np.array(range(int(T*2**l+1)))/2**l
plt.plot(obs_time,obs[:,a],label="Observations")
plt.plot(times1,x_true[:,a],label="True signal")
dtimes=np.array(range(int(T/d)))*d+d
l=16
times_sol=np.array(range(int(T/d)))*d
plt.plot(dtimes,sol_pf[:,a],label="PF")
plt.legend()
"""
#%%

#TEST!
"""
# IN the following I'll test the bias of the particle filter and see it's 
# relation to the level.
g_par=cov
N=8000
l0=3
L=10
eles=np.array(range(l0,L+1))
pfs=np.zeros((len(eles),int(T/d),dim))

for i in range(len(eles)):
    print(eles[i])

    [log_weightsl0,x_pfl0]=pff.Cox_PF(T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,eles[i],d,N,dim,resamp_coef,g_den,g_par,Lambda,Lamb_par)[1:]
    #[log_weightsl0,x_pfl0]=pff.PF(T,z,lmax,xin,b_ou,A,Sig_ou,fi,ht,H,eles[0],d,Nl[0],dim,resamp_coef,para=True)[1:]
    phi_pf=phi(x_pfl0,axis=axis_action)
    weightsl0=np.reshape(norm_logweights(log_weightsl0,ax=1),(int(T/d),N,1))
    # The reshape of weightsl0 is such that we can multiply easily the weights
    # by all dimensions of the state space model.
    pf= np.sum(phi_pf*weightsl0,axis=1)
    pfs[i]=pf
"""  
#%%
"""
a=1
biases=np.linalg.norm(pfs-sol_pf,axis=2)[:,-1]
plt.plot(eles,biases)
"""
#%%
"""
a=0
l=10
times1=np.array(range(int(T*2**l+1)))/2**l
plt.plot(obs_time,obs[:,a])
plt.plot(times1,x_true[:,a],label="True signal")
dtimes=np.array(range(int(T/d)))*d+d
l=16
times_sol=np.array(range(int(T/d)))*d
plt.plot(dtimes,sol_pf[:,a],label="True signal")


plt.plot(dtimes,pf[:,a],label="MLPF")


plt.legend() 
"""   
#"""
#%%
"""
print(Nl,eles)
#MLPF(T,z,lmax,xin,b_ou,A,Sig_ou,fi,ht,H,eles,d,Nl,dim,resamp_coef,phi,dim_out,axis_action,para=True)
pf=MLPF_cox(T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,eles,d,Nl,dim,resamp_coef,phi,dim_out,axis_action,g_den,g_par,Lambda,Lamb_par)

"""
#Conclusion: 
#END OF THE TEST

#%%
#TEST!
#We check the variance of the PF with N=8000, which shouldn't be much
"""
g_par=cov
N=8000
l=7
b=100
pfs=np.zeros((b,int(T/d),dim))
with progressbar.ProgressBar(max_value=b) as bar:
    for i in range(b):
        print(i)
        [log_weightsl0,x_pfl0]=pff.Cox_PF(T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,l,d,N,dim,resamp_coef,g_den,g_par,Lambda,Lamb_par)[1:]
        #[log_weightsl0,x_pfl0]=pff.PF(T,z,lmax,xin,b_ou,A,Sig_ou,fi,ht,H,eles[0],d,Nl[0],dim,resamp_coef,para=True)[1:]
        phi_pf=phi(x_pfl0,axis=axis_action)
        weightsl0=np.reshape(norm_logweights(log_weightsl0,ax=1),(int(T/d),N,1))
        # The reshape of weightsl0 is such that we can multiply easily the weights
        # by all dimensions of the state space model.
        pf= np.sum(phi_pf*weightsl0,axis=1)
        pfs[i]=pf
        bar.update(i)
"""
#%%
"""
var=np.var(pfs,axis=0)
print(var)     

"""
#Conclusion: The variance of the Cox_PF at level l=7 is approximately O(10^{-4})
#Thus the results, the mean of the error squared is O(10^{-4}) and thus the 
#mean of the error is about O(10^{4})
#
#END OF THE TEST





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

#In the following we find the analysys for the bias for the 
# parallelized computations.
#"""
#%%

"""
samples=400
np.random.seed(6)
T=10
dim=2
if True==True:
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
    l=10
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    x_true=gen_gen_data(T,xin,l,collection_input)
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

#print(obs_time)
"""
#%%
"""
pfs=np.loadtxt("pfs2.txt", dtype="float")
pfs=np.mean(np.reshape(pfs,(len(eles),samples,int(T/d),dim)),axis=1)
#log_weightss_bias=np.loadtxt("log_weightss_bias1.txt", dtype="float")

sol_pf=np.loadtxt("sol_T10dim2l15N2000s6.txt", dtype="float")
"""
#%%

#print(pfs.shape)
#%%
"""
a=0
l=10
times=2**(-l)*np.array(range(int(T*2**l+1)))
dtimes=np.array(range(int(T/d)))*d+d

#Rbiases=np.mean(np.linalg.norm( pfs[1:]-pfs[:-1],axis=-1),axis=1)
Rbiases_final=np.linalg.norm(pfs[1:]-pfs[:-1],axis=-1)[:,-1]
#the final refer to the final time of the system
Rbiases_all=np.mean(np.linalg.norm(pfs[1:]-pfs[:-1],axis=-1),axis=-1)
print(Rbiases_all.shape,Rbiases_final.shape)
biases=np.linalg.norm(pfs-sol_pf,axis=2)[:,-1]
#print(obs_time)
#plt.plot(obs_time,obs[:,a])
#plt.plot(dtimes,pfs[:,:,a].T)
#plt.plot(times,x_true[:,a] )
#plt.plot(obs_time,obs[:,a])
#plt.plot(dtimes,sol_pf[:,a])
"""
#%%
"""
#plt.plot(eles,np.abs(biases))
ref=Rbiases_final[0]*np.array([2**(-l/2) for l in range(4,14)])*2**(5/2)

[b0_final,b1_final]=coef(eles[1:],np.log2(Rbiases_final))
print([b0_final,b1_final])
print("alpha",-2*b1_final)
[b0_all,b1_all]=coef(eles[1:],np.log2(Rbiases_all))
print(2**b0_all,(2**b0_all/(1-np.sqrt(2)))**2)
print(2**b0_final,(2**b0_final/(1-np.sqrt(2)))**2)
print([b0_all,b1_all])
print("alpha",-2*b1_all)
plt.plot(eles[1:],2**(b0_all)*2**(eles[1:]*b1_all),label="regression")

#plt.plot(eles[1:],2**(b0_final)*2**(eles[1:]*b1_final))
#plt.plot(eles[1:],np.abs(Rbiases_final))
plt.plot(eles[1:],np.abs(Rbiases_all),label="sampled bias")
plt.plot(eles[1:],np.abs(ref),label="Reference")
#plt.plot()
plt.yscale("log")
plt.xlabel("levels")
plt.title("bias of the single particle filter")
plt.legend()
plt.savefig("biases.pdf")
plt.show()
#plt.plot(eles,biases)



"""
#END OF THE ANALYSIS
#CONCLUSIONS: For the average all over the time line (Rbiase_all) the constant
# of proportionality is C=0.0018075469495926954
#F or the bias at the final time (Rbiase_final) the constant
# of proportionality is C=0.00041430479860187646
#I think we should use the bigger one C=0.0018075469495926954
#%% 
#In the following we make an analysis comparaing the variancees of the levels 
# of the MLPF
"""
if True==True:   
    #start=time.time()
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
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    cov=I*1e-0
    g_par=cov
    l=10
    x_true=gen_gen_data(T,xin,l,collection_input)
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
#%%
pfs=np.loadtxt("pfs_beta.txt", dtype="float")
pfs=np.reshape(pfs,(2,samples,int(T/d),dim))
var_pfs0=np.sum(np.var(pfs[0],axis=0),axis=-1)
var_pfs_sum=np.sum(np.var(pfs[1],axis=0),axis=-1)

#%%
a=0
l=10
times=2**(-l)*np.array(range(int(T*2**l+1)))
dtimes=np.array(range(int(T/d)))*d+d

#Rbiases=np.mean(np.linalg.norm( pfs[1:]-pfs[:-1],axis=-1),axis=1)
Rbiases=np.linalg.norm( pfs[1:]-pfs[:-1],axis=-1)[:,-1]
print(Rbiases.shape)
Total_pfs=np.mean(pfs[1],axis=0)
plt.plot(dtimes,Total_pfs[:,a])
#biases=np.linalg.norm(pfs-sol_pf,axis=2)[:,-1]
#print(obs_time)
#plt.plot(obs_time,obs[:,a])
#plt.plot(dtimes,pfs[:,:,a].T)
plt.plot(times,x_true[:,a] )
plt.plot(obs_time,obs[:,a])
#plt.plot(dtimes,sol_pf[:,a])
#%%
mean_var0=np.mean(var_pfs0)
mean_var_sum=np.mean(var_pfs_sum)
plt.plot(dtimes,var_pfs0)
plt.plot(dtimes,var_pfs_sum)
print(mean_var_sum/(mean_var0*(L-l0)))
#%%

"""
#END OF THE ANALYSIS

#RESULTS: with a number l0=3, and L=8 the variance of the level l0 is approx 0.0147
# and the variance of the sum of the rest of the telescoping sum is 0.00289, thus the
# ration between beta and beta0 is beta/beta0=0.03932


   
   
