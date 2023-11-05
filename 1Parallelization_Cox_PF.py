

#Upload the information and the packages 

from Un_cox_PF_functions_def import *
#from PF_functions_def import *
import PF_functions_def as pff
import multiprocessing
import time
import math
import numpy as np
import matplotlib.pyplot as plt 
#import progressbar
from scipy import linalg as la
from scipy.sparse import identity
from scipy.sparse import rand
from scipy.sparse import diags
from scipy.sparse import triu
import copy
#from sklearn.linear_model import LinearRegression
from scipy.stats import ortho_group
from scipy.stats import multivariate_normal
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

def PCox_PF(arg_col):
    [seed_val,collection_input]=arg_col
    [T,xin,b_ou,A,Sig_ou,fi,obs,obs_time,l,d,N,dim\
    ,resamp_coef,g_den,g_par,Lambda,Lamb_par]=collection_input
    np.random.seed(seed_val)
    print(seed_val,l,N)
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
        #print(yi,obti-i*d,)
        #print(x_new,xi)
        #print(xi)
        #Things that could be wrong
        # observations, x_new, weights
        #observations seem to be fine

        log_weights[i]=log_weights[i]+pff.Gox(yi,obti-i*d,xi,Lambda,Lamb_par,l,N,dim,g_den,g_par)
        weights=norm_logweights(log_weights[i],ax=0)
        #print(yi,weights)
        #seed_val=i
        #print(weights.shape)
        x_last=xi[-1]
        x_pf[i]=xi[-1]
        ESS=1/np.sum(weights**2)
        #print(ESS,resamp_coef*N)
        if ESS<resamp_coef*N:
        #if True==False:
            #[part0,part1,x0_new,x1_new]=max_coup_sr(w0,w1,N,xi0[-1],xi1[-1],dim)
            #print(x_new.shape)
            x_new=pff.multi_samp(weights,N,x_last,dim)[1]
            #print(x_new.shape)
        else:
            
            #print("time is",i)
            x_new=x_last
            if i< int(T/d)-1:
                log_weights[i+1]=log_weights[i]
        #print(i)
    
    weights=norm_logweights(log_weights,ax=1)

    
    x_pfmean=np.sum(np.reshape(weights,(int(T/d),N,1))*x_pf,axis=1)
       #x_new=sr(weights,N,x_pf[i],dim)[1]
    #weights=np.reshape(norm_logweights(log_weights,ax=1),(int(T/d),N,1))
    #pf=np.sum(weights*x_pf,axis=1)
    #Filter
    #spots=np.arange(d_steps,2**l*T+1,d_steps,dtype=int)
    #x_pf=x[spots]
    #weights=norm_logweights(log_weights,ax=1)
    #print(x_pf.shape,weights.shape)
    #suma=np.sum(x_pf[:,:,1]*weights,axis=1)
    return x_pfmean

def PM_coup(arg_col):
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
    # process is 2^{l-1}, the distance of resampling, the number ofd
    # particles N, and the dimension of the problem dim=d_x
    # OUTCOMES: x0 and x1 are arrays of rank 3 with dimension 2**(l-1)*d,N,dim 
    # and  2**l*d,N,dim respectively, these arrays represents the paths simulated
    # along the discretized time for a number of particles N.


    [seed_val,collection_input]=arg_col
    [xin0,xin1,b,A,Sig,fi,l,d,N,dim]=collection_input

    steps0=int(np.float_power(2,l-1)*d)
    steps1=int(np.float_power(2,l)*d)
    dt1=1./np.float_power(2,l)
    dt0=2./np.float_power(2,l-1)
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
            # Uncomment the following lines for the nldt process
            #x1[2*t0+s+1]=x1[2*t0+s]+b(x1[2*t0+s],A)*dt1+ dW[s]*\
            #(Sig(x1[2*t0+s],fi))
        # Uncomment the following two lines for the GBM and comment
        # the third line
        #diff=np.einsum("nd,njd->nj",dW[0]+dW[1],Sig(x0[t0],fi))
        #x0[t0+1]=x0[t0]+b(x0[t0],A)*dt0+ diff
        # For the OU process comment the previous two lines and uncomment 
        # the following line
        x0[t0+1]=x0[t0]+b(x0[t0],A)*dt0+ (dW[0]+dW[1])@(Sig(x0[t0],fi).T) 
        # Uncomment the following lines for the nldt process
        #x0[t0+1]=x0[t0]+b(x0[t0],A)*dt0+ (dW[0]+dW[1])*(Sig(x0[t0],fi))

    return [x0,x1]



def PCCPF_new(arg_col):

    [seed_val,collection_input]=arg_col
    [T,xin,b,A,Sig,fi,obs,obs_time,l,d,N,dim\
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
        [xi0,xi1]=M_coup(x0_new,x1_new,b,A,Sig,fi,l,d,N,dim)
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
            #x0_new=multi_samp(w0,N,xi0[-1],dim)[1]
            #x1_new=multi_samp(w1,N,xi1[-1],dim)[1]
            #[part0,part1,x0_new,x1_new]=max_coup_multi(w0,w1,N,xi0[-1],xi1[-1],dim)
            [part0,part1,x0_new,x1_new]=\
            max_coup_multi(w0,w1,N,xi0[-1],xi1[-1],dim)
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

    return [x0_pf,x1_pf,log_weights0,log_weights1]


def PCCPF(arg_col):

    [seed_val,collection_input]=arg_col
    [T,xin,b,A,Sig,fi,obs,obs_time,l,d,N,dim\
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
        [xi0,xi1]=M_coup(x0_new,x1_new,b,A,Sig,fi,l,d,N,dim)
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
            #x0_new=multi_samp(w0,N,xi0[-1],dim)[1]
            #x1_new=multi_samp(w1,N,xi1[-1],dim)[1]
            #[part0,part1,x0_new,x1_new]=max_coup_multi(w0,w1,N,xi0[-1],xi1[-1],dim)
            [part0,part1,x0_new,x1_new]=\
            max_coup_multi(w0,w1,N,xi0[-1],xi1[-1],dim)
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

    pf=np.zeros((int(T/d),dim_out))
    [log_weightsl0,x_pfl0]=pff.Cox_PF(T,xin,b,A,Sig,fi,obs,obs_time,eles[0],d\
    ,Nl[0],dim,resamp_coef,g_den,g_par,Lambda,Lamb_par)
    #[log_weightsl0,x_pfl0]=pff.PF(T,z,lmax,xin,b_ou,A,Sig_ou,fi,ht,H,eles[0],
    #d,Nl[0],dim,resamp_coef,para=True)[1:]
    phi_pf=phi(x_pfl0,ax=2)
    weightsl0=np.reshape(norm_logweights(log_weightsl0,ax=1),(int(T/d),Nl[0],1))
    # The reshape of weightsl0 is such that we can multiply easily the weights
    # by all dimensions of the state space model.
    pf= np.sum(phi_pf*weightsl0,axis=1)
    #pf_levels=np.zeros((2,len(eles),int(T/d),dim))
    #pf_levels[1,0]=pf
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
        #pf_levels[0,i]=np.sum(phi_pf0*weights0,axis=1)
        #pf_levels[1,i]=np.sum(phi_pf1*weights1,axis=1)
        pf=pf+np.sum(phi_pf1*weights1,axis=1)-np.sum(phi_pf0*weights0,axis=1)
        
        
    return pf

def PSUCPF(arg_col):
    [seed_val,collection_input]=arg_col
    (T,xin,b,A,Sig,fi,obs,obs_time,eles,Pl,d,enes,Pp,dim,resamp_coef,phi,\
    g_den,g_par,Lambda,Lamb_par)=collection_input
    np.random.seed(seed_val)
    

    
    # This function computes the Single term unbiased estimator for the 
    # PF with Cox observations. SUCPF stands for Single Unbiased Cox Particle
    # Filter
    # ARGUMENTS: 
    
    # The arguments are basically the same as those for the PF and CPF
    #functions, with changes in
    # l->eles: is a 1 rank array starting with l_0 and ending with Lmax, which is
    # the level of truncation of the time discretization
    # N-> enes: Is the number of particles corresponding to each level.
    # the new parameters are:
    # Pp: rank 1 array with dimention pmax+1, its entries correspond to the 
    # probabilities given each particle level.
    # Pl:  rank 1 array with dimention Lmax+1, its entries correspond to the 
    # probabilities given each time discretization level.
    # phi: function that takes as argument a rank M array and computes a function 
    # along the ax dimension. the dimensions of the output of phi are the 
    # same as the input except the dimension of the axis ax which can be any.
    # Since the argument of phi here is x_pf, then the ax is the axis
    # 2, the one of the dimensions.
    # changed by dim_out
    # g_den: Likelihood function of the observations where g_par 
    #is the parameters of function.


    l=eles[sampling(Pl)]
    p=sampling(Pp)

    print(seed_val,p,l)

    if l==eles[0]:
        log_weights=np.zeros((int(T/d),enes[p]))
        x_pf=np.zeros((int(T/d),enes[p],dim))
        nll=0
        for pi in range(p+1):
            
            [lw,x]=pff.Cox_PF(T, xin, b, A, Sig, fi, obs, obs_time, l, d, enes[pi]-nll\
            , dim, resamp_coef, g_den, g_par, Lambda, Lamb_par)
            log_weights[:,nll:enes[pi]]=lw
            x_pf[:,nll:enes[pi]]=x
            nll=enes[pi]
        
        if p==0:
            weights=norm_logweights(log_weights,ax=1)           
            phi_pfmean=np.sum(np.reshape(weights,(int(T/d),enes[p],1))*phi(x_pf,ax=2),axis=1)
            tel_est=phi_pfmean
        else:
            weightsA=norm_logweights(log_weights,ax=1)           
            phi_pfmeanA=np.sum(np.reshape(weightsA,(int(T/d),enes[p],1))*phi(x_pf,ax=2),axis=1)
            #THE FOLLOWING COMMENTS ARE MADE IN ORDER TO GET MAKE SOME COMPUTATION FOR 
            # THE BIAS WHERE THE SUBTRACTION IS NOT NEEDED
            weightsB=norm_logweights(log_weights[:,:enes[p-1]],ax=1)           
            phi_pfmeanB=np.sum(np.reshape(weightsB,(int(T/d),enes[p-1],1))*phi(x_pf[:,:enes[p-1]],ax=2),axis=1)
            #phi_pfmeanB=0
            tel_est=phi_pfmeanA-phi_pfmeanB
            
    else:
        log_weights0=np.zeros((int(T/d),enes[p]))
        log_weights1=np.zeros((int(T/d),enes[p]))
        x_pf0=np.zeros((int(T/d),enes[p],dim))
        x_pf1=np.zeros((int(T/d),enes[p],dim))
        nll=0
        for pi in range(p+1):
            [lw0,lw1,x0,x1]=CCPF(T, xin, b, A, Sig, fi, obs, obs_time, l, d, enes[pi]-nll\
            , dim, resamp_coef, g_den, g_par, Lambda, Lamb_par)
            log_weights0[:,nll:enes[pi]]=lw0
            log_weights1[:,nll:enes[pi]]=lw1
            x_pf0[:,nll:enes[pi]]=x0
            x_pf1[:,nll:enes[pi]]=x1
            nll=enes[pi]
        
        
        if p==0:
            weights0=norm_logweights(log_weights0,ax=1)           
            weights1=norm_logweights(log_weights1,ax=1)           
            phi_pfmean0=np.sum(np.reshape(weights0,(int(T/d),enes[p],1))*phi(x_pf0,ax=2),axis=1)
            phi_pfmean1=np.sum(np.reshape(weights1,(int(T/d),enes[p],1))*phi(x_pf1,ax=2),axis=1)
            tel_est=phi_pfmean1-phi_pfmean0
        else:
            weights0A=norm_logweights(log_weights0,ax=1)           
            weights1A=norm_logweights(log_weights1,ax=1)           
            phi_pfmean0A=np.sum(np.reshape(weights0A,(int(T/d),enes[p],1))*phi(x_pf0,ax=2),axis=1)
            phi_pfmean1A=np.sum(np.reshape(weights1A,(int(T/d),enes[p],1))*phi(x_pf1,ax=2),axis=1)
            
            weights0B=norm_logweights(log_weights0[:,:enes[p-1]],ax=1)           
            weights1B=norm_logweights(log_weights1[:,:enes[p-1]],ax=1)           
            phi_pfmean0B=np.sum(np.reshape(weights0B,(int(T/d),enes[p-1],1))*phi(x_pf0[:,:enes[p-1]],ax=2),axis=1)
            phi_pfmean1B=np.sum(np.reshape(weights1B,(int(T/d),enes[p-1],1))*phi(x_pf1[:,:enes[p-1]],ax=2),axis=1)
            tel_est=phi_pfmean1A-phi_pfmean0A-(phi_pfmean1B-phi_pfmean0B)
        

    return [tel_est,[l,p]]


"""
#%%
p=1
l=2
seed_val=591517

pmax=13
Lmax=13
pma=8
eNes=np.concatenate(([3,6,12,24,50],100*2**np.array(range(0,pma+1))))
eLes=np.arange(l0,Lmax+1)
Pp_cum_det=np.concatenate((np.zeros(p), 1+np.zeros(pmax+1-p)))
Pl_cum_det=np.concatenate((np.zeros(l), 1+np.zeros(Lmax+1-l)))
#lps[sample]=np.array([1,l,p,samplesv2])
#samplesv2+=1
collection_input=[T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,eLes,Pl_cum_det,d,eNes\
,Pp_cum_det,dim,resamp_coef,phi,g_den,g_par,Norm_Lambda,Lamb_par]
    
arg_col=[seed_val,collection_input]

result=PSUCPF(arg_col)
#print(result)
#%%
W=np.array([0.09910599 ,0.90089401, 0.        ])
print(np.sum(W))
N=3
samp=np.random.choice(len(W),size=N,p=W,replace=True)
print(samp)
"""
#%%
if __name__ == '__main__':  
  
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=13
    l=10
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
    #print(obs_time,obs,len(obs_time))
    start=time.time()
    d=2**(0)
    resamp_coef=0.8    
    dim_out=2
    samples=500000
    #samples=50
    #N0=100
    N0=5
    l0=0
    l0=0
    Lmax=10
    #Lmax=5
    pmax=11
    #pmax=4
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
    
    pfs=np.zeros((samples,int(T/d),dim))
    lps=np.zeros((samples,3))    
    inputs=[]

   
    for sample in range(samples):
            
            
            
            collection_input=[T,xin,b_lan,df,Sig_lan,fi,obs,obs_time,eLes,Pl_cum,d,eNes\
            ,Pp_cum,dim,resamp_coef,phi,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample,collection_input])        

    pool = multiprocessing.Pool(processes=8)
    pool_outputs = pool.map(PSUCPF, inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    
    print("Parallelized processes time:",end-start,"\n")            
    
    for sample in range(samples):
            pfs[sample]=pool_outputs[sample][0]
            lps[sample,:2]=pool_outputs[sample][1]
            lps[sample,2]=sample
    #tel_est=np.reshape(np.array(pool_outputs),(len(eLes),samples,int(T/d),dim))
    
    #log_weightss=log_weightss.flatten()
    pfs=pfs.flatten()
    np.savetxt("Observations&data/SUCPF_lan_T100_v1_1.txt",pfs,fmt="%f")
    np.savetxt("Observations&data/SUCPF_lan_pls_T100_v1_1.txt",lps,fmt="%f")


#%%
"""

    28771470
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=13
    #l=10
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
    #print(obs_time,obs,len(obs_time))
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    samples=500000
    #samples=50
    #N0=100
    N0=5
    l0=0
    l0=0
    Lmax=10
    #Lmax=5
    pmax=11
    #pmax=4
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

    pfs=np.zeros((samples,int(T/d),dim))
    lps=np.zeros((samples,3))
    inputs=[]
    for sample in range(samples):



            collection_input=[T,xin,b_lan,df,Sig_lan,fi,obs,obs_time,eLes,Pl_cum,d,eNes\
            ,Pp_cum,dim,resamp_coef,phi,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample,collection_input])

    pool = multiprocessing.Pool(processes=40)
    pool_outputs = pool.map(PSUCPF, inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()

    print("Parallelized processes time:",end-start,"\n")

    for sample in range(samples):
            pfs[sample]=pool_outputs[sample][0]
            lps[sample,:2]=pool_outputs[sample][1]
            lps[sample,2]=sample
    #tel_est=np.reshape(np.array(pool_outputs),(len(eLes),samples,int(T/d),dim))

    #log_weightss=log_weightss.flatten()
    pfs=pfs.flatten()
    np.savetxt("Observations/SUCPF_lan_T100_v1_1.txt",pfs,fmt="%f")
    np.savetxt("Observations/SUCPF_lan_pls_T100_v1_1.txt",lps,fmt="%f")



    28535584
    
    np.random.seed(1)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=13
    #l=10
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
    #samples=5000000
    samples=1000000
    N0=100
    #N0=10
    l0=0
    Lmax=10
    #Lmax=5
    pmax=5
    #pmax=3
    eLes=np.arange(l0,Lmax+1)
    ps=np.array(range(pmax+1))
    eNes=N0*2**ps
    beta=1/2
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
    Pp_cum=np.cumsum(Ppd)
    Pl_cum=np.cumsum(Pld)

    print(Ppd,Pld,Pp_cum,Pl_cum)

    pfs=np.zeros((samples,int(T/d),dim))
    lps=np.zeros((samples,3))
    inputs=[]


    for sample in range(samples):



            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,eLes,Pl_cum,d,eNes\
            ,Pp_cum,dim,resamp_coef,phi,g_den,g_par,Norm_Lambda,Lamb_par]

            inputs.append([sample+500000,collection_input])

    pool = multiprocessing.Pool(processes=120)
    pool_outputs = pool.map(PSUCPF, inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()

    print("Parallelized processes time:",end-start,"\n")

    for sample in range(samples):
            pfs[sample]=pool_outputs[sample][0]
            lps[sample,:2]=pool_outputs[sample][1]
            lps[sample,2]=sample
    #tel_est=np.reshape(np.array(pool_outputs),(len(eLes),samples,int(T/d),dim))

    #log_weightss=log_weightss.flatten()
    pfs=pfs.flatten()
    np.savetxt("Observations/SUCPF_ou_T100_v1_2.txt",pfs,fmt="%f")
    np.savetxt("Observations/SUCPF_ou_pls_T100_v1_2.txt",lps,fmt="%f")



SUCPF_lan_pls_T100_v1_1.txt
        np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=13
    l=10
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
    #print(obs_time,obs,len(obs_time))
    start=time.time()
    d=2**(0)
    resamp_coef=0.8    
    dim_out=2
    samples=500000
    #samples=50
    #N0=100
    N0=5
    l0=0
    l0=0
    Lmax=10
    #Lmax=5
    pmax=11
    #pmax=4
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
    
    pfs=np.zeros((samples,int(T/d),dim))
    lps=np.zeros((samples,3))    
    inputs=[]

   
    for sample in range(samples):
            
            
            
            collection_input=[T,xin,b_lan,df,Sig_lan,fi,obs,obs_time,eLes,Pl_cum,d,eNes\
            ,Pp_cum,dim,resamp_coef,phi,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample,collection_input])        

    pool = multiprocessing.Pool(processes=8)
    pool_outputs = pool.map(PSUCPF, inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    
    print("Parallelized processes time:",end-start,"\n")            
    
    for sample in range(samples):
            pfs[sample]=pool_outputs[sample][0]
            lps[sample,:2]=pool_outputs[sample][1]
            lps[sample,2]=sample
    #tel_est=np.reshape(np.array(pool_outputs),(len(eLes),samples,int(T/d),dim))
    
    #log_weightss=log_weightss.flatten()
    pfs=pfs.flatten()
    np.savetxt("Observations&data/SUCPF_lan_T100_v1_1.txt",pfs,fmt="%f")
    np.savetxt("Observations&data/SUCPF_lan_pls_T100_v1_1.txt",lps,fmt="%f")


28481342

    SUCPF_ou_T100_v1.txt
    np.random.seed(1)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=13
    #l=10
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


    #samples=5000000
    samples=500000
    N0=100
    #N0=10
    l0=0
    Lmax=10
    #Lmax=5
    pmax=5
    #pmax=3
    eLes=np.arange(l0,Lmax+1)
    ps=np.array(range(pmax+1))
    eNes=N0*2**ps
    beta=1/2
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
    Pp_cum=np.cumsum(Ppd)
    Pl_cum=np.cumsum(Pld)

    print(Ppd,Pld,Pp_cum,Pl_cum)

    pfs=np.zeros((samples,int(T/d),dim))
    lps=np.zeros((samples,3))
    inputs=[]


    for sample in range(samples):



            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,eLes,Pl_cum,d,eNes\
            ,Pp_cum,dim,resamp_coef,phi,g_den,g_par,Norm_Lambda,Lamb_par]

            inputs.append([sample,collection_input])

    pool = multiprocessing.Pool(processes=120)
    pool_outputs = pool.map(PSUCPF, inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()

    print("Parallelized processes time:",end-start,"\n")

    for sample in range(samples):
            pfs[sample]=pool_outputs[sample][0]
            lps[sample,:2]=pool_outputs[sample][1]
            lps[sample,2]=sample
    #tel_est=np.reshape(np.array(pool_outputs),(len(eLes),samples,int(T/d),dim))

    #log_weightss=log_weightss.flatten()
    pfs=pfs.flatten()
    np.savetxt("Observations/SUCPF_ou_T100_v1.txt",pfs,fmt="%f")
    np.savetxt("Observations/SUCPF_ou_pls_T100_v1.txt",lps,fmt="%f")

    
    SUCPF_ou_T100_vtest.txt
    np.random.seed(1)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    #l=13
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
    
    
    #samples=5000000
    samples=50
    #N0=100
    N0=10
    l0=0
    #Lmax=10
    Lmax=5
    #pmax=5
    pmax=3
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
    Pp_cum=np.cumsum(Ppd)
    Pl_cum=np.cumsum(Pld)
    
    print(Ppd,Pld,Pp_cum,Pl_cum)
    
    pfs=np.zeros((samples,int(T/d),dim))
    lps=np.zeros((samples,3))    
    inputs=[]

   
    for sample in range(samples):
            
            
            
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,eLes,Pl_cum,d,eNes\
            ,Pp_cum,dim,resamp_coef,phi,g_den,g_par,Norm_Lambda,Lamb_par]
    
            inputs.append([sample,collection_input])        

    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PSUCPF, inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    
    print("Parallelized processes time:",end-start,"\n")            
    
    for sample in range(samples):
            pfs[sample]=pool_outputs[sample][0]
            lps[sample,:2]=pool_outputs[sample][1]
            lps[sample,2]=sample
    #tel_est=np.reshape(np.array(pool_outputs),(len(eLes),samples,int(T/d),dim))
    
    #log_weightss=log_weightss.flatten()
    pfs=pfs.flatten()
    np.savetxt("Observations&data/SUCPF_ou_T100_vtest.txt",pfs,fmt="%f")
    np.savetxt("Observations&data/SUCPF_ou_pls_T100_vtest.txt",lps,fmt="%f")

np.random.seed(0)
       T=100
       dim=1
       dim_o=dim
       xin=np.zeros(dim)+0.1
       l=13
       #l=10
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
       start=time.time()
       d=2**(0)
       resamp_coef=0.8
       dim_out=2
       #samples=2000000
       samples=500000
       #10
       N0=100
       l0=0
       Lmax=10
       #Lmax=3
       pmax=8
       #pmax=2
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
       Pp_cum=np.cumsum(Ppd)
       Pl_cum=np.cumsum(Pld)

       print(Ppd,Pld,Pp_cum,Pl_cum)

       pfs=np.zeros((samples,int(T/d),dim))
       lps=np.zeros((samples,3))
       inputs=[]


       for sample in range(samples):



           collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,eLes,Pl_cum,d,eNes\
           ,Pp_cum,dim,resamp_coef,phi,g_den,g_par,Norm_Lambda,Lamb_par]

           inputs.append([sample+samples*0,collection_input])
       pool = multiprocessing.Pool(processes=8)
       pool_outputs = pool.map(PSUCPF, inputs)
       pool.close()
       pool.join()
       #blocks_pools.append(pool_outputs)
       xend1=time.time()
       end=time.time()

       print("Parallelized processes time:",end-start,"\n")

       for sample in range(samples):
           pfs[sample]=pool_outputs[sample][0]
           lps[sample,:2]=pool_outputs[sample][1]
           lps[sample,2]=sample
           #tel_est=np.reshape(np.array(pool_outputs),(len(eLes),samples,int(T/d),dim))

           #log_weightss=log_weightss.flatten()
       pfs=pfs.flatten()
       np.savetxt("Observations&data/SUCPF_gbm_T100_v1_1.txt",pfs,fmt="%f")
       np.savetxt("Observations&data/SUCPF_gbm_pls_T100_v1_1.txt",lps,fmt="%f")



28273496


    np.random.seed(0)
       T=100
       dim=1
       dim_o=dim
       xin=np.zeros(dim)+0.1
       l=13
       #l=10
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
       start=time.time()
       d=2**(0)
       resamp_coef=0.8
       dim_out=2
        #samples=2000000
       samples=500000
       N0=100
       l0=0
       Lmax=10
       #Lmax=3
       pmax=8
       #pmax=2
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
       Pp_cum=np.cumsum(Ppd)
       Pl_cum=np.cumsum(Pld)

       print(Ppd,Pld,Pp_cum,Pl_cum)

       pfs=np.zeros((samples,int(T/d),dim))
       lps=np.zeros((samples,3))
       inputs=[]


       for sample in range(samples):



           collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,eLes,Pl_cum,d,eNes\
           ,Pp_cum,dim,resamp_coef,phi,g_den,g_par,Norm_Lambda,Lamb_par]

           inputs.append([sample+samples*1,collection_input])
       pool = multiprocessing.Pool(processes=120)
       pool_outputs = pool.map(PSUCPF, inputs)
       pool.close()
       pool.join()
       #blocks_pools.append(pool_outputs)
       xend1=time.time()
       end=time.time()
       print("Parallelized processes time:",end-start,"\n")
       for sample in range(samples):
           pfs[sample]=pool_outputs[sample][0]
           lps[sample,:2]=pool_outputs[sample][1]
           lps[sample,2]=sample
           #tel_est=np.reshape(np.array(pool_outputs),(len(eLes),samples,int(T/d),dim))

           #log_weightss=log_weightss.flatten()
       pfs=pfs.flatten()
       np.savetxt("Observations/SUCPF_gbm_T100_v1_2.txt",pfs,fmt="%f")
       np.savetxt("Observations/SUCPF_gbm_pls_T100_v1_2.txt",lps,fmt="%f")


28273487
28340478
np.random.seed(0)
       T=100
       dim=1
       dim_o=dim
       xin=np.zeros(dim)+0.1
       l=13
       #l=10
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
       start=time.time()
       d=2**(0)
       resamp_coef=0.8
       dim_out=2
        #samples=2000000
       samples=500000
       N0=100
       l0=0
       Lmax=10
       #Lmax=3
       pmax=8
       #pmax=2
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
       Pp_cum=np.cumsum(Ppd)
       Pl_cum=np.cumsum(Pld)

       print(Ppd,Pld,Pp_cum,Pl_cum)

       pfs=np.zeros((samples,int(T/d),dim))
       lps=np.zeros((samples,3))
       inputs=[]


       for sample in range(samples):



           collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,eLes,Pl_cum,d,eNes\
           ,Pp_cum,dim,resamp_coef,phi,g_den,g_par,Norm_Lambda,Lamb_par]

           inputs.append([sample,collection_input])
       pool = multiprocessing.Pool(processes=120)
       pool_outputs = pool.map(PSUCPF, inputs)
       pool.close()
       pool.join()
       #blocks_pools.append(pool_outputs)
       xend1=time.time()
       end=time.time()

       print("Parallelized processes time:",end-start,"\n")

       for sample in range(samples):
           pfs[sample]=pool_outputs[sample][0]
           lps[sample,:2]=pool_outputs[sample][1]
           lps[sample,2]=sample
           #tel_est=np.reshape(np.array(pool_outputs),(len(eLes),samples,int(T/d),dim))

           #log_weightss=log_weightss.flatten()
       pfs=pfs.flatten()
       np.savetxt("Observations/SUCPF_gbm_T100_v1_1.txt",pfs,fmt="%f")
       np.savetxt("Observations/SUCPF_gbm_pls_T100_v1_1.txt",lps,fmt="%f")




28193588

       np.random.seed(0)
       T=100
       dim=1
       dim_o=dim
       xin=np.zeros(dim)+0.1
       l=13
       #l=10
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
       start=time.time()
       d=2**(0)
       resamp_coef=0.8
       dim_out=2

       samples=2000000
       #samples=20
       N0=100
       l0=0
       Lmax=10
       #Lmax=3
       pmax=8
       #pmax=2
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
       Pp_cum=np.cumsum(Ppd)
       Pl_cum=np.cumsum(Pld)

       print(Ppd,Pld,Pp_cum,Pl_cum)

       pfs=np.zeros((samples,int(T/d),dim))
       lps=np.zeros((samples,3))
       inputs=[]


       for sample in range(samples):



           collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,eLes,Pl_cum,d,eNes\
           ,Pp_cum,dim,resamp_coef,phi,g_den,g_par,Norm_Lambda,Lamb_par]

           inputs.append([sample,collection_input])
       pool = multiprocessing.Pool(processes=120)
       pool_outputs = pool.map(PSUCPF, inputs)
       pool.close()
       pool.join()
       #blocks_pools.append(pool_outputs)
       xend1=time.time()
       end=time.time()

       print("Parallelized processes time:",end-start,"\n")

       for sample in range(samples):
           pfs[sample]=pool_outputs[sample][0]
           lps[sample,:2]=pool_outputs[sample][1]
           lps[sample,2]=sample
           #tel_est=np.reshape(np.array(pool_outputs),(len(eLes),samples,int(T/d),dim))

           #log_weightss=log_weightss.flatten()
       pfs=pfs.flatten()
       np.savetxt("Observations/SUCPF_gbm_T100_v1.txt",pfs,fmt="%f")
       np.savetxt("Observations/SUCPF_gbm_pls_T100_v1.txt",lps,fmt="%f")



28162545

    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+0
    l=13
    #l=10
    collection_input=[]
    I=identity(dim).toarray()
    A=np.array([[0]])
    fi=1
    collection_input=[dim, b_ou,A,Sig_nldt,fi]
    cov=I*1e0
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=2/9
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    samples=2000000
    #samples=20
    N0=5
    l0=0
    Lmax=10
    #Lmax=2
    pmax=11
    #pmax=2
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
    Pp_cum=np.cumsum(Ppd)
    Pl_cum=np.cumsum(Pld)
    print(Ppd,Pld,Pp_cum,Pl_cum)
    pfs=np.zeros((samples,int(T/d),dim))
    lps=np.zeros((samples,3))
    inputs=[]
    for sample in range(samples):

        collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,eLes,Pl_cum,d,eNes\
        ,Pp_cum,dim,resamp_coef,phi,g_den,g_par,Norm_Lambda,Lamb_par]

        inputs.append([sample,collection_input])

    pool = multiprocessing.Pool(processes=120)
    pool_outputs = pool.map(PSUCPF, inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")

    for sample in range(samples):
        pfs[sample]=pool_outputs[sample][0]
        lps[sample,:2]=pool_outputs[sample][1]
        lps[sample,2]=sample
        #tel_est=np.reshape(np.array(pool_outputs),(len(eLes),samples,int(T/d),dim))

        #log_weightss=log_weightss.flatten()
    pfs=pfs.flatten()
    np.savetxt("Observations/SUCPF_nldf_T100_v1.txt",pfs,fmt="%f")
    np.savetxt("Observations/SUCPF_nldf_pls_T100_v1.txt",lps,fmt="%f")


    SUCPF_nldf_pls_T100_vtest.txt

    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+0
    #l=13
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    A=np.array([[0]])
    fi=1
    collection_input=[dim, b_ou,A,Sig_nldt,fi]
    cov=I*1e0
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=2/9
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8    
    dim_out=2
    #samples=2000000
    samples=20
    N0=5
    l0=0
    #Lmax=10
    Lmax=2
    #pmax=11
    pmax=2
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
    Pp_cum=np.cumsum(Ppd)
    Pl_cum=np.cumsum(Pld)
    print(Ppd,Pld,Pp_cum,Pl_cum)
    pfs=np.zeros((samples,int(T/d),dim))
    lps=np.zeros((samples,3))    
    inputs=[]
    for sample in range(samples):

        collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,eLes,Pl_cum,d,eNes\
        ,Pp_cum,dim,resamp_coef,phi,g_den,g_par,Norm_Lambda,Lamb_par]

        inputs.append([sample,collection_input])        

    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PSUCPF, inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()

    print("Parallelized processes time:",end-start,"\n")            

    for sample in range(samples):
        pfs[sample]=pool_outputs[sample][0]
        lps[sample,:2]=pool_outputs[sample][1]
        lps[sample,2]=sample
        #tel_est=np.reshape(np.array(pool_outputs),(len(eLes),samples,int(T/d),dim))

        #log_weightss=log_weightss.flatten()
    pfs=pfs.flatten()
    np.savetxt("Observations&data/SUCPF_nldf_T100_vtest.txt",pfs,fmt="%f")
    np.savetxt("Observations&data/SUCPF_nldf_pls_T100_vtest.txt",lps,fmt="%f")


    PPF_cox_ou_Truth_T100_scaled1_v1.txt 
    28049878
    
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
    sca=1e-4
    Lamb_par=1.33
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*1e5
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca*5e2
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=1
    es=1e-9
    C0=0.00066
    C=np.sqrt(2)*5e-6
    K=2.331505041965383e-08
    N=int(2*C0/es)
    L=int(np.ceil(-np.log2(es/(2*np.sqrt(K)))/2,dtype=float))
    l0=L
    Lmin=L
    Lmax=L
    d=1
    eLes=np.array(np.arange(Lmin,Lmax+1))
    pfs=np.zeros((len(eLes),samples,int(T/d),dim))
    inputs=[]
    collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,L,d,N,dim,resamp_coef,\
    g_den,g_par,Norm_Lambda,Lamb_par]
    inputs.append([0,collection_input])
    pool = multiprocessing.Pool(processes=1)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")
    pfs[0,0]=pool_outputs[0]
    pfs=pfs.flatten()
    np.savetxt("Observations/PPF_cox_ou_Truth_T100_scaled1_v1.txt",pfs,fmt="%f")



    .28049823
    PPF_cox_ou_t100_scaled1_v1.txt
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
    sca=1e-4
    Lamb_par=1.33
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*1e5
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca*5e2
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=25
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=0
    Lmin=0
    Lmax=5
    eLes=np.array(np.arange(Lmin,Lmax+1))
    print(eLes)
    C0=0.00066
    C=np.sqrt(2)*5e-6
    K=2.331505041965383e-08

    pfs=np.zeros((len(eLes),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eLes)):
        for sample in range(samples):
            l=eLes[i]
            scale=2e-2
            N=int(C0*2**(2*l)*scale/K)
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim,resamp_coef,\
            g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([samples*i+sample,collection_input])
    pool = multiprocessing.Pool(processes=40)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")
    for i in range(len(eLes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations/PPF_cox_ou_t100_scaled1_v1.txt",pfs,fmt="%f")



    28049810
    PPF_cox_ou_Truth_T100_scaled3_vtest.txt

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
    sca=1e-4
    Lamb_par=1.33
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*1e5
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca*5e2
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=1
    es=1e-6
    C0=0.00066
    C=np.sqrt(2)*5e-6
    K=2.331505041965383e-08
    N=int(2*C0/es)
    L=int(np.ceil(-np.log2(es/(2*np.sqrt(K)))/2,dtype=float))
    l0=L
    Lmin=L
    Lmax=L
    d=1
    eLes=np.array(np.arange(Lmin,Lmax+1))
    pfs=np.zeros((len(eLes),samples,int(T/d),dim))
    inputs=[]
    collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,L,d,N,dim,resamp_coef,\
    g_den,g_par,Norm_Lambda,Lamb_par]
    inputs.append([0,collection_input])
    pool = multiprocessing.Pool(processes=1)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")
    pfs[0,0]=pool_outputs[0]
    pfs=pfs.flatten()
    np.savetxt("Observations/PPF_cox_ou_Truth_T100_scaled3_vtest.txt",pfs,fmt="%f")


    PMLPF_cox_ou_T100_scaled1_v1.txt 
    28049434
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
    sca=1e-4
    Lamb_par=1.33
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*1e5
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca*5e2
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=0
    Lmin=0
    Lmax=5
    d=25
    eLes=np.array(np.arange(Lmin,Lmax+1))
    print(eLes)
    C0=0.00066
    C=np.sqrt(2)*5e-6
    K=2.331505041965383e-08
    pfs=np.zeros((len(eLes),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eLes)):
            for sample in range(samples):
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
                collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,eles,d,eNes,dim,resamp_coef,phi,\
                dim_out,g_den,g_par,Norm_Lambda,Lamb_par]
                inputs.append([samples*i+sample,collection_input])

    pool = multiprocessing.Pool(processes=40)
    pool_outputs = pool.map(PMLPF_cox,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")
    for i in range(len(eLes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations/PMLPF_cox_ou_T100_scaled1_v1.txt",pfs,fmt="%f")



    28049780
    PPF_cox_ou_t100_scaled1_vtest.txt
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
    sca=1e-4
    Lamb_par=1.33
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*1e5
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca*5e2
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=25
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=10
    l0=0
    Lmin=0
    Lmax=2
    eLes=np.array(np.arange(Lmin,Lmax+1))
    print(eLes)
    C0=0.00066
    C=np.sqrt(2)*5e-6
    K=2.331505041965383e-08

    pfs=np.zeros((len(eLes),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eLes)):
        for sample in range(samples):
            l=eLes[i]
            scale=1e-1
            N=int(C0*2**(2*l)*scale/K)
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim,resamp_coef,\
            g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([samples*i+sample,collection_input])
    pool = multiprocessing.Pool(processes=40)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")
    for i in range(len(eLes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations/PPF_cox_ou_t100_scaled1_vtest.txt",pfs,fmt="%f")



    CCPF_ou100_scaled3_v11_2.txt
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
    sca=1e-4
    Lamb_par=1.33
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*1e5
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca*5e2
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=25
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=20
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=5000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled3_v11_2.txt",pfs,fmt="%f")


version of the last iteration
CCPF_ou100_scaled3_enes_v11.txt
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
    sca=1e-4
    Lamb_par=1.33
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*1e5
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca*5e2
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=25
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    p=8
    N0=10
    enes=np.array([2**i for i in range(p+1)])*N0
    pfs=np.zeros((len(enes),samples,int(T/d),dim))
    inputs=[]
    l=5
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            
            inputs.append([sample+samples*i,collection_input])             
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    
    np.savetxt("Observations&data/CCPF_ou100_scaled3_enes_v11.txt",pfs,fmt="%f")

we increase the scaling and Lamb_par
CCPF_ou100_scaled3_v11.txt
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
    sca=1e-4
    Lamb_par=1.33
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*1e5
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca*5e2
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=25
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled3_v11.txt",pfs,fmt="%f")




version of the preovios iteration
    CCPF_ou100_scaled3_enes_v10.txt
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
    sca=1e-5
    Lamb_par=1.33/2
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*1e5
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca*5e2
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=25
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    p=8
    N0=10
    enes=np.array([2**i for i in range(p+1)])*N0
    pfs=np.zeros((len(enes),samples,int(T/d),dim))
    inputs=[]
    l=5
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            
            inputs.append([sample+samples*i,collection_input])             
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    
    np.savetxt("Observations&data/CCPF_ou100_scaled3_enes_v10.txt",pfs,fmt="%f")


    CCPF_ou100_scaled3_enes_v10.txt
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
    sca=1e-4
    Lamb_par=1.33/2
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*1e5
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca*5e2
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=25
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    p=8
    N0=10
    enes=np.array([2**i for i in range(p+1)])*N0
    pfs=np.zeros((len(enes),samples,int(T/d),dim))
    inputs=[]
    l=5
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            
            inputs.append([sample+samples*i,collection_input])             
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    
    np.savetxt("Observations&data/CCPF_ou100_scaled3_enes_v10.txt",pfs,fmt="%f")



    CCPF_ou100_scaled3_v10.txt
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
    sca=1e-5
    Lamb_par=1.33/2
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*1e5
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca*5e2
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=25
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled3_v10.txt",pfs,fmt="%f")



    version of CCPF_ou100_scaled3_v9.txt with N=1000
    CCPF_ou100_scaled3_v9_2.txt
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
    sca=1e-4
    Lamb_par=1.33/2
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*1e5
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca*5e2
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=25
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled3_v9_2.txt",pfs,fmt="%f")



    CCPF_ou100_scaled3_enes_v9.txt 
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
    sca=1e-4
    Lamb_par=1.33/2
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*1e5
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca*5e2
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=25
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    p=8
    N0=10
    enes=np.array([2**i for i in range(p+1)])*N0
    pfs=np.zeros((len(enes),samples,int(T/d),dim))
    inputs=[]
    l=5
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            
            inputs.append([sample+samples*i,collection_input])             
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    
    np.savetxt("Observations&data/CCPF_ou100_scaled3_enes_v9.txt",pfs,fmt="%f")



    CCPF_ou100_scaled3_v9.txt
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
    sca=1e-4
    Lamb_par=1.33/2
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*1e5
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca*5e2
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=25
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
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
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled3_v9.txt",pfs,fmt="%f")




27992474
PPF_cox_lan_Truth_T100_scaled3_v1_2.txt
np.random.seed(0)
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

    start=time.time()
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=1
    es=1e-8
    C0=0.025
    C=0.0003243
    K=5.67e-6
    N=int(2*C0/es)
    L=int(np.ceil(-np.log2(es/(2*np.sqrt(K)))/2,dtype=float))
    l0=L
    Lmin=L
    Lmax=L
    d=1
    eLes=np.array(np.arange(Lmin,Lmax+1))
    pfs=np.zeros((len(eLes),samples,int(T/d),dim))
    inputs=[]
    collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,L,d,N,dim,resamp_coef,\
    g_den,g_par,Norm_Lambda,Lamb_par]
    inputs.append([0,collection_input])
    pool = multiprocessing.Pool(processes=1)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")
    pfs[0,0]=pool_outputs[0]
    pfs=pfs.flatten()
    np.savetxt("Observations/PPF_cox_lan_Truth_T100_scaled3_v1_2.txt",pfs,fmt="%f")
 


27992196
np.random.seed(0)
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
    start=time.time()
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=0
    Lmin=0
    Lmax=5
    d=25
    eLes=np.array(np.arange(Lmin,Lmax+1))
    print(eLes)
    C0=0.025
    C=0.0003243
    K=5.67e-6
    pfs=np.zeros((len(eLes),samples,int(T/d),dim))
    inputs=[]

for i in range(len(eLes)):
        for sample in range(samples):
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
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,eles,d,eNes,dim,resamp_coef,phi,\
            dim_out,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([samples*i+sample,collection_input])

    pool = multiprocessing.Pool(processes=40)
    pool_outputs = pool.map(PMLPF_cox,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")
    for i in range(len(eLes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations/PMLPF_cox_lan_T100_scaled1_v1_2.txt",pfs,fmt="%f")


27991849
this is the second version of this file since the fist one failed for some reason
PPF_cox_lan_t100_scaled1_v1.txt
np.random.seed(0)
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
    start=time.time()
    d=25
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=0
    Lmin=0
    Lmax=5
    eLes=np.array(np.arange(Lmin,Lmax+1))
    print(eLes)
    C0=0.025
    C=0.0003243
    K=5.67e-6

    pfs=np.zeros((len(eLes),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eLes)):
        for sample in range(samples):
            l=eLes[i]
            scale=1e-1
            N=int(C0*2**(2*l)*scale/K)
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim,resamp_coef,\
            g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([samples*i+sample,collection_input])
    pool = multiprocessing.Pool(processes=40)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")
    for i in range(len(eLes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations/PPF_cox_lan_t100_scaled1_v1.txt",pfs,fmt="%f")



Iteration of th elast one.
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
    sca=1e-4
    Lamb_par=1.33*2
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*1e5
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca*5e2
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=25
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    p=8
    N0=10
    enes=np.array([2**i for i in range(p+1)])*N0
    pfs=np.zeros((len(enes),samples,int(T/d),dim))
    inputs=[]
    l=5
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            
            inputs.append([sample+samples*i,collection_input])             
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    
    np.savetxt("Observations&data/CCPF_ou100_scaled3_enes_v8.txt",pfs,fmt="%f")

we chage the Lamb_par bcs it's not well scaled.
CCPF_ou100_scaled3_v8.txt 
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
    sca=1e-4
    Lamb_par=1.33*2
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*1e5
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca*5e2
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=25
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled3_v8.txt",pfs,fmt="%f")


we change the scale in order to get the right rates.
CCPF_ou100_scaled3_v7.txt
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
    sca=1e-4
    Lamb_par=1.33/5
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*1e5
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca*5e2
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=25
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled3_v7.txt",pfs,fmt="%f")

    version of the previous iteration
    CCPF_ou100_scaled3_enes_v6.txt
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
    sca=1e-5
    Lamb_par=1.33/5
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*1e5
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca*5e2
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=25
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    p=8
    N0=10
    enes=np.array([2**i for i in range(p+1)])*N0
    pfs=np.zeros((len(enes),samples,int(T/d),dim))
    inputs=[]
    l=5
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            
            inputs.append([sample+samples*i,collection_input])             
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    
    np.savetxt("Observations&data/CCPF_ou100_scaled3_enes_v6.txt",pfs,fmt="%f")


We increase the intensity and the covariance in order to have stronger coupling with the right rates.
CCPF_ou100_scaled3_v6.txt 
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
    sca=1e-5
    Lamb_par=1.33/5
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*1e5
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca*5e2
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)a
    start=time.time()
    d=25
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled3_v6.txt",pfs,fmt="%f")


version of the previous iteration
CCPF_ou100_scaled3_v5.txt
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
    sca=1e-5
    Lamb_par=1.33/10
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*1e4
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca*5e2
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=25
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled3_v5.txt",pfs,fmt="%f")



we scale everything by 10

    CCPF_ou100_scaled3_enes_v5.txt
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
    sca=1e-5
    Lamb_par=1.33/10
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*1e4
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca*5e2
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=25
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    p=8
    N0=10
    enes=np.array([2**i for i in range(p+1)])*N0
    pfs=np.zeros((len(enes),samples,int(T/d),dim))
    inputs=[]
    l=5
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            
            inputs.append([sample+samples*i,collection_input])             
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    
    np.savetxt("Observations&data/CCPF_ou100_scaled3_enes_v5.txt",pfs,fmt="%f")


    version of the lsast iteration
    CCPF_ou100_scaled3_enes_v4.txt
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
    sca=1e-4
    Lamb_par=1.33
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*1e4
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca*5e2
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=25
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    p=8
    N0=10
    enes=np.array([2**i for i in range(p+1)])*N0
    pfs=np.zeros((len(enes),samples,int(T/d),dim))
    inputs=[]
    l=5
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            
            inputs.append([sample+samples*i,collection_input])             
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    
    np.savetxt("Observations&data/CCPF_ou100_scaled3_enes_v4.txt",pfs,fmt="%f")


    we increase the cov to strenghten the coupling.
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
    sca=1e-4
    Lamb_par=1.33
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*1e4
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca*5e2
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=25
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled3_v4.txt",pfs,fmt="%f")


    we increase the number of particles in order to denoise the rates
    CCPF_ou100_scaled3_v3_2.txt 
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
    sca=1e-4
    Lamb_par=1.33
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*1e3
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca*5e2
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=25
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled3_v3_2.txt",pfs,fmt="%f")


version of th previous iteration
    CCPF_ou100_scaled3_enes_v3.txt 
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
    sca=1e-4
    Lamb_par=1.33
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*1e3
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca*5e2
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=25
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    p=8
    N0=10
    enes=np.array([2**i for i in range(p+1)])*N0
    pfs=np.zeros((len(enes),samples,int(T/d),dim))
    inputs=[]
    l=5
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            
            inputs.append([sample+samples*i,collection_input])             
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    
    np.savetxt("Observations&data/CCPF_ou100_scaled3_enes_v3.txt",pfs,fmt="%f")


    we make the process less stochastic.
    CCPF_ou100_scaled3_v3.txt 
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
    sca=1e-4
    Lamb_par=1.33
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*1e3
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca*5e2
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=25
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
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
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled3_v3.txt",pfs,fmt="%f")


we decrease the scale and the covariance and check
CCPF_ou100_scaled3_enes_v2.txt
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
    cov=I*1e3
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca*1e2
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=25
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    p=8
    N0=10
    enes=np.array([2**i for i in range(p+1)])*N0
    pfs=np.zeros((len(enes),samples,int(T/d),dim))
    inputs=[]
    l=5
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            
            inputs.append([sample+samples*i,collection_input])             
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    
    np.savetxt("Observations&data/CCPF_ou100_scaled3_enes_v2.txt",pfs,fmt="%f")


    version of the previous iteration
    CCPF_ou100_scaled3_enes_v1.txt
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
    sca=1e-1
    Lamb_par=1.33
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*1e5
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca*1e1
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=25
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    p=8
    N0=10
    enes=np.array([2**i for i in range(p+1)])*N0
    pfs=np.zeros((len(enes),samples,int(T/d),dim))
    inputs=[]
    l=5
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            
            inputs.append([sample+samples*i,collection_input])             
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    
    np.savetxt("Observations&data/CCPF_ou100_scaled3_enes_v1.txt",pfs,fmt="%f")



    we try to do something similar to what we did with the langgevin
    CCPF_ou100_scaled3_v1.txt
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
    sca=1e-1
    Lamb_par=1.33
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*1e5
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca*1e1
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=25
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
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
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled3_v1.txt",pfs,fmt="%f")

    PPF_cox_lan_Truth_T100_scaled3_v1.txt
    27956153
    np.random.seed(0)
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

    start=time.time()
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=1
    es=1e-8
    C0=0.025
    C=0.0003243
    K=5.67e-6
    N=int(2*C0/es)
    L=int(np.ceil(-np.log2(es/(2*np.sqrt(K)))/2,dtype=float))
    l0=L
    Lmin=L
    Lmax=L
    d=2**0
    eLes=np.array(np.arange(Lmin,Lmax+1))
    pfs=np.zeros((len(eLes),samples,int(T/d),dim))
    print(pfs)
    inputs=[]
    collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,L,d,N,dim,resamp_coef,\
    g_den,g_par,Norm_Lambda,Lamb_par]
    inputs.append([0,collection_input])
    pool = multiprocessing.Pool(processes=1)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")
    pfs[0,0]=pool_outputs[0]
    pfs=pfs.flatten()
    np.savetxt("Observations/PPF_cox_lan_Truth_T100_scaled3_v1.txt",pfs,fmt="%f")



    np.random.seed(0)
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

    start=time.time()
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=1
    es=1e-4
    C0=0.025
    C=0.0003243
    K=5.67e-6
    N=int(2*C0/es)
    L=int(np.ceil(-np.log2(es/(2*np.sqrt(K)))/2,dtype=float))
    l0=L
    Lmin=L
    Lmax=L
    d=2**0
    eLes=np.array(np.arange(Lmin,Lmax+1))
    pfs=np.zeros((len(eLes),samples,int(T/d),dim))
    print(pfs)
    inputs=[]
    collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,L,d,N,dim,resamp_coef,\
    g_den,g_par,Norm_Lambda,Lamb_par]
    inputs.append([0,collection_input])
    pool = multiprocessing.Pool(processes=1)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")
    pfs[0,0]=pool_outputs[0]
    pfs=pfs.flatten()
    np.savetxt("Observations/PPF_cox_lan_Truth_T100_scaled3_vtest.txt",pfs,fmt="%f")

    

    PPF_cox_lan_t100_scaled1_v1
    27956009
    np.random.seed(0)
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
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=0
    Lmin=0
    Lmax=5
    eLes=np.array(np.arange(Lmin,Lmax+1))
    print(eLes)
    C0=0.025
    C=0.0003243
    K=5.67e-6

    pfs=np.zeros((len(eLes),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eLes)):
        for sample in range(samples):
            l=eLes[i]
            scale=1e-1
            N=int(C0*2**(2*l)*scale/K)
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim,resamp_coef,\
            g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([samples*i+sample,collection_input])
    pool = multiprocessing.Pool(processes=40)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")
    for i in range(len(eLes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations/PPF_cox_lan_t100_scaled1_v1.txt",pfs,fmt="%f")


    PPF_cox_lan_t100_scaled1_vtest.txt
    np.random.seed(0)
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
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=10
    l0=0
    Lmin=0
    Lmax=2
    eLes=np.array(np.arange(Lmin,Lmax+1))
    print(eLes)
    C0=0.025
    C=0.0003243
    K=5.67e-6

    pfs=np.zeros((len(eLes),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eLes)):
        for sample in range(samples):
            l=eLes[i]
            scale=1e-1
            N=int(C0*2**(2*l)*scale/K)
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim,resamp_coef,\
            g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([samples*i+sample,collection_input])
    pool = multiprocessing.Pool(processes=40)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")
    for i in range(len(eLes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations/PPF_cox_lan_t100_scaled1_vtest.txt",pfs,fmt="%f")


    27955920
    PMLPF_cox_lan_T100_scaled1_v1.txt 
    np.random.seed(0)
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
    start=time.time()
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=0
    Lmin=0
    Lmax=5
    d=25
    eLes=np.array(np.arange(Lmin,Lmax+1))
    print(eLes)
    C0=0.025
    C=0.0003243
    K=5.67e-6
    pfs=np.zeros((len(eLes),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eLes)):
        for sample in range(samples):
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
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,eles,d,eNes,dim,resamp_coef,phi,\
            dim_out,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([samples*i+sample,collection_input])

    pool = multiprocessing.Pool(processes=80)
    pool_outputs = pool.map(PMLPF_cox,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")
    for i in range(len(eLes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations/PMLPF_cox_lan_T100_scaled1_v1.txt",pfs,fmt="%f")


version of the previous iteration. 
CCPF_lan100_scaled1_v62.txt
np.random.seed(0)
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
    start=time.time()
    d=25
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
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
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])           
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v62.txt",pfs,fmt="%f")

CCPF_lan100_scaled1_enes_v62.txt
we change the scale and the covariance
np.random.seed(0)
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
    start=time.time()
    d=25
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    p=8
    N0=10
    enes=np.array([2**i for i in range(p+1)])*N0
    pfs=np.zeros((len(enes),samples,int(T/d),dim))
    inputs=[]
    l=5
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_enes_v62.txt",pfs,fmt="%f") 


    version of the previous iteration
    CCPF_lan100_scaled1_enes_v61.txt
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=(1e-2)/2
    pars=[sca*1e2,df]
    fi=np.array([[1]])*np.sqrt(sca)
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    cov=I*1e2
    Lamb_par=2/10
    print("Lamb_par",Lamb_par)
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=25
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    p=8
    N0=10
    enes=np.array([2**i for i in range(p+1)])*N0
    pfs=np.zeros((len(enes),samples,int(T/d),dim))
    inputs=[]
    l=5
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_enes_v61.txt",pfs,fmt="%f") 


we change the covariance and the drift scale.
CCPF_lan100_scaled1_v61.txt
np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=(1e-2)/2
    pars=[sca*1e2,df]
    fi=np.array([[1]])*np.sqrt(sca)
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    cov=I*1e2
    Lamb_par=2/10
    print("Lamb_par",Lamb_par)
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=25
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
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
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])           
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v61.txt",pfs,fmt="%f")


version of the previous iteration.
    CCPF_lan100_scaled1_v60.txt
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=(1e-2)/2
    pars=[sca*1e1,df]
    fi=np.array([[1]])*np.sqrt(sca)
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    cov=I*1e4
    Lamb_par=2/10
    print("Lamb_par",Lamb_par)
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=25
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
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
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])           
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v60.txt",pfs,fmt="%f") 



since I got good results with the previous iteration, I will try change the 
parameters tath give a better bias

CCPF_lan100_scaled1_enes_v60.txt
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=(1e-2)/2
    pars=[sca*1e1,df]
    fi=np.array([[1]])*np.sqrt(sca)
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    cov=I*1e4
    Lamb_par=2/10
    print("Lamb_par",Lamb_par)
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=25
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    p=8
    N0=10
    enes=np.array([2**i for i in range(p+1)])*N0
    pfs=np.zeros((len(enes),samples,int(T/d),dim))
    inputs=[]
    l=5
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_enes_v60.txt",pfs,fmt="%f")


version of the previous iteration
CCPF_lan100_scaled1_enes_v59.txt 
np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=(1e-2)/2
    pars=[sca*1e1,df]
    fi=np.array([[1]])*np.sqrt(sca)
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    cov=I*1e6
    Lamb_par=2/20
    print("Lamb_par",Lamb_par)
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=25
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    p=8
    N0=10
    enes=np.array([2**i for i in range(p+1)])*N0
    pfs=np.zeros((len(enes),samples,int(T/d),dim))
    inputs=[]
    l=5
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_enes_v59.txt",pfs,fmt="%f")   

I increase the drift coefficent scale to check what's happening in taht case.
CCPF_lan100_scaled1_v59.txt
np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=(1e-2)/2
    pars=[sca*1e1,df]
    fi=np.array([[1]])*np.sqrt(sca)
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    cov=I*1e6
    Lamb_par=2/20
    print("Lamb_par",Lamb_par)
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=25
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
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
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])           
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v59.txt",pfs,fmt="%f") 


CCPF_lan100_scaled1_enes_v58.txt

np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=(1e-2)/2
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    cov=I*1e6
    Lamb_par=2/20
    print("Lamb_par",Lamb_par)
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=25
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    p=8
    N0=10
    enes=np.array([2**i for i in range(p+1)])*N0
    pfs=np.zeros((len(enes),samples,int(T/d),dim))
    inputs=[]
    l=5
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_enes_v58.txt",pfs,fmt="%f") 

I increase all the parameters that potentially give a good coupling.
    CCPF_lan100_scaled1_v58.txt
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=(1e-2)/2
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    cov=I*1e6
    Lamb_par=2/20
    print("Lamb_par",Lamb_par)
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=25
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
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
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])           
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v58.txt",pfs,fmt="%f") 
    


    CCPF_lan100_scaled1_v57.txt
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=(1e-2)
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    cov=I*1e5
    Lamb_par=2/15
    print("Lamb_par",Lamb_par)
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=25
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])           
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v57.txt",pfs,fmt="%f")     
    


    
    we keep decreasing the parameters
CCPF_lan100_scaled1_enes_v57.txt
np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=(1e-2)
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    cov=I*1e5
    Lamb_par=2/15
    print("Lamb_par",Lamb_par)
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=25
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    p=8
    N0=10
    enes=np.array([2**i for i in range(p+1)])*N0
    pfs=np.zeros((len(enes),samples,int(T/d),dim))
    inputs=[]
    l=5
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_enes_v57.txt",pfs,fmt="%f")


    CCPF_lan100_scaled1_enes_v56.txt
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=(1e-2)
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    cov=I*1e4
    Lamb_par=2/10
    print("Lamb_par",Lamb_par)
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=25
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    p=8
    N0=10
    enes=np.array([2**i for i in range(p+1)])*N0
    pfs=np.zeros((len(enes),samples,int(T/d),dim))
    inputs=[]
    l=5
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_enes_v56.txt",pfs,fmt="%f")


    CCPF_lan100_scaled1_v56.txt 

    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=(1e-2)
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    cov=I*1e4
    Lamb_par=2/10
    print("Lamb_par",Lamb_par)
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=25
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])           
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v56.txt",pfs,fmt="%f") 


    we drecrease to d=1 just to check its effect. 
    CCPF_lan100_scaled1_v55.txt
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=(1e-2)
    pars=[sca,df]
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
    start=time.time()
    d=1
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])           
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v55.txt",pfs,fmt="%f") 


    we keep increasing cov.
    CCPF_lan100_scaled1_v54.txt
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=(1e-2)
    pars=[sca,df]
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
    start=time.time()
    d=25
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])           
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v54.txt",pfs,fmt="%f") 



    CCPF_lan100_scaled1_v53.txt 
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=(1e-2)
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    cov=I*1e2
    Lamb_par=2/10
    print("Lamb_par",Lamb_par)
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=25
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])           
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v53.txt",pfs,fmt="%f") 



CCPF_lan100_scaled1_enes_v51.txt

this is a version of CCPF_lan100_scaled1_v51

    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=(1e-2)
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    cov=I*1e1
    Lamb_par=2/10
    print("Lamb_par",Lamb_par)
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=10
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    p=8
    N0=10
    enes=np.array([2**i for i in range(p+1)])*N0
    pfs=np.zeros((len(enes),samples,int(T/d),dim))
    inputs=[]
    l=5
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_enes_v51.txt",pfs,fmt="%f")


we change d to d=25
CCPF_lan100_scaled1_v52.txt 

    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=(1e-2)
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    cov=I*1e1
    Lamb_par=2/10
    print("Lamb_par",Lamb_par)
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=25
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])           
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v52.txt",pfs,fmt="%f") 



we scale by 1/5 and also choose d=10

    CCPF_lan100_scaled1_v51,txxt
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=(1e-2)
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    cov=I*1e1
    Lamb_par=2/10
    print("Lamb_par",Lamb_par)
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=10
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])           
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v51.txt",pfs,fmt="%f") 


    In this instansce we check the hypothesis that the bias decreases with the
    convaricne of the observations and if the lamb_par is related to this. 

    
    CCPF_lan100_scaled1_v50.txt
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=(1e-1)/2
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    cov=I*1e1
    Lamb_par=2/10
    print("Lamb_par",Lamb_par)
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=5
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])           
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v50.txt",pfs,fmt="%f")


    we check the hypothesis that the bias decreases with the covariance of the observations
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=(1e-1)/2
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    cov=I*1e0
    Lamb_par=2/6
    print("Lamb_par",Lamb_par)
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=5
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])           
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v49.txt",pfs,fmt="%f") 




    CCPF_lan100_scaled1_v48.txt 
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=(1e-1)/2
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    cov=I*1e2
    Lamb_par=2/6
    print("Lamb_par",Lamb_par)
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=5
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])           
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v48.txt",pfs,fmt="%f") 



    we increase the covariance and check
    CCPF_lan100_scaled1_enes_v47.txt 
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=(1e-1)/2
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    cov=I*1e2
    Lamb_par=2/6
    print("Lamb_par",Lamb_par)
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=5
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    p=8
    N0=10
    enes=np.array([2**i for i in range(p+1)])*N0
    pfs=np.zeros((len(enes),samples,int(T/d),dim))
    inputs=[]
    l=5
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_enes_v47.txt",pfs,fmt="%f")  



    enes version of the last iteration
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=(1e-1)/2
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    cov=I*1e1
    Lamb_par=2/6
    print("Lamb_par",Lamb_par)
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=5
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    p=8
    N0=10
    enes=np.array([2**i for i in range(p+1)])*N0
    pfs=np.zeros((len(enes),samples,int(T/d),dim))
    inputs=[]
    l=5
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,df,Sig_lan,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_enes_v46.txt",pfs,fmt="%f")  

    CCPF_lan100_scaled1_v46.txt 
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=(1e-1)/2
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    cov=I*1e1
    Lamb_par=2/6
    print("Lamb_par",Lamb_par)
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=5
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])           
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v46.txt",pfs,fmt="%f")


    CCPF_lan100_scaled1_v45.txt 
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=(1e-2)/2
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    cov=I*1e1
    Lamb_par=2/6
    print("Lamb_par",Lamb_par)
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=5
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])           
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v45.txt",pfs,fmt="%f")



    we change the covariance of the observatinos wrt to the last iteration
    CCPF_lan100_scaled1_v44.txt
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=(1e-2)/2
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    cov=I*1e1
    Lamb_par=2/6
    print("Lamb_par",Lamb_par)
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(1)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])           
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v44.txt",pfs,fmt="%f")



We scale everything in time. 
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10 
    sca=(1e-2)/2
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    cov=I*1e4
    Lamb_par=2/6
    print("Lamb_par",Lamb_par)
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(1)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])           
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v43.txt",pfs,fmt="%f")


    we change the scaling for one ten times smaller
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=1e-2
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    cov=I*1e5
    Lamb_par=2/3
    print("Lamb_par",Lamb_par)
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])           
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v42.txt",pfs,fmt="%f")


    CCPF_lan100_scaled1_v41-txt
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=1e-1
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    cov=I*1e5
    Lamb_par=2/3
    print("Lamb_par",Lamb_par)
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])           
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v41.txt",pfs,fmt="%f")



    we make this to check the bias of the estimator since it's not yielding the correct rates 
    CCPF_lan100_scaled1_v40.txt
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=1e-2
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    cov=I*1e4
    Lamb_par=3/3
    print("Lamb_par",Lamb_par)
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])           
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v40.txt",pfs,fmt="%f")


    CCPF_lan100_scaled1_v39.txt
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=1e-1
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    cov=I*1e4
    Lamb_par=3/3
    print("Lamb_par",Lamb_par)
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])           
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v39.txt",pfs,fmt="%f")


we start experimenting with the langevin dynamics again.
np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=1e-1
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    cov=I*1e4
    Lamb_par=3/3
    print("Lamb_par",Lamb_par)
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
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
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])           
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v38.txt",pfs,fmt="%f")



    CCPF_ou100_scaled2_12.txt
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
    sca=1e-1/8
    Lamb_par=1.33*(1.5/5)
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*1e5
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=5
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
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
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled2_12.txt",pfs,fmt="%f")


we just need to have more definiiton
CCPF_ou100_scaled2_10_2.txt
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
    sca=1/8
    Lamb_par=1.33*(1.5/5)
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*1e5
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=5
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled2_10_2.txt",pfs,fmt="%f")



version of the previous iteration.

CCPF_ou100_scaled2_enes_v11.txt
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
    sca=1/8
    Lamb_par=1.33*(1.5/5)
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*1e5
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=10
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    p=8
    N0=10
    enes=np.array([2**i for i in range(p+1)])*N0
    pfs=np.zeros((len(enes),samples,int(T/d),dim))
    inputs=[]
    l=5
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            
            inputs.append([sample+samples*i,collection_input])             
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    
    np.savetxt("Observations&data/CCPF_ou100_scaled2_enes_v11.txt",pfs,fmt="%f")


we change d to d=10
CCPF_ou100_scaled2_11-txt
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
    sca=1/8
    Lamb_par=1.33*(1.5/5)
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*1e5
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=10
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
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
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled2_11.txt",pfs,fmt="%f")

Iteration of the last iteration. 

CCPF_ou100_scaled2_enes_v10.txt
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
    sca=1/8
    Lamb_par=1.33*(1.5/5)
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*1e5
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=5
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    p=8
    N0=10
    enes=np.array([2**i for i in range(p+1)])*N0
    pfs=np.zeros((len(enes),samples,int(T/d),dim))
    inputs=[]
    l=5
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            
            inputs.append([sample+samples*i,collection_input])             
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    
    np.savetxt("Observations&data/CCPF_ou100_scaled2_enes_v10.txt",pfs,fmt="%f")




we return to CCPF_ou100_scaled2_7.txt and change the d=5.
CCPF_ou100_scaled2_10.txt
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
    sca=1/8
    Lamb_par=1.33*(1.5/5)
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*1e5
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=5
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
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
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled2_10.txt",pfs,fmt="%f")

we reduce the lamb_par in order to account for the increase of the signal.

CCPF_ou100_scaled2_9.txt
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
    sca=1/16
    Lamb_par=1.33*(1.5/10)
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*1e5
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=1
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
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
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled2_9.txt",pfs,fmt="%f")


version of the last iteration
CCPF_ou100_scaled2_8.txt
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
    sca=1/16
    Lamb_par=1.33*(1.5/5)
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*1e5
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=1
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
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
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled2_8.txt",pfs,fmt="%f")



we change the scaling to check its effect

CCPF_ou100_scaled2_enes_v8.txt 
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
    sca=1/16
    Lamb_par=1.33*(1.5/5)
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*1e5
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=1
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    p=8
    N0=10
    enes=np.array([2**i for i in range(p+1)])*N0
    pfs=np.zeros((len(enes),samples,int(T/d),dim))
    inputs=[]
    l=5
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            
            inputs.append([sample+samples*i,collection_input])             
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    
    np.savetxt("Observations&data/CCPF_ou100_scaled2_enes_v8.txt",pfs,fmt="%f")


version of th elats iteration.
CCPF_ou100_scaled2_enes_v7.txt

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
    sca=1/8
    Lamb_par=1.33*(1.5/5)
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*1e5
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=1
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    p=8
    N0=10
    enes=np.array([2**i for i in range(p+1)])*N0
    pfs=np.zeros((len(enes),samples,int(T/d),dim))
    inputs=[]
    l=5
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            
            inputs.append([sample+samples*i,collection_input])             
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled2_enes_v7.txt",pfs,fmt="%f")



CCPF_ou100_scaled2_7.txt we change parameters to strenghthen the coupling

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
    sca=1/8
    Lamb_par=1.33*(1.5/5)
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*1e5
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=1
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
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
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled2_7.txt",pfs,fmt="%f")


CCPF_ou100_scaled2_enes_v6.txt version of the last iteration
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
    sca=1/8
    Lamb_par=1.33*(1.5/3)
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*1e4
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=1
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    p=8
    N0=10
    enes=np.array([2**i for i in range(p+1)])*N0
    pfs=np.zeros((len(enes),samples,int(T/d),dim))
    inputs=[]
    l=5
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            
            inputs.append([sample+samples*i,collection_input])             
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled2_enes_v6.txt",pfs,fmt="%f")


we include some scaling to improve the rates.
    CCPF_ou100_scaled2_6.txt

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
    sca=1/8
    Lamb_par=1.33*(1.5/3)
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*1e4
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=1
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
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
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled2_6.txt",pfs,fmt="%f")


we changed the number of particles to 100 and the covariance wrt to the last iteration.
    CCPF_ou100_scaled2_v5.txt 
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
    sca=1e-0
    Lamb_par=1.33*(1.5/3)
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*1e4
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=1
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
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
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled2_v5.txt",pfs,fmt="%f")



version of the previous iteration. 
CCPF_ou100_scaled2_v4.txt

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
    sca=1e-0
    Lamb_par=1.33*(1.5/3)
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*1e3
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=1
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled2_v4.txt",pfs,fmt="%f")




we reduce the scaling.
CCPF_ou100_scaled2_enes_v4.txt
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
    sca=1e-0
    Lamb_par=1.33*(1.5/3)
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*1e3
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=1
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    p=8
    N0=10
    enes=np.array([2**i for i in range(p+1)])*N0
    pfs=np.zeros((len(enes),samples,int(T/d),dim))
    inputs=[]
    l=5
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            
            inputs.append([sample+samples*i,collection_input])             
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled2_enes_v4.txt",pfs,fmt="%f")


    CCPF_ou100_scaled2_enes_v3.txt 
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
    sca=5e-2
    Lamb_par=1.33*(1.5/3)
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*1e3
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=1
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    p=8
    N0=10
    enes=np.array([2**i for i in range(p+1)])*N0
    pfs=np.zeros((len(enes),samples,int(T/d),dim))
    inputs=[]
    l=5
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            
            inputs.append([sample+samples*i,collection_input])             
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled2_enes_v3.txt",pfs,fmt="%f")



    yet, a new set of parameters 
    CCPF_ou100_scaled2_v3.txt
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
    sca=5e-2
    Lamb_par=1.33*(1.5/3)
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*1e3
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=1
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled2_v3.txt",pfs,fmt="%f")



    change of parameter for the CCPF
    CCPF_ou100_scaled2_v2.txt
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
    sca=1e-1
    sca=5e-2
    Lamb_par=1.33*(2/3)
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*1e2
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=1
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled2_v2.txt",pfs,fmt="%f")


    change of parameters for the CCPF
    CCPF_ou100_scaled2_enes_v2.txt
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
    sca=1e-1
    sca=5e-2
    Lamb_par=1.33*(2/3)
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*1e2
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=1
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    p=8
    N0=10
    enes=np.array([2**i for i in range(p+1)])*N0
    pfs=np.zeros((len(enes),samples,int(T/d),dim))
    inputs=[]
    l=5
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            
            inputs.append([sample+samples*i,collection_input])             
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled2_enes_v2.txt",pfs,fmt="%f")




    we test the variance of the PF in this case.
    CCPF_ou100_scaled2_enes_v1.txt
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
    sca=1e-1
    Lamb_par=1.33
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*1e1
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=1
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    p=8
    N0=10
    enes=np.array([2**i for i in range(p+1)])*N0
    pfs=np.zeros((len(enes),samples,int(T/d),dim))
    inputs=[]
    l=5
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            
            inputs.append([sample+samples*i,collection_input])             
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled2_enes_v1.txt",pfs,fmt="%f")


    we test the rates and the maginitude of the coupling with the NEW version of the system
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
    sca=1e-1
    Lamb_par=1.33
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*1e1
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=1
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled2_v1.txt",pfs,fmt="%f")



    CCPF_ou100_scaled1_lw_v16_34.txt 
    np.random.seed(1)
    T=10
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+10
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    comp_matrix=np.array([[1]])
    inv_mat=la.inv(comp_matrix)
    S=diags(np.random.normal(0.99,0.1,dim),0).toarray()*1e1
    sca=1e-2
    Lamb_par=1.33/10
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*5e4
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    #obs_time=np.array(range(1,T))-2.**(-8)
    start=time.time()
    d=1
    #obs=np.array([])
    #obs_time=np.array([])
    resamp_coef=0
    dim_out=2
    g_par=cov
    samples=10
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=10
    xs=np.zeros((2,len(eles),samples,int(T/d),N,dim))
    log_ws=np.zeros((2,len(eles),samples,int(T/d),N))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF_new,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            xs[:,i,sample]=pool_outputs[i*samples+sample][:2]
            log_ws[:,i,sample]=pool_outputs[i*samples+sample][2:]
    xs=xs.flatten()
    log_ws=log_ws.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled1_x_v16_34.txt",xs,fmt="%f")
    np.savetxt("Observations&data/CCPF_ou100_scaled1_lw_v16_34.txt",log_ws,fmt="%f")



in this iteration we shift the observations by 2**(-15)
    CCPF_ou100_scaled1_x_v16_33.txt
    np.random.seed(1)
    T=10
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+10
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    comp_matrix=np.array([[1]])
    inv_mat=la.inv(comp_matrix)
    S=diags(np.random.normal(0.99,0.1,dim),0).toarray()*1e1
    sca=1e-1
    Lamb_par=1.33/10
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*5e4
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    obs_time=np.array(range(1,T))-2.**(-15)

    start=time.time()
    d=1
    #obs=np.array([])
    #obs_time=np.array([])
    resamp_coef=0
    dim_out=2
    g_par=cov
    samples=10
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=10
    xs=np.zeros((2,len(eles),samples,int(T/d),N,dim))
    log_ws=np.zeros((2,len(eles),samples,int(T/d),N))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF_new,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            xs[:,i,sample]=pool_outputs[i*samples+sample][:2]
            log_ws[:,i,sample]=pool_outputs[i*samples+sample][2:]
    xs=xs.flatten()
    log_ws=log_ws.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled1_x_v16_33.txt",xs,fmt="%f")
    np.savetxt("Observations&data/CCPF_ou100_scaled1_lw_v16_33.txt",log_ws,fmt="%f")






    this one is shifted with a seemenly  random number 
    CCPF_ou100_scaled1_x_v16_32.txt 
    np.random.seed(1)
    T=10
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+10
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    comp_matrix=np.array([[1]])
    inv_mat=la.inv(comp_matrix)
    S=diags(np.random.normal(0.99,0.1,dim),0).toarray()*1e1
    sca=1e-1
    Lamb_par=1.33/10
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*5e4
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    obs_time=np.array(range(1,T))-0.123456788

    start=time.time()
    d=1
    #obs=np.array([])
    #obs_time=np.array([])
    resamp_coef=0
    dim_out=2
    g_par=cov
    samples=10
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=10
    xs=np.zeros((2,len(eles),samples,int(T/d),N,dim))
    log_ws=np.zeros((2,len(eles),samples,int(T/d),N))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF_new,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            xs[:,i,sample]=pool_outputs[i*samples+sample][:2]
            log_ws[:,i,sample]=pool_outputs[i*samples+sample][2:]
    xs=xs.flatten()
    log_ws=log_ws.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled1_x_v16_32.txt",xs,fmt="%f")
    np.savetxt("Observations&data/CCPF_ou100_scaled1_lw_v16_32.txt",log_ws,fmt="%f")


in this example we compute the observations times to be shifted to the left be Delta_10
    np.random.seed(1)
    T=10
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+10
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    comp_matrix=np.array([[1]])
    inv_mat=la.inv(comp_matrix)
    S=diags(np.random.normal(0.99,0.1,dim),0).toarray()*1e1
    sca=1e-1
    Lamb_par=1.33/10
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*5e4
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    obs_time=np.array(range(1,T))-2.**(-10)

    start=time.time()
    d=1
    #obs=np.array([])
    #obs_time=np.array([])
    resamp_coef=0
    dim_out=2
    g_par=cov
    samples=10
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=10
    xs=np.zeros((2,len(eles),samples,int(T/d),N,dim))
    log_ws=np.zeros((2,len(eles),samples,int(T/d),N))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF_new,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            xs[:,i,sample]=pool_outputs[i*samples+sample][:2]
            log_ws[:,i,sample]=pool_outputs[i*samples+sample][2:]
    xs=xs.flatten()
    log_ws=log_ws.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled1_x_v16_31.txt",xs,fmt="%f")
    np.savetxt("Observations&data/CCPF_ou100_scaled1_lw_v16_31.txt",log_ws,fmt="%f")


    in this example we compute the observations times to be shifted to the left be Delta_5
    CCPF_ou100_scaled1_x_v16_30.txt 
    np.random.seed(1)
    T=10
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+10
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    comp_matrix=np.array([[1]])
    inv_mat=la.inv(comp_matrix)
    S=diags(np.random.normal(0.99,0.1,dim),0).toarray()*1e1
    sca=1e-1
    Lamb_par=1.33/10
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*5e4
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    obs_time=np.array(range(1,T))-2.**(-5)

    start=time.time()
    d=1
    #obs=np.array([])
    #obs_time=np.array([])
    resamp_coef=0
    dim_out=2
    g_par=cov
    samples=10
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=10
    xs=np.zeros((2,len(eles),samples,int(T/d),N,dim))
    log_ws=np.zeros((2,len(eles),samples,int(T/d),N))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF_new,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            xs[:,i,sample]=pool_outputs[i*samples+sample][:2]
            log_ws[:,i,sample]=pool_outputs[i*samples+sample][2:]
    xs=xs.flatten()
    log_ws=log_ws.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled1_x_v16_30.txt",xs,fmt="%f")
    np.savetxt("Observations&data/CCPF_ou100_scaled1_lw_v16_30.txt",log_ws,fmt="%f")



This iteration is made so we can iterate with more particles and samples.
    CCPF_ou100_scaled1_x_v16_29.txt
    np.random.seed(1)
    T=10
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+10
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    comp_matrix=np.array([[1]])
    inv_mat=la.inv(comp_matrix)
    S=diags(np.random.normal(0.99,0.1,dim),0).toarray()*1e1
    sca=1e-1
    Lamb_par=1.33/10
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*5e4
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    obs_time=np.array(range(1,T))-0.5

    start=time.time()
    d=1
    #obs=np.array([])
    #obs_time=np.array([])
    resamp_coef=0.
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=100
    xs=np.zeros((2,len(eles),samples,int(T/d),N,dim))
    log_ws=np.zeros((2,len(eles),samples,int(T/d),N))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            
            inputs.append([sample+samples*i,collection_input])   
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF_new,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            xs[:,i,sample]=pool_outputs[i*samples+sample][:2]
            log_ws[:,i,sample]=pool_outputs[i*samples+sample][2:]
    xs=xs.flatten()
    log_ws=log_ws.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled1_x_v16_28.txt",xs,fmt="%f")
    np.savetxt("Observations&data/CCPF_ou100_scaled1_lw_v16_28.txt",log_ws,fmt="%f")


    we check the interpolation problem with observation 2**(-3) to the left of the limits

    np.random.seed(1)
    T=10
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+10
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    comp_matrix=np.array([[1]])
    inv_mat=la.inv(comp_matrix)
    S=diags(np.random.normal(0.99,0.1,dim),0).toarray()*1e1
    sca=1e-1
    Lamb_par=1.33/10
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*5e4
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    obs_time=np.array(range(1,T))-2.**(-3)

    start=time.time()
    d=1
    #obs=np.array([])
    #obs_time=np.array([])
    resamp_coef=0
    dim_out=2
    g_par=cov
    samples=10
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=10
    xs=np.zeros((2,len(eles),samples,int(T/d),N,dim))
    log_ws=np.zeros((2,len(eles),samples,int(T/d),N))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF_new,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            xs[:,i,sample]=pool_outputs[i*samples+sample][:2]
            log_ws[:,i,sample]=pool_outputs[i*samples+sample][2:]
    xs=xs.flatten()
    log_ws=log_ws.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled1_x_v16_26.txt",xs,fmt="%f")
    np.savetxt("Observations&data/CCPF_ou100_scaled1_lw_v16_26.txt",log_ws,fmt="%f")



    we modify the observations times' so they0re at slightly to the right of the middle of
    the interval d
    CCPF_ou100_scaled1_x_v16_25.txt
    np.random.seed(1)
    T=10
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+10
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    comp_matrix=np.array([[1]])
    inv_mat=la.inv(comp_matrix)
    S=diags(np.random.normal(0.99,0.1,dim),0).toarray()*1e1
    sca=1e-1
    Lamb_par=1.33/10
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*5e4
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    obs_time=np.array(range(1,T))-0.54

    start=time.time()
    d=1
    #obs=np.array([])
    #obs_time=np.array([])
    resamp_coef=0
    dim_out=2
    g_par=cov
    samples=10
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=10
    xs=np.zeros((2,len(eles),samples,int(T/d),N,dim))
    log_ws=np.zeros((2,len(eles),samples,int(T/d),N))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF_new,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            xs[:,i,sample]=pool_outputs[i*samples+sample][:2]
            log_ws[:,i,sample]=pool_outputs[i*samples+sample][2:]
    xs=xs.flatten()
    log_ws=log_ws.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled1_x_v16_25.txt",xs,fmt="%f")
    np.savetxt("Observations&data/CCPF_ou100_scaled1_lw_v16_25.txt",log_ws,fmt="%f")



    CCPF_ou100_scaled1_x_v16_24.txt in htis file we modify the observations times' so
    they are precisely at the middle of the interval d.

    np.random.seed(1)
    T=10
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+10
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    comp_matrix=np.array([[1]])
    inv_mat=la.inv(comp_matrix)
    S=diags(np.random.normal(0.99,0.1,dim),0).toarray()*1e1
    sca=1e-1
    Lamb_par=1.33/10
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*5e4
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    obs_time=np.array(range(1,T))-0.5

    start=time.time()
    d=1
    #obs=np.array([])
    #obs_time=np.array([])
    resamp_coef=0
    dim_out=2
    g_par=cov
    samples=10
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=10
    xs=np.zeros((2,len(eles),samples,int(T/d),N,dim))
    log_ws=np.zeros((2,len(eles),samples,int(T/d),N))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF_new,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            xs[:,i,sample]=pool_outputs[i*samples+sample][:2]
            log_ws[:,i,sample]=pool_outputs[i*samples+sample][2:]
    xs=xs.flatten()
    log_ws=log_ws.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled1_x_v16_24.txt",xs,fmt="%f")
    np.savetxt("Observations&data/CCPF_ou100_scaled1_lw_v16_24.txt",log_ws,fmt="%f")




    In this iteration we modify the observations so we don't need interpolation since the 
    time of observations are at end of the interval d
    np.random.seed(1)
    T=10
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+10
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    comp_matrix=np.array([[1]])
    inv_mat=la.inv(comp_matrix)
    S=diags(np.random.normal(0.99,0.1,dim),0).toarray()*1e1
    sca=1e-1
    Lamb_par=1.33/10
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*5e4
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    obs_time=np.array(range(1,T+1))
    start=time.time()
    d=1
    #obs=np.array([])
    #obs_time=np.array([])
    resamp_coef=0
    dim_out=2
    g_par=cov
    samples=10
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=10
    xs=np.zeros((2,len(eles),samples,int(T/d),N,dim))
    log_ws=np.zeros((2,len(eles),samples,int(T/d),N))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF_new,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            xs[:,i,sample]=pool_outputs[i*samples+sample][:2]
            log_ws[:,i,sample]=pool_outputs[i*samples+sample][2:]
    xs=xs.flatten()
    log_ws=log_ws.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled1_x_v16_23.txt",xs,fmt="%f")
    np.savetxt("Observations&data/CCPF_ou100_scaled1_lw_v16_23.txt",log_ws,fmt="%f")



In this iteration we modify Gox so lw is the the particles interpolated.

np.random.seed(1)
    T=10
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+10
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    comp_matrix=np.array([[1]])
    inv_mat=la.inv(comp_matrix)
    S=diags(np.random.normal(0.99,0.1,dim),0).toarray()*1e1
    sca=1e-1
    Lamb_par=1.33/10
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*5e4
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=1
    #obs=np.array([])
    #obs_time=np.array([])
    resamp_coef=0
    dim_out=2
    g_par=cov
    samples=10
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=10
    xs=np.zeros((2,len(eles),samples,int(T/d),N,dim))
    log_ws=np.zeros((2,len(eles),samples,int(T/d),N))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF_new,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            xs[:,i,sample]=pool_outputs[i*samples+sample][:2]
            log_ws[:,i,sample]=pool_outputs[i*samples+sample][2:]
    xs=xs.flatten()
    log_ws=log_ws.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled1_x_v16_22.txt",xs,fmt="%f")
    np.savetxt("Observations&data/CCPF_ou100_scaled1_lw_v16_22.txt",log_ws,fmt="%f")



checking if the rates show with just a small number of particles

    np.random.seed(1)
    T=10
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+10
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    comp_matrix=np.array([[1]])
    inv_mat=la.inv(comp_matrix)
    S=diags(np.random.normal(0.99,0.1,dim),0).toarray()*1e1
    sca=1e-1
    Lamb_par=1.33/10
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*5e4
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=1
    #obs=np.array([])
    #obs_time=np.array([])
    resamp_coef=0
    dim_out=2
    g_par=cov
    samples=10
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=10
    xs=np.zeros((2,len(eles),samples,int(T/d),N,dim))
    log_ws=np.zeros((2,len(eles),samples,int(T/d),N))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF_new,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            xs[:,i,sample]=pool_outputs[i*samples+sample][:2]
            log_ws[:,i,sample]=pool_outputs[i*samples+sample][2:]
    xs=xs.flatten()
    log_ws=log_ws.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled1_x_v16_21.txt",xs,fmt="%f")
    np.savetxt("Observations&data/CCPF_ou100_scaled1_lw_v16_21.txt",log_ws,fmt="%f")


    we modified Gox so only the exponential of the intensity and the multiplicatory of the 
    intensity are computed on the weights.
    np.random.seed(1)
    T=10
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+10
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    comp_matrix=np.array([[1]])
    inv_mat=la.inv(comp_matrix)
    S=diags(np.random.normal(0.99,0.1,dim),0).toarray()*1e1
    sca=1e-1
    Lamb_par=1.33/10
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*5e4
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=1
    #obs=np.array([])
    #obs_time=np.array([])
    resamp_coef=0
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    xs=np.zeros((2,len(eles),samples,int(T/d),N,dim))
    log_ws=np.zeros((2,len(eles),samples,int(T/d),N))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF_new,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            xs[:,i,sample]=pool_outputs[i*samples+sample][:2]
            log_ws[:,i,sample]=pool_outputs[i*samples+sample][2:]
    xs=xs.flatten()
    log_ws=log_ws.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled1_x_v16_20.txt",xs,fmt="%f")
    np.savetxt("Observations&data/CCPF_ou100_scaled1_lw_v16_20.txt",log_ws,fmt="%f")



    this iteration uses the g_den_test function.
    CCPF_ou100_scaled1_lw_v16_19.txt 
    np.random.seed(1)
    T=10
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+10
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    comp_matrix=np.array([[1]])
    inv_mat=la.inv(comp_matrix)
    S=diags(np.random.normal(0.99,0.1,dim),0).toarray()*1e1
    sca=1e-1
    Lamb_par=1.33/10
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*5e4
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=1
    #obs=np.array([])
    #obs_time=np.array([])
    resamp_coef=0
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    xs=np.zeros((2,len(eles),samples,int(T/d),N,dim))
    log_ws=np.zeros((2,len(eles),samples,int(T/d),N))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den_test,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF_new,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            xs[:,i,sample]=pool_outputs[i*samples+sample][:2]
            log_ws[:,i,sample]=pool_outputs[i*samples+sample][2:]
    xs=xs.flatten()
    log_ws=log_ws.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled1_x_v16_19.txt",xs,fmt="%f")
    np.savetxt("Observations&data/CCPF_ou100_scaled1_lw_v16_19.txt",log_ws,fmt="%f")



    In this iteration we check that the original Gox gaves the wrong rates here

    np.random.seed(1)
    T=10
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+10
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    comp_matrix=np.array([[1]])
    inv_mat=la.inv(comp_matrix)
    S=diags(np.random.normal(0.99,0.1,dim),0).toarray()*1e1
    sca=1e-1
    Lamb_par=1.33/10
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*5e4
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=1
    #obs=np.array([])
    #obs_time=np.array([])
    resamp_coef=0
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    xs=np.zeros((2,len(eles),samples,int(T/d),N,dim))
    log_ws=np.zeros((2,len(eles),samples,int(T/d),N))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF_new,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            xs[:,i,sample]=pool_outputs[i*samples+sample][:2]
            log_ws[:,i,sample]=pool_outputs[i*samples+sample][2:]
    xs=xs.flatten()
    log_ws=log_ws.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled1_x_v16_18.txt",xs,fmt="%f")
    np.savetxt("Observations&data/CCPF_ou100_scaled1_lw_v16_18.txt",log_ws,fmt="%f")


    In this iteration we Modify the function gox to get the exponential of the intensiyy integral
    and the multiplicatory of the likelihood of the observations .
    np.random.seed(1)
    T=10
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+10
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    comp_matrix=np.array([[1]])
    inv_mat=la.inv(comp_matrix)
    S=diags(np.random.normal(0.99,0.1,dim),0).toarray()*1e1
    sca=1e-1
    Lamb_par=1.33/10
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*5e4
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=1
    #obs=np.array([])
    #obs_time=np.array([])
    resamp_coef=0
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    xs=np.zeros((2,len(eles),samples,int(T/d),N,dim))
    log_ws=np.zeros((2,len(eles),samples,int(T/d),N))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF_new,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            xs[:,i,sample]=pool_outputs[i*samples+sample][:2]
            log_ws[:,i,sample]=pool_outputs[i*samples+sample][2:]
    xs=xs.flatten()
    log_ws=log_ws.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled1_x_v16_17.txt",xs,fmt="%f")
    np.savetxt("Observations&data/CCPF_ou100_scaled1_lw_v16_17.txt",log_ws,fmt="%f")

    CCPF_ou100_scaled1_lw_v16_17.txt



    CCPF_ou100_scaled1_x_v16_16.txt
    this one is made without resampling.
    np.random.seed(1)
    T=10
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+10
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    comp_matrix=np.array([[1]])
    inv_mat=la.inv(comp_matrix)
    S=diags(np.random.normal(0.99,0.1,dim),0).toarray()*1e1
    sca=1e-1
    Lamb_par=1.33/10
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*5e4
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=1
    #obs=np.array([])
    #obs_time=np.array([])
    resamp_coef=0
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    xs=np.zeros((2,len(eles),samples,int(T/d),N,dim))
    log_ws=np.zeros((2,len(eles),samples,int(T/d),N))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF_new,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            xs[:,i,sample]=pool_outputs[i*samples+sample][:2]
            log_ws[:,i,sample]=pool_outputs[i*samples+sample][2:]
    xs=xs.flatten()
    log_ws=log_ws.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled1_x_v16_16.txt",xs,fmt="%f")
    np.savetxt("Observations&data/CCPF_ou100_scaled1_lw_v16_16.txt",log_ws,fmt="%f")





    In this iteration we modify the Gox function so we only consider the 
    multiplicatory of the intensity function and the exponential of the negative
    integral of the intensity. This is done in order to  
    check if the interpolation is the one driving the wrong rates.
    np.random.seed(1)
    T=10
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+10
    l=10
    collection_input=[] 
    I=identity(dim).toarray()
    comp_matrix=np.array([[1]])
    inv_mat=la.inv(comp_matrix)
    S=diags(np.random.normal(0.99,0.1,dim),0).toarray()*1e1
    sca=1e-1
    Lamb_par=1.33/10
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*5e4
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=1
    #obs=np.array([])
    #obs_time=np.array([])
    resamp_coef=1
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    xs=np.zeros((2,len(eles),samples,int(T/d),N,dim))
    log_ws=np.zeros((2,len(eles),samples,int(T/d),N))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den_test,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF_new,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            xs[:,i,sample]=pool_outputs[i*samples+sample][:2]
            log_ws[:,i,sample]=pool_outputs[i*samples+sample][2:]
    xs=xs.flatten()
    log_ws=log_ws.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled1_x_v16_15.txt",xs,fmt="%f")
    np.savetxt("Observations&data/CCPF_ou100_scaled1_lw_v16_15.txt",log_ws,fmt="%f")





    CCPF_ou100_scaled1_x_v16_14.txt 
    this iteration is made so we test the CCPF with a test g_den function.

    np.random.seed(1)
    T=10
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+10
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    comp_matrix=np.array([[1]])
    inv_mat=la.inv(comp_matrix)
    S=diags(np.random.normal(0.99,0.1,dim),0).toarray()*1e1
    sca=1e-1
    Lamb_par=1.33/10
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*5e4
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=1
    #obs=np.array([])
    #obs_time=np.array([])
    resamp_coef=1
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    xs=np.zeros((2,len(eles),samples,int(T/d),N,dim))
    log_ws=np.zeros((2,len(eles),samples,int(T/d),N))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den_test,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF_new,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            xs[:,i,sample]=pool_outputs[i*samples+sample][:2]
            log_ws[:,i,sample]=pool_outputs[i*samples+sample][2:]
    xs=xs.flatten()
    log_ws=log_ws.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled1_x_v16_14.txt",xs,fmt="%f")
    np.savetxt("Observations&data/CCPF_ou100_scaled1_lw_v16_14.txt",log_ws,fmt="%f")


    CCPF_ou100_scaled1_lw_v16_13.txt
    in this experiment we check that the problem is not the g_den function used (we defined two of them
    one on Un_Cox... and another on PF_def...).
    np.random.seed(1)
    T=10
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+10
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    comp_matrix=np.array([[1]])
    inv_mat=la.inv(comp_matrix)
    S=diags(np.random.normal(0.99,0.1,dim),0).toarray()*1e1
    sca=1e-1
    Lamb_par=1.33/10
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*5e4
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=1
    #obs=np.array([])
    #obs_time=np.array([])
    resamp_coef=1
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    xs=np.zeros((2,len(eles),samples,int(T/d),N,dim))
    log_ws=np.zeros((2,len(eles),samples,int(T/d),N))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,pff.g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF_new,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            xs[:,i,sample]=pool_outputs[i*samples+sample][:2]
            log_ws[:,i,sample]=pool_outputs[i*samples+sample][2:]
    xs=xs.flatten()
    log_ws=log_ws.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled1_x_v16_13.txt",xs,fmt="%f")
    np.savetxt("Observations&data/CCPF_ou100_scaled1_lw_v16_13.txt",log_ws,fmt="%f")



    this iteration is made with the data of 
    CCPF_ou100_scaled1_x_v16_11.txt in order to check if we get the 
    rates for the coupling of the PF(not just the importance sampling)
CCPF_ou100_scaled1_v21.txt
np.random.seed(1)
    T=10
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+10
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    comp_matrix=np.array([[1]])
    inv_mat=la.inv(comp_matrix)
    S=diags(np.random.normal(0.99,0.1,dim),0).toarray()*1e1
    sca=1e-1
    Lamb_par=1.33/10
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*5e4
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=1
    obs=np.array([])
    obs_time=np.array([])
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    eles=np.array(range(l0,L+1))
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled1_v21.txt",pfs,fmt="%f")


    CCPF_ou100_scaled1_x_v16_12.txt
    This one is the future experiments referenced in CCPF_ou100_scaled1_x_v16_11.txt 


    np.random.seed(1)
    T=10
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+10
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    comp_matrix=np.array([[1]])
    inv_mat=la.inv(comp_matrix)
    S=diags(np.random.normal(0.99,0.1,dim),0).toarray()*1e1
    sca=1e-1
    Lamb_par=1.33/10
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*5e4
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=1
    #obs=np.array([])
    #obs_time=np.array([])
    resamp_coef=0
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    xs=np.zeros((2,len(eles),samples,int(T/d),N,dim))
    log_ws=np.zeros((2,len(eles),samples,int(T/d),N))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF_new,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            xs[:,i,sample]=pool_outputs[i*samples+sample][:2]
            log_ws[:,i,sample]=pool_outputs[i*samples+sample][2:]
    xs=xs.flatten()
    log_ws=log_ws.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled1_x_v16_12.txt",xs,fmt="%f")
    np.savetxt("Observations&data/CCPF_ou100_scaled1_lw_v16_12.txt",log_ws,fmt="%f")


CCPF_ou100_scaled1_lw_v16_11
    # in this iteration we change the Lamb_par wrt to the previous iteration in order to 
    get some observations (we set them to zero observations here but we will build some 
    in the future that are not zero) and check whether the observations are the ones
    driving the rates wrongly.
np.random.seed(1)
    T=10
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+10
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    comp_matrix=np.array([[1]])
    inv_mat=la.inv(comp_matrix)
    S=diags(np.random.normal(0.99,0.1,dim),0).toarray()*1e1
    sca=1e-1
    Lamb_par=1.33/10
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*5e4
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=1
    obs=np.array([])
    obs_time=np.array([])
    resamp_coef=0
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    xs=np.zeros((2,len(eles),samples,int(T/d),N,dim))
    log_ws=np.zeros((2,len(eles),samples,int(T/d),N))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF_new,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            xs[:,i,sample]=pool_outputs[i*samples+sample][:2]
            log_ws[:,i,sample]=pool_outputs[i*samples+sample][2:]
    xs=xs.flatten()
    log_ws=log_ws.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled1_x_v16_11.txt",xs,fmt="%f")
    np.savetxt("Observations&data/CCPF_ou100_scaled1_lw_v16_11.txt",log_ws,fmt="%f")

This iteration is made with no observation so we can check whether the observations(the likelihood of observations)
are the one driving the rates wrongly.

CCPF_ou100_scaled1_x_v16_10.txt

np.random.seed(1)
    T=10
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+10
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    comp_matrix=np.array([[1]])
    inv_mat=la.inv(comp_matrix)
    S=diags(np.random.normal(0.99,0.1,dim),0).toarray()*1e1
    sca=1e-1
    Lamb_par=1.33/100
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*5e4
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=1
    obs=np.array([])
    obs_time=np.array([])
    resamp_coef=0
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    xs=np.zeros((2,len(eles),samples,int(T/d),N,dim))
    log_ws=np.zeros((2,len(eles),samples,int(T/d),N))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF_new,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            xs[:,i,sample]=pool_outputs[i*samples+sample][:2]
            log_ws[:,i,sample]=pool_outputs[i*samples+sample][2:]
    xs=xs.flatten()
    log_ws=log_ws.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled1_x_v16_10.txt",xs,fmt="%f")
    np.savetxt("Observations&data/CCPF_ou100_scaled1_lw_v16_10.txt",log_ws,fmt="%f")



this iteration is made with a lot of changed parameters, basically we make the process
relatively large and a really small Lamb_par.
CCPF_ou100_scaled1_x_v16_9 .txt

np.random.seed(1)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+10
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    comp_matrix=np.array([[1]])
    inv_mat=la.inv(comp_matrix)
    S=diags(np.random.normal(0.99,0.1,dim),0).toarray()*1e1
    sca=1e-1
    Lamb_par=1.33/100
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*5e4
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    print("B and fi are",B,fi)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=1
    resamp_coef=0
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    xs=np.zeros((2,len(eles),samples,int(T/d),N,dim))
    log_ws=np.zeros((2,len(eles),samples,int(T/d),N))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF_new,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            xs[:,i,sample]=pool_outputs[i*samples+sample][:2]
            log_ws[:,i,sample]=pool_outputs[i*samples+sample][2:]
    xs=xs.flatten()
    log_ws=log_ws.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled1_x_v16_9.txt",xs,fmt="%f")
    np.savetxt("Observations&data/CCPF_ou100_scaled1_lw_v16_9.txt",log_ws,fmt="%f")


    this iteration is made with a relatively small Lamb_par, so we check if we decrease the 
    effect of the weights, i.e., the rates of the strong error not being Delta_l^2.
    CCPF_ou100_scaled1_x_v16_8.txt
    np.random.seed(1)
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
    sca=1e-2
    Lamb_par=1.33/20
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*5e4
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=1
    resamp_coef=0
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    xs=np.zeros((2,len(eles),samples,int(T/d),N,dim))
    log_ws=np.zeros((2,len(eles),samples,int(T/d),N))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF_new,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            xs[:,i,sample]=pool_outputs[i*samples+sample][:2]
            log_ws[:,i,sample]=pool_outputs[i*samples+sample][2:]
    xs=xs.flatten()
    log_ws=log_ws.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled1_x_v16_8.txt",xs,fmt="%f")
    np.savetxt("Observations&data/CCPF_ou100_scaled1_lw_v16_8.txt",log_ws,fmt="%f")


    This iteration is made with a rescaling so we access larger levels of discretization. 
    CCPF_ou100_scaled1_x_v16_7.txt 
    np.random.seed(1)
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
    sca=1e-2
    Lamb_par=1.33/2
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*5e4
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=1
    resamp_coef=0
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    xs=np.zeros((2,len(eles),samples,int(T/d),N,dim))
    log_ws=np.zeros((2,len(eles),samples,int(T/d),N))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF_new,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            xs[:,i,sample]=pool_outputs[i*samples+sample][:2]
            log_ws[:,i,sample]=pool_outputs[i*samples+sample][2:]
    
    xs=xs.flatten()
    log_ws=log_ws.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled1_x_v16_7.txt",xs,fmt="%f")
    np.savetxt("Observations&data/CCPF_ou100_scaled1_lw_v16_7.txt",log_ws,fmt="%f")



    we do this experiment in order to check the strong error of the
    signal system with the likelihood function evaluated at all times.
    this time with d=1
    CCPF_ou100_scaled1_lw_v16_6.txt
    np.random.seed(1)
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
    sca=1e0
    Lamb_par=1.33/2
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*5e4
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=1
    resamp_coef=0
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    xs=np.zeros((2,len(eles),samples,int(T/d),N,dim))
    log_ws=np.zeros((2,len(eles),samples,int(T/d),N))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF_new,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            xs[:,i,sample]=pool_outputs[i*samples+sample][:2]
            log_ws[:,i,sample]=pool_outputs[i*samples+sample][2:]
    
    xs=xs.flatten()
    log_ws=log_ws.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled1_x_v16_6.txt",xs,fmt="%f")
    np.savetxt("Observations&data/CCPF_ou100_scaled1_lw_v16_6.txt",log_ws,fmt="%f")


    We compute this in order to check the strong error including the evaluation of the 
    likelihood function for all times, not just the first interval d.
    CCPF_ou100_scaled1_lw_v16_5.txt
    np.random.seed(1)
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
    sca=1e0
    Lamb_par=1.33/2
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*5e4
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=5
    resamp_coef=0
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    xs=np.zeros((2,len(eles),samples,int(T/d),N,dim))
    log_ws=np.zeros((2,len(eles),samples,int(T/d),N))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF_new,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            xs[:,i,sample]=pool_outputs[i*samples+sample][:2]
            log_ws[:,i,sample]=pool_outputs[i*samples+sample][2:]
    
    xs=xs.flatten()
    log_ws=log_ws.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled1_x_v16_5.txt",xs,fmt="%f")
    np.savetxt("Observations&data/CCPF_ou100_scaled1_lw_v16_5.txt",log_ws,fmt="%f")


    this sample is made in order to check if the hypothesis of the trade off of d improves the rates of 
    CCPF_ou100_scaled1_enes_v20.txt 
    CCPF_ou100_scaled1_enes_v21.txt
    np.random.seed(1)
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
    sca=1e0
    Lamb_par=1.33/4
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*1e5
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    p=8
    N0=100
    enes=np.array([2**i for i in range(p+1)])*N0
    pfs=np.zeros((len(enes),samples,int(T/d),dim))
    inputs=[]
    l=5
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            
            inputs.append([sample+samples*i,collection_input])             
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled1_enes_v21.txt",pfs,fmt="%f")


    check the variance of the cppf
    CCPF_ou100_scaled1_enes_v20.txt 
    np.random.seed(1)
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
    sca=1e0
    Lamb_par=1.33/4
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*1e5
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=1
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    p=8
    N0=100
    enes=np.array([2**i for i in range(p+1)])*N0
    pfs=np.zeros((len(enes),samples,int(T/d),dim))
    inputs=[]
    l=5
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            
            inputs.append([sample+samples*i,collection_input])             
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled1_enes_v20.txt",pfs,fmt="%f")



    this experiment was run in order to decrease the coupling constant

    CCPF_ou100_scaled1_v20.txt
    np.random.seed(1)
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
    sca=1e0
    Lamb_par=1.33/4
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*1e5
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=1
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled1_v20.txt",pfs,fmt="%f")


    this experiment is made so we get the rates with smaller d, which can be better 
    depending on the trade off. 
    CCPF_ou100_scaled1_v19.txt 

    np.random.seed(1)
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
    sca=1e0
    Lamb_par=1.33/2
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*5e4
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=1
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled1_v19.txt",pfs,fmt="%f")


    In this experiments we check again the same rates as in experiment 
    CCPF_ou100_scaled1_lw_v16_5.txt 
    np.random.seed(1)
    T=10
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    comp_matrix=np.array([[1]])
    inv_mat=la.inv(comp_matrix)
    S=diags(np.random.normal(0.99,0.1,dim),0).toarray()    
    sca=1e0
    Lamb_par=1.33/2
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*5e4
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=10
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    xs=np.zeros((2,len(eles),samples,int(T/d),N,dim))
    log_ws=np.zeros((2,len(eles),samples,int(T/d),N))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF_new,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            xs[:,i,sample]=pool_outputs[i*samples+sample][:2]
            log_ws[:,i,sample]=pool_outputs[i*samples+sample][2:]
    xs=xs.flatten()
    log_ws=log_ws.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled1_x_v16_6.txt",xs,fmt="%f")
    np.savetxt("Observations&data/CCPF_ou100_scaled1_lw_v16_6.txt",log_ws,fmt="%f")


    In this experiment we decrease the interval d (and T=1) in order to check the strong error
    whenever we haven't resampled, we observe in some experiments that it's not correct, i.e., it's of 
    order Delta_l instead of Delta_l^2.

    CCPF_ou100_scaled1_x_v16_5.txt 
    np.random.seed(1)
    T=1
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    comp_matrix=np.array([[1]])
    inv_mat=la.inv(comp_matrix)
    S=diags(np.random.normal(0.99,0.1,dim),0).toarray()    
    sca=1e0
    Lamb_par=1.33/2
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*5e4
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=1
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    xs=np.zeros((2,len(eles),samples,int(T/d),N,dim))
    log_ws=np.zeros((2,len(eles),samples,int(T/d),N))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF_new,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            xs[:,i,sample]=pool_outputs[i*samples+sample][:2]
            log_ws[:,i,sample]=pool_outputs[i*samples+sample][2:]
    xs=xs.flatten()
    log_ws=log_ws.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled1_x_v16_5.txt",xs,fmt="%f")
    np.savetxt("Observations&data/CCPF_ou100_scaled1_lw_v16_5.txt",log_ws,fmt="%f")



    we make this experiment in order to compare the constants of the coupling with the constant
    of the variance.
    np.random.seed(1)
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
    sca=1e0
    Lamb_par=1.33/2
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*5e4
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=4
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    p=8
    N0=100
    enes=np.array([2**i for i in range(p+1)])*N0
    pfs=np.zeros((len(enes),samples,int(T/d),dim))
    inputs=[]
    l=5
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            
            inputs.append([sample+samples*i,collection_input])             
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled1_enes_v3.txt",pfs,fmt="%f")



    We make this experiment to check the actual strong error of the signal system in 
    whenever we don't have any normalization of the likelihood.
    CCPF_ou100_scaled1_x_v16_4.txt 
    np.random.seed(1)
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
    sca=1e0
    Lamb_par=1.33/2
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*5e4
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=4
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    xs=np.zeros((2,len(eles),samples,int(T/d),N,dim))
    log_ws=np.zeros((2,len(eles),samples,int(T/d),N))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF_new,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            xs[:,i,sample]=pool_outputs[i*samples+sample][:2]
            log_ws[:,i,sample]=pool_outputs[i*samples+sample][2:]
    
    xs=xs.flatten()
    log_ws=log_ws.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled1_x_v16_4.txt",xs,fmt="%f")
    np.savetxt("Observations&data/CCPF_ou100_scaled1_lw_v16_4.txt",log_ws,fmt="%f")



    We make this experiment to check the actual strong error of the signal system,
    meaning that there is no resampling and just propagation of particles. 

    CCPF_ou100_scaled1_lw_v16_3.txt 

    np.random.seed(1)
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
    sca=1e0
    Lamb_par=1.33/2*0 
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*5e4
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=4
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    xs=np.zeros((2,len(eles),samples,int(T/d),N,dim))
    log_ws=np.zeros((2,len(eles),samples,int(T/d),N))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF_new,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            xs[:,i,sample]=pool_outputs[i*samples+sample][:2]
            log_ws[:,i,sample]=pool_outputs[i*samples+sample][2:]
    
    xs=xs.flatten()
    log_ws=log_ws.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled1_x_v16_3.txt",xs,fmt="%f")
    np.savetxt("Observations&data/CCPF_ou100_scaled1_lw_v16_3.txt",log_ws,fmt="%f")


    we notice taht CCPF_ou100_scaled1_v16_2.txt's second moment is Delta_l when it should be 
    Delta_l^2 since no resampling has occured.     

    CCPF_ou100_scaled1_v16_2.txt 
    np.random.seed(1)
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
    sca=1e0
    Lamb_par=1.33/2*0 
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*5e4
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=4
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled1_v16_2.txt",pfs,fmt="%f")


    In this experiments we are going to compute the strong error in a different and more 
    appropiate way where we compute two expectations, the first its with respect to the measure 
    of the particle filter, the second w.r.t many filters. 

    CCPF_ou100_scaled1_lw_v19.txt
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
    sca=1e0
    Lamb_par=1.33/2
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*5e4
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=4
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    xs=np.zeros((2,len(eles),samples,int(T/d),N,dim))
    log_ws=np.zeros((2,len(eles),samples,int(T/d),N))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            xs[:,i,sample]=pool_outputs[i*samples+sample][:2]
            log_ws[:,i,sample]=pool_outputs[i*samples+sample][2:]
    
    xs=xs.flatten()
    log_ws=log_ws.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled1_x_v19.txt",xs,fmt="%f")
    np.savetxt("Observations&data/CCPF_ou100_scaled1_lw_v19.txt",log_ws,fmt="%f")


    CCPF_ou100_scaled1_v17 gave the Delta_l^2 rate(which is the irght one) at the first step,
    since this is the first step, we haven't had resampling and we should expect Delta_l^2, are the 
    weights lipshitz?
    CCPF_ou100_scaled1_v18.txt 
    np.random.seed(1)
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
    sca=5e-1
    Lamb_par=1.33/4*0
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*5e4
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=10
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled1_v18.txt",pfs,fmt="%f")



We start checking the variance particle filter 
    CCPF_ou100_scaled1_enes_v2.txt
    np.random.seed(1)
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
    sca=1e0
    Lamb_par=1.33/2
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*5e4
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    p=8
    N0=100
    enes=np.array([2**i for i in range(p+1)])*N0
    pfs=np.zeros((len(enes),samples,int(T/d),dim))
    inputs=[]
    l=5
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            
            inputs.append([sample+samples*i,collection_input])             
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled1_enes_v2.txt",pfs,fmt="%f")



    In this experiment we check the coupling with the full CPF with some small scaling wrt to
    CCPF_ou100_scaled1_v16.txt
    CCPF_ou100_scaled1_v17.txt
    np.random.seed(1)
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
    sca=5e-1
    Lamb_par=1.33/4
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*5e4
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=10
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled1_v17.txt",pfs,fmt="%f")



    in this experiment we check the coupling with the full CPF and the same scaling as
    in  CCPF_ou100_scaled1_v15.txt 
    CCPF_ou100_scaled1_v16.txt

    np.random.seed(1)
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
    sca=1e0
    Lamb_par=1.33/2
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*5e4
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=4
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled1_v16.txt",pfs,fmt="%f")


    In htis iteration we check the strong error of the signal, we no scaling.
    CCPF_ou100_scaled1_v15.txt 
    np.random.seed(1)
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
    sca=1e0
    Lamb_par=1.33/20*0
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*5e4
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=10
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled1_v15.txt",pfs,fmt="%f")


    we check if the strong error of the signal is still order Delta^2
    CCPF_ou100_scaled1_v14.txt
    np.random.seed(1)
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
    Lamb_par=1.33/20*0
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*5e4
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=10
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled1_v14.txt",pfs,fmt="%f")


    WE Keep scaling the system to check if the rates behave.
    CCPF_ou100_scaled1_v13.txt 
    np.random.seed(1)
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
    Lamb_par=1.33/20
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*5e4
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=10
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled1_v13.txt",pfs,fmt="%f") 


we test the thehypothesis that we can scale the system so we get the right rates 
CCPF_ou100_scaled1_v12.txt
    np.random.seed(1)
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
    sca=5e-2
    Lamb_par=1.33/10
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*5e4
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=10
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled1_v12.txt",pfs,fmt="%f") 



we want to check the hypothesis that decreasing the scaling, Lamb_par and increasing d we get a
pseudoscaling that should show the Delta_l rate. 
CCPF_lan100_scaled1_v41.txt
np.random.seed(1)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=1e-2
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    cov=I*1e5
    Lamb_par=2/3
    print("Lamb_par",Lamb_par)
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=10
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0 ,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])           
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v41.txt",pfs,fmt="%f")


we want to  check if the Delta_l^2 rate shows when Lamb_par=0.
CCPF_ou100_scaled1_v11.txt 
    np.random.seed(1)
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
    sca=1e-1
    Lamb_par=1.33/5*0
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*5e4
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=10
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled1_v11.txt",pfs,fmt="%f") 


    CCPF_lan100_scaled1_v40.txt 
    np.random.seed(1)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=1e-1
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    cov=I*5e4
    Lamb_par=3/3
    print("Lamb_par",Lamb_par)
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=10
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0 ,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])           
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v40.txt",pfs,fmt="%f") 


    CCPF_ou100_scaled1_v11.txt 
    np.random.seed(1)
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
    sca=1e-1
    Lamb_par=1.33/5
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*5e4
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=10
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_ou100_scaled1_v11.txt",pfs,fmt="%f")


    CCPF_ou100_scaled1_v10.txt 
    np.random.seed(1)
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
    sca=1e-1
    Lamb_par=1.33/5
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*5e4
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=10
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_ou100_scaled1_v10.txt",pfs,fmt="%f")

CCPF_ou100_scaled1_v9.txt 
np.random.seed(1)
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
    sca=1e-1
    Lamb_par=1.33/5
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*5e4
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=4
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_ou100_scaled1_v9.txt",pfs,fmt="%f")


CCPF_lan100_scaled1_v39.txt 
np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=1e-0
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    cov=I*1e4
    Lamb_par=3/3
    print("Lamb_par",Lamb_par)
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=10
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])           
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v39.txt",pfs,fmt="%f")




    np.random.seed(1)
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
    sca=1e-1
    Lamb_par=1.33/5
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*5e4
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_ou100_scaled1_v8.txt",pfs,fmt="%f")


    CCPF_lan100_scaled1_v38.txt 
    np.random.seed(1)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=1e-1
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    cov=I*5e4
    Lamb_par=3/3
    print("Lamb_par",Lamb_par)
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0 ,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])           
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v38.txt",pfs,fmt="%f")  

    CCPF_ou100_scaled1_v6_5.txt 
    np.random.seed(1)
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
    sca=1e-0
    Lamb_par=1.33/4
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*5e4
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_ou100_scaled1_v6_5.txt",pfs,fmt="%f")


    CPF_lan100_scaled1_v37_5.txt 
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=1e-0
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    cov=I*5e4
    Lamb_par=3/3
    print("Lamb_par",Lamb_par)
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])           
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v37_5.txt",pfs,fmt="%f")  


    CCPF_ou100_scaled1_v6_4.txt
    np.random.seed(1)
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
    sca=1e-0
    Lamb_par=1.33/2
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*1e4
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_ou100_scaled1_v6_4.txt",pfs,fmt="%f")


    CCPF_lan100_scaled1_v37_4.txt 
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=1e-0
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    cov=I*1e4
    Lamb_par=3/3
    print("Lamb_par",Lamb_par)
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=10
    L=10
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])           
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v37_4.txt",pfs,fmt="%f")



    CCPF_ou100_scaled1_v6_3.txt 
    np.random.seed(1)
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
    sca=1e-0
    Lamb_par=1.33*2
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*1e4
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_ou100_scaled1_v6_3.txt",pfs,fmt="%f")


    CCPF_lan100_scaled1_v37_3.txt 
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=1e-0
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    cov=I*1e4
    Lamb_par=3/3
    print("Lamb_par",Lamb_par)
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])           
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v37_3.txt",pfs,fmt="%f")


    CCPF_ou100_scaled1_v6_2.txt 
    np.random.seed(0)
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
    sca=1e-0
    Lamb_par=1.33
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*1e5
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_ou100_scaled1_v6_2.txt",pfs,fmt="%f")


    CCPF_ou100_scaled1_v8.txt 
    np.random.seed(0)
    T=50
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    comp_matrix=np.array([[1]])
    inv_mat=la.inv(comp_matrix)
    S=diags(np.random.normal(0.99,0.1,dim),0).toarray()    
    sca=1e-4
    Lamb_par=1.33*0
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*1e5
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_ou100_scaled1_v8.txt",pfs,fmt="%f")


    CCPF_lan100_scaled1_v37_2.txt 
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=1e-0
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    cov=I*1e4
    Lamb_par=5/3
    print("Lamb_par",Lamb_par)
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])           
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v37_2.txt",pfs,fmt="%f")



np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=1e-0
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    cov=I*1e4
    Lamb_par=5/3*0
    print("Lamb_par",Lamb_par)
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])           
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v37.txt",pfs,fmt="%f")

    CCPF_ou100_scaled1_v7.txt 
    np.random.seed(0)
    T=50
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    comp_matrix=np.array([[1]])
    inv_mat=la.inv(comp_matrix)
    S=diags(np.random.normal(0.99,0.1,dim),0).toarray()    
    sca=1e-0
    Lamb_par=1.33
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*1e5
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=10
    L=10
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_ou100_scaled1_v7.txt",pfs,fmt="%f")


    CCPF_ou100_scaled1_v6.txt 
    np.random.seed(0)
    T=50
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    comp_matrix=np.array([[1]])
    inv_mat=la.inv(comp_matrix)
    S=diags(np.random.normal(0.99,0.1,dim),0).toarray()    
    sca=1e-0
    Lamb_par=1.33
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*1e5
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_ou100_scaled1_v6.txt",pfs,fmt="%f")

CCPF_ou100_scaled1_v5.txt  
np.random.seed(0)
    T=50
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    comp_matrix=np.array([[1]])
    inv_mat=la.inv(comp_matrix)
    S=diags(np.random.normal(0.99,0.1,dim),0).toarray()    
    sca=1e-0
    Lamb_par=1.33*5
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*1e5
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_ou100_scaled1_v5.txt",pfs,fmt="%f")

np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=1e-4
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)*5e-1
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    cov=I*1e4
    Lamb_par=5/3*0
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])           
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v36.txt",pfs,fmt="%f")


    CCPF_lan100_scaled1_v16_3.txt   
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=1e-3
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)*5e-1
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    cov=I*1e4
    Lamb_par=5/3*0
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])           
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v16_3.txt",pfs,fmt="%f")


CCPF_ou100_scaled1_v4.txt 
    np.random.seed(0)
    T=10
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    comp_matrix=np.array([[1]])
    inv_mat=la.inv(comp_matrix)
    S=diags(np.random.normal(0.99,0.1,dim),0).toarray()    
    sca=1e-0
    Lamb_par=0
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*1e4
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_ou100_scaled1_v4.txt",pfs,fmt="%f")

    CCPF_lan100_scaled1_v35.txt 
    np.random.seed(0)
    T=10
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=1e-1*0
    pars=[sca,df]
    fi=np.array([[1]])*5e-1
    cov=I*1e2
    Lamb_par=0 
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    obs_time=np.array([])
    obs=np.array([])
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])           
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v35.txt",pfs,fmt="%f")

    CCPF_lan100_scaled1_v34.txt 
    np.random.seed(0)
    T=10
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=1e-1
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)*5e-1*0
    cov=I*1e2
    Lamb_par=0 
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    obs_time=np.array([])
    obs=np.array([])
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])           
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v34.txt",pfs,fmt="%f")


    
    CCPF_lan100_scaled1_v33.txt 
    np.random.seed(0)
    T=10
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=1e-1
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)*5e-1
    cov=I*1e2
    Lamb_par=0 
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    obs_time=np.array([])
    obs=np.array([])
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])           
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v33.txt",pfs,fmt="%f")


    

    CCPF_lan100_scaled1_v32.txt 
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=5e-4
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)*5e-1
    cov=I*1e2
    Lamb_par=0 
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)

    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    obs_time=np.array([])
    obs=np.array([])
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])           
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v32.txt",pfs,fmt="%f")



    CCPF_lan100_scaled1_v31.txt 
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=1e-4
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)*5e-1
    cov=I*1e2
    Lamb_par=0 
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)

    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    obs_time=np.array([])
    obs=np.array([])
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])           
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v31.txt",pfs,fmt="%f")


    CCPF_lan100_scaled1_v30.txt 
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=1e-3
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)*5e-1
    cov=I*1e2
    Lamb_par=0 
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)

    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    obs_time=np.array([])
    obs=np.array([])
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])           
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v30.txt",pfs,fmt="%f")


    CCPF_lan100_scaled1_v29.txt 
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=1e-1
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)*5e-1
    cov=I*1e2
    Lamb_par=0 
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)

    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    obs_time=np.array([])
    obs=np.array([])
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])           
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v29.txt",pfs,fmt="%f")


    CCPF_lan100_scaled1_v28.txt 
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=1e-1
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)*5e-1
    cov=I*1e2
    Lamb_par=3/3
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)

    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    obs_time=np.array([])
    obs=np.array([])
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])           
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v28.txt",pfs,fmt="%f")


    CCPF_lan100_scaled1_v27.txt 
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=1e0
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)*5e-1
    cov=I*1e2
    Lamb_par=3/3
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)

    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    obs_time=np.array([])
    obs=np.array([])
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])           
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v27.txt",pfs,fmt="%f")


    CCPF_lan100_scaled1_v26.txt 
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=1e-5
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)*5e-1
    cov=I*1e2
    Lamb_par=3/3
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])           
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v26.txt",pfs,fmt="%f")



    CCPF_lan100_scaled1_v25.txt 
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=1e-5
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)*5e-1
    cov=I*1e3
    Lamb_par=2/3
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])           
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v25.txt",pfs,fmt="%f")




    CCPF_lan100_scaled1_v24_2.txt 
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=1e-4
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)*5e-1
    cov=I*1e3
    Lamb_par=2/3
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=10
    L=10
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])           
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v24_2.txt",pfs,fmt="%f")



    CCPF_lan100_scaled1_v24.txt  
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=1e-4
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)*5e-1
    cov=I*1e3
    Lamb_par=2/3
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])           
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v24.txt",pfs,fmt="%f")



    CCPF_lan100_scaled1_v23.txt 
np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=1e-3
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)*5e-1
    cov=I*1e4
    Lamb_par=2/3
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])           
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v23.txt",pfs,fmt="%f")

    CCPF_lan100_scaled1_v16_2.txt 
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=1e-3
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)*5e-1
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    cov=I*1e4
    Lamb_par=5/3
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=10
    L=10
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])           
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v16_2.txt",pfs,fmt="%f")



CCPF_ou100_scaled1_3.txt 
    np.random.seed(0)
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
    sca=1e-0
    Lamb_par=1.33*5
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*1e4
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_ou100_scaled1_3.txt",pfs,fmt="%f")



CCPF_ou100_scaled1_v1.txt
np.random.seed(0)
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
    sca=1e-1
    Lamb_par=1.33
    fi=(inv_mat@S@comp_matrix)*np.sqrt(sca)
    cov=I*1e1
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=(inv_mat@B@comp_matrix)*sca
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou_pf,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_ou100_scaled1_v1.txt",pfs,fmt="%f")


CCPF_lan100_scaled1_v22.txt 
np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=1e-3
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)*5e-1
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    cov=I*1e4
    Lamb_par=5/3
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=10
    L=10
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])           
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v22.txt",pfs,fmt="%f")



np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=1e-4
    fi=np.array([[1]])*np.sqrt(sca)*5e-1
    cov=I*1e4
    Lamb_par=5/3
    pars=[sca,df]
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])           
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v20.txt",pfs,fmt="%f")



     CCPF_lan100_scaled1_v19.txt
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=1e-1
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)*5e-1
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    cov=I*1e6
    Lamb_par=3/3
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])           
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v19.txt",pfs,fmt="%f")


    CCPF_lan100_scaled1_v19.txt 
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=1e-1
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)*5e-1
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    cov=I*1e4
    Lamb_par=10/3
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])           
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v19.txt",pfs,fmt="%f")



    CCPF_lan100_scaled1_v18.txt 
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=1e-1
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)*5e-1
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    cov=I*1e4
    Lamb_par=5/3
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])           
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v18.txt",pfs,fmt="%f")



    CCPF_lan100_scaled1_v17.txt 
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=5e-4
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)*5e-1
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    cov=I*1e4
    Lamb_par=5/3
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])           
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v17.txt",pfs,fmt="%f")


    CCPF_lan100_scaled1_v16.txt
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=1e-3
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)*5e-1
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    cov=I*1e4
    Lamb_par=5/3
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])           
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v16.txt",pfs,fmt="%f")


    CCPF_lan100_scaled1_v15_2.txt 

    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=5e-3
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)*5e-1
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    cov=I*1e4
    Lamb_par=5/3
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=200
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=5000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])           
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v15_2.txt",pfs,fmt="%f")



    CCPF_lan100_scaled1_v15.txt 
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=5e-3
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)*5e-1
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    cov=I*1e4
    Lamb_par=5/3
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])           
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v15.txt",pfs,fmt="%f")


    CCPF_lan100_scaled1_v14_2.txt 
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=1e-2
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)*5e-1
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    cov=I*1e4
    Lamb_par=5/3
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=200
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=5000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])           
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v14_2.txt",pfs,fmt="%f")

    CCPF_lan100_scaled1_v14.txt 
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=1e-2
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)*5e-1
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    cov=I*1e4
    Lamb_par=5/3
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])           
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v14.txt",pfs,fmt="%f")


    CCPF_lan100_scaled1_v13.txt 
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=1e-1
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)*5e-1
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    cov=I*1e4
    Lamb_par=5/3
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])           
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v13.txt",pfs,fmt="%f")

    CCPF_lan100_scaled1_v12.txt 
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=1e-1
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)*1e-1
    print("sca and fi",sca,fi)
    
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    cov=I*1e5
    Lamb_par=4/3
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])           
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v12.txt",pfs,fmt="%f")



    CCPF_lan100_scaled1_v11.txt 
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=5e-3
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)*1e-1
    print("sca and fi",sca,fi)
    Sig_ou
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    cov=I*1e5
    Lamb_par=4/3
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])           
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v11.txt",pfs,fmt="%f")


    CCPF_lan100_scaled1_v10.txt 
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=1e-2
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)*1e-1
    print("sca and fi",sca,fi)
    Sig_ou
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    cov=I*1e5
    Lamb_par=4/3
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])           
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v10.txt",pfs,fmt="%f")



CCPF_lan100_scaled1_v9.txt 
np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=1e-2
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_ou,fi]
    cov=I*1e5
    Lamb_par=4/3
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_ou,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])           
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v9.txt",pfs,fmt="%f")



    CCPF_lan100_scaled1_v8.txt 
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=1e-2
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    cov=I*1e5
    Lamb_par=4/3
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])           
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v8.txt",pfs,fmt="%f")



CCPF_lan100_scaled1_v7.txt 

np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=1e-2
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    cov=I*1e4
    Lamb_par=4/3
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=200
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=5000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])           
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v7.txt",pfs,fmt="%f")



CCPF_lan100_scaled1_v6.txt 
np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=1e-2
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    cov=I*1e3
    Lamb_par=4/3
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=200
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=5000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])           
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v6.txt",pfs,fmt="%f")


CCPF_lan100_scaled1_enes_v6.txt 
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=1e-2
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)
    print("sca and fi",sca,fi)
    collection_input=[dim, b_lan_new,pars,Sig_lan,fi]
    cov=I*1e3
    Lamb_par=4/3
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    p=8
    N0=100
    enes=np.array([2**i for i in range(p+1)])*N0
    pfs=np.zeros((len(enes),samples,int(T/d),dim))
    inputs=[]
    l=5
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_enes_v6.txt",pfs,fmt="%f")


CCPF_lan100_scaled1_enes_v5.txt  
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=1e-1
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)
    print("sca and fi",sca,fi)
    collection_input=[dim, b_lan_new,pars,Sig_lan,fi]
    cov=I*1e3
    Lamb_par=4/3
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    p=8
    N0=100
    enes=np.array([2**i for i in range(p+1)])*N0
    pfs=np.zeros((len(enes),samples,int(T/d),dim))
    inputs=[]
    l=5
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_enes_v5.txt",pfs,fmt="%f")


    CCPF_lan100_scaled1_v5.txt 
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=1e-1
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    cov=I*1e3
    Lamb_par=4/3
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=200
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=5000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])           
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v5.txt",pfs,fmt="%f")



CCPF_lan100_scaled1_enes_v4_2.txt 
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=1e-1
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)
    print("sca and fi",sca,fi)
    collection_input=[dim, b_lan_new,pars,Sig_lan,fi]
    cov=I*5e2
    Lamb_par=5/3
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    p=8
    N0=100
    enes=np.array([2**i for i in range(p+1)])*N0
    pfs=np.zeros((len(enes),samples,int(T/d),dim))
    inputs=[]
    l=5
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_enes_v4_2.txt",pfs,fmt="%f") 


np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=1e-1
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)
    print("sca and fi",sca,fi)
    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    cov=I*5e2
    Lamb_par=5/3
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=200
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=5000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])           
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v4_2.txt",pfs,fmt="%f")


 CCPF_lan100_scaled1_enes_v4.txt 
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=1e-1
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)
    print("sca and fi",sca,fi)
    collection_input=[dim, b_lan_new,pars,Sig_lan,fi]
    cov=I*5e2
    Lamb_par=5/3
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    p=8
    N0=10
    enes=np.array([2**i for i in range(p+1)])*N0
    pfs=np.zeros((len(enes),samples,int(T/d),dim))
    inputs=[]
    l=5
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_enes_v4.txt",pfs,fmt="%f") 


CCPF_lan100_scaled1_v4.txt 
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=1e-1
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)
    print("sca and fi",sca,fi)

    collection_input=[dim,b_lan_new,pars,Sig_lan,fi]
    cov=I*5e2
    Lamb_par=5/3
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v4.txt",pfs,fmt="%f")


    CCPF_lan100_scaled1_enes_v3_2.txt
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=1e-1
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)
    print("sca and fi",sca,fi)
    collection_input=[dim, b_lan_new,pars,Sig_lan,fi]
    cov=I*5e1
    Lamb_par=5/3
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    p=8
    N0=20
    enes=np.array([2**i for i in range(p+1)])*N0
    pfs=np.zeros((len(enes),samples,int(T/d),dim))
    inputs=[]
    l=5
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_enes_v3_2.txt",pfs,fmt="%f")  


    CCPF_lan100_scaled1_enes_v3.txt
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=1e-1
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)
    print("sca and fi",sca,fi)
    collection_input=[dim, b_lan_new,pars,Sig_lan,fi]
    cov=I*5e1
    Lamb_par=5/3
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    p=8
    N0=10
    enes=np.array([2**i for i in range(p+1)])*N0
    pfs=np.zeros((len(enes),samples,int(T/d),dim))
    inputs=[]
    l=5
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_enes_v3.txt",pfs,fmt="%f") 

    
    CCPF_lan100_scaled1_v3.txt 
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    sca=1e-1
    pars=[sca,df]
    fi=np.array([[1]])*np.sqrt(sca)
    print("sca and fi",sca,fi)
    collection_input=[dim, b_lan_new,pars,Sig_lan,fi]
    cov=I*5e1
    Lamb_par=5/3
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan_new,pars,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v3.txt",pfs,fmt="%f")



27439708
PMLPF_cox_gbm100_scaled1_v6.txt 
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
    Lamb_par=.3
    sigs=np.abs(np.random.normal(np.sqrt(2),0,dim))
    mu=mu*c
    sigs=sigs*np.sqrt(c)
    print(mu,sigs)
    comp_matrix=np.array([[1]])
    inv_mat=comp_matrix.T
    S=diags(np.random.normal(1,0,dim),0).toarray()
    Sig=inv_mat@S@comp_matrix
    fi=[sigs,Sig]
    collection_input=[dim, b_gbm,mu,Sig_gbm,fi]
    g_pars=[dim,cov]
    g_par=cov
    np.random.seed(0)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    Lmin=6
    Lmax=6
    eLes=np.array(np.arange(Lmin,Lmax+1))
    print(eLes)
    C0=0.03540286765887861
    C=0.00025205321130822886
    K=1.765425864219179e-09
    pfs=np.zeros((len(eLes),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eLes)):
        for sample in range(samples):
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
            collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,eles,d,eNes,dim,resamp_coef,phi,\
            dim_out,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([samples*i+sample,collection_input])
    pool = multiprocessing.Pool(processes=40)
    pool_outputs = pool.map(PMLPF_cox,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]

    pfs=pfs.flatten()

    np.savetxt("Observations/PMLPF_cox_gbm100_scaled1_v6.txt",pfs,fmt="%f")



CCPF_lan100_scaled1_enes_v1.txt 
np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    fi=np.array([[1]])
    collection_input=[dim, b_lan,df,Sig_lan,fi]
    cov=I*5e1
    Lamb_par=3/3
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    p=8
    N0=10
    enes=np.array([2**i for i in range(p+1)])*N0
    pfs=np.zeros((len(enes),samples,int(T/d),dim))
    inputs=[]
    l=5
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan,df,Sig_lan,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_lan100_scaled1_enes_v1.txt",pfs,fmt="%f")   



np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    fi=np.array([[1]])
    collection_input=[dim, b_lan,df,Sig_lan,fi]
    cov=I*5e1
    Lamb_par=3/3

    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan,df,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_lan100_scaled1_v2.txt",pfs,fmt="%f")


27435126
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
    Lamb_par=.3
    sigs=np.abs(np.random.normal(np.sqrt(2),0,dim))
    mu=mu*c
    sigs=sigs*np.sqrt(c)
    print(mu,sigs)
    comp_matrix=np.array([[1]])
    inv_mat=comp_matrix.T
    S=diags(np.random.normal(1,0,dim),0).toarray()
    Sig=inv_mat@S@comp_matrix
    fi=[sigs,Sig]
    collection_input=[dim, b_gbm,mu,Sig_gbm,fi]
    g_pars=[dim,cov]
    g_par=cov
    np.random.seed(0)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    Lmin=l0
    Lmax=5
    eLes=np.array(np.arange(Lmin,Lmax+1))
    print(eLes)
    C0=0.03540286765887861
    C=0.00025205321130822886
    K=1.765425864219179e-09
    pfs=np.zeros((len(eLes),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eLes)):
        for sample in range(samples):
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
            collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,eles,d,eNes,dim,resamp_coef,phi,\
            dim_out,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([samples*i+sample,collection_input])
    pool = multiprocessing.Pool(processes=40)
    pool_outputs = pool.map(PMLPF_cox,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]

    pfs=pfs.flatten()

    np.savetxt("Observations/PMLPF_cox_gbm100_scaled1_v5.txt",pfs,fmt="%f")


27411831
PPF_cox_nldt_T100_scaled3_v1.txt 
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
    start=time.time()
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=1
    es=1e-10
    C0=0.000117
    C=2.8284271247461907e-06
    K=8e-11
    N=int(2*C0/es)
    L=int(np.ceil(-np.log2(es/(2*np.sqrt(K)))/2,dtype=float))
    l0=L
    Lmin=L
    Lmax=L
    d=2**0
    eLes=np.array(np.arange(Lmin,Lmax+1))
    pfs=np.zeros((len(eLes),samples,int(T/d),dim))
    print(pfs)
    inputs=[]
    collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,L,d,N,dim,resamp_coef,\
    g_den,g_par,Norm_Lambda,Lamb_par]
    inputs.append([0,collection_input])
    pool = multiprocessing.Pool(processes=1)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")
    pfs[0,0]=pool_outputs[0]
    pfs=pfs.flatten()
    np.savetxt("Observations/PPF_cox_nldt_T100_scaled3_v1.txt",pfs,fmt="%f")


    27411369
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
    start=time.time()
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=1
    es=1e-4
    C0=0.000117
    C=2.8284271247461907e-06
    K=8e-11
    N=int(2*C0/es)
    L=int(np.ceil(-np.log2(es/(2*np.sqrt(K)))/2,dtype=float))
    l0=L
    Lmin=L
    Lmax=L
    d=2**0
    eLes=np.array(np.arange(Lmin,Lmax+1))
    pfs=np.zeros((len(eLes),samples,int(T/d),dim))
    print(pfs)
    inputs=[]
    collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,L,d,N,dim,resamp_coef,\
    g_den,g_par,Norm_Lambda,Lamb_par]
    inputs.append([0,collection_input])
    pool = multiprocessing.Pool(processes=1)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")
    pfs[0,0]=pool_outputs[0]
    pfs=pfs.flatten()
    np.savetxt("Observations/PPF_cox_nldt_T100_scaled3_vtest.txt",pfs,fmt="%f")


    27407227
    PPF_cox_gbmt100_scaled1_v3.txt
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
    Lamb_par=.3
    sigs=np.abs(np.random.normal(np.sqrt(2),0,dim))
    mu=mu*c
    sigs=sigs*np.sqrt(c)
    print(mu,sigs)
    comp_matrix=np.array([[1]])
    inv_mat=comp_matrix.T
    S=diags(np.random.normal(1,0,dim),0).toarray()
    Sig=inv_mat@S@comp_matrix
    fi=[sigs,Sig]
    collection_input=[dim, b_gbm,mu,Sig_gbm,fi]
    g_pars=[dim,cov]
    g_par=cov
    np.random.seed(0)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=0
    Lmin=6
    Lmax=6
    eLes=np.array(np.arange(Lmin,Lmax+1))
    print(eLes)
    C0=0.03283620263287618
    C=0.00025205321130822886
    K=1.765425864219179e-09
    pfs=np.zeros((len(eLes),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eLes)):
        for sample in range(samples):
            l=eLes[i]

            N=int(C0*2**(2*l)*1e-4/K)
            collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,N,dim,resamp_coef,\
            g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([samples*i+sample,collection_input])
    pool = multiprocessing.Pool(processes=40)
    pool_outputs = pool.map(PCox_PF,inputs)

    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")
    for i in range(len(eLes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]

    pfs=pfs.flatten()
    np.savetxt("Observations/PPF_cox_gbmt100_scaled1_v3.txt",pfs,fmt="%f")



    27406675
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
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=0
    Lmin=0
    Lmax=5
    eLes=np.array(np.arange(Lmin,Lmax+1))
    print(eLes)
    C0=0.000117
    C=2.8284271247461907e-06
    K= 8e-11

    pfs=np.zeros((len(eLes),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eLes)):
        for sample in range(samples):
            l=eLes[i]
            scale=1e-4
            N=int(C0*2**(2*l)*scale/K)
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,N,dim,resamp_coef,\
            g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([samples*i+sample,collection_input])
    pool = multiprocessing.Pool(processes=40)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")
    for i in range(len(eLes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations/PPF_cox_nldt100_scaled3_v1.txt",pfs,fmt="%f")

    
    
    PMLPF_cox_nldt_T100_scaled3_v1.txt 
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
    start=time.time()
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=0
    Lmin=0
    Lmax=5
    d=2**0
    eLes=np.array(np.arange(Lmin,Lmax+1))
    print(eLes)
    C0=0.000117
    C=2.8284271247461907e-06
    K=8e-11
    pfs=np.zeros((len(eLes),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eLes)):
        for sample in range(samples):
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
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,eles,d,eNes,dim,resamp_coef,phi,\
            dim_out,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([samples*i+sample,collection_input])

    pool = multiprocessing.Pool(processes=40)
    pool_outputs = pool.map(PMLPF_cox,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")
    for i in range(len(eLes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations/PMLPF_cox_nldt_T100_scaled3_v1.txt",pfs,fmt="%f")

PMLPF_cox_nldt_T100_scaled3_vtest.txt 
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
    start=time.time()
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=10
    l0=0
    Lmin=0
    Lmax=2
    d=2**0
    eLes=np.array(np.arange(Lmin,Lmax+1))
    print(eLes)
    C0=0.000117
    C=2.8284271247461907e-06
    K= 8e-11
    pfs=np.zeros((len(eLes),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eLes)):
        for sample in range(samples):
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
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,eles,d,eNes,dim,resamp_coef,phi,\
            dim_out,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([samples*i+sample,collection_input])        
           
    pool = multiprocessing.Pool(processes=40)
    pool_outputs = pool.map(PMLPF_cox,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eLes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations/PMLPF_cox_nldt_T100_scaled3_vtest.txt",pfs,fmt="%f")


PPF_cox_gbmt100_scaled1_v2.txt

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
    Lamb_par=.3
    sigs=np.abs(np.random.normal(np.sqrt(2),0,dim))
    mu=mu*c
    sigs=sigs*np.sqrt(c)
    print(mu,sigs)
    comp_matrix=np.array([[1]])
    inv_mat=comp_matrix.T
    S=diags(np.random.normal(1,0,dim),0).toarray()
    Sig=inv_mat@S@comp_matrix
    fi=[sigs,Sig]
    collection_input=[dim, b_gbm,mu,Sig_gbm,fi]
    g_pars=[dim,cov]
    g_par=cov
    np.random.seed(0)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=0
    Lmin=6
    Lmax=6
    eLes=np.array(np.arange(Lmin,Lmax+1))
    print(eLes)
    C0=0.03283620263287618
    C=0.00025205321130822886
    K=1.765425864219179e-09
    pfs=np.zeros((len(eLes),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eLes)):
        for sample in range(samples):
            l=eLes[i]

            N=int(C0*2**(2*l)*1e-4/K)
            collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,N,dim,resamp_coef,\
            g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([samples*i+sample,collection_input])
    pool = multiprocessing.Pool(processes=50)
    pool_outputs = pool.map(PCox_PF,inputs)

    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")
    for i in range(len(eLes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]

    pfs=pfs.flatten()
    np.savetxt("Observations/PPF_cox_gbmt100_scaled1_v2.txt",pfs,fmt="%f")


PMLPF_cox_gbm100_scaled1_v4.txt 


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
    Lamb_par=.3
    sigs=np.abs(np.random.normal(np.sqrt(2),0,dim))
    mu=mu*c
    sigs=sigs*np.sqrt(c)
    print(mu,sigs)
    comp_matrix=np.array([[1]])
    inv_mat=comp_matrix.T
    S=diags(np.random.normal(1,0,dim),0).toarray()
    Sig=inv_mat@S@comp_matrix
    fi=[sigs,Sig]
    collection_input=[dim, b_gbm,mu,Sig_gbm,fi]
    g_pars=[dim,cov]
    g_par=cov
    np.random.seed(0)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    Lmin=l0
    Lmax=5
    eLes=np.array(np.arange(Lmin,Lmax+1))
    print(eLes)
    C0=0.03540286765887861
    C=0.00025205321130822886
    K=1.765425864219179e-09
    pfs=np.zeros((len(eLes),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eLes)):
        for sample in range(samples):
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
            collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,eles,d,eNes,dim,resamp_coef,phi,\
            dim_out,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([samples*i+sample,collection_input])
    pool = multiprocessing.Pool(processes=50)
    pool_outputs = pool.map(PMLPF_cox,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]

    pfs=pfs.flatten()

    np.savetxt("Observations/PMLPF_cox_gbm100_scaled1_v4.txt",pfs,fmt="%f")


    Cox_PF_gbm100_p_scaled_v16_2.txt
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
    Lamb_par=.3
    sigs=np.abs(np.random.normal(np.sqrt(2),0,dim))
    mu=mu*c
    sigs=sigs*np.sqrt(c)
    print(mu,sigs)
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
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l=5
    N0=10
    p=8
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    enes=N0*np.array([2**i for i in range(p)])

    pfs=np.zeros((len(enes),samples,int(T/d),dim))
    inputs=[]
    
    
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/Cox_PF_gbm100_p_scaled_v16_2.txt",pfs,fmt="%f")




PPF_cox_nldt100_scaled2_v1.txt 

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
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=0
    Lmin=0
    Lmax=3
    eLes=np.array(np.arange(Lmin,Lmax+1))
    print(eLes)
    C0=0.00997
    C=0.0004998
    K= 1.9131764860557113e-06
    pfs=np.zeros((len(eLes),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eLes)):
        for sample in range(samples):
            l=eLes[i]
            N=int(C0*2**(2*l)/K)
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,N,dim,resamp_coef,\
            g_den,g_par,Norm_Lambda,Lamb_par]   
            inputs.append([samples*i+sample,collection_input])
    pool = multiprocessing.Pool(processes=6)
    pool_outputs = pool.map(PCox_PF,inputs)

    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eLes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
    np.savetxt("Observations&data/PPF_cox_nldt100_scaled2_v1.txt",pfs,fmt="%f")



CCPF_nldt100_scaled3_v17_3.txt
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
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=400
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=10000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_nldt100_scaled3_v17_3.txt",pfs,fmt="%f") 


CCPF_nldt100_scaled3_v17_2.txt 

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
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=200
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=5000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_nldt100_scaled3_v17_2.txt",pfs,fmt="%f")



CCPF_nldt100_enes_scaled3_v17.txt
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
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    p=8
    N0=10
    enes=np.array([2**i for i in range(p+1)])*N0
    pfs=np.zeros((2,len(enes),samples,int(T/d),dim))
    inputs=[]
    l=5
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()   
    np.savetxt("Observations&data/CCPF_nldt100_enes_scaled3_v17.txt",pfs,fmt="%f")




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
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_nldt100_scaled3_v17.txt",pfs,fmt="%f") 


    CCPF_nldt100_scaled3_v16.txt 
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
    cov=I*1e-2
    Lamb_par=3/20
    fi=1/np.sqrt(sc)
    print(fi)
    collection_input=[dim, b_ou,A,Sig_nldt,fi]
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_nldt100_scaled3_v16.txt",pfs,fmt="%f") 




    CCPF_nldt100_scaled3_v15.txt 
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
    cov=I*3e-2
    Lamb_par=6/20
    fi=1/np.sqrt(sc)
    print(fi)
    collection_input=[dim, b_ou,A,Sig_nldt,fi]
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_nldt100_scaled3_v15.txt",pfs,fmt="%f") 


    CCPF_nldt100_enes_scaled3_v14.txt 
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
    cov=I*3e-2
    Lamb_par=3/20
    fi=1/np.sqrt(sc)
    print(fi)
    collection_input=[dim, b_ou,A,Sig_nldt,fi]
    
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    p=8
    N0=10
    enes=np.array([2**i for i in range(p+1)])*N0
    pfs=np.zeros((2,len(enes),samples,int(T/d),dim))
    inputs=[]
    l=5
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()   
    np.savetxt("Observations&data/CCPF_nldt100_enes_scaled3_v14.txt",pfs,fmt="%f")

    CCPF_nldt100_scaled3_v14.txt 
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
    cov=I*3e-2
    Lamb_par=3/20
    fi=1/np.sqrt(sc)
    print(fi)
    collection_input=[dim, b_ou,A,Sig_nldt,fi]
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_nldt100_scaled3_v14.txt",pfs,fmt="%f") 



    Truth_PF_cox_gbmt100_scaled1_v2.txt  
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
    Lamb_par=.3
    sigs=np.abs(np.random.normal(np.sqrt(2),0,dim))
    mu=mu*c
    sigs=sigs*np.sqrt(c)
    print(mu,sigs)
    comp_matrix=np.array([[1]])
    inv_mat=comp_matrix.T
    S=diags(np.random.normal(1,0,dim),0).toarray()
    Sig=inv_mat@S@comp_matrix
    fi=[sigs,Sig]
    collection_input=[dim, b_gbm,mu,Sig_gbm,fi]
    g_pars=[dim,cov]
    g_par=cov
    np.random.seed(0)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=1
    l0=0
    Lmin=7
    Lmax=7
    eLes=np.array(np.arange(Lmin,Lmax+1))
    print(eLes)
    C0=0.03283620263287618
    C=0.00025205321130822886
    K=1.765425864219179e-09
    pfs=np.zeros((len(eLes),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eLes)):
        for sample in range(samples):
            l=eLes[i]

            N=int(C0*2**(2*l)*1e-4/K)
            collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,N,dim,resamp_coef,\
            g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([samples*i+sample,collection_input])
    pool = multiprocessing.Pool(processes=1)
    pool_outputs = pool.map(PCox_PF,inputs)

    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")
    for i in range(len(eLes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]

    pfs=pfs.flatten()
    np.savetxt("Observations/Truth_PF_cox_gbmt100_scaled1_v2.txt",pfs,fmt="%f")


    CCPF_nldt100_scaled3_v13_2.txt  
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
    cov=I*2e-2
    Lamb_par=2/20
    fi=1/np.sqrt(sc)
    print(fi)
    collection_input=[dim, b_ou,A,Sig_nldt,fi]
    
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=200
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=5000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_nldt100_scaled3_v13_2.txt",pfs,fmt="%f") 

CCPF_nldt100_enes_scaled3_v13.txt 
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
    cov=I*2e-2
    Lamb_par=2/20
    fi=1/np.sqrt(sc)
    print(fi)
    collection_input=[dim, b_ou,A,Sig_nldt,fi]
    
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    p=8
    N0=10
    enes=np.array([2**i for i in range(p+1)])*N0
    pfs=np.zeros((2,len(enes),samples,int(T/d),dim))
    inputs=[]
    l=5
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()   
    np.savetxt("Observations&data/CCPF_nldt100_enes_scaled3_v13.txt",pfs,fmt="%f")


CCPF_nldt100_scaled3_v13.txt 
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
    cov=I*2e-2
    Lamb_par=2/20
    fi=1/np.sqrt(sc)
    print(fi)
    collection_input=[dim, b_ou,A,Sig_nldt,fi]
    
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_nldt100_scaled3_v13.txt",pfs,fmt="%f")  


CCPF_nldt100_scaled3_v12.txt 
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
    cov=I*5e-2
    Lamb_par=2/20
    fi=1/np.sqrt(sc)
    print(fi)
    collection_input=[dim, b_ou,A,Sig_nldt,fi]
    
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_nldt100_scaled3_v12.txt",pfs,fmt="%f") 

CCPF_nldt100_enes_scaled3_v12.txt
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
    cov=I*5e-2
    Lamb_par=2/20
    fi=1/np.sqrt(sc)
    print(fi)
    collection_input=[dim, b_ou,A,Sig_nldt,fi]
    
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    p=8
    N0=10
    enes=np.array([2**i for i in range(p+1)])*N0
    pfs=np.zeros((2,len(enes),samples,int(T/d),dim))
    inputs=[]
    l=5
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()   
    np.savetxt("Observations&data/CCPF_nldt100_enes_scaled3_v12.txt",pfs,fmt="%f")

CCPF_nldt100_enes_scaled3_v11.txt 
np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+4
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    A=np.array([[0]])
    sc=np.float_power(2,12)
    cov=I*5e-2
    Lamb_par=2/20
    fi=1/np.sqrt(sc)
    print(fi)
    collection_input=[dim, b_ou,A,Sig_nldt,fi]
    
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    p=8
    N0=10
    enes=np.array([2**i for i in range(p+1)])*N0
    pfs=np.zeros((2,len(enes),samples,int(T/d),dim))
    inputs=[]
    l=5
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()   
    np.savetxt("Observations&data/CCPF_nldt100_enes_scaled3_v11.txt",pfs,fmt="%f") 



    CCPF_nldt100_scaled3_v11.txt 
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+4
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    A=np.array([[0]])
    sc=np.float_power(2,12)
    cov=I*5e-2
    Lamb_par=2/20
    fi=1/np.sqrt(sc)
    print(fi)
    collection_input=[dim, b_ou,A,Sig_nldt,fi]
    
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_nldt100_scaled3_v11.txt",pfs,fmt="%f")  


CCPF_nldt100_scaled3_v10.txt 
np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+4
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    A=np.array([[0]])
    sc=np.float_power(2,12)
    cov=I*1e-1
    Lamb_par=3/20
    fi=1/np.sqrt(sc)
    print(fi)
    collection_input=[dim, b_ou,A,Sig_nldt,fi]
    
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_nldt100_scaled3_v10.txt",pfs,fmt="%f") 


CCPF_nldt100_enes_scaled3_v10.txt 
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+4
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    A=np.array([[0]])
    sc=np.float_power(2,12)
    cov=I*1e-1
    Lamb_par=3/20
    fi=1/np.sqrt(sc)
    print(fi)
    collection_input=[dim, b_ou,A,Sig_nldt,fi]
    
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    p=8
    N0=10
    enes=np.array([2**i for i in range(p+1)])*N0
    pfs=np.zeros((2,len(enes),samples,int(T/d),dim))
    inputs=[]
    l=5
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()  
        
    np.savetxt("Observations&data/CCPF_nldt100_enes_scaled3_v10.txt",pfs,fmt="%f")  

    Truth_PF_cox_gbmt100_scaled1_v1.txt 
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
    Lamb_par=.3
    sigs=np.abs(np.random.normal(np.sqrt(2),0,dim))
    mu=mu*c
    sigs=sigs*np.sqrt(c)
    print(mu,sigs)
    comp_matrix=np.array([[1]])
    inv_mat=comp_matrix.T
    S=diags(np.random.normal(1,0,dim),0).toarray()
    Sig=inv_mat@S@comp_matrix
    fi=[sigs,Sig]
    collection_input=[dim, b_gbm,mu,Sig_gbm,fi]
    g_pars=[dim,cov]
    g_par=cov
    np.random.seed(0)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=1
    l0=0
    Lmin=8
    Lmax=8
    eLes=np.array(np.arange(Lmin,Lmax+1))
    print(eLes)
    C0=0.03283620263287618
    C=0.00025205321130822886
    K=1.765425864219179e-09
    pfs=np.zeros((len(eLes),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eLes)):
        for sample in range(samples):
            l=eLes[i]

            N=int(C0*2**(2*l)*1e-4/K)
            collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,N,dim,resamp_coef,\
            g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([samples*i+sample,collection_input])
    pool = multiprocessing.Pool(processes=1)
    pool_outputs = pool.map(PCox_PF,inputs)

    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")
    for i in range(len(eLes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]

    pfs=pfs.flatten()
    np.savetxt("Observations/Truth_PF_cox_gbmt100_scaled1_v1.txt",pfs,fmt="%f")


CCPF_nldt100_enes_scaled3_v9.txt  
np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+4
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    A=np.array([[0]])
    sc=np.float_power(2,12)
    cov=I*1e-1
    Lamb_par=1.5/20
    fi=1/np.sqrt(sc)
    print(fi)
    collection_input=[dim, b_ou,A,Sig_nldt,fi]
    
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    p=8
    N0=10
    enes=np.array([2**i for i in range(p+1)])*N0
    pfs=np.zeros((2,len(enes),samples,int(T/d),dim))
    inputs=[]
    l=5
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()  
        
    np.savetxt("Observations&data/CCPF_nldt100_enes_scaled3_v9.txt",pfs,fmt="%f") 

    CCPF_nldt100_scaled3_v9.txt 
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+4
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    A=np.array([[0]])
    sc=np.float_power(2,12)
    cov=I*1e-1
    Lamb_par=1.5/20
    fi=1/np.sqrt(sc)
    print(fi)
    collection_input=[dim, b_ou,A,Sig_nldt,fi]
    
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_nldt100_scaled3_v9.txt",pfs,fmt="%f")   


CCPF_nldt100_scaled3_v8.txt
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+4
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    A=np.array([[0]])
    sc=np.float_power(2,12)
    cov=I*1e-1
    Lamb_par=1/20
    fi=1/np.sqrt(sc)
    print(fi)
    collection_input=[dim, b_ou,A,Sig_nldt,fi]
    
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_nldt100_scaled3_v8.txt",pfs,fmt="%f")    



CCPF_nldt100_enes_scaled3_v8.txt 

    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+4
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    A=np.array([[0]])
    sc=np.float_power(2,12)
    cov=I*1e-1
    Lamb_par=1/20
    fi=1/np.sqrt(sc)
    print(fi)
    collection_input=[dim, b_ou,A,Sig_nldt,fi]
    
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    p=8
    N0=10
    enes=np.array([2**i for i in range(p+1)])*N0
    pfs=np.zeros((2,len(enes),samples,int(T/d),dim))
    inputs=[]
    l=5
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()  
        
    np.savetxt("Observations&data/CCPF_nldt100_enes_scaled3_v8.txt",pfs,fmt="%f")


CCPF_nldt100_enes_scaled3_v7.txt 
np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+4
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    A=np.array([[0]])
    sc=np.float_power(2,12)
    cov=I*1e0
    Lamb_par=2
    fi=1/np.sqrt(sc)
    print(fi)
    collection_input=[dim, b_ou,A,Sig_nldt,fi]
    
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    p=8
    N0=10
    enes=np.array([2**i for i in range(p+1)])*N0
    pfs=np.zeros((2,len(enes),samples,int(T/d),dim))
    inputs=[]
    l=5
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()  
        
    np.savetxt("Observations&data/CCPF_nldt100_enes_scaled3_v7.txt",pfs,fmt="%f")    


    CCPF_nldt100_scaled3_v7.txt 
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+4
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    A=np.array([[0]])
    sc=np.float_power(2,12)
    cov=I*1e0
    Lamb_par=2
    fi=1/np.sqrt(sc)
    print(fi)
    collection_input=[dim, b_ou,A,Sig_nldt,fi]
    
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_nldt100_scaled3_v7.txt",pfs,fmt="%f")    




    PPF_cox_gbmt100_scaled1_v1.txt 
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
    Lamb_par=.3
    sigs=np.abs(np.random.normal(np.sqrt(2),0,dim))
    mu=mu*c
    sigs=sigs*np.sqrt(c)
    print(mu,sigs)
    comp_matrix=np.array([[1]])
    inv_mat=comp_matrix.T
    S=diags(np.random.normal(1,0,dim),0).toarray()
    Sig=inv_mat@S@comp_matrix
    fi=[sigs,Sig]
    collection_input=[dim, b_gbm,mu,Sig_gbm,fi]
    g_pars=[dim,cov]
    g_par=cov
    np.random.seed(0)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=0
    Lmin=0
    Lmax=5
    eLes=np.array(np.arange(Lmin,Lmax+1))
    print(eLes)
    C0=0.03283620263287618
    C=0.00025205321130822886
    K=1.765425864219179e-09
    pfs=np.zeros((len(eLes),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eLes)):
        for sample in range(samples):
            l=eLes[i]

            N=int(C0*2**(2*l)*1e-4/K)
            collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,N,dim,resamp_coef,\
            g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([samples*i+sample,collection_input])
    pool = multiprocessing.Pool(processes=80)
    pool_outputs = pool.map(PCox_PF,inputs)

    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")
    for i in range(len(eLes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]

    pfs=pfs.flatten()
    np.savetxt("Observations/PPF_cox_gbmt100_scaled1_v1.txt",pfs,fmt="%f")



    CCPF_nldt100_scaled3_v6.txt 
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+4
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    A=np.array([[0]])
    sc=np.float_power(2,12)
    cov=I*1e0
    Lamb_par=9/9
    fi=1/np.sqrt(sc)
    print(fi)
    collection_input=[dim, b_ou,A,Sig_nldt,fi]
    
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_nldt100_scaled3_v6.txt",pfs,fmt="%f")   


    CCPF_nldt100_enes_scaled3_v6.txt
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+4
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    A=np.array([[0]])
    sc=np.float_power(2,12)
    cov=I*1e0
    Lamb_par=9/9
    fi=1/np.sqrt(sc)
    print(fi)
    collection_input=[dim, b_ou,A,Sig_nldt,fi]
    
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    p=8
    N0=10
    enes=np.array([2**i for i in range(p+1)])*N0
    pfs=np.zeros((2,len(enes),samples,int(T/d),dim))
    inputs=[]
    l=5
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()  
        
    np.savetxt("Observations&data/CCPF_nldt100_enes_scaled3_v6.txt",pfs,fmt="%f")
    


    CCPF_nldt100_enes_scaled3_v5.txt 
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+4
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    A=np.array([[0]])
    sc=np.float_power(2,12)
    cov=I*1e0
    Lamb_par=5/9
    fi=1/np.sqrt(sc)
    print(fi)
    collection_input=[dim, b_ou,A,Sig_nldt,fi]
    
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    p=8
     N0=10
    enes=np.array([2**i for i in range(p+1)])*N0
    pfs=np.zeros((2,len(enes),samples,int(T/d),dim))
    inputs=[]
    l=5
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_nldt100_enes_scaled3_v5.txt",pfs,fmt="%f")


    CCPF_nldt100_scaled3_v5.txt 
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+4
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    A=np.array([[0]])
    sc=np.float_power(2,12)
    cov=I*1e0
    Lamb_par=5/9
    fi=1/np.sqrt(sc)
    print(fi)
    collection_input=[dim, b_ou,A,Sig_nldt,fi]
    
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_nldt100_scaled3_v5.txt",pfs,fmt="%f")    




    PMLPF_cox_gbm100_scaled1_v3.txt 
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
    Lamb_par=.3
    sigs=np.abs(np.random.normal(np.sqrt(2),0,dim))
    mu=mu*c
    sigs=sigs*np.sqrt(c)
    print(mu,sigs)
    comp_matrix=np.array([[1]])
    inv_mat=comp_matrix.T
    S=diags(np.random.normal(1,0,dim),0).toarray()
    Sig=inv_mat@S@comp_matrix
    fi=[sigs,Sig]
    collection_input=[dim, b_gbm,mu,Sig_gbm,fi]
    g_pars=[dim,cov]
    g_par=cov
    np.random.seed(0)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=0
    Lmin=0
    Lmax=5
    eLes=np.array(np.arange(Lmin,Lmax+1))
    print(eLes)
    C0=0.0328362
    C=0.00025205321130822886
    K=1.765425864219179e-09
    pfs=np.zeros((len(eLes),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eLes)):
        for sample in range(samples):
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
            collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,eles,d,eNes,dim,resamp_coef,phi,\
            dim_out,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([samples*i+sample,collection_input])

    pool = multiprocessing.Pool(processes=50)
    pool_outputs = pool.map(PMLPF_cox,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]

    pfs=pfs.flatten()

    np.savetxt("Observations/PMLPF_cox_gbm100_scaled1_v3.txt",pfs,fmt="%f")


    CCPF_nldt100_scaled3_v4.txt 
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+4
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    A=np.array([[0]])
    sc=np.float_power(2,12)
    cov=I*1e-1
    Lamb_par=3/9
    fi=1/np.sqrt(sc)
    print(fi)
    collection_input=[dim, b_ou,A,Sig_nldt,fi]
    
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=200
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=5000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_nldt100_scaled3_v4.txt",pfs,fmt="%f")




CCPF_nldt100_enes_scaled3_v3.txt 

    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+4
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    A=np.array([[0]])
    sc=np.float_power(2,12)
    cov=I*1e-1
    Lamb_par=3/9
    fi=1/np.sqrt(sc)
    print(fi)
    collection_input=[dim, b_ou,A,Sig_nldt,fi]
    
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    p=8
    N0=10
    enes=np.array([2**i for i in range(p+1)])*N0
    pfs=np.zeros((2,len(enes),samples,int(T/d),dim))
    inputs=[]
    l=5
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_nldt100_enes_scaled3_v3.txt",pfs,fmt="%f")



CCPF_nldt100_scaled3_v3.txt 
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+4
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    A=np.array([[0]])
    sc=np.float_power(2,12)
    fi=1/np.sqrt(sc)
    print(fi)
    collection_input=[dim, b_ou,A,Sig_nldt,fi]
    cov=I*1e-1
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=3/9
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_nldt100_scaled3_v3.txt",pfs,fmt="%f")


    PMLPF_cox_gbm100_scaled1_v2.txt 
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
    Lamb_par=.3
    sigs=np.abs(np.random.normal(np.sqrt(2),0,dim))
    mu=mu*c
    sigs=sigs*np.sqrt(c)
    print(mu,sigs)
    comp_matrix=np.array([[1]])
    inv_mat=comp_matrix.T
    S=diags(np.random.normal(1,0,dim),0).toarray()
    Sig=inv_mat@S@comp_matrix
    fi=[sigs,Sig]
    collection_input=[dim, b_gbm,mu,Sig_gbm,fi]
    g_pars=[dim,cov]
    g_par=cov
    np.random.seed(0)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=0
    Lmin=6
    Lmax=6
    eLes=np.array(np.arange(Lmin,Lmax+1))
    print(eLes)
    C0=0.01302386758
    C=0.00025205321130822886
    K=1.765425864219179e-09
    pfs=np.zeros((len(eLes),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eLes)):
        for sample in range(samples):
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
            collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,eles,d,eNes,dim,resamp_coef,phi,\
            dim_out,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([samples*i+sample,collection_input])

    pool = multiprocessing.Pool(processes=50)
    pool_outputs = pool.map(PMLPF_cox,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]

    pfs=pfs.flatten()

    np.savetxt("Observations/PMLPF_cox_gbm100_scaled1_v2.txt",pfs,fmt="%f")




CCPF_nldt100_scaled3_v2.txt 

np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+4
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    A=np.array([[0]])
    sc=np.float_power(2,12)
    fi=1/np.sqrt(sc)
    print(fi)
    collection_input=[dim, b_ou,A,Sig_nldt,fi]
    cov=I*1e-1
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=2/9
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_nldt100_scaled3_v2.txt",pfs,fmt="%f")


    CCPF_nldt100_enes_scaled3_v2.txt
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+4
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    A=np.array([[0]])
    sc=np.float_power(2,12)
    cov=I*1e-1
    fi=1/np.sqrt(sc)
    print(fi)
    collection_input=[dim, b_ou,A,Sig_nldt,fi]
    
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=2/9
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    p=8
    N0=10
    enes=np.array([2**i for i in range(p+1)])*N0
    pfs=np.zeros((2,len(enes),samples,int(T/d),dim))
    inputs=[]
    l=5
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_nldt100_enes_scaled3_v2.txt",pfs,fmt="%f")


    PMLPF_cox_gbm100_scaled1_v1.txt  

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
    Lamb_par=.3
    sigs=np.abs(np.random.normal(np.sqrt(2),0,dim))
    mu=mu*c
    sigs=sigs*np.sqrt(c)
    print(mu,sigs)
    comp_matrix=np.array([[1]])
    inv_mat=comp_matrix.T
    S=diags(np.random.normal(1,0,dim),0).toarray()
    Sig=inv_mat@S@comp_matrix
    fi=[sigs,Sig]
    collection_input=[dim, b_gbm,mu,Sig_gbm,fi]
    g_pars=[dim,cov]
    g_par=cov
    np.random.seed(0)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=0
    Lmin=0
    Lmax=5
    eLes=np.array(np.arange(Lmin,Lmax+1))
    print(eLes)
    C0=0.01302386758
    C=0.00025205321130822886
    K=1.765425864219179e-09
    pfs=np.zeros((len(eLes),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eLes)):
        for sample in range(samples):
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
            collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,eles,d,eNes,dim,resamp_coef,phi,\
            dim_out,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([samples*i+sample,collection_input])

    pool = multiprocessing.Pool(processes=50)
    pool_outputs = pool.map(PMLPF_cox,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]

    pfs=pfs.flatten()

    np.savetxt("Observations/PMLPF_cox_gbm100_scaled1_v1.txt",pfs,fmt="%f")


    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+4
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    A=np.array([[0]])
    sc=np.float_power(2,12)
    cov=I*1e0
    fi=1/np.sqrt(sc)
    print(fi)
    collection_input=[dim, b_ou,A,Sig_nldt,fi]
    
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=2/9
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    p=8
    N0=10
    enes=np.array([2**i for i in range(p+1)])*N0
    pfs=np.zeros((2,len(enes),samples,int(T/d),dim))
    inputs=[]
    l=5
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_nldt100_enes_scaled3_v1.txt",pfs,fmt="%f")



CCPF_nldt100_scaled2_v9.txt
np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+4
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    A=np.array([[0]])
    sc=np.float_power(2,12)
    fi=1/np.sqrt(sc)
    print(fi)
    collection_input=[dim, b_ou,A,Sig_nldt,fi]
    cov=I*1e0
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=2/9
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_nldt100_scaled2_v9.txt",pfs,fmt="%f")



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
    Lamb_par=.3
    sigs=np.abs(np.random.normal(np.sqrt(2),0,dim))
    mu=mu*c
    sigs=sigs*np.sqrt(c)
    print(mu,sigs)
    comp_matrix=np.array([[1]])
    inv_mat=comp_matrix.T
    S=diags(np.random.normal(1,0,dim),0).toarray()
    Sig=inv_mat@S@comp_matrix
    fi=[sigs,Sig]
    collection_input=[dim, b_gbm,mu,Sig_gbm,fi]
    g_pars=[dim,cov]
    g_par=cov
    np.random.seed(0)
    x_true=gen_gen_data(T,xin,l,collection_input)
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=10
    l0=0
    Lmin=0
    Lmax=2
    eLes=np.array(np.arange(Lmin,Lmax+1))
    print(eLes)
    C0=0.01302386758
    C=0.00025205321130822886
    K=1.765425864219179e-09
    pfs=np.zeros((len(eLes),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eLes)):
        for sample in range(samples):
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
            collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,eles,d,eNes,dim,resamp_coef,phi,\
            dim_out,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([samples*i+sample,collection_input])        
     
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PMLPF_cox,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations/PMLPF_cox_gbm100_scaled1_vtest.txt",pfs,fmt="%f")



CCPF_gbm100_p_scaled_v16_2.txt    

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
    Lamb_par=.3
    sigs=np.abs(np.random.normal(np.sqrt(2),0,dim))
    mu=mu*c
    sigs=sigs*np.sqrt(c)
    print(mu,sigs)
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
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=500
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=2000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_gbm100_p_scaled_v16_2.txt",pfs,fmt="%f")



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
    Lamb_par=.2
    sigs=np.abs(np.random.normal(np.sqrt(2),0,dim))
    mu=mu*c
    sigs=sigs*np.sqrt(c)
    print(mu,sigs)
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
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    #l0=1
    #L=10
    l=5
    N0=10
    p=8
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    enes=N0*np.array([2**i for i in range(p)])

    pfs=np.zeros((len(enes),samples,int(T/d),dim))
    inputs=[]
    
    
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/Cox_PF_gbm100_p_scaled_v16.txt",pfs,fmt="%f")

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
    Lamb_par=.3
    sigs=np.abs(np.random.normal(np.sqrt(2),0,dim))
    mu=mu*c
    sigs=sigs*np.sqrt(c)
    print(mu,sigs)
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
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_gbm100_p_scaled_v16.txt",pfs,fmt="%f")


    CCPF_gbm100_p_scaled_v15.txt 
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+4
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    c=2**(-18)
    cov=I*1e-1
    mu=np.abs(np.random.normal(1.01,0,dim))*1e3
    Lamb_par=.3
    sigs=np.abs(np.random.normal(np.sqrt(2),0,dim))
    mu=mu*c
    sigs=sigs*np.sqrt(c)
    print(mu,sigs)
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
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_gbm100_p_scaled_v15.txt",pfs,fmt="%f")

Cox_PF_gbm100_p_scaled_v15.txt 
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+4
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    c=2**(-18)
    cov=I*1e-1
    mu=np.abs(np.random.normal(1.01,0,dim))*1e3
    Lamb_par=.2
    sigs=np.abs(np.random.normal(np.sqrt(2),0,dim))
    mu=mu*c
    sigs=sigs*np.sqrt(c)
    print(mu,sigs)
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
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    #l0=1
    #L=10
    l=5
    N0=10
    p=8
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    enes=N0*np.array([2**i for i in range(p)])

    pfs=np.zeros((len(enes),samples,int(T/d),dim))
    inputs=[]
    
    
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/Cox_PF_gbm100_p_scaled_v15.txt",pfs,fmt="%f")

PPF_cox_nldt100_scaled2_v3.txt
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
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=0
    Lmin=5
    Lmax=5
    eLes=np.array(np.arange(Lmin,Lmax+1))
    print(eLes)
    C0=0.00997
    C=0.0004998
    K= 1.9131764860557113e-06
    pfs=np.zeros((len(eLes),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eLes)):
        for sample in range(samples):
            l=eLes[i]
            N=int(C0*2**(2*l)/K)
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,N,dim,resamp_coef,\
            g_den,g_par,Norm_Lambda,Lamb_par]   
            inputs.append([samples*i+sample,collection_input])
    pool = multiprocessing.Pool(processes=50)
    pool_outputs = pool.map(PCox_PF,inputs)

    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eLes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
    np.savetxt("Observations/PPF_cox_nldt100_scaled2_v3.txt",pfs,fmt="%f")


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
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=10
    l0=0
    Lmin=2
    Lmax=2
    eLes=np.array(np.arange(Lmin,Lmax+1))
    print(eLes)
    C0=0.00997
    C=0.0004998
    K= 1.9131764860557113e-06
    pfs=np.zeros((len(eLes),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eLes)):
        for sample in range(samples):
            l=eLes[i]
            N=int(C0*2**(2*l)/K)
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,N,dim,resamp_coef,\
            g_den,g_par,Norm_Lambda,Lamb_par]   
            inputs.append([samples*i+sample,collection_input])
    pool = multiprocessing.Pool(processes=6)
    pool_outputs = pool.map(PCox_PF,inputs)

    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eLes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
    np.savetxt("Observations/PPF_cox_nldt100_scaled2_vtest.txt",pfs,fmt="%f")


PMLPF_cox_nldt_T100_scaled2_v3.txt 

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
    start=time.time()

    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=0
    Lmin=5
    Lmax=5
    d=2**0
    eLes=np.array(np.arange(Lmin,Lmax+1))
    print(eLes)
    C0=0.0084
    C=0.00049
    K= 1.9131764860557113e-06

    pfs=np.zeros((len(eLes),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eLes)):
        for sample in range(samples):
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
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,eles,d,eNes,dim,resamp_coef,phi,\
            dim_out,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([samples*i+sample,collection_input])

    pool = multiprocessing.Pool(processes=50)
    pool_outputs = pool.map(PMLPF_cox,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")
    for i in range(len(eLes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations/PMLPF_cox_nldt_T100_scaled2_v3.txt",pfs,fmt="%f")




PPF_cox_nldt_T100_scaled2_v2.txt 

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
    start=time.time()

    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=1
    es=1e-9
    C0=0.0084
    C=0.00049
    K= 1.9131764860557113e-06

    N=int(2*C0/es)
    L=int(np.ceil(-np.log2(es/(2*np.sqrt(K)))/2,dtype=float))
    l0=L
    Lmin=L
    Lmax=L
    d=2**0
    eLes=np.array(np.arange(Lmin,Lmax+1))

    pfs=np.zeros((len(eLes),samples,int(T/d),dim))
    print(pfs)
    inputs=[]


    collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,L,d,N,dim,resamp_coef,\
    g_den,g_par,Norm_Lambda,Lamb_par]
    inputs.append([0,collection_input])
    pool = multiprocessing.Pool(processes=6)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")
    pfs[0,0]=pool_outputs[0]
    pfs=pfs.flatten()
    np.savetxt("Observations/PPF_cox_nldt_T100_scaled2_v2.txt",pfs,fmt="%f")


Cox_PF_gbm100_p_scaled_v14.txt 

    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+4
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    c=2**(-18)
    cov=I*1e1
    mu=np.abs(np.random.normal(1.01,0,dim))*1e3
    Lamb_par=.2
    sigs=np.abs(np.random.normal(np.sqrt(2),0,dim))
    mu=mu*c
    sigs=sigs*np.sqrt(c)
    print(mu,sigs)
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
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    #l0=1
    #L=10
    l=5
    N0=10
    p=8
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    enes=N0*np.array([2**i for i in range(p)])

    pfs=np.zeros((len(enes),samples,int(T/d),dim))
    inputs=[]
    
    
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/Cox_PF_gbm100_p_scaled_v14.txt",pfs,fmt="%f")



    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+4
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    c=2**(-18)
    cov=I*1e1
    mu=np.abs(np.random.normal(1.01,0,dim))*1e3
    Lamb_par=.3
    sigs=np.abs(np.random.normal(np.sqrt(2),0,dim))
    mu=mu*c
    sigs=sigs*np.sqrt(c)
    print(mu,sigs)
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
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_gbm100_p_scaled_v14.txt",pfs,fmt="%f")


    CCPF_gbm100_p_scaled_v13.txt 
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+4
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    c=2**(-14)
    cov=I*1e1
    mu=np.abs(np.random.normal(1.01,0,dim))*1e2
    Lamb_par=0.2    
    sigs=np.abs(np.random.normal(np.sqrt(2),0,dim))
    mu=mu*c
    sigs=sigs*np.sqrt(c)
    print(mu,sigs)
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
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_gbm100_p_scaled_v13.txt",pfs,fmt="%f")





    Cox_PF_gbm100_p_scaled_v13.txt 

    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+4
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    c=2**(-14)
    cov=I*1e1
    mu=np.abs(np.random.normal(1.01,0,dim))*1e2
    Lamb_par=0.2    
    sigs=np.abs(np.random.normal(np.sqrt(2),0,dim))
    mu=mu*c
    sigs=sigs*np.sqrt(c)
    print(mu,sigs)
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
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    #l0=1
    #L=10
    l=5
    N0=10
    p=8
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    enes=N0*np.array([2**i for i in range(p)])

    pfs=np.zeros((len(enes),samples,int(T/d),dim))
    inputs=[]
    
    
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/Cox_PF_gbm100_p_scaled_v13.txt",pfs,fmt="%f")


    Cox_PF_gbm100_p_scaled_v12.txt 
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+4
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    c=2**(-14)
    cov=I*1e0
    mu=np.abs(np.random.normal(1.01,0,dim))*1e2
    sigs=np.abs(np.random.normal(np.sqrt(2),0,dim))
    mu=mu*c
    sigs=sigs*np.sqrt(c)
    print(mu,sigs)
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
    Lamb_par=.3
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    #l0=1
    #L=10
    l=5
    N0=10
    p=8
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    enes=N0*np.array([2**i for i in range(p)])

    pfs=np.zeros((len(enes),samples,int(T/d),dim))
    inputs=[]
    
    
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/Cox_PF_gbm100_p_scaled_v12.txt",pfs,fmt="%f")


CCPF_gbm100_p_scaled_v12.txt  

    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+4
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    c=2**(-14)
    cov=I*1e0
    mu=np.abs(np.random.normal(1.01,0,dim))*1e2
    sigs=np.abs(np.random.normal(np.sqrt(2),0,dim))
    mu=mu*c
    sigs=sigs*np.sqrt(c)
    print(mu,sigs)
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
    Lamb_par=.3
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_gbm100_p_scaled_v12.txt",pfs,fmt="%f")

    

    CCPF_gbm100_p_scaled_v11.txt
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+4
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    c=2**(-18)
    cov=I*1e1

    mu=np.abs(np.random.normal(1.01,0,dim))*1e3
    sigs=np.abs(np.random.normal(np.sqrt(2),0,dim))
    mu=mu*c
    sigs=sigs*np.sqrt(c)
    print(mu,sigs)
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
    Lamb_par=.3
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_gbm100_p_scaled_v11.txt",pfs,fmt="%f")

Cox_PF_gbm100_p_scaled_v11.TXT 

    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+4
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    c=2**(-18)
    cov=I*1e1
    mu=np.abs(np.random.normal(1.01,0,dim))*1e3
    sigs=np.abs(np.random.normal(np.sqrt(2),0,dim))
    mu=mu*c
    sigs=sigs*np.sqrt(c)
    print(mu,sigs)
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
    Lamb_par=.3
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    #l0=1
    #L=10
    l=5
    N0=10
    p=8
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    enes=N0*np.array([2**i for i in range(p)])

    pfs=np.zeros((len(enes),samples,int(T/d),dim))
    inputs=[]
    
    
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/Cox_PF_gbm100_p_scaled_v11.txt",pfs,fmt="%f")


Cox_PF_gbm100_p_scaled_v10.txt 
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+4
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    c=2**(-14)
    cov=I*1e1
    mu=np.abs(np.random.normal(1.01,0,dim))*1e2
    sigs=np.abs(np.random.normal(np.sqrt(2),0,dim))
    mu=mu*c
    sigs=sigs*np.sqrt(c)
    print(mu,sigs)
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
    Lamb_par=.3
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    #l0=1
    #L=10
    l=5
    N0=10
    p=8
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    enes=N0*np.array([2**i for i in range(p)])

    pfs=np.zeros((len(enes),samples,int(T/d),dim))
    inputs=[]
    
    
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/Cox_PF_gbm100_p_scaled_v10.txt",pfs,fmt="%f")





CCPF_gbm100_p_scaled_v10.txt 
np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+4
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    c=2**(-14)
    cov=I*1e1

    mu=np.abs(np.random.normal(1.01,0,dim))*1e2
    sigs=np.abs(np.random.normal(np.sqrt(2),0,dim))
    mu=mu*c
    sigs=sigs*np.sqrt(c)
    print(mu,sigs)
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
    Lamb_par=.3
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_gbm100_p_scaled_v10.txt",pfs,fmt="%f")


CCPF_gbm100_p_scaled_v9.txt 
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+4
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    c=2**(-14)
    cov=I*1e0

    mu=np.abs(np.random.normal(1.01,0,dim))*1e2
    sigs=np.abs(np.random.normal(np.sqrt(2),0,dim))
    mu=mu*c
    sigs=sigs*np.sqrt(c)
    print(mu,sigs)
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
    Lamb_par=.3
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_gbm100_p_scaled_v9.txt",pfs,fmt="%f")




    Cox_PF_gbm100_p_scaled_v9.txt 
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+4
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    c=2**(-14)
    cov=I*1e0
    mu=np.abs(np.random.normal(1.01,0,dim))*1e2
    sigs=np.abs(np.random.normal(np.sqrt(2),0,dim))
    mu=mu*c
    sigs=sigs*np.sqrt(c)
    print(mu,sigs)
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
    Lamb_par=.3
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    #l0=1
    #L=10
    l=5
    N0=10
    p=8
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    enes=N0*np.array([2**i for i in range(p)])

    pfs=np.zeros((len(enes),samples,int(T/d),dim))
    inputs=[]
    
    
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/Cox_PF_gbm100_p_scaled_v9.txt",pfs,fmt="%f")


    Cox_PF_gbm100_p_scaled_v8.txt 

    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+4
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    c=2**(-14)
    cov=I*1e0
    mu=np.abs(np.random.normal(1.01,0,dim))*1e1
    sigs=np.abs(np.random.normal(np.sqrt(2),0,dim))
    mu=mu*c
    sigs=sigs*np.sqrt(c)
    print(mu,sigs)
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
    Lamb_par=.3
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    #l0=1
    #L=10
    l=5
    N0=10
    p=8
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    enes=N0*np.array([2**i for i in range(p)])

    pfs=np.zeros((len(enes),samples,int(T/d),dim))
    inputs=[]
    
    
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/Cox_PF_gbm100_p_scaled_v8.txt",pfs,fmt="%f")





CCPF_gbm100_p_scaled_v8.txt 

np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+4
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    c=2**(-14)
    cov=I*1e0

    mu=np.abs(np.random.normal(1.01,0,dim))*1e1
    sigs=np.abs(np.random.normal(np.sqrt(2),0,dim))
    mu=mu*c
    sigs=sigs*np.sqrt(c)
    print(mu,sigs)
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
    Lamb_par=.3
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_gbm100_p_scaled_v8.txt",pfs,fmt="%f")


    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+4
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    c=2**(-18)
    mu=np.abs(np.random.normal(1.01,0,dim))
    sigs=np.abs(np.random.normal(np.sqrt(2),0,dim))
    mu=mu*c
    sigs=sigs*np.sqrt(c)
    print(mu,sigs)
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
    Lamb_par=.3
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_gbm100_p_scaled_v7.txt",pfs,fmt="%f") 


Cox_PF_gbm100_p_scaled_v7.txt 
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+4
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    c=2**(-18)
    mu=np.abs(np.random.normal(1.01,0,dim))
    sigs=np.abs(np.random.normal(np.sqrt(2),0,dim))
    mu=mu*c
    sigs=sigs*np.sqrt(c)
    print(mu,sigs)
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
    Lamb_par=.3
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    #l0=1
    #L=10
    l=5
    N0=10
    p=8
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    enes=N0*np.array([2**i for i in range(p)])

    pfs=np.zeros((len(enes),samples,int(T/d),dim))
    inputs=[]
    
    
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/Cox_PF_gbm100_p_scaled_v7.txt",pfs,fmt="%f")

    Cox_PF_gbm100_p_scaled_v6.txt
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+4
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    c=2**(-14)
    mu=np.abs(np.random.normal(1.01,0,dim))
    sigs=np.abs(np.random.normal(np.sqrt(2),0,dim))
    mu=mu*c
    sigs=sigs*np.sqrt(c)
    print(mu,sigs)
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
    Lamb_par=.3
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    #l0=1
    #L=10
    l=5
    N0=10
    p=8
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    enes=N0*np.array([2**i for i in range(p)])

    pfs=np.zeros((len(enes),samples,int(T/d),dim))
    inputs=[]
    
    
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/Cox_PF_gbm100_p_scaled_v6.txt",pfs,fmt="%f")

CCPF_gbm100_p_scaled_v6.txt 

    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+4
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    c=2**(-14)
    mu=np.abs(np.random.normal(1.01,0,dim))
    sigs=np.abs(np.random.normal(np.sqrt(2),0,dim))
    mu=mu*c
    sigs=sigs*np.sqrt(c)
    print(mu,sigs)
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
    Lamb_par=.3
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_gbm100_p_scaled_v6.txt",pfs,fmt="%f")


    Cox_PF_gbm100_p_scaled_v5.txt 
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+4
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    c=2**(-10)
    mu=np.abs(np.random.normal(1.01,0,dim))
    sigs=np.abs(np.random.normal(np.sqrt(2),0,dim))
    mu=mu*c
    sigs=sigs*np.sqrt(c)
    print(mu,sigs)
    comp_matrix=np.array([[1]])
    inv_mat=comp_matrix.T
    S=diags(np.random.normal(1,0,dim),0).toarray()
    Sig=inv_mat@S@comp_matrix
    fi=[sigs,Sig]
    collection_input=[dim, b_gbm,mu,Sig_gbm,fi]
    cov=I*1e2
    g_pars=[dim,cov]
    g_par=cov
    samples=5
    np.random.seed(0)
    x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=.3
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    #l0=1
    #L=10
    l=5
    N0=10
    p=8
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    enes=N0*np.array([2**i for i in range(p)])

    pfs=np.zeros((len(enes),samples,int(T/d),dim))
    inputs=[]
    
    
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/Cox_PF_gbm100_p_scaled_v5.txt",pfs,fmt="%f")


Cox_PF_gbm100_p_scaled_v4.txt
np.random.seed(0) 

    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+4
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    c=2**(-10)
    mu=np.abs(np.random.normal(1.01,0,dim))
    sigs=np.abs(np.random.normal(np.sqrt(2),0,dim))
    mu=mu*c
    sigs=sigs*np.sqrt(c)
    print(mu,sigs)
    comp_matrix=np.array([[1]])
    inv_mat=comp_matrix.T
    S=diags(np.random.normal(1,0,dim),0).toarray()
    Sig=inv_mat@S@comp_matrix
    fi=[sigs,Sig]
    collection_input=[dim, b_gbm,mu,Sig_gbm,fi]
    cov=I*1e1
    g_pars=[dim,cov]
    g_par=cov
    samples=5
    np.random.seed(0)
    x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=.3
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    #l0=1
    #L=10
    l=5
    N0=10
    p=8
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    enes=N0*np.array([2**i for i in range(p)])

    pfs=np.zeros((len(enes),samples,int(T/d),dim))
    inputs=[]
    
    
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/Cox_PF_gbm100_p_scaled_v4.txt",pfs,fmt="%f")



CCPF_gbm100_p_scaled_v4.txt 

    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+4
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    c=2**(-10)
    mu=np.abs(np.random.normal(1.01,0,dim))
    sigs=np.abs(np.random.normal(np.sqrt(2),0,dim))
    mu=mu*c
    sigs=sigs*np.sqrt(c)
    print(mu,sigs)
    comp_matrix=np.array([[1]])
    inv_mat=comp_matrix.T
    S=diags(np.random.normal(1,0,dim),0).toarray()
    Sig=inv_mat@S@comp_matrix
    fi=[sigs,Sig]
    collection_input=[dim, b_gbm,mu,Sig_gbm,fi]
    cov=I*1e1   
    g_pars=[dim,cov]
    g_par=cov
    samples=5
    np.random.seed(0)
    x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=.3
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_gbm100_p_scaled_v4.txt",pfs,fmt="%f")



Cox_PF_gbm100_p_scaled_v3.txt 

    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+4
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    c=2**(-10)
    mu=np.abs(np.random.normal(1.01,0,dim))
    sigs=np.abs(np.random.normal(np.sqrt(2),0,dim))
    mu=mu*c
    sigs=sigs*np.sqrt(c)
    print(mu,sigs)
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
    Lamb_par=.3
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    #l0=1
    #L=10
    l=5
    N0=10
    p=8
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    enes=N0*np.array([2**i for i in range(p)])

    pfs=np.zeros((len(enes),samples,int(T/d),dim))
    inputs=[]
    
    
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/Cox_PF_gbm100_p_scaled_v3.txt",pfs,fmt="%f")



    CCPF_gbm100_p_scaled_v3.txt

    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+4
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    c=2**(-10)
    mu=np.abs(np.random.normal(1.01,0,dim))
    sigs=np.abs(np.random.normal(np.sqrt(2),0,dim))
    mu=mu*c
    sigs=sigs*np.sqrt(c)
    print(mu,sigs)
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
    Lamb_par=.3
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_gbm100_p_scaled_v3.txt",pfs,fmt="%f")


CCPF_gbm100_p_scaled_v2.txt
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+4
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    c=2**(-10)
    mu=np.abs(np.random.normal(1.01,0,dim))
    sigs=np.abs(np.random.normal(np.sqrt(2),0,dim))
    mu=mu*c
    sigs=sigs*np.sqrt(c)
    print(mu,sigs)
    comp_matrix=np.array([[1]])
    inv_mat=comp_matrix.T
    S=diags(np.random.normal(1,0,dim),0).toarray()
    Sig=inv_mat@S@comp_matrix
    fi=[sigs,Sig]
    collection_input=[dim, b_gbm,mu,Sig_gbm,fi]
    cov=I*1e-1
    g_pars=[dim,cov]
    g_par=cov
    samples=5
    np.random.seed(0)
    x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=.3
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=200
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=5000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_gbm100_p_scaled_v2.txt",pfs,fmt="%f")


CCPF_gbm100_p_scaled_v1.txt
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+4
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    c=2**(-10)
    mu=np.abs(np.random.normal(1.01,0,dim))
    sigs=np.abs(np.random.normal(np.sqrt(2),0,dim))
    mu=mu*c
    sigs=sigs*np.sqrt(c)
    print(mu,sigs)
    comp_matrix=np.array([[1]])
    inv_mat=comp_matrix.T
    S=diags(np.random.normal(1,0,dim),0).toarray()
    Sig=inv_mat@S@comp_matrix
    fi=[sigs,Sig]
    collection_input=[dim, b_gbm,mu,Sig_gbm,fi]
    cov=I*1e-1
    g_pars=[dim,cov]
    g_par=cov
    samples=5
    np.random.seed(0)
    x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=.3
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_gbm100_p_scaled_v1.txt",pfs,fmt="%f")


Cox_PF_gbm100_p_scaled_v1.txt

    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+4
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    c=2**(-10)
    mu=np.abs(np.random.normal(1.01,0,dim))
    sigs=np.abs(np.random.normal(np.sqrt(2),0,dim))
    mu=mu*c
    sigs=sigs*np.sqrt(c)
    print(mu,sigs)
    comp_matrix=np.array([[1]])
    inv_mat=comp_matrix.T
    S=diags(np.random.normal(1,0,dim),0).toarray()
    Sig=inv_mat@S@comp_matrix
    fi=[sigs,Sig]
    collection_input=[dim, b_gbm,mu,Sig_gbm,fi]
    cov=I*1e-1
    g_pars=[dim,cov]
    g_par=cov
    samples=5
    np.random.seed(0)
    x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=.3
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    #l0=1
    #L=10
    l=5
    N0=10
    p=8
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    enes=N0*np.array([2**i for i in range(p)])

    pfs=np.zeros((len(enes),samples,int(T/d),dim))
    inputs=[]
    
    
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/Cox_PF_gbm100_p_scaled_v1.txt",pfs,fmt="%f")



PPF_cox_nldt100_scaled2_v2.txt
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
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=0
    Lmin=4
    Lmax=4
    eLes=np.array(np.arange(Lmin,Lmax+1))
    print(eLes)
    C0=0.00997
    C=0.0004998
    K= 1.9131764860557113e-06
    pfs=np.zeros((len(eLes),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eLes)):
        for sample in range(samples):
            l=eLes[i]
            N=int(C0*2**(2*l)/K)
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,N,dim,resamp_coef,\
            g_den,g_par,Norm_Lambda,Lamb_par]   
            inputs.append([samples*i+sample,collection_input])
    pool = multiprocessing.Pool(processes=6)
    pool_outputs = pool.map(PCox_PF,inputs)

    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eLes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
    np.savetxt("Observations&data/PPF_cox_nldt100_scaled2_v2.txt",pfs,fmt="%f")




PMLPF_cox_nldt_T100_scaled2_v2

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
    start=time.time()
    
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=0
    Lmin=4
    Lmax=4
    d=2**0
    eLes=np.array(np.arange(Lmin,Lmax+1))
    print(eLes)
    C0=0.0084   
    C=0.00049
    K= 1.9131764860557113e-06

    pfs=np.zeros((len(eLes),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eLes)):
        for sample in range(samples):
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
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,eles,d,eNes,dim,resamp_coef,phi,\
            dim_out,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([samples*i+sample,collection_input])        
           
    pool = multiprocessing.Pool(processes=)
    pool_outputs = pool.map(PMLPF_cox,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eLes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/PMLPF_cox_nldt_T100_scaled2_v2.txt",pfs,fmt="%f")


    PPF_cox_nldt100_scaled2_v1.txt 

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
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=0
    Lmin=0
    Lmax=3
    eLes=np.array(np.arange(Lmin,Lmax+1))
    print(eLes)
    C0=0.00997
    C=0.0004998
    K= 1.9131764860557113e-06
    pfs=np.zeros((len(eLes),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eLes)):
        for sample in range(samples):
            l=eLes[i]
            N=int(C0*2**(2*l)/K)
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,N,dim,resamp_coef,\
            g_den,g_par,Norm_Lambda,Lamb_par]   
            inputs.append([samples*i+sample,collection_input])
    pool = multiprocessing.Pool(processes=6)
    pool_outputs = pool.map(PCox_PF,inputs)

    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eLes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
    np.savetxt("Observations&data/PPF_cox_nldt100_scaled2_v1.txt",pfs,fmt="%f")

    


    PMLPF_cox_nldt_T100_scaled2_v1.txt 
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
    start=time.time()
    
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=0
    Lmin=0
    Lmax=3
    d=2**0
    eLes=np.array(np.arange(Lmin,Lmax+1))
    print(eLes)
    C0=0.0084   
    C=0.00049
    K= 1.9131764860557113e-06

    pfs=np.zeros((len(eLes),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eLes)):
        for sample in range(samples):
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
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,eles,d,eNes,dim,resamp_coef,phi,\
            dim_out,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([samples*i+sample,collection_input])        
           
    pool = multiprocessing.Pool(processes=6)
    pool_outputs = pool.map(PMLPF_cox,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eLes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/PMLPF_cox_nldt_T100_scaled2_v1.txt",pfs,fmt="%f")





    CCPF_nldt100_scaled2_v8.txt 
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
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_nldt100_scaled2_v8.txt",pfs,fmt="%f")




PMLPF_cox_nldt_T100_scaled2_v1.txt


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
    start=time.time()
    
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=0
    Lmin=0
    Lmax=3
    d=2**0
    eLes=np.array(np.arange(Lmin,Lmax+1))
    print(eLes)
    C0=0.0084
    C=0.00049
    K= 1.9131764860557113e-06

    pfs=np.zeros((len(eLes),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eLes)):
        for sample in range(samples):
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
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,eles,d,eNes,dim,resamp_coef,phi,\
            dim_out,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([samples*i+sample,collection_input])        
           
    pool = multiprocessing.Pool(processes=6)
    pool_outputs = pool.map(PMLPF_cox,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eLes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/PMLPF_cox_nldt_T100_scaled2_v1.txt",pfs,fmt="%f")


CCPF_nldt100_enes_scaled2_v8.txt
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
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    p=8
    N0=10
    enes=np.array([2**i for i in range(p+1)])*N0
    pfs=np.zeros((2,len(enes),samples,int(T/d),dim))
    inputs=[]
    l=5
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_nldt100_enes_scaled2_v8.txt",pfs,fmt="%f")



CCPF_nldt100_enes_scaled2_v7.txt  

    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+4
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    A=np.array([[0]])
    sc=np.float_power(2,14)
    fi=1/np.sqrt(sc)
    print(fi)
    collection_input=[dim, b_ou,A,Sig_nldt,fi]
    cov=I*1e-2
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=2/9
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    p=8
    N0=10
    enes=np.array([2**i for i in range(p+1)])*N0
    pfs=np.zeros((2,len(enes),samples,int(T/d),dim))
    inputs=[]
    l=5
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_nldt100_enes_scaled2_v7.txt",pfs,fmt="%f")


    CCPF_nldt100_scaled2_v7.txt

    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+4
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    A=np.array([[0]])
    sc=np.float_power(2,14)
    fi=1/np.sqrt(sc)
    print(fi)
    collection_input=[dim, b_ou,A,Sig_nldt,fi]
    cov=I*1e-2
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=2/9
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_nldt100_scaled2_v7.txt",pfs,fmt="%f")


    CCPF_nldt100_scaled2_v6.txt

    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+4
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    A=np.array([[0]])
    sc=np.float_power(2,14)
    fi=1/np.sqrt(sc)
    print(fi)
    collection_input=[dim, b_ou,A,Sig_nldt,fi]
    cov=I*1e-1
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=2/9
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_nldt100_scaled2_v6.txt",pfs,fmt="%f")


CCPF_nldt100_enes_scaled2_v6.txt
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+4
    l=10
    collection_input=[]
    I=identity(dim).toarray()
    A=np.array([[0]])
    sc=np.float_power(2,14)
    fi=1/np.sqrt(sc)
    print(fi)
    collection_input=[dim, b_ou,A,Sig_nldt,fi]
    cov=I*1e-1
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=2/9
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    p=8
    N0=10
    enes=np.array([2**i for i in range(p+1)])*N0
    pfs=np.zeros((2,len(enes),samples,int(T/d),dim))
    inputs=[]
    l=5
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_nldt100_enes_scaled2_v6.txt",pfs,fmt="%f")



CCPF_nldt100_enes_scaled2_v5.txt
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
    cov=I*1e-2
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=2/9
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    p=8
    N0=10
    enes=np.array([2**i for i in range(p+1)])*N0
    pfs=np.zeros((2,len(enes),samples,int(T/d),dim))
    inputs=[]
    l=5
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_nldt100_enes_scaled2_v5.txt",pfs,fmt="%f")

CCPF_nldt100_scaled2_v5.txt


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
    cov=I*1e-2
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=2/9
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/CCPF_nldt100_scaled2_v5.txt",pfs,fmt="%f")




CCPF_nldt100_enes_scaled2_v4.txt

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
    cov=I*1e0
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=2/9
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    p=8
    N0=10
    enes=np.array([2**i for i in range(p+1)])*N0
    pfs=np.zeros((2,len(enes),samples,int(T/d),dim))
    inputs=[]
    l=5
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)

    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_nldt100_enes_scaled2_v4.txt",pfs,fmt="%f")



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
    cov=I*1e0
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=2/9
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_nldt100_scaled2_v4.txt",pfs,fmt="%f")





CCPF_nldt100_scaled2_v3.txt
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+4
    l=10
    collection_input=[]
    I=identity(dim).toarray()

    A=np.array([[0]])
    sc=np.float_power(2,17)
    fi=1/np.sqrt(sc)
    print(fi)
    collection_input=[dim, b_ou,A,Sig_nldt,fi]
    cov=I*1e-5
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=2/9
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_nldt100_scaled2_v3.txt",pfs,fmt="%f")





    CCPF_nldt100_enes_scaled2_v3.txt
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+4
    l=10
    collection_input=[]
    I=identity(dim).toarray()

    A=np.array([[0]])
    sc=np.float_power(2,17)
    fi=1/np.sqrt(sc)
    print(fi)
    collection_input=[dim, b_ou,A,Sig_nldt,fi]
    cov=I*1e-5
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=2/9
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    p=8
    N0=10
    enes=np.array([2**i for i in range(p+1)])*N0
    pfs=np.zeros((2,len(enes),samples,int(T/d),dim))
    inputs=[]
    l=5
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)

    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_nldt100_enes_scaled2_v3.txt",pfs,fmt="%f")



CCPF_nldt100_enes_scaled2_v2.txt


np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+4
    l=10
    collection_input=[]
    I=identity(dim).toarray()

    A=np.array([[0]])
    sc=np.float_power(2,17)
    fi=1/np.sqrt(sc)
    print(fi)
    collection_input=[dim, b_ou,A,Sig_nldt,fi]
    cov=I*1e-3
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=2/9
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    p=8
    N0=10
    enes=np.array([2**i for i in range(p+1)])*N0
    pfs=np.zeros((2,len(enes),samples,int(T/d),dim))
    inputs=[]
    l=5
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)

    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_nldt100_enes_scaled2_v2.txt",pfs,fmt="%f")






CCPF_nldt100_scaled2_v2.txt
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+4
    l=10
    collection_input=[]
    I=identity(dim).toarray()

    A=np.array([[0]])
    sc=np.float_power(2,17)
    fi=1/np.sqrt(sc)
    print(fi)
    collection_input=[dim, b_ou,A,Sig_nldt,fi]
    cov=I*1e-3
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=2/9
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_nldt100_scaled2_v2.txt",pfs,fmt="%f")








    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+4
    l=10
    collection_input=[]
    I=identity(dim).toarray()

    A=np.array([[0]])
    sc=np.float_power(2,17)
    fi=1/np.sqrt(sc)
    print(fi)
    collection_input=[dim, b_ou,A,Sig_nldt,fi]
    cov=I*1e-1
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=2/9
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_nldt100_scaled2_v1.txt",pfs,fmt="%f")



CCPF_nldt100_enes_scaled2_v1.txt, 

    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+4
    l=10
    collection_input=[]
    I=identity(dim).toarray()

    A=np.array([[0]])
    sc=np.float_power(2,17)
    fi=1/np.sqrt(sc)
    print(fi)
    collection_input=[dim, b_ou,A,Sig_nldt,fi]
    cov=I*1e-1
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=2/9
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    p=8
    N0=10
    enes=np.array([2**i for i in range(p+1)])*N0
    pfs=np.zeros((2,len(enes),samples,int(T/d),dim))
    inputs=[]
    l=5
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)

    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_nldt100_enes_scaled2_v1.txt",pfs,fmt="%f")






CCPF_nldt100_enes_scaled_v5.txt
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+4
    l=10
    collection_input=[]
    I=identity(dim).toarray()

    A=np.array([[0]])
    sc=np.float_power(2,13)
    fi=1/np.sqrt(sc)
    print(fi)
    collection_input=[dim, b_ou,A,Sig_nldt,fi]
    cov=I*1e-1
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=2/9
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    p=8
    N0=10
    enes=np.array([2**i for i in range(p+1)])*N0
    pfs=np.zeros((2,len(enes),samples,int(T/d),dim))
    inputs=[]
    l=5
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)

    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_nldt100_enes_scaled_v5.txt",pfs,fmt="%f")



CCPF_nldt100_scaled_v5.txt

    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+4
    l=10
    collection_input=[]
    I=identity(dim).toarray()

    A=np.array([[0]])
    sc=np.float_power(2,13)
    fi=1/np.sqrt(sc)
    print(fi)
    collection_input=[dim, b_ou,A,Sig_nldt,fi]
    cov=I*1e-1
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=2/9
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_nldt100_scaled_v5.txt",pfs,fmt="%f")




    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+4
    l=10
    collection_input=[]
    I=identity(dim).toarray()

    A=np.array([[0]])
    sc=np.float_power(2,13)
    fi=1/np.sqrt(sc)
    print(fi)
    collection_input=[dim, b_ou,A,Sig_nldt,fi]
    cov=I*1e2
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=2/9
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_nldt100_scaled_v4.txt",pfs,fmt="%f")




    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+4
    l=10
    collection_input=[]
    I=identity(dim).toarray()

    A=np.array([[0]])
    sc=np.float_power(2,13)
    fi=1/np.sqrt(sc)
    print(fi)
    collection_input=[dim, b_ou,A,Sig_nldt,fi]
    cov=I*1e2
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=2/9
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    p=8
    N0=10
    enes=np.array([2**i for i in range(p+1)])*N0
    pfs=np.zeros((2,len(enes),samples,int(T/d),dim))
    inputs=[]
    l=5
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)

    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_nldt100_enes_scaled_v4.txt",pfs,fmt="%f")



CCPF_nldt100_enes_scaled_v3.txt

    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+4
    l=10
    collection_input=[]
    I=identity(dim).toarray()

    A=np.array([[0]])
    sc=np.float_power(2,13)
    fi=1/np.sqrt(sc)
    print(fi)
    collection_input=[dim, b_ou,A,Sig_nldt,fi]
    cov=I*1e-2
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=2/9
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    p=8
    N0=10
    enes=np.array([2**i for i in range(p+1)])*N0
    pfs=np.zeros((2,len(enes),samples,int(T/d),dim))
    inputs=[]
    l=5
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_nldt100_enes_scaled_v3.txt",pfs,fmt="%f")


    CCPF_nldt100_scaled_v3.txt
    np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+4
    l=10
    collection_input=[]
    I=identity(dim).toarray()

    A=np.array([[0]])
    sc=np.float_power(2,13)
    fi=1/np.sqrt(sc)
    print(fi)
    collection_input=[dim, b_ou,A,Sig_nldt,fi]
    cov=I*1e-2
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=2/9
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_nldt100_scaled_v3.txt",pfs,fmt="%f")

CCPF_nldt100_enes_scaled_v1_test.txt
np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+4
    l=10
    collection_input=[]
    I=identity(dim).toarray()

    A=np.array([[0]])
    sc=np.float_power(2,13)
    fi=1/np.sqrt(sc)
    print(fi)
    collection_input=[dim, b_ou,A,Sig_nldt,fi]
    cov=I*1e-1
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=2/9
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    p=8
    N0=10
    enes=np.array([2**i for i in range(p+1)])*N0
    pfs=np.zeros((2,len(enes),samples,int(T/d),dim))
    inputs=[]
    l=5
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_nldt100_enes_scaled_v1_test.txt",pfs,fmt="%f")

CCPF_nldt100_enes_scaled_v1.txt
np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+4
    l=10
    collection_input=[]
    I=identity(dim).toarray()

    A=np.array([[0]])
    sc=np.float_power(2,13)
    fi=1/np.sqrt(sc)
    print(fi)
    collection_input=[dim, b_ou,A,Sig_nldt,fi]
    cov=I*1e0
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=2/9
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    p=8
    N0=10
    enes=np.array([2**i for i in range(p+1)])*N0
    pfs=np.zeros((2,len(enes),samples,int(T/d),dim))
    inputs=[]
    l=5
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_nldt100_enes_scaled_v1.txt",pfs,fmt="%f")


This version is made with the scalled process to check if the constants C and C0 are good enough to
run our algorithm.

CCPF_nldt100_scaled_v1.txt

np.random.seed(0)
    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+4
    l=10
    collection_input=[]
    I=identity(dim).toarray()

    A=np.array([[0]])
    sc=np.float_power(2,13)
    fi=1/np.sqrt(sc)
    print(fi)
    collection_input=[dim, b_ou,A,Sig_nldt,fi]
    cov=I*1e0
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=2/9
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=500
    l0=1
    L=10
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=5000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_nldt100_scaled_v1.txt",pfs,fmt="%f")


This version changed the value of the particles depending on the levels to include the first particle

PMLPF_cox_nldt_T100_l03_v3.txt

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
    x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=2/9
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=2
    Lmin=l0
    Lmax=5
    d=2**0
    eLes=np.array(np.arange(Lmin,Lmax+1))
    print(eLes)
    C0=0.439
    C=0.45463086447
    K=0.00017

    pfs=np.zeros((len(eLes),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eLes)):
        for sample in range(samples):
            L=eLes[i]
            CL=2**(1/4)*(2**((L-l0)/4)-1)/(2**(1/4)-1)
            #NB0=(np.sqrt(C0)+np.sqrt(3*C/2)*CL)*np.sqrt(2*C/3)*2**(-(L-Lmin)/4)/K
            #print("NB0 is ",NB0)
            #print("is the error",2**(L/4))
            #N0=np.sqrt(C0)*(np.sqrt(C0)+np.sqrt(3*C/2)*CL)*2**(2*(L))/K
            delta_l0=np.float_power(2,-l0)
            N0=np.sqrt(C0*delta_l0)*(np.sqrt(C0/delta_l0)+np.sqrt(3*C/2)*CL*delta_l0**(-1/4))*np.float_power(2,2*L)/K
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
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,eles,d,eNes,dim,resamp_coef,phi,\
            dim_out,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([samples*i+sample,collection_input])        
        

            
    pool = multiprocessing.Pool(processes=8)
    pool_outputs = pool.map(PMLPF_cox,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eLes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/PMLPF_cox_nldt_T100_l03_v3.txt",pfs,fmt="%f")


PMLPF_cox_nldt_T100_l03_v2.txt

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
    x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=2/9
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=2
    Lmin=2
    Lmax=5
    d=2**0
    eLes=np.array(np.arange(Lmin,Lmax+1))
    print(eLes)
    C0=0.439
    C=0.45463086447
    K=0.00017

    pfs=np.zeros((len(eLes),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eLes)):
        for sample in range(samples):
            L=eLes[i]
            CL=2**(1/4)*(2**((L-l0)/4)-1)/(2**(1/4)-1)
            #NB0=(np.sqrt(C0)+np.sqrt(3*C/2)*CL)*np.sqrt(2*C/3)*2**(-(L-Lmin)/4)/K
            #print("NB0 is ",NB0)
            #print("is the error",2**(L/4))
            #N0=np.sqrt(C0)*(np.sqrt(C0)+np.sqrt(3*C/2)*CL)*2**(2*(L))/K
            N0=np.sqrt(C0)*(np.sqrt(C0)+np.sqrt(3*C/2)*CL)*np.float_power(2,2*L)*np.float_power(2,-l0/2)/K
            #print("N0 is ",N0)
           
            eles=np.array(np.arange(l0,L+1))
            eNes=np.zeros(len(eles),dtype=int)
            scale=5e-2
            eNes[0]=N0*scale
            #Cost[i]+=eNes[0]*2**eLes[0]
            eNes[0]=np.maximum(2,eNes[0])
            eNes[1:]=np.sqrt(2*C/3)*(np.sqrt(C0)+np.sqrt(3*C/2)*CL)*np.float_power(2,2*L)*np.float_power(2,-l0/4)*np.float_power(2,-eles[1:]*3/4)/K*scale
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,eles,d,eNes,dim,resamp_coef,phi,\
            dim_out,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([samples*i+sample,collection_input])        
        

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PMLPF_cox,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eLes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/PMLPF_cox_nldt_T100_l03_v2.txt",pfs,fmt="%f")


PMLPF_cox_nldt100_v4.txt

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
    x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=2/9
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=0
    Lmin=7
    Lmax=7
    eLes=np.array(np.arange(Lmin,Lmax+1))
    print(eLes)
    C0=0.439
    C=0.45463086447
    K=0.00017

    pfs=np.zeros((len(eLes),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eLes)):
        for sample in range(samples):
            L=eLes[i]
            CL=2**(1/4)*(2**(L/4)-1)/(2**(1/4)-1)
            NB0=(np.sqrt(C0)+np.sqrt(3*C/2)*CL)*np.sqrt(2*C/3)*2**(-L/4)/K
            N0=np.sqrt(C0)*(np.sqrt(C0)+np.sqrt(3*C/2)*CL)*2**(2*L)/K
            eles=np.array(np.arange(l0,L+1))
            eNes=np.zeros(len(eles),dtype=int)
            a=5e-2
            eNes[0]=N0*a
            eNes[1:]=NB0*2**(-eles[1:]*3/4)*2**(L*(2+1/4))*a
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,eles,d,eNes,dim,resamp_coef,phi,\
            dim_out,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([samples*i+sample,collection_input])        
        

            
    pool = multiprocessing.Pool(processes=60)
    pool_outputs = pool.map(PMLPF_cox,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eLes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
    np.savetxt("Observations/PMLPF_cox_nldt100_v4.txt",pfs,fmt="%f")


PMLPF_cox_nldt100_v3,txt

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
    x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=2/9
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=0
    Lmin=0
    Lmax=6
    eLes=np.array(np.arange(Lmin,Lmax+1))
    print(eLes)
    C0=0.439
    C=0.45463086447
    K=0.00017

    pfs=np.zeros((len(eLes),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eLes)):
        for sample in range(samples):
            L=eLes[i]
            CL=2**(1/4)*(2**(L/4)-1)/(2**(1/4)-1)
            NB0=(np.sqrt(C0)+np.sqrt(3*C/2)*CL)*np.sqrt(2*C/3)*2**(-L/4)/K
            N0=np.sqrt(C0)*(np.sqrt(C0)+np.sqrt(3*C/2)*CL)*2**(2*L)/K
            eles=np.array(np.arange(l0,L+1))
            eNes=np.zeros(len(eles),dtype=int)
            a=5e-2
            eNes[0]=N0*a
            eNes[1:]=NB0*2**(-eles[1:]*3/4)*2**(L*(2+1/4))*a
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,eles,d,eNes,dim,resamp_coef,phi,\
            dim_out,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([samples*i+sample,collection_input])        
        

            
    pool = multiprocessing.Pool(processes=60)
    pool_outputs = pool.map(PMLPF_cox,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eLes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
    np.savetxt("Observations/PMLPF_cox_nldt100_v3.txt",pfs,fmt="%f")


PPF_cox_nldt100_v1.txt

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
    x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=2/9
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=0
    Lmin=0
    Lmax=6
    eLes=np.array(np.arange(Lmin,Lmax+1))
    print(eLes)

    C=0.3878416411744125
    K=0.0017
    pfs=np.zeros((len(eLes),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eLes)):
        for sample in range(samples):
            l=eLes[i]
            N=int(C*2**(2*l)/K)
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,N,dim,resamp_coef,\
            g_den,g_par,Norm_Lambda,Lamb_par]   
            inputs.append([samples*i+sample,collection_input])
    pool = multiprocessing.Pool(processes=60)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eLes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
    np.savetxt("Observations/PPF_cox_nldt100_v1.txt",pfs,fmt="%f")


PPF_cox_nldt100_v2.txt

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
    x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=2/9
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=0
    Lmin=7
    Lmax=7
    eLes=np.array(np.arange(Lmin,Lmax+1))
    print(eLes)

    C=0.3878416411744125
    K=0.0017
    pfs=np.zeros((len(eLes),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eLes)):
        for sample in range(samples):
            l=eLes[i]
            N=int(C*2**(2*l)/K)
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,N,dim,resamp_coef,\
            g_den,g_par,Norm_Lambda,Lamb_par]   
            inputs.append([samples*i+sample,collection_input])        
        

            
    pool = multiprocessing.Pool(processes=60)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eLes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
    np.savetxt("Observations/PPF_cox_nldt100_v2.txt",pfs,fmt="%f")


PMLPF_cox_nldt100_v2.txt, this file was ran on Ibex. The details are rather here than 
there for consistency.  

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
    x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=2/9
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=0
    Lmin=7
    Lmax=7
    eLes=np.array(np.arange(Lmin,Lmax+1))
    print(eLes)
    C0=0.439
    C=0.45463086447
    K=0.00017

    pfs=np.zeros((len(eLes),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eLes)):
        for sample in range(samples):
            L=eLes[i]
            CL=2**(1/4)*(2**(L/4)-1)/(2**(1/4)-1)
            NB0=(np.sqrt(C0)+np.sqrt(3*C/2)*CL)*np.sqrt(2*C/3)*2**(-L/4)/K
            N0=np.sqrt(C0)*(np.sqrt(C0)+np.sqrt(3*C/2)*CL)*2**(2*L)/K
            eles=np.array(np.arange(l0,L+1))
            eNes=np.zeros(len(eles),dtype=int)
            a=5e-3
            eNes[0]=N0*a
            eNes[1:]=NB0*2**(-eles[1:]*3/4)*2**(L*(2+1/4))*a
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,eles,d,eNes,dim,resamp_coef,phi,\
            dim_out,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([samples*i+sample,collection_input])        
        

            
    pool = multiprocessing.Pool(processes=60)
    pool_outputs = pool.map(PMLPF_cox,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eLes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
    np.savetxt("Observations/PMLPF_cox_nldt100_v2.txt",pfs,fmt="%f")





PMLPF_cox_lan100_v4.txt, this file was ran on Ibex. The details are rather here than 
there for consistency. 



    T=100
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=13
    collection_input=[]
    I=identity(dim).toarray()
    df=10
    fi=np.array([[1]])
    collection_input=[dim, b_lan,df,pff.Sig_ou,fi]
    cov=I*1e0
    g_pars=[dim,cov]
    g_par=cov
    x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=5/3
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=0
    Lmin=7
    Lmax=7
    eLes=np.array(np.arange(Lmin,Lmax+1))
    print(eLes)
    C=2
    C0=0.5 
    K=0.021
    pfs=np.zeros((len(eLes),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eLes)):
        for sample in range(samples):
            L=eLes[i]
            NB0=(np.sqrt(C0)+np.sqrt(3*C/2)*L)*np.sqrt(2*C/3)/(K*L)
            N0=np.sqrt(C0)*(np.sqrt(C0)+np.sqrt(3*C/2)*L)*2**(2*L)/K
            eles=np.array(np.arange(l0,L+1))
            eNes=np.zeros(len(eles),dtype=int)
            scale=3e-1
            eNes[0]=N0*scale
            eNes[1:]=NB0*2**(L*2)*L*scale/2**(eles[1:])
            collection_input=[T,xin,b_lan,df,pff.Sig_ou,fi,obs,obs_time,eles,d,eNes,dim,resamp_coef,phi,\
            dim_out,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([samples*i+sample,collection_input])
    pool = multiprocessing.Pool(processes=40)
    pool_outputs = pool.map(PMLPF_cox,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eLes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    pfs=pfs.flatten()

    np.savetxt("Observations/PMLPF_cox_lan100_v4.txt",pfs,fmt="%f")



SUCPF_gbm_pls_T100_v1.txt, SUCPF_gbm_T100_v1.txt


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
       start=time.time()
       d=2**(0)
       resamp_coef=0.8    
       dim_out=2
           

           
       samples=2000000
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
       Pp_cum=np.cumsum(Ppd)
       Pl_cum=np.cumsum(Pld)

       print(Ppd,Pld,Pp_cum,Pl_cum)

       pfs=np.zeros((samples,int(T/d),dim))
       lps=np.zeros((samples,3))    
       inputs=[]

          
       for sample in range(samples):
                   
                   
                   
           collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,eLes,Pl_cum,d,eNes\
           ,Pp_cum,dim,resamp_coef,phi,g_den,g_par,Norm_Lambda,Lamb_par]
           
           inputs.append([sample,collection_input])        
       pool = multiprocessing.Pool(processes=10)
       pool_outputs = pool.map(PSUCPF, inputs)
       pool.close()
       pool.join()
       #blocks_pools.append(pool_outputs)
       xend1=time.time()
       end=time.time()

       print("Parallelized processes time:",end-start,"\n")            

       for sample in range(samples):
           pfs[sample]=pool_outputs[sample][0]
           lps[sample,:2]=pool_outputs[sample][1]
           lps[sample,2]=sample
           #tel_est=np.reshape(np.array(pool_outputs),(len(eLes),samples,int(T/d),dim))
           
           #log_weightss=log_weightss.flatten()
       pfs=pfs.flatten()
       np.savetxt("Observations&data/SUCPF_gbm_T100_v1.txt",pfs,fmt="%f")
       np.savetxt("Observations&data/SUCPF_gbm_pls_T100_v1.txt",lps,fmt="%f") 



SUCPF_gbm_p_levels_T10O_v1.txt


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

        start=time.time()
        d=2**(0)
        resamp_coef=0.8    
        dim_out=2
        samples=1000
        
        Lmax=8
        l0=0
        p=1
        N0=250
        eLes=np.arange(l0,Lmax+1)
        ps=np.arange(0,p+1)
        enes=N0*2**ps
        print(enes)
        #Pl=(eles+1)*np.log(eles+2)**2/2**(eles)
        #Pl[0]=2*Pl[1]
        #Pl=Pl/np.sum(Pl)
        #Pl=np.cumsum(Pl)
        #Pp=(ps+1)*np.log(ps+2)**2/2**(ps)
        #Pp[0]=2*Pp[1]
        #Pp=Pp/np.sum(Pp)
        #Pp=np.cumsum(Pp)
        #print(Pl, Pp)
        
        pfs=np.zeros((2,len(eLes),samples,int(T/d),dim))
        
        inputs=[]



        for i in range(len(eLes)):
            for sample in range(samples):
                l=eLes[i]
                eles=np.array([l,l+1])
                Pl_cum=np.array([0,1])
                enes=np.array([N0,2*N0])
                Pp_cum=np.array([1,1])
                collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,eles,Pl_cum,d,enes\
                ,Pp_cum,dim,resamp_coef,phi,g_den,g_par,Norm_Lambda,Lamb_par]
        
                inputs.append([samples*i+sample,collection_input])        

        pool = multiprocessing.Pool(processes=10)
        pool_outputs = pool.map(PSUCPF, inputs)
        pool.close()
        pool.join()
        #blocks_pools.append(pool_outputs)
        xend1=time.time()
        end=time.time()
        
        print("Parallelized processes time:",end-start,"\n")            
        for i in range(len(eLes)):
            for sample in range(samples):
                pfs[:,i,sample]=pool_outputs[i*samples+sample][0]
        #tel_est=np.reshape(np.array(pool_outputs),(len(eLes),samples,int(T/d),dim))
        
        #log_weightss=log_weightss.flatten()
        pfs=pfs.flatten()
        np.savetxt("Observations&data/SUCPF_gbm_l_levels_T100_v1.txt",pfs,fmt="%f")


SUCPF_gbm_p_levels_T10O_v1.txt

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

start=time.time()
d=2**(0)
resamp_coef=0.8    
dim_out=2
samples=1000
L=5
l0=4
p=8
N0=4
eLes=np.arange(l0,L+1)
ps=np.arange(0,p+1)
eNes=N0*2**ps
print(eNes)
#Pl=(eles+1)*np.log(eles+2)**2/2**(eles)
#Pl[0]=2*Pl[1]
#Pl=Pl/np.sum(Pl)
#Pl=np.cumsum(Pl)
#Pp=(ps+1)*np.log(ps+2)**2/2**(ps)
#Pp[0]=2*Pp[1]
#Pp=Pp/np.sum(Pp)
#Pp=np.cumsum(Pp)
#print(Pl, Pp)

pfs=np.zeros((2,len(eNes)-1,samples,int(T/d),dim))
inputs=[]

for pit in range(len(eNes)-1):
    for sample in range(samples):        
        
        Pl_cum=np.array([1,1])
        Pp_cum=np.concatenate((np.zeros(pit+1),np.zeros(p-pit)+1))
        collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,eLes,Pl_cum,d,eNes\
                          ,Pp_cum,dim,resamp_coef,phi,g_den,g_par,Norm_Lambda,Lamb_par]
        inputs.append([samples*pit+sample,collection_input])        
pool = multiprocessing.Pool(processes=10)
pool_outputs = pool.map(PSUCPF, inputs)
pool.close()
pool.join()
#blocks_pools.append(pool_outputs)
xend1=time.time()
end=time.time()

print("Parallelized processes time:",end-start,"\n")            
for i in range(len(eNes)-1):
    for sample in range(samples):
        pfs[:,i,sample]=pool_outputs[i*samples+sample][0]
        #tel_est=np.reshape(np.array(pool_outputs),(len(eLes),samples,int(T/d),dim))
        
        #log_weightss=log_weightss.flatten()
pfs=pfs.flatten()
np.savetxt("Observations&data/SUCPF_gbm_p_levels_T10O_v1.txt",pfs,fmt="%f") 

SUCPF_nldf_pls_T100_v1
SUCPF_nldf_T100_v1.txt

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
x_true=gen_gen_data(T,xin,l,collection_input)
Lamb_par=2/9
[obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)

start=time.time()
d=2**(0)
resamp_coef=0.8    
dim_out=2
    

    
samples=2000000
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
Pp_cum=np.cumsum(Ppd)
Pl_cum=np.cumsum(Pld)

print(Ppd,Pld,Pp_cum,Pl_cum)

pfs=np.zeros((samples,int(T/d),dim))
lps=np.zeros((samples,3))    
inputs=[]

   
for sample in range(samples):
            
            
            
    collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,eLes,Pl_cum,d,eNes\
    ,Pp_cum,dim,resamp_coef,phi,g_den,g_par,Norm_Lambda,Lamb_par]
    
    inputs.append([sample,collection_input])        

pool = multiprocessing.Pool(processes=10)
pool_outputs = pool.map(PSUCPF, inputs)
pool.close()
pool.join()
#blocks_pools.append(pool_outputs)
xend1=time.time()
end=time.time()

print("Parallelized processes time:",end-start,"\n")            

for sample in range(samples):
    pfs[sample]=pool_outputs[sample][0]
    lps[sample,:2]=pool_outputs[sample][1]
    lps[sample,2]=sample
    #tel_est=np.reshape(np.array(pool_outputs),(len(eLes),samples,int(T/d),dim))
    
    #log_weightss=log_weightss.flatten()
pfs=pfs.flatten()
np.savetxt("Observations&data/SUCPF_nldf_T100_v1.txt",pfs,fmt="%f")
np.savetxt("Observations&data/SUCPF_nldf_pls_T100_v1.txt",lps,fmt="%f")



SUCPF_lan_pls_T100_v1.txt 
SUCPF_lan_l14_T100_v1.txt

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

start=time.time()
d=2**(0)
resamp_coef=0.8    
dim_out=2
samples=2000
L=12
l0=11
p=1
N0=250
eLes=np.arange(l0,L+1)
ps=np.arange(0,p+1)
eNes=N0*2**ps
print(eNes)
#Pl=(eles+1)*np.log(eles+2)**2/2**(eles)
#Pl[0]=2*Pl[1]
#Pl=Pl/np.sum(Pl)
#Pl=np.cumsum(Pl)
#Pp=(ps+1)*np.log(ps+2)**2/2**(ps)
#Pp[0]=2*Pp[1]
#Pp=Pp/np.sum(Pp)
#Pp=np.cumsum(Pp)
#print(Pl, Pp)

pfs=np.zeros((samples,int(T/d),dim))

inputs=[]



for pit in range(len(eNes)-1):
    for sample in range(samples):
    
    
        Pl_cum=np.array([1,1])
        Pp_cum=np.array([1,1])
        
        
        
        collection_input=[T,xin,b_lan,df,Sig_lan,fi,obs,obs_time,eLes,Pl_cum,d,eNes\
        ,Pp_cum,dim,resamp_coef,phi,g_den,g_par,Norm_Lambda,Lamb_par]
        inputs.append([samples*pit+sample,collection_input])        
pool = multiprocessing.Pool(processes=10)
pool_outputs = pool.map(PSUCPF, inputs)
pool.close()
pool.join()
#blocks_pools.append(pool_outputs)
xend1=time.time()
end=time.time()

print("Parallelized processes time:",end-start,"\n")            
for i in range(len(eNes)-1):
    for sample in range(samples):
        pfs[sample]=pool_outputs[i*samples+sample][0]
        #tel_est=np.reshape(np.array(pool_outputs),(len(eLes),samples,int(T/d),dim))

#log_weightss=log_weightss.flatten()
pfs=pfs.flatten()
np.savetxt("Observations&data/SUCPF_lan_l14_T100_v1.txt",pfs,fmt="%f") 



SUCPF_lan_l_levels_T100_v2.txt

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

start=time.time()
d=2**(0)
resamp_coef=0.8    
dim_out=2
samples=1000

Lmax=8
l0=0
p=1
N0=25
eLes=np.arange(l0,Lmax+1)
ps=np.arange(0,p+1)
enes=N0*2**ps
print(enes)
#Pl=(eles+1)*np.log(eles+2)**2/2**(eles)
#Pl[0]=2*Pl[1]
#Pl=Pl/np.sum(Pl)
#Pl=np.cumsum(Pl)
#Pp=(ps+1)*np.log(ps+2)**2/2**(ps)
#Pp[0]=2*Pp[1]
#Pp=Pp/np.sum(Pp)
#Pp=np.cumsum(Pp)
#print(Pl, Pp)

pfs=np.zeros((2,len(eLes),samples,int(T/d),dim))        
inputs=[]

for i in range(len(eLes)):
    for sample in range(samples):
        l=eLes[i]
        eles=np.array([l,l+1])
        Pl_cum=np.array([0,1])
        enes=np.array([N0,2*N0])
        Pp_cum=np.array([1,1])
        collection_input=[T,xin,b_lan,df,Sig_lan,fi,obs,obs_time,eles,Pl_cum,d,enes\
        ,Pp_cum,dim,resamp_coef,phi,g_den,g_par,Norm_Lambda,Lamb_par]

        inputs.append([samples*i+sample,collection_input])        

pool = multiprocessing.Pool(processes=10)
pool_outputs = pool.map(PSUCPF, inputs)
pool.close()
pool.join()
#blocks_pools.append(pool_outputs)
xend1=time.time()
end=time.time()

print("Parallelized processes time:",end-start,"\n")            
for i in range(len(eLes)):
    for sample in range(samples):
        pfs[:,i,sample]=pool_outputs[i*samples+sample][0]
#tel_est=np.reshape(np.array(pool_outputs),(len(eLes),samples,int(T/d),dim))
#log_weightss=log_weightss.flatten()
pfs=pfs.flatten()
np.savetxt("Observations&data/SUCPF_lan_l_levels_T100_v2.txt",pfs,fmt="%f")



SUCPF_lan_p_levels_T10O_v1.txt
with this iteration we start computing the rates for the Unbiased of the langevin process

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

start=time.time()
d=2**(0)
resamp_coef=0.8    
dim_out=2
samples=1000
L=5
l0=4
p=8
N0=4
eLes=np.arange(l0,L+1)
ps=np.arange(0,p+1)
eNes=N0*2**ps
print(eNes)

pfs=np.zeros((2,len(eNes)-1,samples,int(T/d),dim))
inputs=[]

for pit in range(len(eNes)-1):
    for sample in range(samples):        
        
        Pl_cum=np.array([1,1])
        Pp_cum=np.concatenate((np.zeros(pit+1),np.zeros(p-pit)+1))
        collection_input=[T,xin,b_lan,df,Sig_lan,fi,obs,obs_time,eLes,Pl_cum,d,eNes\
                          ,Pp_cum,dim,resamp_coef,phi,g_den,g_par,Norm_Lambda,Lamb_par]
        
        inputs.append([samples*pit+sample,collection_input])        
pool = multiprocessing.Pool(processes=10)
pool_outputs = pool.map(PSUCPF, inputs)
pool.close()
pool.join()
#blocks_pools.append(pool_outputs)
xend1=time.time()
end=time.time()

print("Parallelized processes time:",end-start,"\n")            
for i in range(len(eNes)-1):
    for sample in range(samples):
        pfs[:,i,sample]=pool_outputs[i*samples+sample][0]
        #tel_est=np.reshape(np.array(pool_outputs),(len(eLes),samples,int(T/d),dim))
        
        #log_weightss=log_weightss.flatten()
pfs=pfs.flatten()
np.savetxt("Observations&data/SUCPF_lan_p_levels_T10O_v1.txt",pfs,fmt="%f") #%%



This iteration is made with large truncation parameters in order to 
get a different the rate 


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

start=time.time()
d=2**(0)
resamp_coef=0.8    
dim_out=2


samples=2000000
l0=0
Lmax=13
Lmaxv1=10
pma=8
eNes=np.concatenate(([3,6,12,24,50],100*2**np.array(range(0,pma+1))))
eLes=np.arange(l0,Lmax+1)

ps=np.arange(len(eNes))

pmax=len(ps)-1

beta=1
Delta0=1/2**eLes[0]
print(eLes,eNes)
Pp=3.75525
Pp0=55.787
Pl=0.0224577
Pl0=0.73559
Pld=np.zeros(Lmax+1-l0)
Ppd=np.zeros(pmax+1)
Pld[0]=Delta0**(beta)*Pl0
Pld[1:]=Pl*np.log2(eLes[1:]+2)**2*(eLes[1:]+1)/2**(beta*eLes[1:])
N0=3 # this N0=100 is just so that we can get the probabilities to match with what 
# we simulatedf
Ppd[0]=Pp0/N0
Ppd[1:]=Pp*np.log2(ps[1:]+2)**2*(ps[1:]+1)/eNes[1:]
Ppd=Ppd/np.sum(Ppd)
Pld=Pld/np.sum(Pld)
print(Ppd,Pld)
Pp_cum=np.cumsum(Ppd)
Pl_cum=np.cumsum(Pld)
    
pmaxv1=5
lpsv1=np.loadtxt("Observations&data/SUCPF_ou_pls_T100_v1.txt",dtype=float)
lpsv1=np.array(lpsv1,dtype=int)

dic=[[[] for j in range(pmaxv1+1)] for i in range(Lmaxv1+1)]


for i in range(len(lpsv1)):
    dic[lpsv1[i,0]][lpsv1[i,1]].append(i)

Num_samples=np.zeros((Lmaxv1+1,pmaxv1+1))
for i in range(Lmaxv1+1):
    for j in range(pmaxv1+1):
        Num_samples[i,j]=len(dic[i][j])


lps=np.zeros((samples,4),dtype=int)    
inputs=[]

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
np.savetxt("Observations&data/SUCPF_ou_T100_v2.txt",pfs,fmt="%f")
np.savetxt("Observations&data/SUCPF_ou_pls_T100_v2.txt",lps,fmt="%f")


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
    x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=2/9
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)

    start=time.time()
    d=2**(0)
    resamp_coef=0.8    
    dim_out=2
    samples=1000
    L=5
    l0=4
    p=8
    N0=4
    eLes=np.arange(l0,L+1)
    ps=np.arange(0,p+1)
    eNes=N0*2**ps
    print(eNes)
    #Pl=(eles+1)*np.log(eles+2)**2/2**(eles)
    #Pl[0]=2*Pl[1]
    #Pl=Pl/np.sum(Pl)
    #Pl=np.cumsum(Pl)
    #Pp=(ps+1)*np.log(ps+2)**2/2**(ps)
    #Pp[0]=2*Pp[1]
    #Pp=Pp/np.sum(Pp)
    #Pp=np.cumsum(Pp)
    #print(Pl, Pp)
    
    pfs=np.zeros((2,len(eNes)-1,samples,int(T/d),dim))
    inputs=[]

    for pit in range(len(eNes)-1):
        for sample in range(samples):        
            
            Pl_cum=np.array([1,1])
            Pp_cum=np.concatenate((np.zeros(pit+1),np.zeros(p-pit)+1))
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,eLes,Pl_cum,d,eNes\
                              ,Pp_cum,dim,resamp_coef,phi,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([samples*pit+sample,collection_input])        
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PSUCPF, inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()

    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eNes)-1):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample][0]
            #tel_est=np.reshape(np.array(pool_outputs),(len(eLes),samples,int(T/d),dim))
            
            #log_weightss=log_weightss.flatten()
    pfs=pfs.flatten()
    np.savetxt("Observations&data/SUCPF_nldt_p_levels_T10O_v2.txt",pfs,fmt="%f") 



PCox_PF_nldt_etc_rate_v5.txt
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
    x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=2/9
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8    
    dim_out=2


    samples=100
    
    es=np.array([1/4**i for i in range(-1,0)])*1e-2

    C0=0.439
    C=0.45463086447
    K=0.00017
    enes=np.array( np.ceil(2*C0/es),dtype=int)
    eles=np.array(np.ceil(-np.log2(es/(2*np.sqrt(K)))/2,dtype=float),dtype=int)
    pfs=np.zeros((len(eles),samples,int(T/d),dim))
    print(enes,eles)
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        N=enes[i]
        for sample in range(samples):
            
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,N,dim,resamp_coef,\
            g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF, inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")


    for i in range(len(eles)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[samples*i+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/PCox_PF_nldt_etc_rate_v5.txt",pfs,fmt="%f")


SUCPF_nldt_l_levels_T100_v2.txt, this is version 2 by accident, it should have ebeen
version 1

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
    x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=2/9
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8    
    dim_out=2
    samples=1000
    
    Lmax=8
    l0=0
    p=1
    N0=250
    eLes=np.arange(l0,Lmax+1)
    ps=np.arange(0,p+1)
    enes=N0*2**ps
    print(enes)
    #Pl=(eles+1)*np.log(eles+2)**2/2**(eles)
    #Pl[0]=2*Pl[1]
    #Pl=Pl/np.sum(Pl)
    #Pl=np.cumsum(Pl)
    #Pp=(ps+1)*np.log(ps+2)**2/2**(ps)
    #Pp[0]=2*Pp[1]
    #Pp=Pp/np.sum(Pp)
    #Pp=np.cumsum(Pp)
    #print(Pl, Pp)
    
    pfs=np.zeros((2,len(eLes),samples,int(T/d),dim))
    
    inputs=[]



    for i in range(len(eLes)):
        for sample in range(samples):
            l=eLes[i]
            eles=np.array([l,l+1])
            Pl_cum=np.array([0,1])
            enes=np.array([N0,2*N0])
            Pp_cum=np.array([1,1])
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,eles,Pl_cum,d,enes\
            ,Pp_cum,dim,resamp_coef,phi,g_den,g_par,Norm_Lambda,Lamb_par]
    
            inputs.append([samples*i+sample,collection_input])        

    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PSUCPF, inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eLes)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample][0]
    #tel_est=np.reshape(np.array(pool_outputs),(len(eLes),samples,int(T/d),dim))
    
    #log_weightss=log_weightss.flatten()
    pfs=pfs.flatten()
    np.savetxt("Observations&data/SUCPF_nldt_l_levels_T100_v2.txt",pfs,fmt="%f")



PCox_PF_nldt_etc_rate_v4.txt was made because the PCox_PF version was not working 
properly.

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
    x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=2/9
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8    
    dim_out=2


    samples=100
    
    es=np.array([1/4**i for i in range(6)])*1e-2

    C0=0.439
    C=0.45463086447
    K=0.00017
    enes=np.array( np.ceil(2*C0/es),dtype=int)
    eles=np.array(np.ceil(-np.log2(es/(2*np.sqrt(K)))/2,dtype=float),dtype=int)
    pfs=np.zeros((len(eles),samples,int(T/d),dim))
    print(enes,eles)
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        N=enes[i]
        for sample in range(samples):
            
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,N,dim,resamp_coef,\
            g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF, inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")


    for i in range(len(eles)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[samples*i+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/PCox_PF_nldt_etc_rate_v4.txt",pfs,fmt="%f")


SUCPF_nldt_l14_T100_v1.txt computation of the truth for l.

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
    x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=2/9
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8    
    dim_out=2
    samples=2000
    L=12
    l0=11
    p=1
    N0=250
    eLes=np.arange(l0,L+1)
    ps=np.arange(0,p+1)
    eNes=N0*2**ps
    print(eNes)
    #Pl=(eles+1)*np.log(eles+2)**2/2**(eles)
    #Pl[0]=2*Pl[1]
    #Pl=Pl/np.sum(Pl)
    #Pl=np.cumsum(Pl)
    #Pp=(ps+1)*np.log(ps+2)**2/2**(ps)
    #Pp[0]=2*Pp[1]
    #Pp=Pp/np.sum(Pp)
    #Pp=np.cumsum(Pp)
    #print(Pl, Pp)

    pfs=np.zeros((samples,int(T/d),dim))
    
    inputs=[]
    


    for pit in range(len(eNes)-1):
        for sample in range(samples):
        
        
            Pl_cum=np.array([1,1])
            Pp_cum=np.array([1,1])
            
            
            
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,eLes,Pl_cum,d,eNes\
            ,Pp_cum,dim,resamp_coef,phi,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([samples*pit+sample,collection_input])        
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PSUCPF, inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()

    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eNes)-1):
        for sample in range(samples):
            pfs[sample]=pool_outputs[i*samples+sample][0]
            #tel_est=np.reshape(np.array(pool_outputs),(len(eLes),samples,int(T/d),dim))

    #log_weightss=log_weightss.flatten()
    pfs=pfs.flatten()
    np.savetxt("Observations&data/SUCPF_nldt_l14_T100_v1.txt",pfs,fmt="%f") 


PCox_PF_nldt_etc_rate_v3.txt


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
    x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=2/9
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8    
    dim_out=2


    samples=100
    
    es=np.array([1/4**i for i in range(6)])*1e-2

    C0=0.439
    C=0.45463086447
    K=0.00017
    enes=np.array( np.ceil(2*C0/es),dtype=int)
    eles=np.array(np.ceil(-np.log2(es/(2*np.sqrt(K)))/2,dtype=float),dtype=int)
    pfs=np.zeros((len(eles),samples,int(T/d),dim))
    print(enes,eles)
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        N=enes[i]
        for sample in range(samples):
            
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,N,dim,resamp_coef,\
            g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF, inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")


    for i in range(len(eles)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[samples*i+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/PCox_PF_nldt_etc_rate_v3.txt",pfs,fmt="%f")




computation of the bias and variance wheen varying p SUCPF_nldt_p_levels_T10O_v1.txt
the output is [phi_pfmean0,phi_pfmean1]
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
    x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=2/9
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)

    start=time.time()
    d=2**(0)
    resamp_coef=0.8    
    dim_out=2
    samples=1000
    L=5
    l0=4
    p=8
    N0=4
    eLes=np.arange(l0,L+1)
    ps=np.arange(0,p+1)
    eNes=N0*2**ps
    print(eNes)
    #Pl=(eles+1)*np.log(eles+2)**2/2**(eles)
    #Pl[0]=2*Pl[1]
    #Pl=Pl/np.sum(Pl)
    #Pl=np.cumsum(Pl)
    #Pp=(ps+1)*np.log(ps+2)**2/2**(ps)
    #Pp[0]=2*Pp[1]
    #Pp=Pp/np.sum(Pp)
    #Pp=np.cumsum(Pp)
    #print(Pl, Pp)
    
    pfs=np.zeros((2,len(eNes)-1,samples,int(T/d),dim))
    inputs=[]

    for pit in range(len(eNes)-1):
        for sample in range(samples):        
            
            Pl_cum=np.array([1,1])
            Pp_cum=np.concatenate((np.zeros(pit+1),np.zeros(p-pit)+1))
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,eLes,Pl_cum,d,eNes\
                              ,Pp_cum,dim,resamp_coef,phi,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([samples*pit+sample,collection_input])        
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PSUCPF, inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()

    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eNes)-1):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample][0]
            #tel_est=np.reshape(np.array(pool_outputs),(len(eLes),samples,int(T/d),dim))
            
            #log_weightss=log_weightss.flatten()
    pfs=pfs.flatten()
    np.savetxt("Observations&data/SUCPF_nldt_p_levels_T10O_v1.txt",pfs,fmt="%f") 



PCox_PF_nldt_etc_rate_v2.txt



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
x_true=gen_gen_data(T,xin,l,collection_input)
Lamb_par=2/9
[obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
start=time.time()
d=2**(0)
resamp_coef=0.8    
dim_out=2


samples=100

es=np.array([1/4**i for i in range(6)])*1e-2

C0=0.439
C=0.45463086447
K=0.00017
enes=np.array( np.ceil(2*C0/es),dtype=int)
eles=np.array(np.ceil(-np.log2(es/(2*np.sqrt(K)))/2,dtype=float),dtype=int)
pfs=np.zeros((len(eles),samples,int(T/d),dim))
print(enes,eles)
inputs=[]
for i in range(len(eles)):
    l=eles[i]
    N=enes[i]
    for sample in range(samples):
        
        collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,N,dim,resamp_coef,\
        g_den,g_par,Norm_Lambda,Lamb_par]
        inputs.append([sample+samples*i,collection_input])
pool = multiprocessing.Pool(processes=10)
pool_outputs = pool.map(PCox_PF, inputs)
pool.close()
pool.join()
#blocks_pools.append(pool_outputs)
xend1=time.time()
end=time.time()
print("Parallelized processes time:",end-start,"\n")


for i in range(len(eles)):
    for sample in range(samples):
        pfs[i,sample]=pool_outputs[samples*i+sample]
pfs=pfs.flatten()
np.savetxt("Observations&data/PCox_PF_nldt_etc_rate_v2.txt",pfs,fmt="%f")


PCox_PF_nldt_etc_rate_v1.txt particle filter error to cost reate for the nldt

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
x_true=gen_gen_data(T,xin,l,collection_input)
Lamb_par=2/9
[obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)

start=time.time()
d=2**(0)
resamp_coef=0.8    
dim_out=2


samples=100
es=np.array([1/4**i for i in range(6)])*1e-2

C0=0.439
C=0.45463086447
K=0.00017
enes=np.array( np.ceil(2*C0/es),dtype=int)
a=99
eles=np.array(np.ceil(-np.log2(es/(2*np.sqrt(K)))/2,dtype=float),dtype=int)

pfs=np.zeros((len(eles),samples,int(T/d),dim))
print(enes,eles)
inputs=[]
for i in range(len(eles)):
    l=eles[i]
    N=enes[i]
    for sample in range(samples):
        
        collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,N,dim,resamp_coef,\
        g_den,g_par,Norm_Lambda,Lamb_par]
            
        inputs.append([sample+samples*i,collection_input])        

pool = multiprocessing.Pool(processes=10)
pool_outputs = pool.map(PCox_PF, inputs)
pool.close()
pool.join()
#blocks_pools.append(pool_outputs)
xend1=time.time()
end=time.time()

print("Parallelized processes time:",end-start,"\n")            

for i in range(len(eles)):
    for sample in range(samples):
        pfs[i,sample]=pool_outputs[samples*i+sample]
pfs=pfs.flatten()
np.savetxt("Observations&data/PCox_PF_nldt_etc_rate_v1.txt",pfs,fmt="%f")




PCox_PF_etc_rate_v2.txt,  this iteration was made in order to go one level deeper


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

    start=time.time()
    d=2**(0)
    resamp_coef=0.8    
    dim_out=2
    
    
    samples=100
    C=0.8
    C0=0.5
    K=0.0012
    l0=0
    Lmin=0
    Lmax=6
    #es=np.array([1/4**i for i in range(6)])*1e-2
    es=np.array([1e-2/4**6])
    #eLes=np.arange(Lmin,Lmax+1,1)
    enes=np.array( np.ceil(2*C0/es),dtype=int)

    eles=np.array(np.ceil(-np.log2(es/(2*np.sqrt(K)))/2,dtype=float),dtype=int)
    pfs=np.zeros((len(eles),samples,int(T/d),dim))
    print(enes,eles)
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        N=enes[i]
        for sample in range(samples):
            
            collection_input=[T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,l,d,N\
            ,dim,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
   
            inputs.append([sample+samples*i,collection_input])        

    pool = multiprocessing.Pool(processes=4)
    pool_outputs = pool.map(PCox_PF, inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    
    print("Parallelized processes time:",end-start,"\n")            
    
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[samples*i+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/PCox_PF_etc_rate_v2.txt",pfs,fmt="%f")
            




PCox_PF_etc_rate_v1.txt corresponds to the samples for the error to cost rate 
of the Plain particle filter, we make this experiments in order to compare with 
our methods. 

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

    start=time.time()
    d=2**(0)
    resamp_coef=0.8    
    dim_out=2
    
    
    samples=100
    C=0.8
    C0=0.5
    K=0.0012
    l0=0
    Lmin=0
    Lmax=6
    es=np.array([1/4**i for i in range(6)])*1e-2
    
    #eLes=np.arange(Lmin,Lmax+1,1)
    enes=np.array( np.ceil(2*C0/es),dtype=int)

    eles=np.array(np.ceil(-np.log2(es/(2*np.sqrt(K)))/2,dtype=float),dtype=int)
    pfs=np.zeros((len(eles),samples,int(T/d),dim))
    
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        N=enes[i]
        for sample in range(samples):
            
            collection_input=[T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,l,d,N\
            ,dim,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
   
            inputs.append([sample+samples*i,collection_input])        

    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF, inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    
    print("Parallelized processes time:",end-start,"\n")            
    
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[samples*i+sample]
    pfs=pfs.flatten()
    np.savetxt("Observations&data/PCox_PF_etc_rate_v1.txt",pfs,fmt="%f")


SUCPF_ou_T100_v1.txt
SUCPF_ou_pls_T100_v1.txt

This iteration is made to get the final Unbiased OU T=100 pf cox process. 
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

    start=time.time()
    d=2**(0)
    resamp_coef=0.8    
    dim_out=2
    
    
    samples=5000000
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
    Pp_cum=np.cumsum(Ppd)
    Pl_cum=np.cumsum(Pld)
    
    print(Ppd,Pld,Pp_cum,Pl_cum)
    
    pfs=np.zeros((samples,int(T/d),dim))
    lps=np.zeros((samples,3))    
    inputs=[]

   
    for sample in range(samples):
            
            
            
            collection_input=[T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,eLes,Pl_cum,d,eNes\
            ,Pp_cum,dim,resamp_coef,phi,g_den,g_par,Norm_Lambda,Lamb_par]
    
            inputs.append([sample,collection_input])        

    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PSUCPF, inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    
    print("Parallelized processes time:",end-start,"\n")            
    
    for sample in range(samples):
            pfs[sample]=pool_outputs[sample][0]
            lps[sample,:2]=pool_outputs[sample][1]
            lps[sample,2]=sample
    #tel_est=np.reshape(np.array(pool_outputs),(len(eLes),samples,int(T/d),dim))
    
    #log_weightss=log_weightss.flatten()
    pfs=pfs.flatten()
    np.savetxt("Observations&data/SUCPF_ou_T100_v1.txt",pfs,fmt="%f")
    np.savetxt("Observations&data/SUCPF_ou_pls_T100_v1.txt",lps,fmt="%f")




SUCPF_ou_pl0_levels_T100_v1.txtf


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

    start=time.time()
    d=2**(0)
    resamp_coef=0.8    
    dim_out=2
    samples=1000
    
    Lmax=6
    l0=5
    p=8
    N0=4
    eLes=np.arange(l0,Lmax+1)
    ps=np.arange(0,p+1)
    eNes=N0*2**ps
    print(eNes)
    #Pl=(eles+1)*np.log(eles+2)**2/2**(eles)
    #Pl[0]=2*Pl[1]
    #Pl=Pl/np.sum(Pl)
    #Pl=np.cumsum(Pl)
    #Pp=(ps+1)*np.log(ps+2)**2/2**(ps)
    #Pp[0]=2*Pp[1]
    #Pp=Pp/np.sum(Pp)
    #Pp=np.cumsum(Pp)
    #print(Pl, Pp)    
    pfs=np.zeros((len(eNes),samples,int(T/d),dim))    
    inputs=[]

    for i in range(len(eNes)):
        for sample in range(samples):
            N=eNes[i]
            
            Pl_cum=np.array([1,1])
            
            enes=N*np.array([1,2])
            Pp_cum=np.array([1,1])
            collection_input=[T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,eLes,Pl_cum,d,enes\
            ,Pp_cum,dim,resamp_coef,phi,g_den,g_par,Norm_Lambda,Lamb_par]
    
            inputs.append([samples*i+sample,collection_input])        

    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PSUCPF, inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eNes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample][0]
    
    #tel_est=np.reshape(np.array(pool_outputs),(len(eLes),samples,int(T/d),dim))
    
    #log_weightss=log_weightss.flatten()
    pfs=pfs.flatten()
    np.savetxt("Observations&data/SUCPF_ou_pl0_levels_T100_v1.txt",pfs,fmt="%f")





This test is done in order to compute cl0 and cp0


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

    start=time.time()
    d=2**(0)
    resamp_coef=0.8    
    dim_out=2
    samples=1000
    
    Lmax=5
    l0=4
    p=8
    N0=4
    eLes=np.arange(l0,Lmax+1)
    ps=np.arange(0,p+1)
    eNes=N0*2**ps
    print(eNes)
    #Pl=(eles+1)*np.log(eles+2)**2/2**(eles)
    #Pl[0]=2*Pl[1]
    #Pl=Pl/np.sum(Pl)
    #Pl=np.cumsum(Pl)
    #Pp=(ps+1)*np.log(ps+2)**2/2**(ps)
    #Pp[0]=2*Pp[1]
    #Pp=Pp/np.sum(Pp)
    #Pp=np.cumsum(Pp)
    #print(Pl, Pp)    
    pfs=np.zeros((len(eNes),samples,int(T/d),dim))    
    inputs=[]

    for i in range(len(eNes)):
        for sample in range(samples):
            N=eNes[i]
            
            Pl_cum=np.array([1,1])
            
            enes=N*np.array([1,2])
            Pp_cum=np.array([1,1])
            collection_input=[T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,eLes,Pl_cum,d,enes\
            ,Pp_cum,dim,resamp_coef,phi,g_den,g_par,Norm_Lambda,Lamb_par]
    
            inputs.append([samples*i+sample,collection_input])        

    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PSUCPF, inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eNes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample][0]
    
    #tel_est=np.reshape(np.array(pool_outputs),(len(eLes),samples,int(T/d),dim))
    
    #log_weightss=log_weightss.flatten()
    pfs=pfs.flatten()
    np.savetxt("Observations&data/SUCPF_ou_pl0_levels_T100_v1.txt",pfs,fmt="%f")


 SUCPF_ou_l_levels_T100_v2.txt this test was made bcs  SUCPF_ou_l_levels_T100_v1.txgt
 didn't have ther right dimension for some reason
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

    start=time.time()
    d=2**(0)
    resamp_coef=0.8    
    dim_out=2
    samples=1000
    
    Lmax=8
    l0=0
    p=1
    N0=250
    eLes=np.arange(l0,Lmax+1)
    ps=np.arange(0,p+1)
    enes=N0*2**ps
    print(enes)
    #Pl=(eles+1)*np.log(eles+2)**2/2**(eles)
    #Pl[0]=2*Pl[1]
    #Pl=Pl/np.sum(Pl)
    #Pl=np.cumsum(Pl)
    #Pp=(ps+1)*np.log(ps+2)**2/2**(ps)
    #Pp[0]=2*Pp[1]
    #Pp=Pp/np.sum(Pp)
    #Pp=np.cumsum(Pp)
    #print(Pl, Pp)
    
    pfs=np.zeros((2,len(eLes),samples,int(T/d),dim))
    
    inputs=[]



    for i in range(len(eLes)):
        for sample in range(samples):
            l=eLes[i]
            eles=np.array([l,l+1])
            Pl_cum=np.array([0,1])
            enes=np.array([N0,2*N0])
            Pp_cum=np.array([1,1])
            collection_input=[T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,eles,Pl_cum,d,enes\
            ,Pp_cum,dim,resamp_coef,phi,g_den,g_par,Norm_Lambda,Lamb_par]
    
            inputs.append([samples*i+sample,collection_input])        

    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PSUCPF, inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eLes)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample][0]
    #tel_est=np.reshape(np.array(pool_outputs),(len(eLes),samples,int(T/d),dim))
    
    #log_weightss=log_weightss.flatten()
    pfs=pfs.flatten()
    np.savetxt("Observations&data/SUCPF_ou_l_levels_T100_v2.txt",pfs,fmt="%f")
            


This test is carried out in order to get the truth for the bias when varying 
l. 
SUCPF_ou_l14_T100_v1.txt


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

    start=time.time()
    d=2**(0)
    resamp_coef=0.8    
    dim_out=2
    samples=2000
    L=12
    l0=11
    p=1
    N0=250
    eLes=np.arange(l0,L+1)
    ps=np.arange(0,p+1)
    eNes=N0*2**ps
    print(eNes)
    #Pl=(eles+1)*np.log(eles+2)**2/2**(eles)
    #Pl[0]=2*Pl[1]
    #Pl=Pl/np.sum(Pl)
    #Pl=np.cumsum(Pl)
    #Pp=(ps+1)*np.log(ps+2)**2/2**(ps)
    #Pp[0]=2*Pp[1]
    #Pp=Pp/np.sum(Pp)
    #Pp=np.cumsum(Pp)
    #print(Pl, Pp)

    pfs=np.zeros((samples,int(T/d),dim))
    
    inputs=[]
    


    for pit in range(len(eNes)-1):
        for sample in range(samples):
        
        
            Pl_cum=np.array([1,1])
            Pp_cum=np.array([1,1])
            
            
            
            collection_input=[T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,eLes,Pl_cum,d,eNes\
            ,Pp_cum,dim,resamp_coef,phi,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([samples*pit+sample,collection_input])        
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PSUCPF, inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()

    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eNes)-1):
        for sample in range(samples):
            pfs[sample]=pool_outputs[i*samples+sample][0]
            #tel_est=np.reshape(np.array(pool_outputs),(len(eLes),samples,int(T/d),dim))

    #log_weightss=log_weightss.flatten()
    pfs=pfs.flatten()
    np.savetxt("Observations&data/SUCPF_ou_l14_T100_v1.txt",pfs,fmt="%f") 



SUCPF_ou_l_levels_T100_v1.txt this test is made in order to compute the 
bias and variance in terms of l. the output is [phi_pfmean0,phi_pfmean1]

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

    start=time.time()
    d=2**(0)
    resamp_coef=0.8    
    dim_out=2
    samples=1000
    
    Lmax=8
    l0=0
    p=1
    N0=250
    eLes=np.arange(l0,Lmax+1)
    ps=np.arange(0,p+1)
    enes=N0*2**ps
    print(enes)
    #Pl=(eles+1)*np.log(eles+2)**2/2**(eles)
    #Pl[0]=2*Pl[1]
    #Pl=Pl/np.sum(Pl)
    #Pl=np.cumsum(Pl)
    #Pp=(ps+1)*np.log(ps+2)**2/2**(ps)
    #Pp[0]=2*Pp[1]
    #Pp=Pp/np.sum(Pp)
    #Pp=np.cumsum(Pp)
    #print(Pl, Pp)
    
    pfs=np.zeros((2,len(eLes),samples,int(T/d),dim))
    
    inputs=[]



    for i in range(len(eLes)):
        for sample in range(samples):
            l=eLes[i]
            eles=np.array([l,l+1])
            Pl_cum=np.array([0,1])
            enes=np.array([N0,2*N0])
            Pp_cum=np.array([1,1])
            collection_input=[T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,eles,Pl_cum,d,enes\
            ,Pp_cum,dim,resamp_coef,phi,g_den,g_par,Norm_Lambda,Lamb_par]
    
            inputs.append([samples*i+sample,collection_input])        

    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PSUCPF, inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eLes)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample][0]
    
    #tel_est=np.reshape(np.array(pool_outputs),(len(eLes),samples,int(T/d),dim))
    
    #log_weightss=log_weightss.flatten()
    pfs=pfs.flatten()
    np.savetxt("Observations&data/SUCPF_ou_l_levels_T100_v1.txt",pfs,fmt="%f")



This is the first test in order to get the constants related to the number of 
particles for OU T=100. SUCPF_ou_p_levels_T10O_v1.txt the output is [phi_pfmean0,phi_pfmean1]

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

    start=time.time()
    d=2**(0)
    resamp_coef=0.8    
    dim_out=2
    samples=1000
    L=5
    l0=4
    p=8
    N0=4
    eLes=np.arange(l0,L+1)
    ps=np.arange(0,p+1)
    eNes=N0*2**ps
    print(eNes)
    #Pl=(eles+1)*np.log(eles+2)**2/2**(eles)
    #Pl[0]=2*Pl[1]
    #Pl=Pl/np.sum(Pl)
    #Pl=np.cumsum(Pl)
    #Pp=(ps+1)*np.log(ps+2)**2/2**(ps)
    #Pp[0]=2*Pp[1]
    #Pp=Pp/np.sum(Pp)
    #Pp=np.cumsum(Pp)
    #print(Pl, Pp)
    
    pfs=np.zeros((2,len(eNes)-1,samples,int(T/d),dim))
    inputs=[]

    for pit in range(len(eNes)-1):
        for sample in range(samples):        
            
            Pl_cum=np.array([1,1])
            Pp_cum=np.concatenate((np.zeros(pit+1),np.zeros(p-pit)+1))
            collection_input=[T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,eLes,Pl_cum,d,eNes\
                              ,Pp_cum,dim,resamp_coef,phi,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([samples*pit+sample,collection_input])        
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PSUCPF, inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()

    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eNes)-1):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample][0]
            #tel_est=np.reshape(np.array(pool_outputs),(len(eLes),samples,int(T/d),dim))
            
            #log_weightss=log_weightss.flatten()
    pfs=pfs.flatten()
    np.savetxt("Observations&data/SUCPF_ou_p_levels_T10O_v1.txt",pfs,fmt="%f") 





SUCPF_ou_T10_v1.txt and SUCPF_ou_pls_T10_v1.txt correspond to the following settings
This is the first test to check how the unbiased estimator performs 

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

    start=time.time()
    d=2**(0)
    resamp_coef=0.8    
    dim_out=2
    
    
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
    
    print(Ppd,Pld,Pp_cum,Pl_cum)
    
    pfs=np.zeros((samples,int(T/d),dim))
    lps=np.zeros((samples,3))    
    inputs=[]

   
    for sample in range(samples):
            
            
            
            collection_input=[T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,eLes,Pl_cum,d,eNes\
            ,Pp_cum,dim,resamp_coef,phi,g_den,g_par,Norm_Lambda,Lamb_par]
    
            inputs.append([sample,collection_input])        

    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PSUCPF, inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    
    print("Parallelized processes time:",end-start,"\n")            
    
    for sample in range(samples):
            pfs[sample]=pool_outputs[sample][0]
            lps[sample,:2]=pool_outputs[sample][1]
            lps[sample,2]=sample
    #tel_est=np.reshape(np.array(pool_outputs),(len(eLes),samples,int(T/d),dim))
    
    #log_weightss=log_weightss.flatten()
    pfs=pfs.flatten()
    np.savetxt("Observations&data/SUCPF_ou_T10_v1.txt",pfs,fmt="%f")
    np.savetxt("Observations&data/SUCPF_ou_pls_T10_v1.txt",lps,fmt="%f")




This iteration is made in order to find the bias when we vary l

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

    start=time.time()
    d=2**(0)
    resamp_coef=0.8    
    dim_out=2
    samples=1000
    
    Lmax=8
    l0=0
    p=1
    N0=250
    eLes=np.arange(l0,Lmax+1)
    ps=np.arange(0,p+1)
    enes=N0*2**ps
    print(enes)
    #Pl=(eles+1)*np.log(eles+2)**2/2**(eles)
    #Pl[0]=2*Pl[1]
    #Pl=Pl/np.sum(Pl)
    #Pl=np.cumsum(Pl)
    #Pp=(ps+1)*np.log(ps+2)**2/2**(ps)
    #Pp[0]=2*Pp[1]
    #Pp=Pp/np.sum(Pp)
    #Pp=np.cumsum(Pp)
    #print(Pl, Pp)
    
    pfs=np.zeros((len(eLes),samples,int(T/d),dim))
    
    inputs=[]



    for i in range(len(eLes)):
        for sample in range(samples):
            l=eLes[i]
            eles=np.array([l,l+1])
            Pl_cum=np.array([1,1])
            enes=np.array([N0,2*N0])
            Pp_cum=np.array([1,1])
            collection_input=[T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,eles,Pl_cum,d,enes\
            ,Pp_cum,dim,resamp_coef,phi,g_den,g_par,Norm_Lambda,Lamb_par]
    
            inputs.append([samples*i+sample,collection_input])        

    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PSUCPF, inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eLes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample][0]
    
    #tel_est=np.reshape(np.array(pool_outputs),(len(eLes),samples,int(T/d),dim))
    
    #log_weightss=log_weightss.flatten()
    pfs=pfs.flatten()
    np.savetxt("Observations&data/SUCPF_ou_biasl_levels_T10_v1.txt",pfs,fmt="%f")




SUCPF_ou_lp_levels_T10_v8.txt 

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
    
    start=time.time()
    d=2**(0)
    resamp_coef=0.8    
    dim_out=2
    samples=100
    
    Lmax=10
    l0=0
    p=6
    N0=100
    eLes=np.arange(l0,Lmax+1)
    ps=np.arange(0,p+1)
    enes=N0*2**ps
    print(enes)
    #Pl=(eles+1)*np.log(eles+2)**2/2**(eles)
    #Pl[0]=2*Pl[1]
    #Pl=Pl/np.sum(Pl)
    #Pl=np.cumsum(Pl)
    #Pp=(ps+1)*np.log(ps+2)**2/2**(ps)
    #Pp[0]=2*Pp[1]
    #Pp=Pp/np.sum(Pp)
    #Pp=np.cumsum(Pp)
    #print(Pl, Pp)
    
    pfs=np.zeros((2,len(eLes),samples,int(T/d),dim))
    
    inputs=[]

    for i in range(len(eLes)):
        for sample in range(samples):
            l=eLes[i]
            eles=np.array([l,l+1])
            Pl_cum=np.array([0,1])

            Pp_cum=np.array([0,0,0,0,0,0,1])
            collection_input=[T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,eles,Pl_cum,d,enes\
            ,Pp_cum,dim,resamp_coef,phi,g_den,g_par,Norm_Lambda,Lamb_par]
    
            inputs.append([samples*i+sample+100000,collection_input])        
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PSUCPF, inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eLes)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample][0]


    #tel_est=np.reshape(np.array(pool_outputs),(len(eLes),samples,int(T/d),dim))
    #log_weightss=log_weightss.flatten()
    pfs=pfs.flatten()
    np.savetxt("Observations&data/SUCPF_ou_lp_levels_T10_v8.txt",pfs,fmt="%f")



SUCPF_ou_lp_levels_T10_v7.txt same experiment as before, we just changed 

  return [[phi_pfmean1B-phi_pfmean0B,phi_pfmean1A-phi_pfmean0A],[l,p]]



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
    
    start=time.time()
    d=2**(0)
    resamp_coef=0.8    
    dim_out=2
    samples=500
    
    Lmax=10
    l0=0
    p=4
    N0=125
    eLes=np.arange(l0,Lmax+1)
    ps=np.arange(0,p+1)
    enes=N0*2**ps
    print(enes)
    #Pl=(eles+1)*np.log(eles+2)**2/2**(eles)
    #Pl[0]=2*Pl[1]
    #Pl=Pl/np.sum(Pl)
    #Pl=np.cumsum(Pl)
    #Pp=(ps+1)*np.log(ps+2)**2/2**(ps)
    #Pp[0]=2*Pp[1]
    #Pp=Pp/np.sum(Pp)
    #Pp=np.cumsum(Pp)
    #print(Pl, Pp)
    
    pfs=np.zeros((2,len(eLes),samples,int(T/d),dim))
    
    inputs=[]

    for i in range(len(eLes)):
        for sample in range(samples):
            l=eLes[i]
            eles=np.array([l,l+1])
            Pl_cum=np.array([0,1])

            Pp_cum=np.array([0,0,0,0,1])
            collection_input=[T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,eles,Pl_cum,d,enes\
            ,Pp_cum,dim,resamp_coef,phi,g_den,g_par,Norm_Lambda,Lamb_par]
    
            inputs.append([samples*i+sample+100000,collection_input])        
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PSUCPF, inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eLes)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample][0]


    #tel_est=np.reshape(np.array(pool_outputs),(len(eLes),samples,int(T/d),dim))
    #log_weightss=log_weightss.flatten()
    pfs=pfs.flatten()
    np.savetxt("Observations&data/SUCPF_ou_lp_levels_T10_v7.txt",pfs,fmt="%f")



This was done in order to check why the sm in terms of l is not working properly.
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
    
    start=time.time()
    d=2**(0)
    resamp_coef=0.8    
    dim_out=2
    samples=500
    
    Lmax=10
    l0=0
    p=4
    N0=125
    eLes=np.arange(l0,Lmax+1)
    ps=np.arange(0,p+1)
    enes=N0*2**ps
    print(enes)
    #Pl=(eles+1)*np.log(eles+2)**2/2**(eles)
    #Pl[0]=2*Pl[1]
    #Pl=Pl/np.sum(Pl)
    #Pl=np.cumsum(Pl)
    #Pp=(ps+1)*np.log(ps+2)**2/2**(ps)
    #Pp[0]=2*Pp[1]
    #Pp=Pp/np.sum(Pp)
    #Pp=np.cumsum(Pp)
    #print(Pl, Pp)
    
    pfs=np.zeros((2,len(eLes),samples,int(T/d),dim))
    
    inputs=[]

    for i in range(len(eLes)):
        for sample in range(samples):
            l=eLes[i]
            eles=np.array([l,l+1])
            Pl_cum=np.array([0,1])

            Pp_cum=np.array([0,0,0,0,1])
            collection_input=[T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,eles,Pl_cum,d,enes\
            ,Pp_cum,dim,resamp_coef,phi,g_den,g_par,Norm_Lambda,Lamb_par]
    
            inputs.append([samples*i+sample+100000,collection_input])        
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PSUCPF, inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eLes)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample][0]


    #tel_est=np.reshape(np.array(pool_outputs),(len(eLes),samples,int(T/d),dim))
    #log_weightss=log_weightss.flatten()
    pfs=pfs.flatten()
    np.savetxt("Observations&data/SUCPF_ou_lp_levels_T10_v6.txt",pfs,fmt="%f")




SUCPF_ou_lp_levels_T10_v5.txt 


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
    
    
    start=time.time()
    d=2**(0)
    resamp_coef=0.8    
    dim_out=2
    samples=1000
    
    Lmax=10
    l0=0
    p=1
    N0=250
    eLes=np.arange(l0,Lmax+1)
    ps=np.arange(0,p+1)
    enes=N0*2**ps
    print(enes)
    #Pl=(eles+1)*np.log(eles+2)**2/2**(eles)
    #Pl[0]=2*Pl[1]
    #Pl=Pl/np.sum(Pl)
    #Pl=np.cumsum(Pl)
    #Pp=(ps+1)*np.log(ps+2)**2/2**(ps)
    #Pp[0]=2*Pp[1]
    #Pp=Pp/np.sum(Pp)
    #Pp=np.cumsum(Pp)
    #print(Pl, Pp)
    
    pfs=np.zeros((2,len(eLes),samples,int(T/d),dim))
    
    inputs=[]



    for i in range(len(eLes)):
        for sample in range(samples):
            l=eLes[i]
            eles=np.array([l,l+1])
            Pl_cum=np.array([0,1])

            Pp_cum=np.array([0,1])
            collection_input=[T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,eles,Pl_cum,d,enes\
            ,Pp_cum,dim,resamp_coef,phi,g_den,g_par,Norm_Lambda,Lamb_par]
    
            inputs.append([samples*i+sample+10000,collection_input])        

    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PSUCPF, inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eLes)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample][0]


    #tel_est=np.reshape(np.array(pool_outputs),(len(eLes),samples,int(T/d),dim))
    #log_weightss=log_weightss.flatten()
    pfs=pfs.flatten()
    np.savetxt("Observations&data/SUCPF_ou_lp_levels_T10_v5.txt",pfs,fmt="%f")



The return is return [[phi_pfmean0A-phi_pfmean0B,phi_pfmean1A-phi_pfmean1B],[l,p]


SUCPF_ou_scaled_lp_levels_T10_v4.txt this is made to compare with the rate already
obtained. 

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
    
    
    start=time.time()
    d=2**(0)
    resamp_coef=0.8    
    dim_out=2
    samples=1000
    
    Lmax=10
    l0=0
    p=1
    N0=250
    eLes=np.arange(l0,Lmax+1)
    ps=np.arange(0,p+1)
    enes=N0*2**ps
    print(enes)
    #Pl=(eles+1)*np.log(eles+2)**2/2**(eles)
    #Pl[0]=2*Pl[1]
    #Pl=Pl/np.sum(Pl)
    #Pl=np.cumsum(Pl)
    #Pp=(ps+1)*np.log(ps+2)**2/2**(ps)
    #Pp[0]=2*Pp[1]
    #Pp=Pp/np.sum(Pp)
    #Pp=np.cumsum(Pp)
    #print(Pl, Pp)
    
    pfs=np.zeros((2,len(eLes),samples,int(T/d),dim))
    
    inputs=[]



    for i in range(len(eLes)):
        for sample in range(samples):
            l=eLes[i]
            eles=np.array([l,l+1])
            Pl_cum=np.array([0,1])

            Pp_cum=np.array([1,1])
            collection_input=[T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,eles,Pl_cum,d,enes\
            ,Pp_cum,dim,resamp_coef,phi,g_den,g_par,Norm_Lambda,Lamb_par]
    
            inputs.append([samples*i+sample+10000,collection_input])        

    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PSUCPF, inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eLes)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample][0]


    #tel_est=np.reshape(np.array(pool_outputs),(len(eLes),samples,int(T/d),dim))
    #log_weightss=log_weightss.flatten()
    pfs=pfs.flatten()
    np.savetxt("Observations&data/SUCPF_ou_scaled_lp_levels_T10_v4.txt",pfs,fmt="%f")




SUCPF_ou_scaled_lp_levels_T10_v3.txt, this iteration is made in order to check the 
second moment in terms of l for the base levle in terms of the number of particles
    np.random.seed(7)
    T=10
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=13
    scale0=1/2**(5)
    collection_input=[]
    I=identity(dim).toarray()
    #comp_matrix = ortho_group.rvs(dim)
    comp_matrix=np.array([[1]])
    inv_mat=la.inv(comp_matrix)
    #S=diags(np.random.normal(1,0.1,dim),0).toarray()
    S=diags(np.random.normal(0.99,0.1,dim),0).toarray()
    #S=np.array([[1.]])
    fi=inv_mat@S@comp_matrix*np.sqrt(scale0)
    #B=diags(np.random.normal(-1,0.1,dim),0).toarray()
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=inv_mat@B@comp_matrix*scale0
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

    start=time.time()
    d=2**(0)
    resamp_coef=0.8    
    dim_out=2
    samples=1000
    
    Lmax=10
    l0=4
    p=1
    N0=250
    eLes=np.arange(l0,Lmax+1)
    ps=np.arange(0,p+1)
    enes=N0*2**ps
    print(enes)
    #Pl=(eles+1)*np.log(eles+2)**2/2**(eles)
    #Pl[0]=2*Pl[1]
    #Pl=Pl/np.sum(Pl)
    #Pl=np.cumsum(Pl)
    #Pp=(ps+1)*np.log(ps+2)**2/2**(ps)
    #Pp[0]=2*Pp[1]
    #Pp=Pp/np.sum(Pp)
    #Pp=np.cumsum(Pp)
    #print(Pl, Pp)
    
    pfs=np.zeros((len(eLes),samples,int(T/d),dim))
    
    inputs=[]



    for i in range(len(eLes)):
        for sample in range(samples):
            l=eLes[i]
            eles=np.array([l,l+1])
            Pl_cum=np.array([0,1])

            Pp_cum=np.array([1,1])
            collection_input=[T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,eles,Pl_cum,d,enes\
            ,Pp_cum,dim,resamp_coef,phi,g_den,g_par,Norm_Lambda,Lamb_par]
    
            inputs.append([samples*i+sample,collection_input])        

    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PSUCPF, inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eLes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample][0]
    
    #tel_est=np.reshape(np.array(pool_outputs),(len(eLes),samples,int(T/d),dim))
    
    #log_weightss=log_weightss.flatten()
    pfs=pfs.flatten()
    np.savetxt("Observations&data/SUCPF_ou_scaled_lp_levels_T10_v3.txt",pfs,fmt="%f")






SUCPF_ou_scaled_lp_levels_T10_v2.txt, this iteration is made because we don't 
have the desired rate of the second moment in terms of l. 


np.random.seed(7)
    T=10
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=13
    scale0=1/2**(5)
    collection_input=[]
    I=identity(dim).toarray()
    #comp_matrix = ortho_group.rvs(dim)
    comp_matrix=np.array([[1]])
    inv_mat=la.inv(comp_matrix)
    #S=diags(np.random.normal(1,0.1,dim),0).toarray()
    S=diags(np.random.normal(0.99,0.1,dim),0).toarray()
    #S=np.array([[1.]])
    fi=inv_mat@S@comp_matrix*np.sqrt(scale0)
    #B=diags(np.random.normal(-1,0.1,dim),0).toarray()
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=inv_mat@B@comp_matrix*scale0
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

    start=time.time()
    d=2**(0)
    resamp_coef=0.8    
    dim_out=2
    samples=100
    
    Lmax=10
    l0=4
    p=5
    N0=125
    eLes=np.arange(l0,Lmax+1)
    ps=np.arange(0,p+1)
    enes=N0*2**ps
    print(enes)
    #Pl=(eles+1)*np.log(eles+2)**2/2**(eles)
    #Pl[0]=2*Pl[1]
    #Pl=Pl/np.sum(Pl)
    #Pl=np.cumsum(Pl)
    #Pp=(ps+1)*np.log(ps+2)**2/2**(ps)
    #Pp[0]=2*Pp[1]
    #Pp=Pp/np.sum(Pp)
    #Pp=np.cumsum(Pp)
    #print(Pl, Pp)
    
    pfs=np.zeros((len(eLes),samples,int(T/d),dim))
    
    inputs=[]



    for i in range(len(eLes)):
        for sample in range(samples):
            l=eLes[i]
            eles=np.array([l,l+1])
            Pl_cum=np.array([0,1])

            Pp_cum=np.array([0,0,0,0,0,1])
            collection_input=[T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,eles,Pl_cum,d,enes\
            ,Pp_cum,dim,resamp_coef,phi,g_den,g_par,Norm_Lambda,Lamb_par]
    
            inputs.append([samples*i+sample,collection_input])        

    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PSUCPF, inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eLes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample][0]
    
    #tel_est=np.reshape(np.array(pool_outputs),(len(eLes),samples,int(T/d),dim))
    
    #log_weightss=log_weightss.flatten()
    pfs=pfs.flatten()
    np.savetxt("Observations&data/SUCPF_ou_scaled_lp_levels_T10_v2.txt",pfs,fmt="%f")





SUCPF_ou_scaled_lp_levels_T10_v1.txt this iteration was made in order to check 
the second moment of the telescoping estimator given that the rate that is 
showing is Delta_l^{2/3} instead of Delta_l. We scaled the process to obtain
"smaller" time steps and see if the relation holds there. 

np.random.seed(7)
    T=10
    dim=1
    dim_o=dim
    xin=np.zeros(dim)+1
    l=13
    scale0=1/2**(5)
    collection_input=[]
    I=identity(dim).toarray()
    #comp_matrix = ortho_group.rvs(dim)
    comp_matrix=np.array([[1]])
    inv_mat=la.inv(comp_matrix)
    #S=diags(np.random.normal(1,0.1,dim),0).toarray()
    S=diags(np.random.normal(0.99,0.1,dim),0).toarray()
    #S=np.array([[1.]])
    fi=inv_mat@S@comp_matrix*np.sqrt(scale0)
    #B=diags(np.random.normal(-1,0.1,dim),0).toarray()
    B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
    B=inv_mat@B@comp_matrix*scale0
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

    start=time.time()
    d=2**(0)
    resamp_coef=0.8    
    dim_out=2
    samples=1000
    
    Lmax=10
    l0=4
    p=2
    N0=125
    eLes=np.arange(l0,Lmax+1)
    ps=np.arange(0,p+1)
    enes=N0*2**ps
    print(enes)
    #Pl=(eles+1)*np.log(eles+2)**2/2**(eles)
    #Pl[0]=2*Pl[1]
    #Pl=Pl/np.sum(Pl)
    #Pl=np.cumsum(Pl)
    #Pp=(ps+1)*np.log(ps+2)**2/2**(ps)
    #Pp[0]=2*Pp[1]
    #Pp=Pp/np.sum(Pp)
    #Pp=np.cumsum(Pp)
    #print(Pl, Pp)
    
    pfs=np.zeros((len(eLes),samples,int(T/d),dim))
    
    inputs=[]



    for i in range(len(eLes)):
        for sample in range(samples):
            l=eLes[i]
            eles=np.array([l,l+1])
            Pl_cum=np.array([0,1])

            Pp_cum=np.array([0,0,1])
            collection_input=[T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,eles,Pl_cum,d,enes\
            ,Pp_cum,dim,resamp_coef,phi,g_den,g_par,Norm_Lambda,Lamb_par]
    
            inputs.append([samples*i+sample,collection_input])        

    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PSUCPF, inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eLes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample][0]
    
    #tel_est=np.reshape(np.array(pool_outputs),(len(eLes),samples,int(T/d),dim))
    
    #log_weightss=log_weightss.flatten()
    pfs=pfs.flatten()
    np.savetxt("Observations&data/SUCPF_ou_scaled_lp_levels_T10_v1.txt",pfs,fmt="%f")



SUCPF_ou_pl0_levels_T10_v1.txt this iteration is made in order to compute 
cp0 and cl0 


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

    start=time.time()
    d=2**(0)
    resamp_coef=0.8    
    dim_out=2
    samples=1000
    
    Lmax=6
    l0=5
    p=8
    N0=4
    eLes=np.arange(l0,Lmax+1)
    ps=np.arange(0,p+1)
    eNes=N0*2**ps
    print(eNes)
    #Pl=(eles+1)*np.log(eles+2)**2/2**(eles)
    #Pl[0]=2*Pl[1]
    #Pl=Pl/np.sum(Pl)
    #Pl=np.cumsum(Pl)
    #Pp=(ps+1)*np.log(ps+2)**2/2**(ps)
    #Pp[0]=2*Pp[1]
    #Pp=Pp/np.sum(Pp)
    #Pp=np.cumsum(Pp)
    #print(Pl, Pp)    
    pfs=np.zeros((len(eNes),samples,int(T/d),dim))    
    inputs=[]

    for i in range(len(eNes)):
        for sample in range(samples):
            N=eNes[i]
            
            Pl_cum=np.array([1,1])
            
            enes=N*np.array([1,2])
            Pp_cum=np.array([1,1])
            collection_input=[T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,eLes,Pl_cum,d,enes\
            ,Pp_cum,dim,resamp_coef,phi,g_den,g_par,Norm_Lambda,Lamb_par]
    
            inputs.append([samples*i+sample,collection_input])        

    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PSUCPF, inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eNes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample][0]
    
    #tel_est=np.reshape(np.array(pool_outputs),(len(eLes),samples,int(T/d),dim))
    
    #log_weightss=log_weightss.flatten()
    pfs=pfs.flatten()
    np.savetxt("Observations&data/SUCPF_ou_pl0_levels_T10_v1.txt",pfs,fmt="%f")



SUCPF_ou_lp_levels_T10_v3.txt is made in order to check why the variance 
of the telescoping levels does not have the correct rate. 

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

    start=time.time()
    d=2**(0)
    resamp_coef=0.8    
    dim_out=2
    samples=1000
    
    Lmax=10
    l0=0
    p=2
    N0=125
    eLes=np.arange(l0,Lmax+1)
    ps=np.arange(0,p+1)
    enes=N0*2**ps
    print(enes)
    #Pl=(eles+1)*np.log(eles+2)**2/2**(eles)
    #Pl[0]=2*Pl[1]
    #Pl=Pl/np.sum(Pl)
    #Pl=np.cumsum(Pl)
    #Pp=(ps+1)*np.log(ps+2)**2/2**(ps)
    #Pp[0]=2*Pp[1]
    #Pp=Pp/np.sum(Pp)
    #Pp=np.cumsum(Pp)
    #print(Pl, Pp)
    
    pfs=np.zeros((len(eLes),samples,int(T/d),dim))
    
    inputs=[]



    for i in range(len(eLes)):
        for sample in range(samples):
            l=eLes[i]
            eles=np.array([l,l+1])
            Pl_cum=np.array([0,1])

            Pp_cum=np.array([0,0,1])
            collection_input=[T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,eles,Pl_cum,d,enes\
            ,Pp_cum,dim,resamp_coef,phi,g_den,g_par,Norm_Lambda,Lamb_par]
    
            inputs.append([samples*i+sample,collection_input])        

    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PSUCPF, inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eLes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample][0]
    
    #tel_est=np.reshape(np.array(pool_outputs),(len(eLes),samples,int(T/d),dim))
    
    #log_weightss=log_weightss.flatten()
    pfs=pfs.flatten()
    np.savetxt("Observations&data/SUCPF_ou_lp_levels_T10_v3.txt",pfs,fmt="%f")



SUCPF_ou_lp_levels_T10_v2.txt
This iteration is done to check why the SUCPF_ou_lp_levels_T10_v1.txt and
SUCPF_ou_l_levels_T10_v1.txt have the same second moment. Answer, there was 
a confusion and the same second moment was being computed.  
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

    start=time.time()
    d=2**(0)
    resamp_coef=0.8    
    dim_out=2
    samples=1000
    
    Lmax=8
    l0=0
    p=1
    N0=250
    eLes=np.arange(l0,Lmax+1)
    ps=np.arange(0,p+1)
    enes=N0*2**ps
    print(enes)
    #Pl=(eles+1)*np.log(eles+2)**2/2**(eles)
    #Pl[0]=2*Pl[1]
    #Pl=Pl/np.sum(Pl)
    #Pl=np.cumsum(Pl)
    #Pp=(ps+1)*np.log(ps+2)**2/2**(ps)
    #Pp[0]=2*Pp[1]
    #Pp=Pp/np.sum(Pp)
    #Pp=np.cumsum(Pp)
    #print(Pl, Pp)
    
    pfs=np.zeros((len(eLes),samples,int(T/d),dim))
    
    inputs=[]


    for i in range(len(eLes)):
        for sample in range(samples):
            l=eLes[i]
            eles=np.array([l,l+1])
            Pl_cum=np.array([0,1])
            enes=np.array([N0,2*N0])
            Pp_cum=np.array([0,1])
            collection_input=[T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,eles,Pl_cum,d,enes\
            ,Pp_cum,dim,resamp_coef,phi,g_den,g_par,Norm_Lambda,Lamb_par]

            inputs.append([samples*i+sample,collection_input])        
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PSUCPF, inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eLes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample][0]
    
    #tel_est=np.reshape(np.array(pool_outputs),(len(eLes),samples,int(T/d),dim))
    
    #log_weightss=log_weightss.flatten()
    pfs=pfs.flatten()
    np.savetxt("Observations&data/SUCPF_ou_lp_levels_T10_v2.txt",pfs,fmt="%f")


SUCPF_ou_l14_T10_v1
 This iteeration was made in order to find a PF with low variance, N=250 and large
 l=11 in order to control the bias. 
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

    start=time.time()
    d=2**(0)
    resamp_coef=0.8    
    dim_out=2
    samples=2000
    L=12
    l0=11
    p=1
    N0=250
    eLes=np.arange(l0,L+1)
    ps=np.arange(0,p+1)
    eNes=N0*2**ps
    print(eNes)
    #Pl=(eles+1)*np.log(eles+2)**2/2**(eles)
    #Pl[0]=2*Pl[1]
    #Pl=Pl/np.sum(Pl)
    #Pl=np.cumsum(Pl)
    #Pp=(ps+1)*np.log(ps+2)**2/2**(ps)
    #Pp[0]=2*Pp[1]
    #Pp=Pp/np.sum(Pp)
    #Pp=np.cumsum(Pp)
    #print(Pl, Pp)

    pfs=np.zeros((samples,int(T/d),dim))
    
    inputs=[]
    


    for pit in range(len(eNes)-1):
        for sample in range(samples):
        
        
            Pl_cum=np.array([1,1])
            Pp_cum=np.array([1,1])
            
            
            
            collection_input=[T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,eLes,Pl_cum,d,eNes\
            ,Pp_cum,dim,resamp_coef,phi,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([samples*pit+sample,collection_input])        
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PSUCPF, inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()

    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eNes)-1):
        for sample in range(samples):
            pfs[sample]=pool_outputs[i*samples+sample][0]
            #tel_est=np.reshape(np.array(pool_outputs),(len(eLes),samples,int(T/d),dim))

    #log_weightss=log_weightss.flatten()
    pfs=pfs.flatten()
    np.savetxt("Observations&data/SUCPF_ou_l14_T10_v1.txt",pfs,fmt="%f") 




SUCPF_ou_pl_levels_T10_v1.txt corresponds to the computation of the constant 
Cp in order to bound the variance. 

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

    start=time.time()
    d=2**(0)
    resamp_coef=0.8    
    dim_out=2
    samples=1000
    L=6
    l0=5
    p=8
    N0=4
    eLes=np.arange(l0,L+1)
    ps=np.arange(0,p+1)
    eNes=N0*2**ps
    print(eNes)
    #Pl=(eles+1)*np.log(eles+2)**2/2**(eles)
    #Pl[0]=2*Pl[1]
    #Pl=Pl/np.sum(Pl)
    #Pl=np.cumsum(Pl)
    #Pp=(ps+1)*np.log(ps+2)**2/2**(ps)
    #Pp[0]=2*Pp[1]
    #Pp=Pp/np.sum(Pp)
    #Pp=np.cumsum(Pp)
    #print(Pl, Pp)

    pfs=np.zeros((2,len(eNes)-1,samples,int(T/d),dim))
    
    inputs=[]
    


    for pit in range(len(eNes)-1):
        for sample in range(samples):
        
        
            Pl_cum=np.array([0,1])
            
            Pp_cum=np.concatenate((np.zeros(pit+1),np.zeros(p-pit)+1))
            
            collection_input=[T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,eLes,Pl_cum,d,eNes\
            ,Pp_cum,dim,resamp_coef,phi,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([samples*pit+sample,collection_input])        
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PSUCPF, inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()

    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eNes)-1):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample][0]
            #tel_est=np.reshape(np.array(pool_outputs),(len(eLes),samples,int(T/d),dim))

    #log_weightss=log_weightss.flatten()
    pfs=pfs.flatten()
    np.savetxt("Observations&data/SUCPF_ou_pl_levels_T10_v1.txt",pfs,fmt="%f") 






SUCPF_ou_lp_levels_T10_v1.txt corresponds to the variation of the time discretiztion
levels l in order to find the constant Cl.
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

    start=time.time()
    d=2**(0)
    resamp_coef=0.8    
    dim_out=2
    samples=1000
    
    Lmax=8
    l0=0
    p=1
    N0=250
    eLes=np.arange(l0,Lmax+1)
    ps=np.arange(0,p+1)
    enes=N0*2**ps
    print(enes)
    #Pl=(eles+1)*np.log(eles+2)**2/2**(eles)
    #Pl[0]=2*Pl[1]
    #Pl=Pl/np.sum(Pl)
    #Pl=np.cumsum(Pl)
    #Pp=(ps+1)*np.log(ps+2)**2/2**(ps)
    #Pp[0]=2*Pp[1]
    #Pp=Pp/np.sum(Pp)
    #Pp=np.cumsum(Pp)
    #print(Pl, Pp)
    
    pfs=np.zeros((len(eLes),samples,int(T/d),dim))
    
    inputs=[]



    for i in range(len(eLes)):
        for sample in range(samples):
            l=eLes[i]
            eles=np.array([l,l+1])
            Pl_cum=np.array([0,1])
            enes=np.array([N0,2*N0])
            Pp_cum=np.array([0,1])
            collection_input=[T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,eles,Pl_cum,d,enes\
            ,Pp_cum,dim,resamp_coef,phi,g_den,g_par,Norm_Lambda,Lamb_par]
    
            inputs.append([samples*i+sample,collection_input])        

    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PSUCPF, inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eLes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample][0]
    
    #tel_est=np.reshape(np.array(pool_outputs),(len(eLes),samples,int(T/d),dim))
    
    #log_weightss=log_weightss.flatten()
    pfs=pfs.flatten()
    np.savetxt("Observations&data/SUCPF_ou_lp_levels_T10_v1.txt",pfs,fmt="%f")


SUCPF_ou_l_levels_T10_v1.txt 
 This iteration is made in order to compute the constants deltaL and C0L that
 control the bias and variance related to the time discretization. These
 constants will be used as hyperparameters of the unbiased estimators.
 IMPORTANT: there is a modification to the function PSUCPF, the output is changed 
 to [[phi_pfmean0,phi_pfmean1],[l,p]]



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

    start=time.time()
    d=2**(0)
    resamp_coef=0.8    
    dim_out=2
    samples=1000
    
    Lmax=8
    l0=0
    p=1
    N0=250
    eLes=np.arange(l0,Lmax+1)
    ps=np.arange(0,p+1)
    enes=N0*2**ps
    print(enes)
    #Pl=(eles+1)*np.log(eles+2)**2/2**(eles)
    #Pl[0]=2*Pl[1]
    #Pl=Pl/np.sum(Pl)
    #Pl=np.cumsum(Pl)
    #Pp=(ps+1)*np.log(ps+2)**2/2**(ps)
    #Pp[0]=2*Pp[1]
    #Pp=Pp/np.sum(Pp)
    #Pp=np.cumsum(Pp)
    #print(Pl, Pp)
    
    pfs=np.zeros((2,len(eLes),samples,int(T/d),dim))
    
    inputs=[]



    for i in range(len(eLes)):
        for sample in range(samples):
            l=eLes[i]
            eles=np.array([l,l+1])
            Pl_cum=np.array([0,1])
            enes=np.array([N0,2*N0])
            Pp_cum=np.array([1,1])
            collection_input=[T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,eles,Pl_cum,d,enes\
            ,Pp_cum,dim,resamp_coef,phi,g_den,g_par,Norm_Lambda,Lamb_par]
    
            inputs.append([samples*i+sample,collection_input])        

    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PSUCPF, inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eLes)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample][0]
    
    #tel_est=np.reshape(np.array(pool_outputs),(len(eLes),samples,int(T/d),dim))
    
    #log_weightss=log_weightss.flatten()
    pfs=pfs.flatten()
    np.savetxt("Observations&data/SUCPF_ou_l_levels_T10_v1.txt",pfs,fmt="%f")





SUCPF_ou_p_levels_T10_v1.txt 
 This iteration is made in order to compute the constants deltaP and Cp0 that
 control the bias and variance related to the number of particles. These
 constants will be used as hyperparameters of the unbiased estimators.
 IMPORTANT: there is a modification to the function PSUCPF, the output is changed 
 to [[phi_pfmeanB,phi_pfmeanA],[l,p]]

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

start=time.time()
d=2**(0)
resamp_coef=0.8    
dim_out=2
samples=1000
L=6
l0=5
p=8
N0=4
eLes=np.arange(l0,L+1)
ps=np.arange(0,p+1)
eNes=N0*2**ps
print(eNes)
#Pl=(eles+1)*np.log(eles+2)**2/2**(eles)
#Pl[0]=2*Pl[1]
#Pl=Pl/np.sum(Pl)
#Pl=np.cumsum(Pl)
#Pp=(ps+1)*np.log(ps+2)**2/2**(ps)
#Pp[0]=2*Pp[1]
#Pp=Pp/np.sum(Pp)
#Pp=np.cumsum(Pp)
#print(Pl, Pp)

pfs=np.zeros((2,len(eNes)-1,samples,int(T/d),dim))

inputs=[]



for pit in range(len(eNes)-1):
    for sample in range(samples):        
        
        Pl_cum=np.array([1,1])
        Pp_cum=np.concatenate((np.zeros(pit+1),np.zeros(p-pit)+1))
        collection_input=[T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,eLes,Pl_cum,d,eNes\
        ,Pp_cum,dim,resamp_coef,phi,g_den,g_par,Norm_Lambda,Lamb_par]
        inputs.append([samples*pit+sample,collection_input])        
pool = multiprocessing.Pool(processes=10)
pool_outputs = pool.map(PSUCPF, inputs)
pool.close()
pool.join()
#blocks_pools.append(pool_outputs)
xend1=time.time()
end=time.time()

print("Parallelized processes time:",end-start,"\n")            
for i in range(len(eNes)-1):
    for sample in range(samples):
        pfs[:,i,sample]=pool_outputs[i*samples+sample][0]
#tel_est=np.reshape(np.array(pool_outputs),(len(eLes),samples,int(T/d),dim))

#log_weightss=log_weightss.flatten()
pfs=pfs.flatten()
np.savetxt("Observations&data/SUCPF_ou_p_levels_T10_v1.txt",pfs,fmt="%f") 






PMLPF_cox_ou100_v1.txt


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
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=0
    Lmin=0
    Lmax=6
    eLes=np.array(np.arange(Lmin,Lmax+1))
    print(eLes)
    C=0.8
    C0=0.5
    K=0.0012
    pfs=np.zeros((len(eLes),samples,int(T/d),dim))
    inputs=[]
    
    for i in range(len(eLes)):
        for sample in range(samples):
            L=eLes[i]
            NB0=(np.sqrt(C0)+np.sqrt(3*C/2)*L)*np.sqrt(2*C/3)/(K*L)
            N0=np.sqrt(C0)*(np.sqrt(C0)+np.sqrt(3*C/2)*L)*2**(2*L)/K
            eles=np.array(np.arange(l0,L+1))
            eNes=np.zeros(len(eles),dtype=int)
            scale=5e-2
            eNes[0]=N0*scale
            eNes[1:]=NB0*2**(L*2)*L*scale/2**(eles[1:])
            collection_input=[T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,eles,d,eNes,dim,resamp_coef,phi,\
            dim_out,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([samples*i+sample,collection_input])        
    

        
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PMLPF_cox,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]

    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/PMLPF_cox_ou100_v1.txt",pfs,fmt="%f") 




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
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=1000
    l0=1
    L=10
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=5000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
            
    pool = multiprocessing.Pool(processes=8)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_ou100_v1.txt",pfs,fmt="%f")







In this iteration of the MLPF we have fixed the issue with the number of particles
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
start=time.time()
d=2**(0)
resamp_coef=0.8
dim_out=2
g_par=cov
samples=100
l0=0
Lmin=0
Lmax6
eLes=np.array(np.arange(Lmin,Lmax+1))
print(eLes)
C=2
C0=0.5 
K=0.021
pfs=np.zeros((len(eLes),samples,int(T/d),dim))
inputs=[]

for i in range(len(eLes)):
    for sample in range(samples):
        L=eLes[i]
            NB0=(np.sqrt(C0)+np.sqrt(3*C/2)*L)*np.sqrt(2*C/3)/(K*L)
            N0=np.sqrt(C0)*(np.sqrt(C0)+np.sqrt(3*C/2)*L)*2**(2*L)/K
            eles=np.array(np.arange(l0,L+1))
            eNes=np.zeros(len(eles),dtype=int)
            scale=3e-1
            eNes[0]=N0*scale
            eNes[1:]=NB0*2**(L*2)*L*scale/2**(eles[1:])
        collection_input=[T,xin,b_lan,df,Sig_lan,fi,obs,obs_time,eles,d,eNes,dim,resamp_coef,phi,\
        dim_out,g_den,g_par,Norm_Lambda,Lamb_par]
        inputs.append([samples*i+sample,collection_input])        
    

        
pool = multiprocessing.Pool(processes=10)
pool_outputs = pool.map(PMLPF_cox,inputs)
pool.close()
pool.join()
#blocks_pools.append(pool_outputs)
xend1=time.time()
end=time.time()
print("Parallelized processes time:",end-start,"\n")            
for i in range(len(eles)):
    for sample in range(samples):
        pfs[i,sample]=pool_outputs[i*samples+sample]

pfs=pfs.flatten()
    
np.savetxt("Observations&data/PMLPF_cox_lan100_v3.txt",pfs,fmt="%f")



The following settings haven't been  applied yet

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
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=1000
    l0=1
    L=10
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=5000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   
            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_ou100_v1.txt",pfs,fmt="%f")



PMLPF_cox_lan100_v2.txt 

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
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=0
    Lmin=0
    Lmax=6
    eLes=np.array(np.arange(Lmin,Lmax+1))
    print(eLes)
    C=9
    C0=0.5 
    K=0.021
    pfs=np.zeros((len(eLes),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eLes)):
        for sample in range(samples):
            L=eLes[i]
            NB0=(np.sqrt(C0)+np.sqrt(3*C/2)*L)*np.sqrt(2*C/3)/(K*L)
            N0=np.sqrt(C0)*(np.sqrt(C0)+np.sqrt(3*C/2)*L)*2**(2*L)/K
            eles=np.array(np.arange(l0,L+1))
            eNes=np.zeros(len(eles),dtype=int)
            scale=2e-1
            eNes[0]=N0*scale
            eNes[1:]=NB0*2**(-eles[1:]*3/4)*2**(L*(2+1/4))*scale
            collection_input=[T,xin,b_lan,df,Sig_lan,fi,obs,obs_time,eles,d,eNes,dim,resamp_coef,phi,\
            dim_out,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([samples*i+sample,collection_input])        
        

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PMLPF_cox,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/PMLPF_cox_lan100_v2.txt",pfs,fmt="%f")



PMLPF_cox_nldt100_v1.txt is computed in order to get the MLPF_cox for the 
nonlinear diffusion coefficient 

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
    x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=2/9
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=0
    Lmin=0
    Lmax=6
    eLes=np.array(np.arange(Lmin,Lmax+1))
    print(eLes)
    C0=0.439
    C=0.45463086447
    K=0.00017

    pfs=np.zeros((len(eLes),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eLes)):
        for sample in range(samples):
            L=eLes[i]
            CL=2**(1/4)*(2**(L/4)-1)/(2**(1/4)-1)
            NB0=(np.sqrt(C0)+np.sqrt(3*C/2)*CL)*np.sqrt(2*C/3)*2**(-L/4)/K
            N0=np.sqrt(C0)*(np.sqrt(C0)+np.sqrt(3*C/2)*CL)*2**(2*L)/K
            eles=np.array(np.arange(l0,L+1))
            eNes=np.zeros(len(eles),dtype=int)
            a=5e-3
            eNes[0]=N0*a
            eNes[1:]=NB0*2**(-eles[1:]*3/4)*2**(L*(2+1/4))*a
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,eles,d,eNes,dim,resamp_coef,phi,\
            dim_out,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([samples*i+sample,collection_input])        
        

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PMLPF_cox,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/PMLPF_cox_nldt100_v1.txt",pfs,fmt="%f")


CCPF_nldt100_enes_v1.txt
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
    x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=2/9
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    p=8
    N0=10
    enes=np.array([2**i for i in range(p+1)])*N0
    pfs=np.zeros((2,len(enes),samples,int(T/d),dim))
    inputs=[]
    l=5
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_nldt100_enes_v1.txt",pfs,fmt="%f")


CCPF_nldt100_v1.txt



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
    x_true=gen_gen_data(T,xin,l,collection_input)
    Lamb_par=2/9
    [obs_time,obs]=gen_obs(x_true,l,T,Norm_Lambda,Lamb_par,g_normal,g_pars)
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=500
    l0=1
    L=10
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=5000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_nldt100_v1.txt",pfs,fmt="%f")


CCPF_nldt10_v1.txt is the first file with experiments of the nonlienear difussion,
the settings of such experiment is 

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
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=500
    l0=1
    L=10
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=5000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_ou,A,Sig_nldt,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_nldt10_v1.txt",pfs,fmt="%f")


CCPF_lan100_v2.txt this iteration corresponds to the extension of time levels
in order to observe the strong error of the coupling.


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
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=11
    L=12
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=5000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan,df,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_lan100_v2.txt",pfs,fmt="%f")


CCPF_lan100_enes_v1.txt 

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
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    p=8
    N0=10
    enes=np.array([2**i for i in range(p+1)])*N0
    pfs=np.zeros((2,len(enes),samples,int(T/d),dim))
    inputs=[]
    l=5
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan,df,Sig_lan,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_lan100_enes_v1.txt",pfs,fmt="%f")    



CCPF_lan100_v1.txt
This iteration is made in order to get the constants in order to apply the 
MLPF.

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
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=1000
    l0=1
    L=10
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=5000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan,df,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_lan100_v1.txt",pfs,fmt="%f")


Cox_PF_lan20_v1.txt was made in order to obtain the constant C 


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
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    p=8
    
    N0=10
    enes=np.array([2**i for i in range(p+1)])*N0
    pfs=np.zeros((len(enes),samples,int(T/d),dim))
    inputs=[]
    l=5
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan,df,Sig_lan,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/Cox_PF_lan20_v1.txt",pfs,fmt="%f")


CCPF_lan20_v1.txt in the following we compute samples of the 
couple particle filter with cox observation wiht a langevin signal 
wiht student's  t distribution. This is made in order to check the rates 
of bias and variance of the coupling and in order to get the constant of hte 
MLPF.



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
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=500
    l0=1
    L=10
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=5000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_lan,df,Sig_lan,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_lan20_v1.txt",pfs,fmt="%f")



CCPF_gbm100_p_i_v1 is for the constants computation for the CCPF,

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
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=500
    l0=1
    L=10
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=5000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_gbm100_p_i_v1.txt",pfs,fmt="%f")



Cox_PF_gbm100_p_i_v1 with this test we check the monte carlo variance in order to 
compute the necessary constants. 

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
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=200
    #l0=1
    #L=10
    l=5
    N0=10
    p=10
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    enes=N0*np.array([2**i for i in range(p)])

    pfs=np.zeros((len(enes),samples,int(T/d),dim))
    inputs=[]
    
    
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,n,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])   

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/Cox_PF_gbm100_p_i_v1.txt",pfs,fmt="%f")





PMLPF_cox_gbm20_p_i_v1 corresponds to the following settings. This iteration of 
the code is made with the independent sampling of the particles, ergo the i in
name of the data file.


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
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=0
    Lmin=0
    Lmax=5
    eLes=np.array(np.arange(Lmin,Lmax+1))
    print(eLes)
    C=20
    C0=11
    K=0.0072
    pfs=np.zeros((len(eLes),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eLes)):
        for sample in range(samples):
            L=eLes[i]
            CL=2**(1/4)*(2**(L/4)-1)/(2**(1/4)-1)
            NB0=(np.sqrt(C0)+np.sqrt(3*C/2)*CL)*np.sqrt(2*C/3)*2**(-L/4)/K
            N0=np.sqrt(C0)*(np.sqrt(C0)+np.sqrt(3*C/2)*CL)*2**(2*L)/K
            eles=np.array(np.arange(l0,L+1))
            eNes=np.zeros(len(eles),dtype=int)
            a=3e-2
            eNes[0]=N0*a
            eNes[1:]=NB0*2**(-eles[1:]*3/4)*2**(L*(2+1/4))*a
            collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,eles,d,eNes,dim,resamp_coef,phi,\
            dim_out,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([samples*i+sample,collection_input])        
        

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PMLPF_cox,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/PMLPF_cox_gbm20_p_i_v1.txt",pfs,fmt="%f")A



Cox_pf_gbm_enes_v11 and ,CCPF_gbm_enes_v11

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
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=200
    l=4
    l0=0
    Lmin=0
    Lmax=5
    a=3e-3
    
    C=20
    C0=11
    K=0.0072
    ELES=np.array(np.arange(1,6))
    print(ELES)
    CL=2**(1/4)*(2**(ELES/4)-1)/(2**(1/4)-1)
    NB0=(np.sqrt(C0)+np.sqrt(3*C/2)*CL)*np.sqrt(2*C/3)*2**(-ELES/4)/K

    enes=np.array(a*NB0*2**(-1*3/4)*2**(ELES*(2+1/4)) ,dtype=int)
    print(enes)
    pfs=np.zeros((2,len(enes),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(enes)):
        for sample in range(samples):
            n=enes[i]
            collection_input=(T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,n,dim,\
            resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par)
            inputs.append([samples*i+sample+1000,collection_input])
            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
            
    pfs=pfs.flatten()

    np.savetxt("Observations&data/CCPF_gbm_enes_v11.txt",pfs,fmt="%f")
    
    start=time.time()

    pfs=np.zeros((len(enes),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(enes)):
        for sample in range(samples):
            n=enes[i]
            collection_input=(T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,n,dim,\
            resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par)
            inputs.append([samples*i+sample+1000,collection_input])
            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
            
    pfs=pfs.flatten()

    np.savetxt("Observations&data/Cox_pf_gbm_enes_v11.txt",pfs,fmt="%f")



Cox_pf_gbm_enes_v10 and CCPF_gbm_enes_v10 correspond to the version in
which the collective resampling issue is resolved. A independant sampling of 
the particles is made. 

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
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=200
    l=2
    l0=0
    Lmin=0
    Lmax=5
    a=3e-2
    
    C=20
    C0=11
    K=0.0072
    ELES=np.array(np.arange(1,6))
    print(ELES)
    CL=2**(1/4)*(2**(ELES/4)-1)/(2**(1/4)-1)
    NB0=(np.sqrt(C0)+np.sqrt(3*C/2)*CL)*np.sqrt(2*C/3)*2**(-ELES/4)/K

    enes=np.array(a*NB0*2**(-1*3/4)*2**(ELES*(2+1/4)) ,dtype=int)
    print(enes)
    pfs=np.zeros((2,len(enes),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(enes)):
        for sample in range(samples):
            n=enes[i]
            collection_input=(T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,n,dim,\
            resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par)
            inputs.append([samples*i+sample+1000,collection_input])
            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
            
    pfs=pfs.flatten()

    np.savetxt("Observations&data/CCPF_gbm_enes_v10.txt",pfs,fmt="%f")
    
    start=time.time()

    pfs=np.zeros((len(enes),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(enes)):
        for sample in range(samples):
            n=enes[i]
            collection_input=(T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,n,dim,\
            resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par)
            inputs.append([samples*i+sample+1000,collection_input])
            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
            
    pfs=pfs.flatten()

    np.savetxt("Observations&data/Cox_pf_gbm_enes_v10.txt",pfs,fmt="%f")



Cox_pf_gbm_enes_v8 and CCPF_gbm_enes_v8 this test was made in order to have a 
test the max_coup_multi_gpt function which implements a little change in the 
computation of r



Cox_pf_gbm_enes_v8 and CCPF_gbm_enes_v8
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
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=200
    l=5
    l0=0
    Lmin=0
    Lmax=5
    a=3e-4
    
    C=20
    C0=11
    K=0.0072
    ELES=np.array(np.arange(1,6))
    print(ELES)
    CL=2**(1/4)*(2**(ELES/4)-1)/(2**(1/4)-1)
    NB0=(np.sqrt(C0)+np.sqrt(3*C/2)*CL)*np.sqrt(2*C/3)*2**(-ELES/4)/K

    enes=np.array(a*NB0*2**(-1*3/4)*2**(ELES*(2+1/4)) ,dtype=int)
    print(enes)
    pfs=np.zeros((2,len(enes),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(enes)):
        for sample in range(samples):
            n=enes[i]
            collection_input=(T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,n,dim,\
            resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par)
            inputs.append([samples*i+sample+1000,collection_input])
            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
            
    pfs=pfs.flatten()

    np.savetxt("Observations&data/CCPF_gbm_enes_v8.txt",pfs,fmt="%f")
    
    start=time.time()

    pfs=np.zeros((len(enes),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(enes)):
        for sample in range(samples):
            n=enes[i]
            collection_input=(T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,n,dim,\
            resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par)
            inputs.append([samples*i+sample+1000,collection_input])
            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
            
    pfs=pfs.flatten()

    np.savetxt("Observations&data/Cox_pf_gbm_enes_v8.txt",pfs,fmt="%f")


CCPF_gbm_enes_v7 Cox_gbm_enes_v7 in this test we change PCCPF to emulate more Cox_pf

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
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=200
    l=2
    l0=0
    Lmin=0
    Lmax=5
    a=3e-2
    
    C=20
    C0=11
    K=0.0072
    ELES=np.array(np.arange(1,6))
    print(ELES)
    CL=2**(1/4)*(2**(ELES/4)-1)/(2**(1/4)-1)
    NB0=(np.sqrt(C0)+np.sqrt(3*C/2)*CL)*np.sqrt(2*C/3)*2**(-ELES/4)/K

    enes=np.array(a*NB0*2**(-1*3/4)*2**(ELES*(2+1/4)) ,dtype=int)
    print(enes)
    pfs=np.zeros((2,len(enes),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(enes)):
        for sample in range(samples):
            n=enes[i]
            collection_input=(T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,n,dim,\
            resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par)
            inputs.append([samples*i+sample+1000,collection_input])
            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
            
    pfs=pfs.flatten()

    np.savetxt("Observations&data/CCPF_gbm_enes_v7.txt",pfs,fmt="%f")
    
    start=time.time()

    pfs=np.zeros((len(enes),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(enes)):
        for sample in range(samples):
            n=enes[i]
            collection_input=(T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,n,dim,\
            resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par)
            inputs.append([samples*i+sample+1000,collection_input])
            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
            
    pfs=pfs.flatten()

    np.savetxt("Observations&data/Cox_pf_gbm_enes_v7.txt",pfs,fmt="%f")

Cox_pf_gbm_enes_v6 and CCPF_gbm_enes_v6

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
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=200
    l=2
    l0=0
    Lmin=0
    Lmax=5
    a=3e-2
    
    C=20
    C0=11
    K=0.0072
    ELES=np.array(np.arange(1,6))
    print(ELES)
    CL=2**(1/4)*(2**(ELES/4)-1)/(2**(1/4)-1)
    NB0=(np.sqrt(C0)+np.sqrt(3*C/2)*CL)*np.sqrt(2*C/3)*2**(-ELES/4)/K

    enes=np.array(a*NB0*2**(-1*3/4)*2**(ELES*(2+1/4)) ,dtype=int)
    print(enes)
    pfs=np.zeros((2,len(enes),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(enes)):
        for sample in range(samples):
            n=enes[i]
            collection_input=(T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,n,dim,\
            resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par)
            inputs.append([samples*i+sample+1000,collection_input])
            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
            
    pfs=pfs.flatten()

    np.savetxt("Observations&data/CCPF_gbm_enes_v6.txt",pfs,fmt="%f")
    
    start=time.time()

    pfs=np.zeros((len(enes),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(enes)):
        for sample in range(samples):
            n=enes[i]
            collection_input=(T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,n,dim,\
            resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par)
            inputs.append([samples*i+sample+1000,collection_input])
            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
            
    pfs=pfs.flatten()

    np.savetxt("Observations&data/Cox_pf_gbm_enes_v6.txt",pfs,fmt="%f")




Cox_pf_gbm_enes_v5 and CCPF_gbm_enes_v5 in this iteration we have a 
different version of the PCCPF in which we comment some lines that 
Cox_pf dont have 
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
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=200
    l=2
    l0=0
    Lmin=0
    Lmax=5
    a=3e-2
    
    C=20
    C0=11
    K=0.0072
    ELES=np.array(np.arange(1,6))
    print(ELES)
    CL=2**(1/4)*(2**(ELES/4)-1)/(2**(1/4)-1)
    NB0=(np.sqrt(C0)+np.sqrt(3*C/2)*CL)*np.sqrt(2*C/3)*2**(-ELES/4)/K

    enes=np.array(a*NB0*2**(-1*3/4)*2**(ELES*(2+1/4)) ,dtype=int)
    print(enes)
    pfs=np.zeros((2,len(enes),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(enes)):
        for sample in range(samples):
            n=enes[i]
            collection_input=(T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,n,dim,\
            resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par)
            inputs.append([samples*i+sample+1000,collection_input])
            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
            
    pfs=pfs.flatten()

    np.savetxt("Observations&data/CCPF_gbm_enes_v5.txt",pfs,fmt="%f")
    
    start=time.time()

    pfs=np.zeros((len(enes),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(enes)):
        for sample in range(samples):
            n=enes[i]
            collection_input=(T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,n,dim,\
            resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par)
            inputs.append([samples*i+sample+1000,collection_input])
            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
            
    pfs=pfs.flatten()

    np.savetxt("Observations&data/Cox_pf_gbm_enes_v5.txt",pfs,fmt="%f")




Cox_pf_gbm_enes_v4 and CCPF_gbm_enes_v4
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
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=200
    l=2
    l0=0
    Lmin=0
    Lmax=5
    a=3e-2
    
    C=20
    C0=11
    K=0.0072
    ELES=np.array(np.arange(1,6))
    print(ELES)
    CL=2**(1/4)*(2**(ELES/4)-1)/(2**(1/4)-1)
    NB0=(np.sqrt(C0)+np.sqrt(3*C/2)*CL)*np.sqrt(2*C/3)*2**(-ELES/4)/K

    enes=np.array(a*NB0*2**(-1*3/4)*2**(ELES*(2+1/4)) ,dtype=int)
    print(enes)
    pfs=np.zeros((2,len(enes),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(enes)):
        for sample in range(samples):
            n=enes[i]
            collection_input=(T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,n,dim,\
            resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par)
            inputs.append([samples*i+sample,collection_input])
            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
            
    pfs=pfs.flatten()

    np.savetxt("Observations&data/CCPF_gbm_enes_v4.txt",pfs,fmt="%f")
    
    start=time.time()

    pfs=np.zeros((len(enes),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(enes)):
        for sample in range(samples):
            n=enes[i]
            collection_input=(T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,n,dim,\
            resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par)
            inputs.append([samples*i+sample,collection_input])
            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
            
    pfs=pfs.flatten()

    np.savetxt("Observations&data/Cox_pf_gbm_enes_v4.txt",pfs,fmt="%f")





Cox_pf_gbm_enes_v3 and CCPF_gbm_enes_v3

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
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=200
    l=2
    l0=0
    Lmin=0
    Lmax=5
    a=3e-2
    
    C=20
    C0=11
    K=0.0072
    ELES=np.array(np.arange(1,6))
    print(ELES)
    CL=2**(1/4)*(2**(ELES/4)-1)/(2**(1/4)-1)
    NB0=(np.sqrt(C0)+np.sqrt(3*C/2)*CL)*np.sqrt(2*C/3)*2**(-ELES/4)/K

    enes=np.array(a*NB0*2**(-1*3/4)*2**(ELES*(2+1/4)) ,dtype=int)
    print(enes)
    pfs=np.zeros((2,len(enes),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(enes)):
        for sample in range(samples):
            n=enes[i]
            collection_input=(T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,n,dim,\
            resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par)
            inputs.append([samples*i+sample,collection_input])
            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
            
    pfs=pfs.flatten()

    np.savetxt("Observations&data/CCPF_gbm_enes_v3.txt",pfs,fmt="%f")
    
    start=time.time()

    pfs=np.zeros((len(enes),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(enes)):
        for sample in range(samples):
            n=enes[i]
            collection_input=(T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,n,dim,\
            resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par)
            inputs.append([samples*i+sample,collection_input])
            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
            
    pfs=pfs.flatten()

    np.savetxt("Observations&data/Cox_pf_gbm_enes_v3.txt",pfs,fmt="%f")












Cox_pf_gbm_enes_v2 and CCPF_gbm_enes_v2 correspond to the following set 
of settings 


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
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=300
    l=1
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

    enes=a*NB0*2**(-1*3/4)*2**(ELES*(2+1/4)) 
    pfs=np.zeros((2,len(enes),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(enes)):
        for sample in range(samples):
            n=enes[i]
            collection_input=(T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,n,dim,\
            resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par)
            inputs.append([samples*i+sample,collection_input])
            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
            
    pfs=pfs.flatten()

    np.savetxt("Observations&data/CCPF_gbm_enes_v2.txt",pfs,fmt="%f")
    
    start=time.time()

    pfs=np.zeros((len(enes),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(enes)):
        for sample in range(samples):
            n=enes[i]
            collection_input=(T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,n,dim,\
            resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par)
            inputs.append([samples*i+sample,collection_input])
            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
            
    pfs=pfs.flatten()

    np.savetxt("Observations&data/Cox_pf_gbm_enes_v2.txt",pfs,fmt="%f")

Cox_pf_gbm_enes_v1 and CCPF_gbm_enes_v1 correspond to the following settings

We make this test to see the if the var=C/N holds for l=1, since previous 
experiments have shown that this is not the case for the function in this code.

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
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=300
    l=1
    enes=np.array([10*2**i for i in range(8)])
    pfs=np.zeros((2,len(enes),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(enes)):
        for sample in range(samples):
            n=enes[i]
            collection_input=(T,xin,b_gbm,mu,Sig_gmb,fi,obs,obs_time,l,d,N,dim,\
            resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par)
            inputs.append([samples*i+sample,collection_input])
            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCCPF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            pfs[:,i,sample]=pool_outputs[i*samples+sample]
            
    pfs=pfs.flatten()

    np.savetxt("Observations&data/CCPF_gbm_enes_v1.txt",pfs,fmt="%f")
    
    
    pfs=np.zeros((len(enes),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(enes)):
        for sample in range(samples):
            n=enes[i]
            collection_input=(T,xin,b_gbm,mu,Sig_gmb,fi,obs,obs_time,l,d,N,dim,\
            resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par)
            inputs.append([samples*i+sample,collection_input])
            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PCox_PF,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(enes)):
        n=enes[i]
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
            
    pfs=pfs.flatten()

    np.savetxt("Observations&data/Cox_pf_gbm_enes_v1.txt",pfs,fmt="%f")


PMLPF_cox_gbm20_p_pf_v3 corresonds to the following parameters

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
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=300
    l0=0
    Lmin=0
    Lmax=5
    eLes=np.array(np.arange(Lmin,Lmax+1))
    print(eLes)
    C=20
    C0=11
    K=0.0072
    pfs=np.zeros((len(eLes),samples,int(T/d),dim))
    # In pfs_levels, there are axis with dimension len(eLes), the first one co
    # rresponds to the maximum levles that we are considering, the other 
    # represents the levels within each MLPF
    pfs_levels=np.zeros((2,len(eLes),len(eLes),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eLes)):
        for sample in range(samples):
            L=eLes[i]
            CL=2**(1/4)*(2**(L/4)-1)/(2**(1/4)-1)
            NB0=(np.sqrt(C0)+np.sqrt(3*C/2)*CL)*np.sqrt(2*C/3)*2**(-L/4)/K
            N0=np.sqrt(C0)*(np.sqrt(C0)+np.sqrt(3*C/2)*CL)*2**(2*L)/K
            eles=np.array(np.arange(l0,L+1))
            eNes=np.zeros(len(eles),dtype=int)
            a=3e-2
            eNes[0]=N0*a
            eNes[1:]=NB0*2**(-eles[1:]*3/4)*2**(L*(2+1/4))*a
            collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,eles,d,eNes,dim,resamp_coef,phi,\
            dim_out,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([samples*i+sample+100*6,collection_input])
            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PMLPF_cox,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eLes)):
        L=eLes[i]
        eles=np.array(np.arange(l0,L+1))
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample][0]
            pfs_levels[:,i,:len(eles),sample]=pool_outputs[i*samples+sample][1]
    pfs=pfs.flatten()
    pfs_levels=pfs_levels.flatten()
    np.savetxt("Observations&data/PMLPF_cox_gbm20_p_pf_v3.txt",pfs,fmt="%f")
    np.savetxt("Observations&data/PMLPF_cox_gbm20_p_levels_v3.txt",pfs_levels,fmt="%f")



PMLPF_cox_gbm20_p_pf_v2 corresponds to 


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
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=0
    Lmin=0
    Lmax=5
    eLes=np.array(np.arange(Lmin,Lmax+1))
    print(eLes)
    C=20
    C0=11
    K=0.0072
    pfs=np.zeros((len(eLes),samples,int(T/d),dim))
    # In pfs_levels, there are axis with dimension len(eLes), the first one co
    # rresponds to the maximum levles that we are considering, the other 
    # represents the levels within each MLPF
    pfs_levels=np.zeros((2,len(eLes),len(eLes),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eLes)):
        for sample in range(samples):
            L=eLes[i]
            CL=2**(1/4)*(2**(L/4)-1)/(2**(1/4)-1)
            NB0=(np.sqrt(C0)+np.sqrt(3*C/2)*CL)*np.sqrt(2*C/3)*2**(-L/4)/K
            N0=np.sqrt(C0)*(np.sqrt(C0)+np.sqrt(3*C/2)*CL)*2**(2*L)/K
            eles=np.array(np.arange(l0,L+1))
            eNes=np.zeros(len(eles),dtype=int)
            a=3e-2
            eNes[0]=N0*a
            eNes[1:]=NB0*2**(-eles[1:]*3/4)*2**(L*(2+1/4))*a
            collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,eles,d,eNes,dim,resamp_coef,phi,\
            dim_out,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([samples*i+sample,collection_input])
            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PMLPF_cox,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eLes)):
        L=eLes[i]
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample][0]
            pfs_levels[:,i,:L,sample]=pool_outputs[i*samples+sample][1]
    pfs=pfs.flatten()
    pfs_levels=pfs_levels.flatten()
    np.savetxt("Observations&data/PMLPF_cox_gbm20_p_pf_v2.txt",pfs,fmt="%f")
    np.savetxt("Observations&data/PMLPF_cox_gbm20_p_levels_v2.txt",pfs,fmt="%f")




The following parameters corresonds to the file PMLPF_cox_gbm20_p_v3.txt
This time we change the distribution of particles along discretization levels,
we simplify this since we where not obtaining satisfactory results 
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
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=10
    l0=0
    Lmin=0
    Lmax=2
    eLes=np.array(np.arange(Lmin,Lmax+1))
    print(eLes)
    C=20
    C0=11
    K=0.0072
    pfs=np.zeros((len(eLes),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eLes)):
        for sample in range(samples):
            L=eLes[i]
            CL=2**(1/4)*(2**(L/4)-1)/(2**(1/4)-1)
            NB0=C*CL*2**(-L/4.)/K
            N0=C0*2**(2*L)/(K)
            eles=np.array(np.arange(l0,L+1))
            eNes=np.zeros(len(eles),dtype=int)
            a=3e-2
            eNes[0]=N0*a
            eNes[1:]=NB0*2**(-eles[1:]*3/4)*2**(L*(2+1/4))*a
            collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,eles,d,eNes,dim,resamp_coef,phi,\
            dim_out,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([samples*i+sample+100*6,collection_input])        
        

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PMLPF_cox,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/PMLPF_cox_gbm20_p_v3.txt",pfs,fmt="%f")



The following settings are not computed yet, they are ready to be tested and run
for T=100 GBM CCPF   
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
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=500
    l0=1
    L=10
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
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
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_gbm100_p_v1.txt",pfs,fmt="%f")


PMLPF_cox_gbm20_p_v2.txt has parameters: 
    
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
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=300
    l0=0
    Lmin=0
    Lmax=5
    eLes=np.array(np.arange(Lmin,Lmax+1))
    print(eLes)
    C=20
    C0=11
    K=0.0072
    pfs=np.zeros((len(eLes),samples,int(T/d),dim))
    inputs=[]

    for i in range(len(eLes)):
        for sample in range(samples):
            L=eLes[i]
            CL=2**(1/4)*(2**(L/4)-1)/(2**(1/4)-1)
            NB0=(np.sqrt(C0)+np.sqrt(3*C/2)*CL)*np.sqrt(2*C/3)*2**(-L/4)/K
            N0=np.sqrt(C0)*(np.sqrt(C0)+np.sqrt(3*C/2)*CL)*2**(2*L)/K
            eles=np.array(np.arange(l0,L+1))
            eNes=np.zeros(len(eles),dtype=int)
            a=3e-2
            eNes[0]=N0*a
            eNes[1:]=NB0*2**(-eles[1:]*3/4)*2**(L*(2+1/4))*a
            collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,eles,d,eNes,dim,resamp_coef,phi,\
            dim_out,g_den,g_par,Norm_Lambda,Lamb_par]
            inputs.append([samples*i+sample+100*6,collection_input])        
        

            
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PMLPF_cox,inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eles)):
        for sample in range(samples):
            pfs[i,sample]=pool_outputs[i*samples+sample]
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/PMLPF_cox_gbm20_p_v2.txt",pfs,fmt="%f")


CCPF_gbm20_p_v5.txt for the following settings 

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
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=500
    l0=1
    L=10
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=5000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    
    
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
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
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_gbm20_p_v5.txt",pfs,fmt="%f")


CCPF_gbm20_p_v4 corresponds to the following settings:
    
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
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=5000
    l0=1
    L=10
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
            collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
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
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_gbm20_p_v4.txt",pfs,fmt="%f")




CCPF_gbm20_p_v3.txt has the data generated with the following settings. 


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
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=500
    l0=1
    L=10
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
            collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Norm_Lambda,Lamb_par]
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
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_gbm20_p_v3.txt",pfs,fmt="%f")


CCPF_gbm20_p_v2 corresponds to 

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
    k=2**(-10)
    mu=np.abs(np.random.normal(.5,0,dim))*k
    sigs=np.abs(np.random.normal(np.sqrt(2),0,dim))*np.sqrt(k)
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
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=1000
    l0=1
    L=10
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=1000
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    
    
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Lambda,Lamb_par]
            inputs.append([sample+samples*i+2000,collection_input])        
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
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_gbm20_p_v2.txt",pfs,fmt="%f")



CCPF_gbm20_p_v1 corresponds to the following settings. This is the first 
reparametrization of time wiht t'=t/k
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
    k=2**(-10)
    mu=np.abs(np.random.normal(.5,0,dim))*k
    sigs=np.abs(np.random.normal(np.sqrt(2),0,dim))*np.sqrt(k)
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
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=10
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
            collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Lambda,Lamb_par]
            inputs.append([sample+samples*i+2000,collection_input])        
    pool = multiprocessing.Pool(processes=8)
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
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_gbm20_p_v1.txt",pfs,fmt="%f")


CCPF_gbm20_v5 has settings 

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
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=200
    l0=8
    L=15
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=500
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    
    
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Lambda,Lamb_par]
            inputs.append([sample+samples*i+2000,collection_input])        
    pool = multiprocessing.Pool(processes=8)
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
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_gbm20_v5.txt",pfs,fmt="%f")




CCPF_gbm20_v3 corresonds to 

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
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=16
    L=17
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=500
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    
    
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])        
    pool = multiprocessing.Pool(processes=6)
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
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_gbm20_v3.txt",pfs,fmt="%f")


CCPF_gbm20_v2 corresponds to 

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
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=14
    L=15
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=500
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    
    
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,N,dim\
            ,resamp_coef,g_den,g_par,Lambda,Lamb_par]
            inputs.append([sample+samples*i,collection_input])        
    pool = multiprocessing.Pool(processes=6)
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
    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_gbm20_v3.txt",pfs,fmt="%f")



CCPF_gbm20_v1 corresponds to 

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
    start=time.time()
    d=2**(0)
    resamp_coef=0.8
    dim_out=2
    g_par=cov
    samples=100
    l0=1
    L=9
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=500
    pfs=np.zeros((2,len(eles),samples,int(T/d),dim))
    inputs=[]
    
    
    for i in range(len(eles)):
        l=eles[i]
        for sample in range(samples):
            collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,N,dim\
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

    
    pfs=pfs.flatten()
        
    np.savetxt("Observations&data/CCPF_gbm20_v1.txt",pfs,fmt="%f")






For CCPF_gbm100_v1.txt the settings are

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
samples=100
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
        collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,l,d,N,dim\
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
    
    
    
    
pfs=pfs.flatten()
    
np.savetxt("Observations&data/CCPF_gbm1000_v1.txt",pfs,fmt="%f")

""" 
    
#%%

"""

MLPF_gbm_v3.txt corresponds to 

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
    start=time.time()
    d=2**(0)
    resamp_coef=0.8    
    dim_out=2
    samples=100
    l0=0
    Lmin=0
    Lmax=5
    eLes=np.array(np.arange(Lmin,Lmax+1))
    print(eLes)
    K=0.305
    C=7.5
    C0=3.30529
    
    tel_est=np.zeros((len(eLes),samples,int(T/d),dim))
    
    inputs=[]



    for i in range(len(eLes)):
        for sample in range(samples):
            L=eLes[i]
            CL=2**(1/4)*(2**(L/4)-1)/(2**(1/4)-1)
            NB0=(np.sqrt(C0)+np.sqrt(3*C/2)*CL)*np.sqrt(2*C/3)*2**(-L/4)/K
            N0=np.sqrt(C0)*(1+np.sqrt(3*C/2)*CL)*2**(2*L)/K
            eles=np.array(np.arange(l0,L+1))
            eNes=np.zeros(len(eles),dtype=int)
            eNes[0]=N0
            eNes[1:]=NB0*2**(-eles[1:]*3/4)*2**(L*(2+1/4))
            collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,eles,d,eNes,dim,resamp_coef,phi,\
            dim_out,g_den,g_par,Lambda,Lamb_par]
            inputs.append([samples*i+sample+100*6,collection_input])        
            

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
            tel_est[i,sample]=pool_outputs[i*samples+sample]
    #tel_est=np.reshape(np.array(pool_outputs),(len(eLes),samples,int(T/d),dim))
    
    #log_weightss=log_weightss.flatten()
    tel_est=tel_est.flatten()
    np.savetxt("Observations&data/MLPF_gbm_v3.txt",tel_est,fmt="%f")


MLPF_gbm_v2.txt corresponds to 
        
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
    start=time.time()
    d=2**(0)
    resamp_coef=0.8    
    dim_out=2
    samples=100
    l0=0
    Lmin=0
    Lmax=5
    eLes=np.array(np.arange(Lmin,Lmax+1))
    print(eLes)
    K=0.305
    C=7.5
    C0=3.30529
    tel_est=np.zeros((len(eLes),samples,int(T/d),dim))
    
    inputs=[]



    for i in range(len(eLes)):
        for sample in range(samples):
            L=eLes[i]
            CL=2**(1/4)*(2**(L/4)-1)/(2**(1/4)-1)
            NB0=(1+np.sqrt(3*C/2)*CL)*np.sqrt(2*C/3)*2**(-L/4)/K
            N0=C0*(1+np.sqrt(3*C/2)*CL)*2**(2*L)/K
            eles=np.array(np.arange(l0,L+1))
            eNes=np.zeros(len(eles),dtype=int)
            eNes[0]=N0
            eNes[1:]=NB0*2**(-eles[1:]*3/4)*2**(L*(2+1/4))
            collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,eles,d,eNes,dim,resamp_coef,phi,\
            dim_out,g_den,g_par,Lambda,Lamb_par]
            inputs.append([samples*i+sample,collection_input])        
            

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
            tel_est[i,sample]=pool_outputs[i*samples+sample]
    #tel_est=np.reshape(np.array(pool_outputs),(len(eLes),samples,int(T/d),dim))
    
    #log_weightss=log_weightss.flatten()
    tel_est=tel_est.flatten()
    np.savetxt("Observations&data/MLPF_gbm_v2.txt",tel_est,fmt="%f")


MLPF_gbm_v1.txt corresponds to the following parameters. This data is failed 
since we there was a mistake in the line 
eNes[1:]=NB0**2**(-eles[1:]*3/4)*2**(L*(2+1/4))
in NB0**2**(-el..., instead, we should have had NB0*2**(-el...

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
    start=time.time()
    d=2**(0)
    resamp_coef=0.8    
    dim_out=2
    samples=100
    l0=0
    Lmin=0
    Lmax=5
    eLes=np.array(np.arange(Lmin,Lmax+1))
    print(eLes)
    K=0.305
    C=7.5
    C0=3.30529
    tel_est=np.zeros((len(eLes),samples,int(T/d),dim))
    
    inputs=[]



    for i in range(len(eLes)):
        for sample in range(samples):
            L=eLes[i]
            CL=2**(1/4)*(2**(L/4)-1)/(2**(1/4)-1)
            NB0=(1+np.sqrt(3*C/2)*CL)*np.sqrt(2*C/3)*2**(-L/4)/K
            N0=C0*(1+np.sqrt(3*C/2)*CL)*2**(2*L)/K
            eles=np.array(np.arange(l0,L+1))
            eNes=np.zeros(len(eles),dtype=int)
            eNes[0]=N0
            eNes[1:]=NB0**2**(-eles[1:]*3/4)*2**(L*(2+1/4))
        
            collection_input=[T,xin,b_gbm,mu,Sig_gbm,fi,obs,obs_time,eles,d,eNes,dim,resamp_coef,phi,\
            dim_out,g_den,g_par,Lambda,Lamb_par]
            inputs.append([samples*i+sample,collection_input])        
            

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
            tel_est[i,sample]=pool_outputs[i*samples+sample]
    #tel_est=np.reshape(np.array(pool_outputs),(len(eLes),samples,int(T/d),dim))
    
    #log_weightss=log_weightss.flatten()
    tel_est=tel_est.flatten()
    #np.savetxt("Observations&data/MLPF_gbm_v1.txt",tel_est,fmt="%f")
    #np.savetxt("Observations&data/SUCPF_v2.txt",pfs,fmt="%f")



Observations&data/SUCPF_p_levels_bias_est_v1.txt corresponds to the following 
setting, the purpose of these experiment is to check the bias of the estimator
additional changes must be made to the function PSUCPF in order to get the desired estimator,
the changes are in lines 
#weightsB=norm_logweights(log_weights[:,:enes[p-1]],ax=1)           
#phi_pfmeanB=np.sum(np.reshape(weightsB,(int(T/d),enes[p-1],1))*phi(x_pf[:,:enes[p-1]],ax=2),axis=1)
phi_pfmeanB=0
where we commented the two lines and added one.

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
    samples=1
    L=6
    l0=5
    p=4
    N0=4*2**8
    eLes=np.arange(l0,L+1)
    ps=np.arange(0,p+1)
    eNes=N0*2**ps
    print(eNes)
    #Pl=(eles+1)*np.log(eles+2)**2/2**(eles)
    #Pl[0]=2*Pl[1]
    #Pl=Pl/np.sum(Pl)
    #Pl=np.cumsum(Pl)
    #Pp=(ps+1)*np.log(ps+2)**2/2**(ps)
    #Pp[0]=2*Pp[1]
    #Pp=Pp/np.sum(Pp)
    #Pp=np.cumsum(Pp)
    #print(Pl, Pp)
    
    tel_est=np.zeros((len(eNes)-1,samples,int(T/d),dim))
    
    inputs=[]



    for pit in range(len(eNes)-1):
        for sample in range(samples):
            
            
            Pl_cum=np.array([1,1])
            
            Pp_cum=np.concatenate((np.zeros(pit+1),np.zeros(p-pit)+1))
            print(Pp_cum)
            collection_input=[T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,eLes,Pl_cum,d,eNes\
            ,Pp_cum,dim,resamp_coef,phi,g_den,g_par,Lambda,Lamb_par]
            inputs.append([samples*pit+sample,collection_input])        

    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PSUCPF, inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eNes)-1):
        for sample in range(samples):
            tel_est[i,sample]=pool_outputs[i*samples+sample]
    #tel_est=np.reshape(np.array(pool_outputs),(len(eLes),samples,int(T/d),dim))
    
    #log_weightss=log_weightss.flatten()
    tel_est=tel_est.flatten()
    np.savetxt("Observations&data/SUCPF_p_levels_bias_est_v2.txt",tel_est,fmt="%f")



Observations&data/SUCPF_p_levels_bias_est_v1.txt corresponds to the following 
setting, the purpose of these experiment is to check the bias of the estimator
additional changes must be made to the function PSUCPF in order to get the desired estimator,
the changes are in lines 
#weightsB=norm_logweights(log_weights[:,:enes[p-1]],ax=1)           
#phi_pfmeanB=np.sum(np.reshape(weightsB,(int(T/d),enes[p-1],1))*phi(x_pf[:,:enes[p-1]],ax=2),axis=1)
phi_pfmeanB=0
where we commented the two lines and added one.



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
    samples=3000
    
    L=6
    l0=5
    p=8
    N0=4
    eLes=np.arange(l0,L+1)
    ps=np.arange(0,p+1)
    eNes=N0*2**ps
    print(eNes)
    #Pl=(eles+1)*np.log(eles+2)**2/2**(eles)
    #Pl[0]=2*Pl[1]
    #Pl=Pl/np.sum(Pl)
    #Pl=np.cumsum(Pl)
    #Pp=(ps+1)*np.log(ps+2)**2/2**(ps)
    #Pp[0]=2*Pp[1]
    #Pp=Pp/np.sum(Pp)
    #Pp=np.cumsum(Pp)
    #print(Pl, Pp)
    
    tel_est=np.zeros((len(eNes)-1,samples,int(T/d),dim))
    
    inputs=[]



    for pit in range(len(eNes)-1):
        for sample in range(samples):
            
            
            Pl_cum=np.array([1,1])
            
            Pp_cum=np.concatenate((np.zeros(pit+1),np.zeros(p-pit)+1))
            collection_input=[T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,eLes,Pl_cum,d,eNes\
            ,Pp_cum,dim,resamp_coef,phi,g_den,g_par,Lambda,Lamb_par]
    
            inputs.append([samples*p+sample,collection_input])        

    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PSUCPF, inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eNes)-1):
        for sample in range(samples):
            tel_est[i,sample]=pool_outputs[i*samples+sample]
    #tel_est=np.reshape(np.array(pool_outputs),(len(eLes),samples,int(T/d),dim))
    
    #log_weightss=log_weightss.flatten()
    tel_est=tel_est.flatten()
    np.savetxt("Observations&data/SUCPF_p_levels_bias_est_v1.txt",tel_est,fmt="%f")
    #np.savetxt("Observations&data/SUCPF_v2.txt",pfs,fmt="%f")



Observations&data/SUCPF_p_levels_v1.txt is obtained so we can test the variance of the 
telescoping terms when we change the number of particles level

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
    samples=3000
    
    L=5
    l0=4
    p=8
    N0=4
    eLes=np.arange(l0,L+1)
    ps=np.arange(0,p+1)
    eNes=N0*2**ps
    print(eNes)
    #Pl=(eles+1)*np.log(eles+2)**2/2**(eles)
    #Pl[0]=2*Pl[1]
    #Pl=Pl/np.sum(Pl)
    #Pl=np.cumsum(Pl)
    #Pp=(ps+1)*np.log(ps+2)**2/2**(ps)
    #Pp[0]=2*Pp[1]
    #Pp=Pp/np.sum(Pp)
    #Pp=np.cumsum(Pp)
    #print(Pl, Pp)
    
    tel_est=np.zeros((len(eNes)-1,samples,int(T/d),dim))
    
    inputs=[]



    for pit in range(len(eNes)-1):
        for sample in range(samples):
            
            
            Pl_cum=np.array([0,1])
            
            Pp_cum=np.concatenate((np.zeros(pit+1),np.zeros(p-pit)+1))
            collection_input=[T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,eLes,Pl_cum,d,eNes\
            ,Pp_cum,dim,resamp_coef,phi,g_den,g_par,Lambda,Lamb_par]
    
            inputs.append([samples*p+sample,collection_input])        

    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PSUCPF, inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eNes)-1):
        for sample in range(samples):
            tel_est[i,sample]=pool_outputs[i*samples+sample]
    #tel_est=np.reshape(np.array(pool_outputs),(len(eLes),samples,int(T/d),dim))
    
    #log_weightss=log_weightss.flatten()
    tel_est=tel_est.flatten()
    np.savetxt("Observations&data/SUCPF_p_levels_v1.txt",tel_est,fmt="%f")



Observations&data/SUCPF_l_levels_v1.txt is obtained so we can test the variance of the 
telescoping terms when we change the time discreatization levels



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
    samples=1000
    
    Lmax=8
    l0=0
    p=1
    N0=250
    eLes=np.arange(l0,Lmax+1)
    ps=np.arange(0,p+1)
    enes=N0*2**ps
    print(enes)
    #Pl=(eles+1)*np.log(eles+2)**2/2**(eles)
    #Pl[0]=2*Pl[1]
    #Pl=Pl/np.sum(Pl)
    #Pl=np.cumsum(Pl)
    #Pp=(ps+1)*np.log(ps+2)**2/2**(ps)
    #Pp[0]=2*Pp[1]
    #Pp=Pp/np.sum(Pp)
    #Pp=np.cumsum(Pp)
    #print(Pl, Pp)
    
    tel_est=np.zeros((len(eLes),samples,int(T/d),dim))
    
    inputs=[]



    for i in range(len(eLes)):
        for sample in range(samples):
            l=eLes[i]
            eles=np.array([l,l+1])
            Pl_cum=np.array([0,1])
            enes=np.array([N0,2*N0])
            Pp_cum=np.array([0,1])
            collection_input=[T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,eles,Pl_cum,d,enes\
            ,Pp_cum,dim,resamp_coef,phi,g_den,g_par,Lambda,Lamb_par]
    
            inputs.append([samples*i+sample,collection_input])        

    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PSUCPF, inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    
    print("Parallelized processes time:",end-start,"\n")            
    for i in range(len(eLes)):
        for sample in range(samples):
            tel_est[i,sample]=pool_outputs[i*samples+sample]
    
    #tel_est=np.reshape(np.array(pool_outputs),(len(eLes),samples,int(T/d),dim))
    
    #log_weightss=log_weightss.flatten()
    tel_est=tel_est.flatten()
    np.savetxt("Observations&data/SUCPF_l_levels_v1.txt",tel_est,fmt="%f")
    #np.savetxt("Observations&data/SUCPF_v2.txt",pfs,fmt="%f")



Observations&data/SUCPF_v2.txt was made in order to test the single term
estimator.


if __name__ == '__main__':
    
    
    
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
    Pl=np.cumsum(Pl)
    Pp=(ps+1)*np.log(ps+2)**2/2**(ps)
    Pp[0]=2*Pp[1]
    Pp=Pp/np.sum(Pp)
    Pp=np.cumsum(Pp)
    print(Pl, Pp)
    
    pfs=np.zeros((len(eles),samples,int(T/d),dim))
    
    inputs=[]



    for sample in range(samples):
            
        collection_input=[T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,eles,Pl,d,enes\
        ,Pp,dim,resamp_coef,phi,g_den,g_par,Lambda,Lamb_par]
    
        inputs.append([sample+1000,collection_input])        
        
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PSUCPF, inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    
    print("Parallelized processes time:",end-start,"\n")            
    for sample in range(samples):
        pfs[:,sample]=pool_outputs[sample]
    
    #x_pfs=np.reshape(np.array(pool_outputs),(len(eles),samples,int(T/d),dim))
    
    #log_weightss=log_weightss.flatten()
    pfs=pfs.flatten()
    
    np.savetxt("Observations&data/SUCPF_v2.txt",pfs,fmt="%f")


Observations&data/SUCPF_v1.txt was made in order to test the single term
estimator.




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
    samples=1000
    
    lmax=7
    l0=0
    pmax=7
    N0=50
    
    eles=np.arange(l0,lmax+1)

    ps=np.arange(0,pmax+1)
    enes=N0*2**ps
    print(enes)
    Pl=(eles+1)*np.log(eles+2)**2/2**(eles)
    Pl=Pl/np.sum(Pl)
    Pl=np.cumsum(Pl)
    Pp=(ps+1)*np.log(ps+2)**2/2**(ps)
    Pp=Pp/np.sum(Pp)
    Pp=np.cumsum(Pp)

    
    pfs=np.zeros((len(eles),samples,int(T/d),dim))
    
    inputs=[]



    for sample in range(samples):
            
        collection_input=[T,xin,b_ou,B,Sig_ou,fi,obs,obs_time,eles,Pl,d,enes\
        ,Pp,dim,resamp_coef,phi,g_den,g_par,Lambda,Lamb_par]
    
        inputs.append([sample,collection_input])        
        
    pool = multiprocessing.Pool(processes=10)
    pool_outputs = pool.map(PSUCPF, inputs)
    pool.close()
    pool.join()
    #blocks_pools.append(pool_outputs)
    xend1=time.time()
    end=time.time()
    
    print("Parallelized processes time:",end-start,"\n")            
    for sample in range(samples):
        pfs[:,sample]=pool_outputs[sample]
    
    #x_pfs=np.reshape(np.array(pool_outputs),(len(eles),samples,int(T/d),dim))
    
    #log_weightss=log_weightss.flatten()
    pfs=pfs.flatten()
    
    np.savetxt("Observations&data/SUCPF_v1.txt",x_pfs,fmt="%f")




Observations&data/CCPF_gbm_v6.txt
was made in order to get nicer plots of the bias and variance of the PF


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
    
    L=15
    l0=0
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=200
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
            x_pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    #x_pfs=np.reshape(np.array(pool_outputs),(len(eles),samples,int(T/d),dim))
    
    #log_weightss=log_weightss.flatten()
    x_pfs=x_pfs.flatten()
    
    np.savetxt("Observations&data/CCPF_gbm_v6.txt",x_pfs,fmt="%f")


Observations&data/CCPF_gbm_v5.txt
was made in order to get nicer plots of the bias and variance of the PF



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
    
    L=15
    l0=4
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=500
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
            x_pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    #x_pfs=np.reshape(np.array(pool_outputs),(len(eles),samples,int(T/d),dim))
    
    #log_weightss=log_weightss.flatten()
    x_pfs=x_pfs.flatten()
    
    np.savetxt("Observations&data/CCPF_gbm_v5.txt",x_pfs,fmt="%f")



Observations&data/CCPF_gbm_v4.txt was obtained in order to get more levels
of the GBM so we can get nice plots.
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
    
    L=12
    l0=1
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eles=np.array(range(l0,L+1))
    N=500
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
            x_pfs[:,i,sample]=pool_outputs[i*samples+sample]
    
    #x_pfs=np.reshape(np.array(pool_outputs),(len(eles),samples,int(T/d),dim))
    
    #log_weightss=log_weightss.flatten()
    x_pfs=x_pfs.flatten()
    
    np.savetxt("Observations&data/CCPF_gbm_v4.txt",x_pfs,fmt="%f")






bservations&data/MLPF_cox_v11.txt corresponds to the following parametrs
and it was made in order to find the proportionality constant of the variance 
of the MLPF


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
    samples=200
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
    
    np.savetxt("Observations&data/MLPF_cox_v11.txt",pfs,fmt="%f")







np.savetxt("Observations&data/MLPF_cox_v10.txt",pfs,fmt="%f")


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
    samples=200
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
            
            
            inputs.append([sample+300,collection_input])        
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
    
    np.savetxt("Observations&data/MLPF_cox_v10.txt",pfs,fmt="%f")




Observations&data/MLPF_cox_v9.txt corresponds to the following parameters 


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
    samples=200
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
    
    np.savetxt("Observations&data/MLPF_cox_v9.txt",pfs,fmt="%f")




Observations&data/MLPF_cox_v7.txt corresponds to    

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
    samples=50
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
            
            
            inputs.append([sample+50,collection_input])        
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
    
    np.savetxt("Observations&data/MLPF_cox_v7.txt",pfs,fmt="%f")




Observations&data/MLPF_cox_v6.txt corresponds to the parameters 

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
    samples=50
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
    
    np.savetxt("Observations&data/MLPF_cox_v6.txt",pfs,fmt="%f")




Observations&data/MLPF_cox_v4.txt corresponds to 

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
    
    np.savetxt("Observations&data/MLPF_cox_v4.txt",pfs,fmt="%f")


bservations&data/MLPF_cox_v3.txt corresponds to 



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
    samples=500
    Lmin=0
    Lmax=7
    l0=0
    # In this iteration we have eLes and eles, do not confuse. eLes respresents
    # the range of maximum number of levels that we take, eles is a direct 
    # argument to the MLPF_cox, its the number of levels that we in one ML. 
    eLes=np.array(range(Lmin,Lmax+1))
    N0=75
    pfs=np.zeros((2,len(eLes),samples,int(T/d),dim))
    
    inputs=[]





Observations&data/MLPF_cox_v2.txt corresponds to de settings


if __name__ == '__main__':
    
    
    
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
    samples=500
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
    
    np.savetxt("Observations&data/MLPF_cox_v2.txt",pfs,fmt="%f")




pfs_shifted_d1_pc2.txt corresponds to

    np.random.seed(6)
    T=100
    dim=1
    dim_o=dim
    xin=np.random.normal(0,5,dim)
    l=1
    collection_input=[]
    I=identity(dim).toarray()
    
    #comp_matrix = ortho_group.rvs(dim)
    comp_matrix=np.array([[1]])
    inv_mat=la.inv(comp_matrix)
    S=diags(np.random.normal(.001,0.1,dim),0).toarray()
    fi=inv_mat@S@comp_matrix
    B=diags(np.random.normal(-.001,0.1,dim),0).toarray()*(2/3)
    B=inv_mat@B@comp_matrix
    print(B)
    #print(B)
    #B=comp_matrix-comp_matrix.T  +B 
    
    collection_input=[dim, b_ou,B,Sig_ou,fi]
    cov=I*5e-3
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
    samples=2400
    g_par=cov
    l0=1
    L=10
    N=6000
    eles=np.array(range(l0,L+1))
    x_pfs=np.zeros((len(eles),samples,int(T/d),dim))
    inputs=[]


pfs_shifted_d2_pc2.txt corresponds to 

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
    samples=2400
    g_par=cov
    l0=1
    L=7
    N=6000
    eles=np.array(range(l0,L+1))
    x_pfs=np.zeros((len(eles),samples,int(T/d),dim))
    inputs=[]


pfs_shifted_d1_pc2.txt corresponds to the settings


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
    samples=1200
    g_par=cov
    l0=1
    L=7
    N=6000
    eles=np.array(range(l0,L+1))
    x_pfs=np.zeros((len(eles),samples,int(T/d),dim))
    inputs=[]



pfs_shifted_pc2.txt corresponds to the configuration

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



sol_T10dim2l15N8000s6samp4001.txt corresponds to the following setting.
We compute this so we can lower the variance of the estimated "analytical"
solution.

blocks_pools=[]
    start1=0
    end1=0
    start=time.time()
    samples=160
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
    l0=15
    L=15
    eles=np.array(range(l0,L+1))
    
    log_weightss=np.zeros((len(eles),samples,int(T/d),N))
    x_pfs=np.zeros((len(eles),samples,int(T/d),N,dim))
    pfs=np.zeros((len(eles),samples,int(T/d),dim))
    


sol_T10dim2l15N8000s6samp400.txt corresponds to the following setting.
We compute this so we can lower the variance of the estimated "analytical"
solution.

blocks_pools=[]
    start1=0
    end1=0
    start=time.time()
    samples=10
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
    l0=15
    L=15
    eles=np.array(range(l0,L+1))
    
    log_weightss=np.zeros((len(eles),samples,int(T/d),N))
    x_pfs=np.zeros((len(eles),samples,int(T/d),N,dim))
    pfs=np.zeros((len(eles),samples,int(T/d),dim))
    



log_weightss_bias1.txt and x_pfs_bias1.txt corresponds to the system 

blocks_pools=[]
    start1=0
    end1=0
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


pfs1.txt corresponds to the system (where the observations have been changed so 
they correspond to the actual system)
I actually lost this data when running a test :(
blocks_pools=[]
    start1=0
    end1=0
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
    
    

pfs2.txt corresponds to the system (where the observations have been changed so 
they correspond to the actual system)
I actually lost this data when running a test :(
blocks_pools=[]
    start1=0
    end1=0
    start=time.time()
    samples=400
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

"""
# %%
