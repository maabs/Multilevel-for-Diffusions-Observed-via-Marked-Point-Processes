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
# THIS FILE IS CREATED FOR THE CODE OF THE SMOOTHER OF A DIFFUSION PROCESS WITH COX
# TIME OBSERVATIONS. THE SMOOTHER IS BASED ON THE PARTICLE FILTER. WE MAKE USE OF THE 
# FUNCTIONS DEFINED IN THE FILE PF_functions_def.py AND Un_cox_PF_functions_def.py
# ADDITIONALLY, USING THE SMOOTHER WE APPROXITAME THE LIKELIHOOD AND OPTIMIZE IT 
# USING STOCHASTIC GRADIENT DESCENT.
# %%
# FIRST WE CONSTRUCT THE SMOOTHER AND WE TEST IT. Do we have results for the smoother? in 
# the linear Gaussian case? 

#def Cox_PF(T,xin,b_ou,A,Sig_ou,fi,obs,obs_time,l,d,N,dim,resamp_coef,g_den,g_par,Lambda,Lamb_par):

def Smooth_Cox(T,xin,b_ou,A,Sig_ou,fi,obs,obs_time,l,d,N,dim,resamp_coef,g_den,g_par,Lambda,Lamb_par):


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
        xi=M(x_new,b_ou,A,Sig_ou,fi,l,d,N,dim)
        #print(yi,obti-i*d,)
        #print(x_new,xi)
        #print(xi)
        #Things that could be wrong
        # observations, x_new, weights
        #observations seem to be fine

        log_weights[i]=log_weights[i]+Gox(yi,obti-i*d,xi,Lambda,Lamb_par,l,N,dim,g_den,g_par)
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
            x_new=multi_samp(weights,N,x_last,dim)[1]
            #print(x_new.shape)
        else:
            
            #print("time is",i)
            x_new=x_last
            if i< int(T/d)-1:
                log_weights[i+1]=log_weights[i]
            
        #print(i)
        