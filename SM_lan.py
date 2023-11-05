# In this file I test the smoother for the langevin equation 
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
np.random.seed(4)
T=100
dim=1
dim_o=dim
xin=np.zeros(dim)+1
l=8
collection_input=[]
I=identity(dim).toarray()
df=10
fi=np.array([[1]])
collection_input=[dim, b_lan,df,Sig_ou,fi]
cov=I*1e0
g_pars=[dim,cov]
g_par=cov
x_true=gen_gen_data(T,xin,l,collection_input)
Lamb_par=5
[obs_time,obs]=gen_obs(x_true,l,T,pff.Lambda,Lamb_par,g_normal,g_pars)
print(obs_time,obs,len(obs_time))
times=2**(-l)*np.array(range(int(T*2**l+1)))
b_numb_par=1
Lamb_par_numb=1
plt.plot(times,x_true,label="True signal")
plt.plot(obs_time,obs,label="Observations")
plt.legend()
d=2**(0)
N=40
resamp_coef=0.8
# %%

#%%
df_init=df+20
Lamb_par_init=Lamb_par+2
g_par_init=g_par+3
print(df_init,Lamb_par_init,g_par_init)

d=1
N=1000
results=pff.Cox_SM_W(T,xin,b_lan,df,pff.Sig_ou,fi,pff.Grad_b_lan,b_numb_par,obs,obs_time,l,d,N,dim,resamp_coef,\
g_den,pff.Grad_Log_g_den,g_par,pff.Lambda,pff.Grad_Lambda,pff.Grad_Log_Lambda,Lamb_par,Lamb_par_numb)
#%%
print(np.sum(results[2][-1]*(norm_logweights(results[0][-1],ax=0)[:,np.newaxis]),axis=0)/T)
print(np.sum(results[3][-1]*(norm_logweights(results[0][-1],ax=0)[:,np.newaxis]),axis=0)/T)
print(np.sum(results[4][-1]*(norm_logweights(results[0][-1],ax=0)[:,np.newaxis,np.newaxis]),axis=0)/T)
# %%
df_init=df-2
Lamb_par_init=Lamb_par+2
g_par_init=g_par+3
print(df_init,Lamb_par_init,g_par_init)
#%%
b_numb_par=1
Lamb_par_numb=1
N=100
l=8 
d=2
resamp_coef=0.8
beta=3/4.
step_0=1.5
results=pff.Cox_PE(T,xin,b_lan,df_init,Sig_ou,fi,pff.Grad_b_lan,b_numb_par,obs,obs_time,l,d,N,\
dim,resamp_coef,pff.g_den,pff.Grad_Log_g_den,g_par_init,pff.Lambda,pff.Grad_Lambda,pff.Grad_Log_Lambda,\
Lamb_par_init,Lamb_par_numb,step_0,beta)
#%%
#"""
Lambda_pars,g_pars,dfs=results[2],results[3],results[4]
print(Lamb_par,Lamb_par_init)
print("g_par",g_par[0,0])
print(Lambda_pars)
plt.plot(range(int(T/d)+1),Lambda_pars,label=r"$\theta_\lambda$")
plt.plot(range(int(T/d)+1),Lamb_par+np.zeros(int(T/d)+1),label=r"$\bar{\theta}_\lambda$",ls="--")
plt.xlabel("Iterations")
#plt.ylabel(r"$\theta_\lambda$")
print(g_par,g_par_init)
print(g_pars)
plt.plot(range(int(T/d)+1),g_pars[:,0,0],label=r"$\theta_\Sigma$")
plt.plot(range(int(T/d)+1),g_par[0,0]+np.zeros(int(T/d)+1),label=r"$\bar{\theta}_\Sigma$",ls="--")

print(df,df_init)
print(dfs)
plt.plot(range(int(T/d)+1),dfs[:,0],label=r"$\theta_b$")
plt.plot(range(int(T/d)+1),df+np.zeros(int(T/d)+1),label=r"$\bar{\theta}_b$",ls="--")
plt.legend()
#plt.savefig("images/SGD_ou_T5000_d2_N200_l8.pdf")
#"""
# %%
print(Lambda_pars[:])
# %%
