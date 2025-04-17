#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 03:52:53 2025

@author: nephilim
"""

from multiprocessing import Pool
import numpy as np
import time
import scipy.signal
from config_parameters import config
from add_cpml import Add_CPML
from forward_modeling import ForwardModeling
from fwi_parameters import fwi_config
from clutter_removal import ClutterRemoval
from wavelet_creation import WaveletType
from regularization_method import Regularization
import update_field_forward
import update_field_adjoint

def calculate_gradient(sigma,iepsilon,index,CPML_Params):
    #Get Forward Params
    ep0=8.841941282883074e-12
    f=WaveletType(config.t, fwi_config.fwi_freq, fwi_config.wavelet_type).create_wavelet()
    #True Model Profile Data
    true_trace=config.true_profile[:,index]
    #Get Forward Data ----> <Generator>
    forward_generator=update_field_forward.time_loop(config.xl,config.zl,config.dx,config.dz,config.dt,
                                                     sigma,iepsilon.copy(),CPML_Params,f,config.k_max,
                                                     config.source_site[index],config.receiver_site[index])
    #Get Generator Data
    forward_field=[]
    prediction_trace=np.zeros(config.k_max)
    for idx in range(config.k_max):
        tmp=forward_generator.__next__()
        forward_field.append(np.array(tmp[0]))
        prediction_trace[idx]=tmp[1]
    prediction_trace-=fwi_config.air_trace
    #Get Residual Data
    # True * Fake_Ref
    true_trace_tmp=scipy.signal.convolve(true_trace,fwi_config.prediction_ref_profile[:,index])
    true_trace_conv=true_trace_tmp[:len(prediction_trace)]
    # Fake * True_Ref
    preditcion_trace_tmp=scipy.signal.convolve(prediction_trace,config.true_ref_profile[:,index])
    preditcion_trace_conv=preditcion_trace_tmp[:len(config.true_ref_profile)]
    rhs_trace=preditcion_trace_conv-true_trace_conv
    rhs_trace_source=scipy.signal.correlate(rhs_trace,config.true_ref_profile[:,index],mode='full')
    rhs_trace_source=rhs_trace_source[int(len(rhs_trace_source)/2):]
    #Get Reversion Data ----> <Generator>
    adjoint_generator=update_field_adjoint.reverse_time_loop(config.xl,config.zl,config.dx,config.dz,config.dt,
                                                             sigma,iepsilon.copy(),CPML_Params,config.k_max,\
                                                             config.receiver_site[index],rhs_trace_source)
    #Get Generator Data
    adjoint_field=[]
    for i in range(config.k_max):
        tmp=adjoint_generator.__next__()
        adjoint_field.append(np.array(tmp[0]))
    adjoint_field.reverse()
    
    time_sum_eps=np.zeros_like(iepsilon)

    for k in range(1,config.k_max-1):
        u1=forward_field[k+1]
        u0=forward_field[k-1]
        p1=adjoint_field[k]
        time_sum_eps+=p1*(u1-u0)/config.dt/2

    g_eps=ep0*time_sum_eps
    
    g_eps[:10+config.air_layer,:]=0
    
    g_eps[:10,:]=0
    g_eps[-10:,:]=0
    g_eps[:,:10]=0
    g_eps[:,-10:]=0

    return rhs_trace.flatten(),g_eps.flatten()    

def misfit(iepsilon): 
    sigma=np.zeros_like(iepsilon)
    start_time=time.time()  
    CPML_Params=Add_CPML(config.xl,config.zl,sigma,iepsilon.copy(),config.dx,config.dz,config.dt)
    
    prediction_profile=ForwardModeling(sigma,iepsilon.copy(),fwi_config.wavelet_type,fwi_config.fwi_freq)._forward_2d()
    prediction_profile-=np.tile(fwi_config.air_trace,(prediction_profile.shape[1],1)).T
    ###########################################################################
    fwi_config.prediction_ref_profile=ClutterRemoval(prediction_profile,max_iter=1000,rank=1,lam=1e-4,method='GoDec').clutter_removal()
    g_eps=0.0
    rhs=[]
    NumOfProcess=int(500//(iepsilon.size*config.k_max/1024/1024/1024*8*2)-5)
    if NumOfProcess>=128:
        NumOfProcess=128
    pool=Pool(processes=NumOfProcess)
    res_l=[]
    
    for index,value in enumerate(config.source_site):
        res=pool.apply_async(calculate_gradient,args=(sigma.copy(),iepsilon.copy(),index,CPML_Params))
        res_l.append(res)
    pool.close()
    pool.join()
    
    for res in res_l:
        result=res.get()
        rhs.append(result[0])
        g_eps+=result[1]
        del result
    rhs=np.array(rhs)      
    
    f=0.5*np.linalg.norm(rhs.flatten(),2)**2
    
    #Get Modified Toltal Variation
    if config.regularization_key:
        # f_penalty_epsilon,g_penalty_epsilon=Regularization(config.regularization_method,alpha=1e-2,rho=1e-1,tau=1e-3,max_iter=50).penalty_value_gradient(iepsilon[10+config.air_layer:-10,10:-10].copy())
        f_penalty_epsilon,g_penalty_epsilon=Regularization(config.regularization_method,alpha=1e-2,rho=1e-1,tau=1e-3,max_iter=50).penalty_value_gradient(iepsilon.copy())
        # g_penalty_epsilon=np.zeros_like(iepsilon)
        # g_penalty_epsilon[10+config.air_layer:-10,10:-10]=g_penalty_epsilon_
        g_penalty_epsilon=g_penalty_epsilon.flatten()

            #Update Lambda
        lambda_=(np.linalg.norm(g_eps.flatten(),2))/(np.linalg.norm(g_penalty_epsilon.flatten(),2))*config.initial_weight
        
        print(f"****lambda={lambda_:15.5e}, \n"
              f"****fd={f:15.5e}, g={np.linalg.norm(g_eps, 2):15.5e}, \n"
              f"****f_penalty_epsilon={f_penalty_epsilon:15.5e}, \n"
              f"****g_penalty_epsilon={np.linalg.norm(g_penalty_epsilon, 2):15.5e} \n")
        
        f+=lambda_*f_penalty_epsilon
        g_eps+=lambda_*g_penalty_epsilon
    pool.terminate() 
    print('Misfit elapsed time is %s seconds !'%str(time.time()-start_time))
    return f,g_eps




