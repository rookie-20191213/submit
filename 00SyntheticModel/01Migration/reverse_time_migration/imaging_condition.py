#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 06:03:20 2025

@author: nephilim
"""

from multiprocessing import Pool
import numpy as np
import time

from config_parameters import config
from add_cpml import Add_CPML
from wavelet_creation import WaveletType
import update_field_forward
import update_field_reverse 


def imaging_condition_explosion(sigma,epsilon,index,CPML_Params):
    reverse_generator=update_field_reverse.reverse_time_loop(config.xl,config.zl,config.dx,config.dz,
                                                             config.dt,sigma,epsilon.copy(),
                                                             CPML_Params,config.k_max,
                                                             config.receiver_site[index],config.true_profile[:,index])
    for idx in range(config.k_max):
        reverse_data=reverse_generator.__next__()
        reverse_data=np.array(reverse_data[0])
    return reverse_data
    
def imaging_condition_correlation(sigma,epsilon,index,CPML_Params):
    #Get Forward Params
    f=WaveletType(config.t, config.frequency, config.wavelet_type).create_wavelet()
    #Get Forward Data ----> <Generator>
    forward_generator=update_field_forward.time_loop(config.xl,config.zl,config.dx,config.dz,config.dt,
                                                     sigma,epsilon.copy(),CPML_Params,f,config.k_max,
                                                     config.source_site[index],config.receiver_site[index])
    #Get Generator Data
    forward_field=[]
    for idx in range(config.k_max):
        tmp=forward_generator.__next__()
        forward_field.append(np.array(tmp[0]))
    

    #Get Reversion Data ----> <Generator>
    reverse_generator=update_field_reverse.reverse_time_loop(config.xl,config.zl,config.dx,config.dz,
                                                             config.dt,sigma,epsilon.copy(),
                                                             CPML_Params,config.k_max,
                                                             config.receiver_site[index],config.true_profile[:,index])
    #Get Generator Data
    reverse_field=[]
    for idx in range(config.k_max):
        tmp=reverse_generator.__next__()
        reverse_field.append(np.array(tmp[0]))
    reverse_field.reverse()
    
    time_sum0=np.zeros((config.xl+2*CPML_Params.npml,config.zl+2*CPML_Params.npml))
    time_sum_source0=np.zeros((config.xl+2*CPML_Params.npml,config.zl+2*CPML_Params.npml))
    time_sum_source1=np.zeros((config.xl+2*CPML_Params.npml,config.zl+2*CPML_Params.npml))

    for k in range(config.k_max):
        f_f=forward_field[k]
        r_f=reverse_field[k]
        time_sum0+=f_f*r_f
        time_sum_source0+=f_f**2
        time_sum_source1+=r_f**2
        
    I_image0=time_sum0
    I_source0=time_sum_source0
    I_source1=time_sum_source1
    
    return I_image0,I_source0,I_source1  

def imaging_condition(iepsilon): 
    sigma=np.zeros_like(iepsilon)
    start_time=time.time()  
    CPML_Params=Add_CPML(config.xl,config.zl,sigma,iepsilon.copy(),config.dx,config.dz,config.dt)    
    
    pool=Pool(processes=128)
    res_l=[]
    if config.imaging_condition=='explosion':
        imaging_result=0
        for index,value in enumerate(config.receiver_site):
            res=pool.apply_async(imaging_condition_explosion,args=(sigma.copy(),iepsilon.copy(),index,CPML_Params))
            res_l.append(res)
        pool.close()
        pool.join()
        
        for res in res_l:
            result=res.get()
            imaging_result+=result
            del result
        pool.terminate() 
        print('Misfit elapsed time is %s seconds !'%str(time.time()-start_time))
        return imaging_result
        
    elif config.imaging_condition=='correlation':
        imaging_result=0
        source_result=0
        receiver_result=0
        for index,value in enumerate(config.receiver_site):
            res=pool.apply_async(imaging_condition_correlation,args=(sigma.copy(),iepsilon.copy(),index,CPML_Params))
            res_l.append(res)
        pool.close()
        pool.join()
        
        for res in res_l:
            result=res.get()
            imaging_result+=result[0]
            source_result+=result[1]
            receiver_result+=result[2]
            del result
        pool.terminate() 
        print('Misfit elapsed time is %s seconds !'%str(time.time()-start_time))
        return imaging_result,source_result,receiver_result
