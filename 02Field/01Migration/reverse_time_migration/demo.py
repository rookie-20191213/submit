#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 06:03:34 2025

@author: nephilim
"""

from config_parameters import config
from clutter_removal import ClutterRemoval
from forward_modeling import ForwardModeling
import numpy as np
import time
import skimage
import imaging_condition

if __name__=='__main__': 
    start_time=time.time() 
    # Forward modeling for the synthetic model
    # epsilon_=np.ones((config.xl,config.zl))*4
    # epsilon_[20:30,40:60]=1
    # epsilon=np.zeros((config.xl+20,config.zl+20))
    # epsilon[10:-10,10:-10]=epsilon_
    # epsilon[:10,:]=epsilon[10,:]
    # epsilon[-10:,:]=epsilon[-10-1,:]
    # epsilon[:,:10]=epsilon[:,10].reshape((len(epsilon[:,10]),-1))
    # epsilon[:,-10:]=epsilon[:,-10-1].reshape((len(epsilon[:,-10-1]),-1))
    # epsilon[:10+config.air_layer,:]=1
    # sigma=np.zeros_like(epsilon)
    # true_profile=ForwardModeling(sigma,epsilon.copy(),config.wavelet_type,config.frequency)._forward_2d()
    # air_trace=ForwardModeling(np.zeros((config.xl+20,config.zl+20)),np.ones((config.xl+20,config.zl+20)),config.wavelet_type,config.frequency)._forward_2d_air()
    # true_profile=true_profile-np.tile(air_trace,(true_profile.shape[1],1)).T
    # true_profile=skimage.transform.resize(true_profile,(config.k_max,len(config.source_site)))
    # config.true_profile=true_profile-ClutterRemoval(true_profile,max_iter=1000,rank=1,lam=1e-4,method='GoDec').clutter_removal()
    
    true_profile=np.load('without_ref_data.npy')
    config.true_profile=skimage.transform.resize(true_profile,(config.k_max,len(config.source_site)))
    
    # # Explosion
    # config.imaging_condition='explosion'
    # iepsilon_=np.load('InitModel1.npy')
    # iepsilon_=skimage.filters.gaussian(iepsilon_,sigma=1)
    # iepsilon=np.zeros((config.xl+20,config.zl+20))
    # iepsilon[10:-10,10:-10]=iepsilon_
    # iepsilon[:10,:]=iepsilon[10,:]
    # iepsilon[-10:,:]=iepsilon[-10-1,:]
    # iepsilon[:,:10]=iepsilon[:,10].reshape((len(iepsilon[:,10]),-1))
    # iepsilon[:,-10:]=iepsilon[:,-10-1].reshape((len(iepsilon[:,-10-1]),-1))
    # iepsilon[:10+config.air_layer,:]=1
    # iepsilon*=4
    # I=imaging_condition.imaging_condition(iepsilon)
    # np.save('explosion_rtm.npy',I)
    
    #Correlation
    # config.imaging_condition='explosion'
    config.imaging_condition='correlation'
    iepsilon_=np.load('InitModel1.npy')*0.9
    # iepsilon_=skimage.filters.gaussian(iepsilon_,sigma=30)
    iepsilon=np.zeros((config.xl+20,config.zl+20))
    iepsilon[10:-10,10:-10]=iepsilon_
    iepsilon[:10,:]=iepsilon[10,:]
    iepsilon[-10:,:]=iepsilon[-10-1,:]
    iepsilon[:,:10]=iepsilon[:,10].reshape((len(iepsilon[:,10]),-1))
    iepsilon[:,-10:]=iepsilon[:,-10-1].reshape((len(iepsilon[:,-10-1]),-1))
    iepsilon[:10+config.air_layer,:]=1
    # iepsilon*=4
    I=imaging_condition.imaging_condition(iepsilon)
    np.save('correlation_rtm_result.npy',I[0])
    np.save('correlation_rtm_source.npy',I[1])
    np.save('correlation_rtm_receiver.npy',I[2])
    
