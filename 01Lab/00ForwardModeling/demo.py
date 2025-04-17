#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 07:48:44 2025

@author: nephilim
"""

import numpy as np
from matplotlib import pyplot,cm
from config_parameters import config
from forward_modeling import ForwardModeling
import model_creation
import clutter_removal

if __name__=='__main__':
    epsilon=np.load('OverThrust.npy')
    epsilon=model_creation.expand_pml(epsilon)
    sigma=np.zeros_like(epsilon)
    profile=ForwardModeling(sigma.copy(),epsilon.copy(),wavelet_type=config.wavelet_type,freq=config.frequency)._forward_2d()
    air_trace=ForwardModeling(np.zeros_like(epsilon),np.ones_like(epsilon),wavelet_type=config.wavelet_type,freq=config.frequency)._forward_2d_air()
    profile-=np.tile(air_trace,(profile.shape[1],1)).T
    
    pyplot.figure()
    pyplot.imshow(profile,extent=(0,2,1,0),cmap=cm.jet)
    np.save('forward_without_air.npy',profile)
    
    ref_data=clutter_removal.ClutterRemoval(profile.copy(),max_iter=50,rank=1,lam=1e-4,method='GoDec').clutter_removal()
    np.save('direct.npy',ref_data)

    pyplot.figure()
    pyplot.imshow(profile-ref_data,extent=(0,2,1,0),cmap=cm.jet)
