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
from mpl_toolkits.axes_grid1 import make_axes_locatable

if __name__=='__main__':
    epsilon=np.load('fwi_data.npy')
    epsilon=model_creation.expand_pml(epsilon)
    sigma=np.zeros_like(epsilon)
    profile=ForwardModeling(sigma.copy(),epsilon.copy(),wavelet_type=config.wavelet_type,freq=config.frequency)._forward_2d()
    air_trace=ForwardModeling(np.zeros_like(epsilon),np.ones_like(epsilon),wavelet_type=config.wavelet_type,freq=config.frequency)._forward_2d_air()
    profile-=np.tile(air_trace,(profile.shape[1],1)).T
    
    pyplot.figure()
    pyplot.figure(figsize=(8,6))
    pyplot.rcParams.update({'font.size': 15})
    gci=pyplot.imshow(profile,extent=(0,400,75,0),cmap=cm.gray_r,vmin=-1e2,vmax=1e2)
    ax=pyplot.gca()
    ax.set_xticks(np.linspace(0,400,5))
    ax.set_xticklabels([0,2,4,6,8])
    ax.set_yticks(np.linspace(0,75,5))
    ax.set_yticklabels([0,5,10,15,20])
    pyplot.xlabel('Distance (m)')
    pyplot.ylabel('Time (ns)')
    divider=make_axes_locatable(ax)
    cax=divider.append_axes('right', size='4%', pad=0.15)
    cbar=pyplot.colorbar(gci,cax=cax)
    cbar.set_label('Amplitude')
    cbar.formatter.set_powerlimits((-1, 1))
    pyplot.savefig('fwi_forward.png', dpi=1000)
    # np.save('forward_without_air.npy',profile)
    
    # # ref_data=clutter_removal.ClutterRemoval(profile.copy(),max_iter=50,rank=1,lam=1e-4,method='GoDec').clutter_removal()
    # # np.save('direct.npy',ref_data)

    # # pyplot.figure()
    # # pyplot.imshow(profile-ref_data,extent=(0,2,1,0),cmap=cm.jet)
    
    # data=np.load('Field.npy')
    # pyplot.figure()
    # pyplot.imshow(data,extent=(0,400,75,0),cmap=cm.gray_r)
