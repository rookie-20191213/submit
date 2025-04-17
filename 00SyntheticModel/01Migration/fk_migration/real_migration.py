#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 03:01:13 2025

@author: nephilim
"""

import numpy as np
import Wavelet
from matplotlib import pyplot,cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

if __name__=='__main__':
    ep=np.load('OverThrust.npy')
    
    # x_position=50
    # z_position=200
    # Radium=20
    
    # for idx_x in range(ep.shape[0]):
    #     for idx_z in range(ep.shape[1]):
    #         if (idx_x-x_position)**2+(idx_z-z_position)**2<=Radium**2:
    #             ep[idx_x,idx_z]=1
    # # ep=ep[:,::10]
    
    # # ep=5*np.ones((100,200))
    # # ep[:50,:]=7
    
    f=3e9
    t=np.arange(-89,100)*4e-11
    v=Wavelet.ricker(t,f)
    R=(np.sqrt(1/ep[:-1,:])-np.sqrt(1/ep[1:,:]))/(np.sqrt(1/ep[:-1,:])+np.sqrt(1/ep[1:,:]))
    Re=[]
    for idx in range(400):
        Re.append(np.convolve(v,R[:,idx],mode='same'))
        
    Re=np.array(Re)
    Re=Re.T
    # pyplot.figure()
    # pyplot.imshow(R,extent=(0,2,1,0),cmap=cm.seismic,vmin=-0.01,vmax=0.01)
    # # pyplot.savefig('Ceof.png',dpi=1000)
    # pyplot.figure()
    # pyplot.imshow(Re,extent=(0,2,1,0),cmap=cm.seismic,vmin=-0.4,vmax=0.4)
    # # pyplot.savefig('Migration.png',dpi=1000)
    # pyplot.figure()
    # pyplot.imshow(ep,extent=(0,2,1,0),cmap=cm.seismic,vmin=1,vmax=12)
    # # pyplot.savefig('Ep.png',dpi=1000)
    
    pyplot.figure(figsize=(8,6))
    pyplot.rcParams.update({'font.size': 15})
    gci=pyplot.imshow(Re,extent=(0,400,200,0),cmap=cm.gray_r,vmin=-0.2,vmax=0.2)
    ax=pyplot.gca()
    ax.set_xticks(np.linspace(0,400,5))
    ax.set_xticklabels([0,2,4,6,8])
    ax.set_yticks(np.linspace(0,200,5))
    ax.set_yticklabels([0,1,2,3,4])
    pyplot.xlabel('Distance (m)')
    pyplot.ylabel('Depth (m)')
    divider=make_axes_locatable(ax)
    cax=divider.append_axes('right', size='4%', pad=0.15)
    cbar=pyplot.colorbar(gci,cax=cax)
    cbar.formatter.set_powerlimits((-1, 1))
    cbar.set_label('Amplitude')
    pyplot.savefig('REAL.png',dpi=1000)
