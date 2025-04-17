#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 15:24:01 2025

@author: nephilim
"""

import numpy as np
from matplotlib import pyplot,cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

if __name__=='__main__':
    data=np.load('explosion_rtm.npy')
    data=data[10:-10,10:-10]
    pyplot.figure(figsize=(8,6))
    pyplot.rcParams.update({'font.size': 15})
    gci=pyplot.imshow(data,extent=(0,400,200,0),cmap=cm.gray_r,vmin=-30,vmax=30)
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
    cbar.set_label('Amplitude')
    pyplot.savefig('00_SyntheticExplosion.png',dpi=1000)
    
    data=np.load('correlation_rtm_result.npy')
    data=data[10:-10,10:-10]
    pyplot.figure(figsize=(8,6))
    pyplot.rcParams.update({'font.size': 15})
    gci=pyplot.imshow(data,extent=(0,400,200,0),cmap=cm.gray_r,vmin=-2e4,vmax=2e4)
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
    pyplot.savefig('01_SyntheticCorrelationResult.png',dpi=1000)
    
    data=np.load('correlation_rtm_result.npy')
    data_=np.load('correlation_rtm_source.npy')
    data=data[10:-10,10:-10]/data_[10:-10,10:-10]
    pyplot.figure(figsize=(8,6))
    pyplot.rcParams.update({'font.size': 15})
    gci=pyplot.imshow(data,extent=(0,400,200,0),cmap=cm.gray_r,vmin=-5e-4,vmax=5e-4)
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
    pyplot.savefig('02_SyntheticCorrelationSource.png',dpi=1000)
    
    data=np.load('correlation_rtm_result.npy')
    data_=np.load('correlation_rtm_receiver.npy')
    data=data[10:-10,10:-10]/data_[10:-10,10:-10]
    pyplot.figure(figsize=(8,6))
    pyplot.rcParams.update({'font.size': 15})
    gci=pyplot.imshow(data,extent=(0,400,200,0),cmap=cm.gray_r,vmin=-60,vmax=60)
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
    cbar.set_label('Amplitude')
    pyplot.savefig('03_SyntheticCorrelationReceiver.png',dpi=1000)