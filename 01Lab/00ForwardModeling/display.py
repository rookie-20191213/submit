#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 15:24:01 2025

@author: nephilim
"""

import numpy as np
from matplotlib import pyplot,cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from clutter_removal import ClutterRemoval
import time

if __name__=='__main__':
    data=np.load('./LabData.npy')
    pyplot.figure(figsize=(8,6))
    pyplot.rcParams.update({'font.size': 15})
    gci=pyplot.imshow(data,extent=(0,400,100,0),cmap=cm.gray_r,vmin=-1,vmax=1)
    ax=pyplot.gca()
    ax.set_xticks(np.linspace(0,400,5))
    ax.set_xticklabels([0,1,2,3,4])
    ax.set_yticks(np.linspace(0,100,7))
    ax.set_yticklabels([0,2,4,6,8,10,12])
    pyplot.xlabel('Distance (m)')
    pyplot.ylabel('Time (ns)')
    divider=make_axes_locatable(ax)
    cax=divider.append_axes('right', size='4%', pad=0.15)
    cbar=pyplot.colorbar(gci,cax=cax)
    cbar.set_label('Amplitude')
    pyplot.savefig('01_LabProfile.png',dpi=1000)
    
    start_time=time.time()
    max_iter=1000
    rank=1
    lam=1e-4
    method='GoDec'
    direct1=ClutterRemoval(data,max_iter,rank,lam,method).clutter_removal()    
    print(time.time()-start_time)
    pyplot.figure(figsize=(8,6))
    pyplot.rcParams.update({'font.size': 15})
    gci=pyplot.imshow(direct1,extent=(0,400,100,0),cmap=cm.gray_r,vmin=-1,vmax=1)
    ax=pyplot.gca()
    ax.set_xticks(np.linspace(0,400,5))
    ax.set_xticklabels([0,1,2,3,4])
    ax.set_yticks(np.linspace(0,100,7))
    ax.set_yticklabels([0,2,4,6,8,10,12])
    pyplot.xlabel('Distance (m)')
    pyplot.ylabel('Time (ns)')
    divider=make_axes_locatable(ax)
    cax=divider.append_axes('right', size='4%', pad=0.15)
    cbar=pyplot.colorbar(gci,cax=cax)
    cbar.set_label('Amplitude')
    pyplot.savefig('02_LabProfileDirect_GoDec.png',dpi=1000)
    
    start_time=time.time()
    method='RNMF'
    direct2=ClutterRemoval(data,max_iter,rank,lam,method).clutter_removal()    
    print(time.time()-start_time)
    pyplot.figure(figsize=(8,6))
    pyplot.rcParams.update({'font.size': 15})
    gci=pyplot.imshow(direct2,extent=(0,400,100,0),cmap=cm.gray_r,vmin=-1,vmax=1)
    ax=pyplot.gca()
    ax.set_xticks(np.linspace(0,400,5))
    ax.set_xticklabels([0,1,2,3,4])
    ax.set_yticks(np.linspace(0,100,7))
    ax.set_yticklabels([0,2,4,6,8,10,12])
    pyplot.xlabel('Distance (m)')
    pyplot.ylabel('Time (ns)')
    divider=make_axes_locatable(ax)
    cax=divider.append_axes('right', size='4%', pad=0.15)
    cbar=pyplot.colorbar(gci,cax=cax)
    cbar.set_label('Amplitude')
    pyplot.savefig('03_LabProfileDirect_RNMF.png',dpi=1000)
    
    data=np.load('./LabData.npy')-direct1
    pyplot.figure(figsize=(8,6))
    pyplot.rcParams.update({'font.size': 15})
    gci=pyplot.imshow(data,extent=(0,400,100,0),cmap=cm.gray_r,vmin=-1,vmax=1)
    ax=pyplot.gca()
    ax.set_xticks(np.linspace(0,400,5))
    ax.set_xticklabels([0,1,2,3,4])
    ax.set_yticks(np.linspace(0,100,7))
    ax.set_yticklabels([0,2,4,6,8,10,12])
    pyplot.xlabel('Distance (m)')
    pyplot.ylabel('Time (ns)')
    divider=make_axes_locatable(ax)
    cax=divider.append_axes('right', size='4%', pad=0.15)
    cbar=pyplot.colorbar(gci,cax=cax)
    cbar.set_label('Amplitude')
    pyplot.savefig('04_LabProfileWithouDirect_GoDec.png',dpi=1000)
    
    data=np.load('./LabData.npy')-direct2
    pyplot.figure(figsize=(8,6))
    pyplot.rcParams.update({'font.size': 15})
    gci=pyplot.imshow(data,extent=(0,400,100,0),cmap=cm.gray_r,vmin=-1,vmax=1)
    ax=pyplot.gca()
    ax.set_xticks(np.linspace(0,400,5))
    ax.set_xticklabels([0,1,2,3,4])
    ax.set_yticks(np.linspace(0,100,7))
    ax.set_yticklabels([0,2,4,6,8,10,12])
    pyplot.xlabel('Distance (m)')
    pyplot.ylabel('Time (ns)')
    divider=make_axes_locatable(ax)
    cax=divider.append_axes('right', size='4%', pad=0.15)
    cbar=pyplot.colorbar(gci,cax=cax)
    cbar.set_label('Amplitude')
    pyplot.savefig('05_LabProfileWithouDirect_GoDec.png',dpi=1000)