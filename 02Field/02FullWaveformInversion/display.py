#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 15:24:01 2025

@author: nephilim
"""

import numpy as np
from matplotlib import pyplot,cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import create_model
from pathlib import Path
import skimage

if __name__=='__main__':
    select_index=285
    
    iepsilon_=np.load('InitModel.npy')
    data=create_model.Initial_Smooth_Model(iepsilon_,12,0)
    data[:2,:]=1
    initial_model=data[:,select_index]
    pyplot.figure(figsize=(8,6))
    pyplot.rcParams.update({'font.size': 15})
    gci=pyplot.imshow(data,extent=(0,400,75,0),cmap=cm.gray_r,vmin=1,vmax=7.5)
    ax=pyplot.gca()
    ax.set_xticks(np.linspace(0,400,5))
    ax.set_xticklabels([0,2,4,6,8])
    ax.set_yticks(np.linspace(0,75,4))
    ax.set_yticklabels([0,0.5,1,1.5])
    pyplot.xlabel('Distance (m)')
    pyplot.ylabel('Depth (m)')
    divider=make_axes_locatable(ax)
    cax=divider.append_axes('right', size='4%', pad=0.15)
    cbar=pyplot.colorbar(gci,cax=cax)
    cbar.set_label('$\epsilon_r$')
    pyplot.savefig('00_FWI_Initial.png',dpi=1000)
    
    ###########################################################################
    FWI_INFO=[]
    
    freq=2e8
    dir_path='./NoReg/%sHz_imodel_file'%freq
    file_num=int(len(list(Path(dir_path).iterdir())))
    data=np.load('%s/%s_imodel.npy'%(dir_path,file_num//2-1))
    
    info=np.load('%s/%s_info.npy'%(dir_path,file_num//2-1))
    FWI_INFO.append(info)
    
    data=data.reshape((95,-1))
    data=data[10:-10,10:-10]
    noreg_model2e8=data[:,select_index]
    # pyplot.figure(figsize=(8,6))
    # pyplot.rcParams.update({'font.size': 15})
    # gci=pyplot.imshow(data,extent=(0,300,75,0),cmap=cm.gray_r,vmin=1,vmax=7.5)
    # ax=pyplot.gca()
    # ax.set_xticks(np.linspace(0,400,5))
    # ax.set_xticklabels([0,2,4,6,8])
    # ax.set_yticks(np.linspace(0,200,5))
    # ax.set_yticklabels([0,1,2,3,4])
    # pyplot.xlabel('Distance (m)')
    # pyplot.ylabel('Depth (m)')
    # divider=make_axes_locatable(ax)
    # cax=divider.append_axes('right', size='4%', pad=0.15)
    # cbar=pyplot.colorbar(gci,cax=cax)
    # cbar.formatter.set_powerlimits((-1, 1))
    # cbar.set_label('$\epsilon_r$')
    # pyplot.savefig('01_without_regularization_2e8.png',dpi=1000)
    
    freq=4e8
    dir_path='./NoReg/%sHz_imodel_file'%freq
    file_num=int(len(list(Path(dir_path).iterdir())))
    data=np.load('%s/%s_imodel.npy'%(dir_path,file_num//2-1))
    
    info=np.load('%s/%s_info.npy'%(dir_path,file_num//2-1))
    FWI_INFO.append(info)

    data=data.reshape((95,-1))
    data=data[10:-10,10:-10]
    noreg_model4e8=data[:,select_index]
    pyplot.figure(figsize=(8,6))
    pyplot.rcParams.update({'font.size': 15})
    gci=pyplot.imshow(data,extent=(0,400,75,0),cmap=cm.gray_r,vmin=1,vmax=7.5)
    ax=pyplot.gca()
    ax.set_xticks(np.linspace(0,400,5))
    ax.set_xticklabels([0,2,4,6,8])
    ax.set_yticks(np.linspace(0,75,4))
    ax.set_yticklabels([0,0.5,1,1.5])
    pyplot.xlabel('Distance (m)')
    pyplot.ylabel('Depth (m)')
    divider=make_axes_locatable(ax)
    cax=divider.append_axes('right', size='4%', pad=0.15)
    cbar=pyplot.colorbar(gci,cax=cax)
    cbar.formatter.set_powerlimits((-1, 1))
    cbar.set_label('$\epsilon_r$')
    pyplot.savefig('02_without_regularization_4e8.png',dpi=1000)

    data_=[]
    for info in FWI_INFO:
        for info_ in info:
            data_.append(info_[3])
    np.save('./NoReg/Loss.npy',data_)
    
    with open('./NoReg/history.txt','w') as fid:
        for dd in FWI_INFO:
            fid.write(str(dd))
            fid.write('\n')
    
    pyplot.figure(figsize=(8,6))
    pyplot.plot(data_)
    pyplot.grid('on')
    ax=pyplot.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.set_aspect(abs((xlim[1] - xlim[0]) / (ylim[1] - ylim[0])) / 2)
    # pyplot.yscale('log')
    # with open('history.txt', 'r') as fid:
    #     read_list = [eval(line.strip()) for line in fid if line.strip()]
    pyplot.savefig('03_without_regularization_loss.png',dpi=1000)

    ###########################################################################    
    
    ###########################################################################
    FWI_INFO=[]
    
    freq=2e8
    dir_path='./TIK/%sHz_imodel_file'%freq
    file_num=int(len(list(Path(dir_path).iterdir())))
    data=np.load('%s/%s_imodel.npy'%(dir_path,file_num//2-1))
    
    info=np.load('%s/%s_info.npy'%(dir_path,file_num//2-1))
    FWI_INFO.append(info)
    
    data=data.reshape((95,-1))
    data=data[10:-10,10:-10]
    tikhonov_model2e8=data[:,select_index]
    # pyplot.figure(figsize=(8,6))
    # pyplot.rcParams.update({'font.size': 15})
    # gci=pyplot.imshow(data,extent=(0,300,75,0),cmap=cm.gray_r,vmin=1,vmax=7.5)
    # ax=pyplot.gca()
    # ax.set_xticks(np.linspace(0,400,5))
    # ax.set_xticklabels([0,2,4,6,8])
    # ax.set_yticks(np.linspace(0,200,5))
    # ax.set_yticklabels([0,1,2,3,4])
    # pyplot.xlabel('Distance (m)')
    # pyplot.ylabel('Depth (m)')
    # divider=make_axes_locatable(ax)
    # cax=divider.append_axes('right', size='4%', pad=0.15)
    # cbar=pyplot.colorbar(gci,cax=cax)
    # cbar.formatter.set_powerlimits((-1, 1))
    # cbar.set_label('$\epsilon_r$')
    # pyplot.savefig('04_tikhonov_2e8.png',dpi=1000)
    
    freq=4e8
    dir_path='./TIK/%sHz_imodel_file'%freq
    file_num=int(len(list(Path(dir_path).iterdir())))
    data=np.load('%s/%s_imodel.npy'%(dir_path,file_num//2-1))
    
    info=np.load('%s/%s_info.npy'%(dir_path,file_num//2-1))
    FWI_INFO.append(info)

    data=data.reshape((95,-1))
    data=data[10:-10,10:-10]
    tikhonov_model4e8=data[:,select_index]
    pyplot.figure(figsize=(8,6))
    pyplot.rcParams.update({'font.size': 15})
    gci=pyplot.imshow(data,extent=(0,400,75,0),cmap=cm.gray_r,vmin=1,vmax=7.5)
    ax=pyplot.gca()
    ax.set_xticks(np.linspace(0,400,5))
    ax.set_xticklabels([0,2,4,6,8])
    ax.set_yticks(np.linspace(0,75,4))
    ax.set_yticklabels([0,0.5,1,1.5])
    pyplot.xlabel('Distance (m)')
    pyplot.ylabel('Depth (m)')
    divider=make_axes_locatable(ax)
    cax=divider.append_axes('right', size='4%', pad=0.15)
    cbar=pyplot.colorbar(gci,cax=cax)
    cbar.formatter.set_powerlimits((-1, 1))
    cbar.set_label('$\epsilon_r$')
    pyplot.savefig('05_tikhonov_4e8.png',dpi=1000)

    data_=[]
    for info in FWI_INFO:
        for info_ in info:
            data_.append(info_[3])
    np.save('./TIK/Loss.npy',data_)
    
    with open('./TIK/history.txt','w') as fid:
        for dd in FWI_INFO:
            fid.write(str(dd))
            fid.write('\n')
    
    pyplot.figure(figsize=(8,6))
    pyplot.plot(data_)
    pyplot.grid('on')
    ax=pyplot.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.set_aspect(abs((xlim[1] - xlim[0]) / (ylim[1] - ylim[0])) / 2)
    
    # pyplot.yscale('log')
    # with open('history.txt', 'r') as fid:
    #     read_list = [eval(line.strip()) for line in fid if line.strip()]
    pyplot.savefig('06_tikhonov_loss.png',dpi=1000)

    ###########################################################################    
    FWI_INFO=[]
    
    freq=2e8
    dir_path='./MTV/%sHz_imodel_file'%freq
    file_num=int(len(list(Path(dir_path).iterdir())))
    data=np.load('%s/%s_imodel.npy'%(dir_path,file_num//2-1))
    
    info=np.load('%s/%s_info.npy'%(dir_path,file_num//2-1))
    FWI_INFO.append(info)
        
    data=data.reshape((95,-1))
    data=data[10:-10,10:-10]
    mtv_model2e8=data[:,select_index]
    # pyplot.figure(figsize=(8,6))
    # pyplot.rcParams.update({'font.size': 15})
    # gci=pyplot.imshow(data,extent=(0,300,75,0),cmap=cm.gray_r,vmin=1,vmax=7.5)
    # ax=pyplot.gca()
    # ax.set_xticks(np.linspace(0,400,5))
    # ax.set_xticklabels([0,2,4,6,8])
    # ax.set_yticks(np.linspace(0,200,5))
    # ax.set_yticklabels([0,1,2,3,4])
    # pyplot.xlabel('Distance (m)')
    # pyplot.ylabel('Depth (m)')
    # divider=make_axes_locatable(ax)
    # cax=divider.append_axes('right', size='4%', pad=0.15)
    # cbar=pyplot.colorbar(gci,cax=cax)
    # cbar.formatter.set_powerlimits((-1, 1))
    # cbar.set_label('$\epsilon_r$')
    # pyplot.savefig('10_mtv_2e8.png',dpi=1000)

    freq=4e8
    dir_path='./MTV/%sHz_imodel_file'%freq
    file_num=int(len(list(Path(dir_path).iterdir())))
    data=np.load('%s/%s_imodel.npy'%(dir_path,file_num//2-1))
    
    info=np.load('%s/%s_info.npy'%(dir_path,file_num//2-1))
    FWI_INFO.append(info)
        
    data=data.reshape((95,-1))
    data=data[10:-10,10:-10]
    mtv_model4e8=data[:,select_index]
    pyplot.figure(figsize=(8,6))
    pyplot.rcParams.update({'font.size': 15})
    gci=pyplot.imshow(data,extent=(0,400,75,0),cmap=cm.gray_r,vmin=1,vmax=7.5)
    ax=pyplot.gca()
    ax.set_xticks(np.linspace(0,400,5))
    ax.set_xticklabels([0,2,4,6,8])
    ax.set_yticks(np.linspace(0,75,4))
    ax.set_yticklabels([0,0.5,1,1.5])
    pyplot.xlabel('Distance (m)')
    pyplot.ylabel('Depth (m)')
    divider=make_axes_locatable(ax)
    cax=divider.append_axes('right', size='4%', pad=0.15)
    cbar=pyplot.colorbar(gci,cax=cax)
    cbar.formatter.set_powerlimits((-1, 1))
    cbar.set_label('$\epsilon_r$')
    pyplot.savefig('11_mtv_4e8.png',dpi=1000)

    data_=[]
    for info in FWI_INFO:
        for info_ in info:
            data_.append(info_[3])
    np.save('./MTV/Loss.npy',data_)
    
    with open('./MTV/history.txt','w') as fid:
        for dd in FWI_INFO:
            fid.write(str(dd))
            fid.write('\n')
    
    pyplot.figure(figsize=(8,6))
    pyplot.plot(data_,'r')
    # pyplot.grid('on')
    # ax=pyplot.gca()
    # xlim = ax.get_xlim()
    # ylim = ax.get_ylim()
    # ax.set_aspect(abs((xlim[1] - xlim[0]) / (ylim[1] - ylim[0])) / 2)
    # # pyplot.yscale('log')
    # # with open('history.txt', 'r') as fid:
    # #     read_list = [eval(line.strip()) for line in fid if line.strip()]
    
    pyplot.grid('on')
    ax=pyplot.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.set_aspect(abs((xlim[1] - xlim[0]) / (ylim[1] - ylim[0])) / 2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Value of Misfit Function') 
    
    
    pyplot.savefig('12_mtv_loss.png',dpi=1000)

    pyplot.figure(figsize=(12,10))
    pyplot.rcParams.update({'font.size': 15})
    l1=pyplot.plot(initial_model,'c:',label='Initial Model')
    l2=pyplot.plot(noreg_model4e8,'g-.',label='Without Regularization')
    # l3=pyplot.plot(tikhonov_model4e8,'b--',label='Tikhonov Regularization')
    # l4=pyplot.plot(tv_model4e8,'g-.',label='TV Regularization')
    l5=pyplot.plot(mtv_model4e8,'r-',label='MTV Regularization')
    ax=pyplot.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.set_aspect(abs((xlim[1] - xlim[0]) / (ylim[1] - ylim[0])) / 4)
    pyplot.grid('on')
    ax=pyplot.gca()
    ax.set_xticks(np.linspace(xlim[0],xlim[1],9))
    ax.set_xticklabels([0,0.5,1,1.5,2,2.5,3,3.5,4])
    ax.set_xlabel('Depth (m)')
    ax.set_ylabel('$\epsilon_r$') 
    lns=l1+l2+l5
    labs=[l.get_label() for l in lns]
    ax.legend(lns,labs,loc='best', fontsize=12)
    pyplot.grid(linestyle='--')
    pyplot.savefig('13_fwi_results.png',dpi=1000)
    # pyplot.gca().invert_yaxis()
    # ax=pyplot.gca()
    # ax.set_yticks(np.linspace(0,200,5))
    # ax.set_yticklabels([0,0.5,1,1.5,2])
    # pyplot.grid('on')
    # ax=pyplot.gca()
    # ax.set_ylabel('Depth (m)')
    # ax.set_xlabel('$\epsilon_r$') 
    # lns=l0+l1+l2+l3+l4
    # labs=[l.get_label() for l in lns]
    # ax.legend(lns,labs,loc='best', fontsize=12)
    # pyplot.grid(linestyle='--')

    
    
    data1=np.load('./NoReg/Loss.npy')
    data2=np.load('./TIK/Loss.npy')
    data3=np.load('./MTV/Loss.npy')
    
    pyplot.figure(figsize=(8,6))
    l1=pyplot.plot(data1,'g-.',label='Without Regularization')
    # l2=pyplot.plot(data2,'b--',label='Tikhonov Regularization')
    l3=pyplot.plot(data3,'r-',label='MTV Regularization')
    pyplot.grid('on')
    ax=pyplot.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.set_aspect(abs((xlim[1] - xlim[0]) / (ylim[1] - ylim[0])) / 2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Value of Misfit Function') 
    lns=l1+l3
    labs=[l.get_label() for l in lns]
    ax.legend(lns,labs,loc='best', fontsize=12)
    pyplot.grid(linestyle='--')
    pyplot.savefig('14_fwi_results.png',dpi=1000)
