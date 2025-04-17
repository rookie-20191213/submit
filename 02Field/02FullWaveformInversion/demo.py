#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 06:03:34 2025

@author: nephilim
"""

from config_parameters import config
from fwi_parameters import fwi_config
from clutter_removal import ClutterRemoval
from optimization import Optimization
from forward_modeling import ForwardModeling
import gradient_calculation
import numpy as np
import time
from pathlib import Path
from matplotlib import pyplot,cm
import create_model
import skimage

if __name__=='__main__': 
    start_time=time.time() 
    
    true_profile=np.load('./FieldData1.npy')
    true_profile=skimage.transform.resize(true_profile,(config.k_max,len(config.source_site)))
    # true_profile=multi_scale.apply_filter(true_profile, fs=1/config.dt, cutoff=freq_)
    
    config.true_profile=true_profile
    config.true_ref_profile=ClutterRemoval(config.true_profile,max_iter=1000,rank=1,lam=1e-4,method='GoDec').clutter_removal()
    
    print('Forward Done !')
    print('Elapsed time is %s seconds !'%str(time.time()-start_time))
    ###########################################################################

    ###########################################################################
    #Multiscale
    FWI_INFO=[]
    frequence=[2e8,4e8]
    for idx,freq_ in enumerate(frequence):
        # if idx==0:
        #     continue
        #Ricker wavelet main frequence
        
        fwi_config.fwi_freq=freq_
        fwi_config.air_trace=ForwardModeling(np.zeros((config.xl+20,config.zl+20)),np.ones((config.xl+20,config.zl+20)),fwi_config.wavelet_type,fwi_config.fwi_freq)._forward_2d_air()
        #If the first frequence,Create Initial Model
        if idx==0:
            iepsilon_=np.load('InitModel.npy')
            iepsilon_=create_model.Initial_Smooth_Model(iepsilon_,10,0)
            # iepsilon_ = (iepsilon_ - iepsilon_.min()) / (iepsilon_.max() - iepsilon_.min())
            # iepsilon_ = iepsilon_ * (epsilon_.max() - epsilon_.min()) + epsilon_.min() 
            iepsilon=np.zeros((config.xl+20,config.zl+20))
            iepsilon[10:-10,10:-10]=iepsilon_
            iepsilon[:10,:]=iepsilon[10,:]
            iepsilon[-10:,:]=iepsilon[-10-1,:]
            iepsilon[:,:10]=iepsilon[:,10].reshape((len(iepsilon[:,10]),-1))
            iepsilon[:,-10:]=iepsilon[:,-10-1].reshape((len(iepsilon[:,-10-1]),-1))
            iepsilon[:10+config.air_layer,:]=1
        #If the first frequence,Using the last final model
        else:
            dir_path='./%sHz_imodel_file'%frequence[idx-1]
            file_num=int(len(list(Path(dir_path).iterdir())))
            data=np.load('./%sHz_imodel_file/%s_imodel.npy'%(frequence[idx-1],file_num//2-1))
            iepsilon=data.reshape((config.xl+20,-1))
                
        # # Test Gradient
        # f,g=gradient_calculation.misfit(iepsilon)
        # pyplot.figure()
        # pyplot.imshow(g.reshape((config.xl+20,-1)))

        
        #Options Params
        Optimization_=Optimization(gradient_calculation.misfit,iepsilon,tol=1e-3,maxiter=50)
        imodel,info=Optimization_.optimization()
        
        FWI_INFO.append(info)
        pyplot.figure()
        pyplot.imshow(imodel.reshape((config.xl+20,-1)),cmap=cm.jet)
        
        # Plot Error Data
    pyplot.figure()
    data_=[]
    for info in FWI_INFO:
        for info_ in info:
            data_.append(info_[3])
    pyplot.plot(data_)
    pyplot.yscale('log')
    print('Elapsed time is %s seconds !'%str(time.time()-start_time))
    np.save('Loss.npy',data_)
    
    with open('history.txt','w') as fid:
        for dd in FWI_INFO:
            fid.write(str(dd))
            fid.write('\n')
            
    with open('history.txt', 'r') as fid:
        read_list = [eval(line.strip()) for line in fid if line.strip()]
