#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 17:58:37 2018

@author: nephilim
"""
from numba import jit
import numpy as np
from skimage import filters
import scipy.io as scio
#
@jit(nopython=True)
def Abnormal_Model(xl,zl,CPML):
    epsilon=np.ones((xl+2*CPML,zl+2*CPML))*4
    p=34+CPML
    l=14
    w=3
    for i in range(p-l,p+l):
        for j in range(p-w,p+w):
            epsilon[i][j]=1
    for i in range(p-w,p+w):
        for j in range(p-l,p+l):
            epsilon[i][j]=1
    p = zl+2*CPML-p
    for i in range(p-l,p+l):
        for j in range(p-w,p+w):
            epsilon[i][j]=8
    for i in range(p-w,p+w):
        for j in range(p-l,p+l):
            epsilon[i][j]=8
    return epsilon

@jit(nopython=True)
def Create_iModel(xl,zl,CPML,ep):
    epsilon=np.ones((xl+2*CPML,zl+2*CPML))*ep
    return epsilon

#Create Initial_Overthrust Model
def Initial_Smooth_Model(epsilon_,sig, layer):
    iepsilon=filters.gaussian(epsilon_,sigma=sig)
    iepsilon[:layer,:]=1
    return iepsilon