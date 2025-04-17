#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 05:03:14 2025

@author: nephilim
"""
from dataclasses import dataclass, field
import numpy as np

class PartiallyFrozen:
    _frozen_fields = {'wavelet_type'}

    def __setattr__(self, key, value):
        if key in self._frozen_fields and key in self.__dict__:
            raise AttributeError(f"Field '{key}' is frozen and cannot be modified.")
        super().__setattr__(key, value)
        
@dataclass
class FWIParameters(PartiallyFrozen):
    wavelet_type: str = 'gaussian'
    
    fwi_freq: int = field(init=False)
    air_trace: np.ndarray = field(default_factory=lambda: np.array([]), init=False)
    prediction_ref_profile: np.ndarray = field(default_factory=lambda: np.array([]), init=False)

fwi_config = FWIParameters()