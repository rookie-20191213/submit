#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 00:03:48 2025

@author: nephilim
"""

import numpy as np
import matplotlib.pyplot as plt

data=np.load('./MTV/200000000.0Hz_imodel_file/11_imodel.npy')
ep=data.reshape((220,-1))
velocity_model=ep[10:-10,10:-10]

# 计算水平方向梯度作为局部突变指标
# 可以考虑同时计算垂直方向梯度，但这里主要以水平梯度为例
gradient_x = np.abs(np.gradient(velocity_model, axis=1))

# 采用指数参数调整权重映射：
# 对于全局平滑权重，我们采用较小的指数p，使得在平滑区域梯度较小时权重更高（宽松约束）
# 对于局部边缘权重，我们采用较大的指数q，使得在突变区域梯度较大时权重迅速趋近于1（严格约束）
p = 1.1  # 全局权重指数（宽松检测）
q = 1.1    # 局部权重指数（严格检测）

global_weight = 1 / (1 + gradient_x**p)
local_weight = gradient_x**q / (1 + gradient_x**q)

global_weight=(global_weight-np.min(global_weight))/(np.max(global_weight)-np.min(global_weight))

# 构建局部突变权重映射：在梯度较大区域权重大，可以使用归一化梯度
# 这里采用梯度/(1+梯度) 的形式，使得梯度小的区域接近0，梯度大的区域接近1
local_weight=(local_weight-np.min(local_weight))/(np.max(local_weight)-np.min(local_weight))

# 绘制图形
# fig, axes = plt.subplots(1, 3, figsize=(18, 5))
# 绘制速度模型
plt.figure()
im = plt.imshow(velocity_model, cmap='viridis', aspect='auto',extent=(0,2,1,0))


# 绘制全局平滑权重映射
plt.figure()
im = plt.imshow(global_weight, cmap='seismic',extent=(0,2,1,0))
plt.savefig('fig1.png',dpi=1000)
# 绘制局部突变权重映射
plt.figure()
im = plt.imshow(local_weight, cmap='seismic', extent=(0,2,1,0))
plt.savefig('fig2.png',dpi=1000)

plt.tight_layout()
plt.show()
