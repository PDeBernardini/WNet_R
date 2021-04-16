import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

vertical_sobel=torch.nn.Parameter(torch.from_numpy(np.array([[[[1,  0,  -1], 
                                            [1,  0,  -1], 
                                            [1,  0,  -1]]]])).float().cuda(), requires_grad=True)

horizontal_sobel=torch.nn.Parameter(torch.from_numpy(np.array([[[[1,   1,  1], 
                                              [0,   0,  0], 
                                              [-1 ,-1, -1]]]])).float().cuda(), requires_grad=True) 


def soft_ncut(softmax, device='cuda'):
    vert=torch.cat([F.conv2d(softmax[i, :].unsqueeze(1), vertical_sobel) for i in range(softmax.shape[0])], 1)
    hori=torch.cat([F.conv2d(softmax[i, :].unsqueeze(1), horizontal_sobel) for i in range(softmax.shape[0])], 1)
    
    # adding err prevents nan values in backpropagation
    err = 1e-16
    mag = torch.sqrt(torch.pow(vert, 2) + torch.pow(hori, 2) + err) 
    mean = torch.mean(mag)
    return mean
                                            