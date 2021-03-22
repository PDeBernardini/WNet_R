import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from WNet import WNet
from configure import Config
from DataLoader import torch_loader
vertical_sobel=torch.nn.Parameter(torch.from_numpy(np.array([[[[1,  0,  -1], 
                                            [1,  0,  -1], 
                                            [1,  0,  -1]]]])).float().cuda(), requires_grad=False)

horizontal_sobel=torch.nn.Parameter(torch.from_numpy(np.array([[[[1,   1,  1], 
                                              [0,   0,  0], 
                                              [-1 ,-1, -1]]]])).float().cuda(), requires_grad=False) 

def gradient_regularization(softmax, device='cuda'):
    #Approximation of the Ncut loss provided in the taolin implementation.
    vert=torch.cat([F.conv2d(softmax[i, :].unsqueeze(1), vertical_sobel) for i in range(softmax.shape[0])], 1)
    hori=torch.cat([F.conv2d(softmax[i, :].unsqueeze(1), horizontal_sobel) for i in range(softmax.shape[0])], 1)
    mag = torch.pow(torch.pow(vert, 2) + torch.pow(hori, 2), 0.5)
    mean = torch.mean(mag)
    return mean

def WNet_tt(model, config):
  # training and testing of the WNet architecture. Settings are managed through the
  # configure.py file.
  model.cuda()

  # Importing dataset
  dataloader = torch_loader(config)

  # Training
  if (config.mode == "train"):

    # Optimizer declaration and scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=config.init_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_decay_iter, gamma=config.lr_decay)
    for epoch in range(config.max_iter):
        print("Epoch: " + str(epoch))

        for step, input in enumerate(dataloader):
          input = input.cuda()

          #NCuts Loss
          optimizer.zero_grad()
          enc = model(input, returns='enc')
          soft_enc = F.softmax(enc, dim=1)
          n_cuts_loss = gradient_regularization(soft_enc) 
          n_cuts_loss.backward()
          optimizer.step()
          
          #Reconstruction Loss
          optimizer.zero_grad()
          enc, dec = model(input)
          rec_loss = F.mse_loss(dec, input, reduction='mean')
          rec_loss.backward()
          optimizer.step()

          #Check the values of the loss functions every (n=10) steps
          if (step % 10 == 0):
            print("step: " + str(step) + ", NCuts Loss: " + str(n_cuts_loss.item()))
            print("step: " + str(step) + ", Reconstruction Loss: " + str(rec_loss.item()))
          scheduler.step()
    
    return model


if __name__ == '__main__':

  # Importing configurations
  config = Config("train")
  model = WNet(config.K, config.ch_mul, config.in_chans, config.out_chans)
  trained_model = WNet_tt(model, config)

   

