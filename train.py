import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from WNet import WNet
from configure import Config
from DataLoader import torch_loader
from torchinfo import summary

vertical_sobel=torch.nn.Parameter(torch.from_numpy(np.array([[[[1,  0,  -1], 
                                            [1,  0,  -1], 
                                            [1,  0,  -1]]]])).float().cuda(), requires_grad=False)

horizontal_sobel=torch.nn.Parameter(torch.from_numpy(np.array([[[[1,   1,  1], 
                                              [0,   0,  0], 
                                              [-1 ,-1, -1]]]])).float().cuda(), requires_grad=False) 

def gradient_regularization(softmax, device='cuda'):
    # Approximation of the Ncut loss provided in the taolin implementation.
    vert=torch.cat([F.conv2d(softmax[i, :].unsqueeze(1), vertical_sobel) for i in range(softmax.shape[0])], 1)
    hori=torch.cat([F.conv2d(softmax[i, :].unsqueeze(1), horizontal_sobel) for i in range(softmax.shape[0])], 1)
    mag = torch.pow(torch.pow(vert, 2) + torch.pow(hori, 2), 0.5)
    mean = torch.mean(mag)
    return mean

def WNet_train(model, config, summary = True):
  # training of the WNet architecture. Settings are managed through the
  # configure.py file.
  model.cuda()
  model.train()

  # Importing dataset
  dataloader = torch_loader(config)
  num_iter = len(dataloader)

  # Optimizer declaration and scheduler
  optimizer = torch.optim.SGD(model.parameters(), lr=config.init_lr) #Adam gives NaN problems
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_decay_iter, gamma=config.lr_decay)

  if summary:
      writer = SummaryWriter()

  for epoch in tqdm(range(config.epochs)):

    for step, input in enumerate(dataloader):
      input = input.cuda()
      
      # NCuts Loss
      optimizer.zero_grad()
      enc = model(input, returns='enc')
      soft_enc = F.softmax(enc, dim=1)
      n_cuts_loss = gradient_regularization(soft_enc) 
      n_cuts_loss.backward()
      optimizer.step()
      
      # Reconstruction Loss
      optimizer.zero_grad()
      enc, dec = model(input)
      rec_loss = F.mse_loss(dec, input, reduction='mean')
      rec_loss.backward()
      optimizer.step()

      # summary and reporting 
      if summary:
        writer.add_scalars("Loss", {'NCut': n_cuts_loss.item(), 'Reconstraction':rec_loss.item()}, (epoch * num_iter) + step)
      if (step % 10 == 0):
        print("\n")
        print("Epoch: " + str(epoch) + " /step: " + str(step) + " /NCuts Loss: " + str(n_cuts_loss.item()))
        print("Epoch: " + str(epoch) + " /step: " + str(step) + " /Reconstruction Loss: " + str(rec_loss.item()))
      
    scheduler.step()

  if summary:
    writer.close()
    return model, writer
  return model



if __name__ == '__main__':

  #importing configurations
  config = Config("train")

  #inizialization of a new WNet model
  model = WNet(config.K, config.dropout, config.ch_mul, config.in_chans, config.out_chans)

  #summary of the model 
  #summary(model, input_size=(config.BatchSize, config.in_chans, config.inputsize[0], config.inputsize[1]))

  #training
  model, summary_writer = WNet_train(model, config, summary = True)
  #%tensorboard --logdir runs/

  torch.save(model.state_dict(), config.saving_path)
   

