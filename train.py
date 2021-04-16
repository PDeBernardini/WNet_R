import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math 
import pdb

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchinfo import summary
from torch import autograd

from WNet import WNet
from gaussian_ncut import NCutLoss2D
from configure import Config
from ncut import soft_ncut
from DataLoader import torch_loader


def WNet_train(model, config, summary = True):
  model.train()
  model.cuda()
  # Importing dataset
  dataloader = torch_loader(config)
  num_iter = len(dataloader)
  print(num_iter)
  # Optimizer declaration and scheduler
  optimizer = torch.optim.Adam(model.parameters(), lr=config.init_lr) 
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_decay_iter, gamma=config.lr_decay, last_epoch = -1, verbose = False)
  # Loss declaration 
  MSE_loss = nn.MSELoss(reduction = "mean") # Mean Square Error loss
  # Summary
  if summary:
      writer = SummaryWriter()
  print("--- TRAINING --- \n")
  # Training
  for epoch in tqdm(range(config.epochs)):
    for step, input in enumerate(dataloader):
      input = input.cuda()
      optimizer.zero_grad()
      enc, dec = model(input)
      rec_loss = MSE_loss(dec, input)
      rec_loss.backward()
      optimizer.step()
      # summary and reporting 
      if summary:
        writer.add_scalars("Loss", {'Reconstruction':rec_loss.item()}, (epoch * num_iter) + step)
      if (step % 10 == 0):
        print("\n")
        print("Epoch: " + str(epoch) + " /step: " + str(step) + " /Reconstruction Loss: " + str(rec_loss.item()))
    scheduler.step()
  if summary:
    writer.flush()
    writer.close()
  return model


def WNet_tune(model, config, summary = True):
  model.train()
  model.cuda()
  # Importing dataset
  dataloader = torch_loader(config)
  num_iter = len(dataloader)
  # Optimizer declaration and scheduler
  optimizer = torch.optim.Adam(model.parameters(), lr=config.lr_tune) 
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_decay_iter, gamma=config.lr_decay, last_epoch = -1, verbose = False)
  # Losses declaration 
  SNC_loss = NCutLoss2D() # Soft Normalized Cut loss
  MSE_loss = nn.MSELoss(reduction = "mean") # Mean Square Error loss
  alpha = config.alpha
  # Summary
  if summary:
      writer = SummaryWriter()
  print("--- FINE TUNING --- \n")
  # Training
  for epoch in tqdm(range(config.epochs_tune)):
    for step, input in enumerate(dataloader):
      input = input.cuda()
      optimizer.zero_grad()
      enc, dec = model(input)
      # NCuts Loss
      soft_enc = F.softmax(enc, dim=1)
      n_cuts_loss = SNC_loss(soft_enc, input)
      # Reconstruction Loss
      rec_loss = MSE_loss(dec, input)
      # Backward
      loss = alpha * n_cuts_loss + rec_loss
      loss.backward()
      optimizer.step()
      # summary and reporting 
      if summary:
        writer.add_scalars("Loss", {'NCut': n_cuts_loss.item(), 'Reconstruction':rec_loss.item()}, (epoch * num_iter) + step)
      if (step % 10 == 0):
        print("\n")
        print("Epoch: " + str(epoch) + " /step: " + str(step) + " /NCuts Loss: " + str(n_cuts_loss.item()))
        print("Epoch: " + str(epoch) + " /step: " + str(step) + " /Reconstruction Loss: " + str(rec_loss.item()))
      
      if (step == config.early_stop):
        return model
    scheduler.step()
  if summary:
    writer.flush()
    writer.close()
  return model
