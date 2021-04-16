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
from configure import Config
from ncut import soft_ncut
from DataLoader import torch_loader


def WNet_train(model, config, summary = True):
  model.train()
  model.cuda()

  # Importing dataset
  dataloader = torch_loader(config)
  num_iter = len(dataloader)

  # Optimizer declaration and scheduler
  optimizer = torch.optim.Adam(model.parameters(), lr=config.init_lr) 
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_decay_iter, gamma=config.lr_decay, last_epoch = -1, verbose = True)

  if summary:
      writer = SummaryWriter()

  for epoch in tqdm(range(config.epochs)):
    for step, input in enumerate(dataloader):
      input = input.cuda()


      # NCuts Loss
#      optimizer.zero_grad()
#      enc = model(input, returns='enc')
#      soft_enc = F.softmax(enc, dim=1) 
#      n_cuts_loss = soft_ncut(soft_enc)
#      n_cuts_loss.retain_grad()
#      n_cuts_loss.backward()
#      optimizer.step()

      # Reconstruction Loss
      optimizer.zero_grad()
      enc, dec = model(input)
      rec_loss = F.mse_loss(dec, input, reduction='mean')
      rec_loss.backward()
      optimizer.step()
      
      # summary and reporting 
      if summary:
        writer.add_scalars("Loss", {'Reconstruction':rec_loss.item()}, (epoch * num_iter) + step)
      if (step % 10 == 0):
        print("\n")
       # print("Epoch: " + str(epoch) + " /step: " + str(step) + " /NCuts Loss: " + str(n_cuts_loss.item()))
        print("Epoch: " + str(epoch) + " /step: " + str(step) + " /Reconstruction Loss: " + str(rec_loss.item()))

    scheduler.step()

  if summary:
    writer.flush()
    writer.close()
  return model

