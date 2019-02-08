# Model.py
# TODO: The module provides functions to save the training model and
#  resume the training model


import os
import torch
import torch.nn as nn
import torchvision.models as models
from ..models.hg_2D_res_CLSTM import Hourglass2DPrediction

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

#Re-init optimizer
def getModel(opt): 
  if 'hg' in opt.arch:
    model = Hourglass2DPrediction(opt)
    optimizer = torch.optim.RMSprop(model.parameters(), opt.LR, 
                                    alpha = opt.alpha,
                                    eps = opt.epsilon,
                                    weight_decay = opt.weightDecay,
                                    momentum = opt.momentum)
  else:
    print("=> using pre-trained model '{}'".format(opt.arch))
    model = models.__dict__[opt.arch](pretrained=True)
    if opt.arch.startswith('resnet'):
      model.avgpool = nn.AvgPool2d(8, stride=1)
      if '18' in opt.arch:
        model.fc = nn.Linear(512 * 1, opt.outputDim)
      else :
        model.fc = nn.Linear(512 * 4, opt.outputDim)
      print('reset classifier', opt.outputDim)
    if opt.arch.startswith('densenet'):
      if '161' in opt.arch:
        model.classifier = nn.Linear(2208, opt.outputDim)
      elif '201' in opt.arch:
        model.classifier = nn.Linear(1920, opt.outputDim)
      else:
        model.classifier = nn.Linear(1024, opt.outputDim)
    if opt.arch.startswith('vgg'):
      feature_model = list(model.classifier.children())
      feature_model.pop()
      feature_model.append(nn.Linear(4096, opt.outputDim))
      model.classifier = nn.Sequential(*feature_model)
    optimizer = torch.optim.SGD(model.parameters(), opt.LR,
                            momentum=0.9,
                            weight_decay=1e-4)

  if opt.loadModel != 'none':
    checkpoint = torch.load(opt.loadModel)
    if type(checkpoint) == type({}):
      state_dict = checkpoint['state_dict']
    else:
      state_dict = checkpoint.state_dict()  
    model.load_state_dict(state_dict)
    
  return model, optimizer
  
def saveModel(model, optimizer, path):
  torch.save({'state_dict': model.state_dict(), 
              'optimizer': optimizer.state_dict()}, path)
