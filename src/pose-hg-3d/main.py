import os
import time
import datetime

import ref
from opts import opts
from utils.logger import Logger

import torch
import torch.utils.data
from utils.utils import adjust_learning_rate
from datasets.fusion import Fusion
from datasets.h36m import H36M
from datasets.mpii import MPII
from models.hg_3d import HourglassNet3D
from train import train, val

def main():
  # Parse the options
  opt = opts().parse()
  # Record the start time
  now = datetime.datetime.now()
  # (optional)tensorboard, show things
  logger = Logger(opt.saveDir + '/logs_{}'.format(now.isoformat()))

  # Load Model
  ## no loadModel as option
  if opt.loadModel != 'none':
    model = torch.load(opt.loadModel).cuda()
  ## input a loadModel as option
  else:
    model = HourglassNet3D(opt.nStack, opt.nModules, opt.nFeats, opt.nRegModules).cuda()
  # Set the Criterion and Optimizer(set in ref)
  criterion = torch.nn.MSELoss().cuda()
  optimizer = torch.optim.RMSprop(model.parameters(), opt.LR, 
                                  alpha = ref.alpha, 
                                  eps = ref.epsilon, 
                                  weight_decay = ref.weightDecay, 
                                  momentum = ref.momentum)
  # If there is 3D data or not(Compare ratio3D to epsilon from ref)
  # and choose the validation dataset
  if opt.ratio3D < ref.eps:
    val_loader = torch.utils.data.DataLoader(
        MPII(opt, 'val', returnMeta = True), 
        batch_size = 1, 
        shuffle = False,
        num_workers = int(ref.nThreads)
    )
  else:
    val_loader = torch.utils.data.DataLoader(
        H36M(opt, 'val'), 
        batch_size = 1, 
        shuffle = False,
        num_workers = int(ref.nThreads)
    )
  
  # Test if set test in the option, finish after test
  if opt.test:
    val(0, opt, val_loader, model, criterion)
    return

  # Load the train data, set the trainloader
  train_loader = torch.utils.data.DataLoader(
      Fusion(opt, 'train'), 
      batch_size = opt.trainBatch, 
      shuffle = True if opt.DEBUG == 0 else False,
      num_workers = int(ref.nThreads)
  )

  # Train, with set epoch(default 60)
  for epoch in range(1, opt.nEpochs + 1):
    ## Train the model
    loss_train, acc_train, mpjpe_train, loss3d_train = train(epoch, opt, train_loader, model, criterion, optimizer)
    ## Logger record train
    logger.scalar_summary('loss_train', loss_train, epoch)
    logger.scalar_summary('acc_train', acc_train, epoch)
    logger.scalar_summary('mpjpe_train', mpjpe_train, epoch)
    logger.scalar_summary('loss3d_train', loss3d_train, epoch)
    ## Intervals to show
    if epoch % opt.valIntervals == 0:
      ### Validation
      loss_val, acc_val, mpjpe_val, loss3d_val = val(epoch, opt, val_loader, model, criterion)
      ### Logger record validation
      logger.scalar_summary('loss_val', loss_val, epoch)
      logger.scalar_summary('acc_val', acc_val, epoch)
      logger.scalar_summary('mpjpe_val', mpjpe_val, epoch)
      logger.scalar_summary('loss3d_val', loss3d_val, epoch)
      ### Update the model
      torch.save(model, os.path.join(opt.saveDir, 'model_{}.pth'.format(epoch)))
      logger.write('{:8f} {:8f} {:8f} {:8f} {:8f} {:8f} {:8f} {:8f} \n'.format(loss_train, acc_train, mpjpe_train, loss3d_train, loss_val, acc_val, mpjpe_val, loss3d_val))
    else:
      logger.write('{:8f} {:8f} {:8f} {:8f} \n'.format(loss_train, acc_train, mpjpe_train, loss3d_train))
    ## Use the optimizer to adjust learning rate
    adjust_learning_rate(optimizer, epoch, opt.dropLR, opt.LR)
  # Close the logger
  logger.close()

# The name of the running module
if __name__ == '__main__':
  main()
