import cv2
import torch
import numpy as np
from progress.bar import Bar

import ref
from utils.utils import AverageMeter
from utils.eval import Accuracy, getPreds, MPJPE
from utils.debugger import Debugger
from models.layers.FusionCriterion import FusionCriterion

# Basic step for train or evaluation
def step(split, epoch, opt, dataLoader, model, criterion, optimizer = None):
  
  # Choose the mode
  ## Train mode 
  if split == 'train':
    model.train()
  ## Evaluate mode(Normally without Dropout and BatchNorm)
  else:
    model.eval()

  # Load default values
  Loss, Acc, Mpjpe = AverageMeter(), AverageMeter(), AverageMeter()
    
  # Show iteration using Bar
  nIters = len(dataLoader)
  bar = Bar('==>', max=nIters)
  # Loop in dataloader
  for i, (input, target2D, target3D, meta) in enumerate(dataLoader):
    ## Wraps tensors and records the operations applied to it
    input_var = torch.autograd.Variable(input).float().cuda()
    target2D_var = torch.autograd.Variable(target2D).float().cuda()
    # For now, 3D data is not needed
    # target3D_var = torch.autograd.Variable(target3D).float().cuda()
    
    ## Forwad propagation
    output = model(input_var)
    ## Original Output of Regression
    reg = output[opt.nStack]
    ## Debug level >= 2, show the 2D prediction compared with 2D groundtruth
    if opt.DEBUG >= 2:
      gt = getPreds(target2D.cpu().numpy()) * 4
      pred = getPreds((output[opt.nStack - 1].data).cpu().numpy()) * 4
      debugger = Debugger()
      debugger.addImg((input[0].numpy().transpose(1, 2, 0)*256).astype(np.uint8))
      debugger.addPoint2D(pred[0], (255, 0, 0))
      debugger.addPoint2D(gt[0], (0, 0, 255))
      debugger.showImg()
      debugger.saveImg('debug/{}.png'.format(i))
    # Fuse the criterion(init(regWeight, varWeight))((input, target))
    loss = FusionCriterion(opt.regWeight, opt.varWeight)(reg, target3D_var)
    # Update the 3D loss
    Loss3D.update(loss.data[0], input.size(0))
    for k in range(opt.nStack):
      loss += criterion(output[k], target2D_var)

    Loss.update(loss.data[0], input.size(0))
    Acc.update(Accuracy((output[opt.nStack - 1].data).cpu().numpy(), (target2D_var.data).cpu().numpy()))
    mpjpe, num3D = MPJPE((output[opt.nStack - 1].data).cpu().numpy(), (reg.data).cpu().numpy(), meta)
    if num3D > 0:
      Mpjpe.update(mpjpe, num3D)
    if split == 'train':
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
 
    Bar.suffix = '{split} Epoch: [{0}][{1}/{2}]| Total: {total:} | ETA: {eta:} | Loss {loss.avg:.6f} \
        | Loss3D {loss3d.avg:.6f} | Acc {Acc.avg:.6f} | Mpjpe {Mpjpe.avg:.6f} ({Mpjpe.val:.6f})'.format(epoch, \
        i, nIters, total=bar.elapsed_td, eta=bar.eta_td, loss=Loss, Acc=Acc, split = split, Mpjpe=Mpjpe, loss3d = Loss3D)
    bar.next()

  bar.finish()
  return Loss.avg, Acc.avg, Mpjpe.avg
  

def train(epoch, opt, train_loader, model, criterion, optimizer):
  return step('train', epoch, opt, train_loader, model, criterion, optimizer)
  
def val(epoch, opt, val_loader, model, criterion):
  return step('val', epoch, opt, val_loader, model, criterion)