import sys
import torch
from opts import opts
import ref
from utils.visualize import Debugger
from utils.eval import getPreds
import cv2
import numpy as np

from functools import partial
import pickle 

def main():
  # Parse the options
  opt = opts().parse()
  # Prepare to load the model for python 3 using partial and pickle
  pickle.load = partial(pickle.load, encoding="latin1")
  pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")

  # No loadModel in the options
  if opt.loadModel != 'none':
    model = torch.load(opt.loadModel, map_location=lambda storage,  loc: storage, pickle_module=pickle).cuda()
  # loadModel in the options
  else:
    model = torch.load('models/hgreg-3d.pth', map_location=lambda storage,  loc: storage, pickle_module=pickle).cuda()
  # Using opencv to load the image
  img = cv2.imread(opt.demo)
  # Turn the image data from opencv to tensor
  input = torch.Tensor(img.transpose(2, 0, 1)).float() / 256.
  # Reshape the tensor with view
  input = input.view(1, input.size(0), input.size(1), input.size(2))
  # Enable autograd to compute the gradiants automatically
  input_var = torch.autograd.Variable(input).float().cuda()
  # Forward propagation
  output = model(input_var)
  # Get prediction
  pred = getPreds((output[-2].data).cpu().numpy())[0] * 4
  # Get regrression
  reg = (output[-1].data).cpu().numpy().reshape(pred.shape[0], 1)
  # Debug
  debugger = Debugger()
  ## Load image
  debugger.addImg((input[0].numpy().transpose(1, 2, 0)*256).astype(np.uint8))
  ## Load 2D points of pose
  debugger.addPoint2D(pred, (255, 0, 0))
  ## Load 3D points with regression
  debugger.addPoint3D(np.concatenate([pred, (reg + 1) / 2. * 256], axis = 1))
  ## Show RGB image
  debugger.showImg(pause = True)
  ## Show 3D image
  debugger.show3D()

if __name__ == '__main__':
  main()
