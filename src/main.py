import os
import time
import datetime

import torch
import torch.utils.data as td
import scipy.io as sio

from opts import Opts
#from train import train, val
#from model import getModel, saveModel

def main():
    # parse the opts
    opt = Opts().parse()
    print(opt.expID, opt.task)
    # record the time
    time_start = opt.exp_time
    # adjust images to fit for the dataloader
    # Set training dataloader
    train_loader = td.DataLoader(
        H36M(opt, 'train'),
        batch_size = 1,
        shuffle = False,
        num_workers = int(opt.num_workers)
    )
    # Set validation dataloader
    val_loader = td.DataLoader(
        H36M(opt, 'val'),
        batch_size = 1,
        shuffle = False,
        num_workers = int(ref.nThreads)
    )

    # #TODO: choose the Model, Optimizer and Criterion
    #
    # #TODO: train and val
    #
    # #TODO: write the LSTM model for action
    # # Load Model
    # model = LSTM().cuda()
    # # Set the Criterion and Optimizer
    # criterion = torch.nn.MSELoss().cuda()
    # optimizer = torch.optim.RMSprop(model.parameters(), opt.LR,
    #                               alpha = ref.alpha,
    #                               eps = ref.epsilon,
    #                               weight_decay = ref.weightDecay,
    #                               momentum = ref.momentum)
    # # Train and Eval
    # for epoch in range(1, opt.nEpochs + 1):
    #     ## Train the model
    #     loss_train, acc_train, mpjpe_train, loss3d_train = train(epoch, opt, train_loader, model, criterion, optimizer)
    #     ## Show results
    #     f"loss_train: {loss_train} | acc_train: {acc_train} | mpjpe_train: {mpjpe_train}\n"
    #     ## Intervals to show eval results
    #     if epoch % opt.valIntervals == 0:
    #       ### Validation
    #       loss_val, acc_val, mpjpe_val, loss3d_val = val(epoch, opt, val_loader, model, criterion)
    #       f"loss_train: {loss_train} | acc_train: {acc_train} | mpjpe_train: {mpjpe_train}\n"
    #       ### Update the model
    #       torch.save(model, os.path.join(opt.saveDir, f"model_{epoch}.pth"))
    #     #TODO: make adjust_learning_rate work
    #     ## Use the optimizer to adjust learning rate
    #     #adjust_learning_rate(optimizer, epoch, opt.dropLR, opt.LR)

# The name of the running module
if __name__ == '__main__':
  main()