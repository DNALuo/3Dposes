import os
import time
import torch
import torch.utils.data as tud
# import scipy.io as sio

from opts import Opts
from datasets.penn import PENN_CROP
from models.hg_2D_res_CLSTM import Hourglass2DPrediction
from train import train, val
from utils.utils import adjust_learning_rate


# from utils.model import getModel, saveModel


def main():
    # Parse the options
    opts = Opts().parse()
    opts.device = torch.device('cuda:{}'.format(opts.gpu[0]))
    print(opts.expID, opts.task)
    # Record the start time
    time_start = time.time()
    # TODO: select the dataset by the options
    # Set up dataset
    train_loader_unit = PENN_CROP(opts, 'train')
    train_loader = tud.DataLoader(
        train_loader_unit,
        batch_size = opts.trainBatch,
        shuffle = False,
        num_workers = int(opts.num_workers)
    )
    val_loader = tud.DataLoader(
        PENN_CROP(opts, 'val'),
        batch_size = 1,
        shuffle = False,
        num_workers = int(opts.num_workers)
    )

    # Read number of joints(dim of output) from dataset
    opts.nJoints = train_loader_unit.part.shape[1]
    # Create the Model, Optimizer and Criterion
    if opts.loadModel == 'none':
        model = Hourglass2DPrediction(opts).cuda(device=opts.device)
    else:
        model = torch.load(opts.loadModel).cuda(device=opts.device)
    # Set the Criterion and Optimizer
    criterion = torch.nn.MSELoss(reduce=False).cuda(device=opts.device)
    # opts.nOutput = len(model.outnode.children)
    optimizer = torch.optim.RMSprop(model.parameters(), opts.LR,
                                    alpha=opts.alpha,
                                    eps=opts.epsilon,
                                    weight_decay=opts.weightDecay,
                                    momentum=opts.momentum)
    # If TEST, just validate
    # TODO: save the validate results to mat or hdf5
    if opts.test:
        loss_test, pck_test = val(0, opts, val_loader, model, criterion)
        print(f"test: | loss_test: {loss_test}| PCK_val: {pck_test}\n")
        ## TODO: save the predictions for the test
        #sio.savemat(os.path.join(opts.saveDir, 'preds.mat'), mdict = {'preds':preds})
        return
    # NOT TEST, Train and Validate
    for epoch in range(1, opts.nEpochs + 1):
        ## Train the model
        loss_train, pck_train = train(epoch, opts, train_loader, model, criterion, optimizer)
        ## Show results and elapsed time
        time_elapsed = time.time() - time_start
        print(f"epoch: {epoch} | loss_train: {loss_train} | PCK_train: {pck_train} | {time_elapsed//60:.0f}min {time_elapsed%60:.0f}s\n")
        ## Intervals to show eval results
        if epoch % opts.valIntervals == 0:
            # TODO: Test the validation part
            ### Validation
            loss_val, pck_val = val(epoch, opts, val_loader, model, criterion)
            print(f"epoch: {epoch} | loss_val: {loss_val}| PCK_val: {pck_val}\n")
            ### Save the model
            torch.save(model, os.path.join(opts.save_dir, f"model_{epoch}.pth"))
            ### TODO: save the preds for the validation
            #sio.savemat(os.path.join(opts.saveDir, f"preds_{epoch}.mat"), mdict={'preds':preds})
        # Use the optimizer to adjust learning rate
        if epoch % opts.dropLR == 0:
            lr = adjust_learning_rate(optimizer, epoch, opts.dropLR, opts.LR)
            print(f"Drop LR to {lr}\n")


# The name of the running module
if __name__ == '__main__':
    main()