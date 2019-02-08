import torch
from progress.bar import Bar

from utils.eval import AverageMeter, get_preds, get_ref, original_coordinate, error, accuracy
from utils.utils import set_sequence_length


# Basic step for training or evaluation
def step(phase, epoch, opt, dataloader, model, criterion, optimizer=None):
    # Choose the phase(Evaluate phase-Normally without Dropout and BatchNorm)
    if phase == 'train':
        model.train()
    else:
        model.eval()
    # Load default values
    Loss, Err, Acc = AverageMeter(), AverageMeter(), AverageMeter()
    seqlen = set_sequence_length(opt.MinSeqLenIndex, opt.MaxSeqLenIndex, epoch)
    # Show iteration using Bar
    nIters = len(dataloader)
    bar = Bar(f'{opt.expID}', max=nIters)
    # Loop in dataloader
    for i, gt in enumerate(dataloader):
        ## Wraps tensors and records the operations applied to it
        input, label = gt['input'], gt['label']
        gtpts, center, scale = gt['gtpts'], gt['center'], gt['scale']
        input_var = input[:, 0, ].float().cuda(device=opt.device, non_blocking=True)
        label_var = label.float().cuda(device=opt.device, non_blocking=True)
        Loss.reset()
        Err.reset()
        Acc.reset()
        ### if it is 3D, may need the nOutput to get the different target, not just only the heatmap
        ## Forwad propagation
        output = model(input_var)
        ## Get model outputs and calculate loss
        loss = criterion(output, label_var)
        ## Backward + Optimize only if in training phase
        if phase == 'train':
            ## Zero the parameter gradients
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()
        Loss.update(loss.sum())
        ## Compute the accuracy
        # acc = Accuracy(opt, output.data.cpu().numpy(), labels_var.data.cpu().numpy())
        ref = get_ref(opt.dataset, scale)
        for j in range(opt.preSeqLen):
            if j <= seqlen:
                pred = get_preds(output[:,j,].float())
                pred = original_coordinate(pred, center, scale, opt.outputRes)
                err, ne = error(pred, gtpts[:,j,], ref)
                acc = accuracy(pred, gtpts[:,j,], ref)
                # acc, na = accuracy(pred, gtpts, opt.outputRes, ref)
                # assert ne == na, "ne must be the same as na"
                # acc[j] = acc/ne
                Err.update(err/ne)
                Acc.update(acc/ne)

        # acc = torch.Tensor(acc)

        Bar.suffix = f'{phase}[{epoch}][{i}/{nIters}]|Total:{bar.elapsed_td}|ETA:{bar.eta_td}|Loss{Loss.val:.6f}|Err{Err.avg:.6f}|Acc{Acc.avg:.6f}'
        bar.next()

    bar.finish()
    return Loss.val, Acc.avg


def train(epoch, opt, dataloader, model, criterion, optimizer):
    return step('train', epoch, opt, dataloader, model, criterion, optimizer)


def val(epoch, opt, dataloader, model, criterion):
    return step('val', epoch, opt, dataloader, model, criterion)