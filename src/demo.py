import os
import cv2
import torch
import numpy as np
import scipy.io as sio
import torch.utils.data as tud

from opts import Opts
from datasets.penn import PENN_CROP
from utils.visualizer import Visualizer
from utils.eval import AverageMeter, get_preds, get_ref, original_coordinate, error, accuracy

def main():
    # Parse the options from parameters
    opts = Opts().parse()
    ## For PyTorch 0.4.1, cuda(device)
    opts.device = torch.device(f'cuda:{opts.gpu[0]}')
    print(opts.expID, opts.task,os.path.dirname(os.path.realpath(__file__)))
    # Load the trained model test
    if opts.loadModel != 'none':
        model_path = os.path.join(opts.root_dir, opts.loadModel)
        model = torch.load(model_path).cuda(device=opts.device)
        model.eval()
    else:
        print('ERROR: No model is loaded!')
        return
    # Read the input image, pass input to gpu
    if opts.img == 'None':
        val_dataset = PENN_CROP(opts, 'val')
        val_loader = tud.DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=int(opts.num_workers)
        )
        opts.nJoints = val_dataset.nJoints
        opts.skeleton = val_dataset.skeleton
        for i, gt in enumerate(val_loader):
            # Test Visualizer, Input and get_preds
            if i == 0:
                input, label = gt['input'], gt['label']
                gtpts, center, scale, proj = gt['gtpts'], gt['center'], gt['scale'], gt['proj']
                input_var = input[:, 0, ].float().cuda(device=opts.device, non_blocking=True)
                # output = label
                output = model(input_var)
                # Test Loss, Err and Acc(PCK)
                Loss, Err, Acc = AverageMeter(), AverageMeter(), AverageMeter()
                ref = get_ref(opts.dataset, scale)
                for j in range(opts.preSeqLen):
                    pred = get_preds(output[:, j, ].cpu().float())
                    pred = original_coordinate(pred, center[:, ], scale, opts.outputRes)
                    err, ne = error(pred, gtpts[:, j, ], ref)
                    acc, na = accuracy(pred, gtpts[:, j, ], ref)
                    # assert ne == na, "ne must be the same as na"
                    Err.update(err)
                    Acc.update(acc)
                    print(j, f"{Err.val:.6f}", Acc.val)
                print('all', f"{Err.avg:.6f}", Acc.avg)
                # Visualizer Object
                ## Initialize
                v = Visualizer(opts.nJoints, opts.skeleton, opts.outputRes)
                # ## Add input image
                # v.add_img(input[0,0,].transpose(2, 0).numpy().astype(np.uint8))
                # ## Get the predicted joints
                # predJoints = get_preds(output[:, 0, ])
                # # ## Add joints and skeleton to the figure
                # v.add_2d_joints_skeleton(predJoints, (0, 0, 255))
                # Transform heatmap to show
                hm_img = output[0,0,].cpu().detach().numpy()
                v.add_hm(hm_img)
                ## Show image
                v.show_img(pause=True)
                break
    else:
        print('NOT ready for the raw input outside the dataset')
        img = cv2.imread(opts.img)
        input = torch.from_numpy(img.tramspose(2, 0, 1)).float()/256.
        input = input.view(1, input.size(0), input.size(1), input.size(2))
        input_var = torch.autograd.variable(input).float().cuda(device=opts.device)
        output = model(input_var)
        predJoints = get_preds(output[-2].data.cpu().numpy())[0] * 4

if __name__ == '__main__':
    main()
