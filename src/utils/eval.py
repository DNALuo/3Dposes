<<<<<<< HEAD
# Eval.py
# TODO: This file providers several functions for model to
#  get the prediction from the output and evaluations
import torch
import numpy as np
from utils.img import Transform


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# Process the scale
def get_ref(dataset, scale):
    if dataset == 'penn-crop':
        return 200 * scale.view(-1) / 1.25


def get_preds(hm):
    """
    :param hm: train_batch x num_joints x w x h
    :return:
    """
    assert len(hm.shape) == 4, 'Input must be a 4-D tensor'
    res = hm.shape[2]
    hm_reshape = hm.reshape(hm.shape[0], hm.shape[1], hm.shape[2] * hm.shape[3]).detach()
    idx = np.argmax(hm_reshape, axis=2)
    preds = torch.zeros((hm.shape[0], hm.shape[1], 2))
    for i in range(hm.shape[0]):
        for j in range(hm.shape[1]):
            preds[i, j, 0], preds[i, j, 1] = idx[i, j] % res, idx[i, j] / res

    return preds


def original_coordinate(pred, center, scale, outputRes, rot = 0):
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            pred[i, j] = torch.from_numpy(Transform(pred[i][j], center[i], scale[i], rot, outputRes, True))

    return pred.double()


def error(prePst, gtpts, ref):
    """
    :param   prePst  : N x num_joints x 2
    :param   gtpts   : N x num_joints x 2
    :param   ref     : N x 1
    :return: error, n: 1
    """
    e, n = torch.zeros(gtpts.size(0)).double(), torch.zeros(gtpts.size(0)).double()
    for i in range(gtpts.size(0)):
        for j in range(gtpts.size(1)):
            if gtpts[i,j,0] != 0 and gtpts[i,j,1] != 0:
                n[i] = n[i] + 1
                e[i] = e[i] + torch.dist(gtpts[i,j], prePst[i,j].double()) / ref[i]
    return e.sum()/n.sum(), n.sum()


def calc_dists(prePst, gt, normalize):
    dists = np.zeros((prePst.shape[1], prePst.shape[0]))
    for i in range(prePst.shape[0]):
        for j in range(prePst.shape[1]):
            if gt[i, j, 0] > 0 and gt[i, j, 1] > 0:
                dists[j][i] = torch.dist(gt[i][j], prePst[i][j]) / normalize[i]
            else:
                dists[j][i] = -1
    return dists


def distAccuracy(dist, thr=0.5):
    dist = dist[dist != -1]
    if len(dist) > 0:
        return 1.0*(dist<thr).sum()/len(dist)
    else:
        return -1


# Accuracy for Penn_crop
# def accuracy(prePst, labels, ref, idxs = None, thr = 0.05):
def accuracy(prePst, labels, ref, idxs = None, thr = 0.5):
    """
    :param prePst:
    :param labels:
    :param ref:
    :param idxs:
    :param thr:
    :return:
    """
    dists = calc_dists(prePst, labels, ref)
    # TODO: import dataset for different configuration
    acc = np.zeros(dists.shape[0])
    avgAcc = 0.0
    badIdxsCount = 0

    if idxs is None:
        for i in range(dists.shape[0]):
            acc[i] = distAccuracy(dists[i], thr)
            if acc[i] >= 0:
                avgAcc += acc[i]
            else:
                badIdxsCount += 1
        if badIdxsCount < dists.shape[0]:
            return avgAcc / (dists.shape[0] - badIdxsCount), dists.shape[0]-badIdxsCount
        else:
            return 0, badIdxsCount
    else:
        for i in range(len(idxs)):
            acc[i] = distAccuracy(dists[idxs[i]], thr)
            if acc[i] >= 0:
                avgAcc += acc[i]
            else:
                badIdxsCount += 1
        if badIdxsCount < len(idxs):
            return avgAcc / (len(idxs) - badIdxsCount), len(idxs) - badIdxsCount
        else:
            return 0


def MPJPE(opt, output2D, output3D, meta):
    meta = meta.numpy()
    p = np.zeros((output2D.shape[0], opt.nJoints, 3))
    p[:, :, :2] = get_preds(output2D).copy()

    hm = output2D.reshape(output2D.shape[0], output2D.shape[1], opt.outputRes, opt.outputRes)
    for i in range(hm.shape[0]):
        for j in range(hm.shape[1]):
            pX, pY = int(p[i, j, 0]), int(p[i, j, 1])
            scores = hm[i, j, pX, pY]
            if 0 < pX < opt.outputRes - 1 and 0 < pY < opt.outputRes - 1:
                diffY = hm[i, j, pX, pY + 1] - hm[i, j, pX, pY - 1]
                diffX = hm[i, j, pX + 1, pY] - hm[i, j, pX - 1, pY]
                p[i, j, 0] = p[i, j, 0] + 0.25 * (1 if diffX >= 0 else -1)
                p[i, j, 1] = p[i, j, 1] + 0.25 * (1 if diffY >= 0 else -1)
    p = p + 0.5

    p[:, :, 2] = (output3D.copy() + 1) / 2 * opt.outputRes
    h36mSumLen = 4296.99233013
    root = 6
    err = 0
    num3D = 0
    for i in range(p.shape[0]):
        s = meta[i].sum()
        if not (s > - opt.eps and s < opt.eps):
            num3D += 1
            lenPred = 0
            for e in opt.edges:
                lenPred += ((p[i, e[0]] - p[i, e[1]]) ** 2).sum() ** 0.5
            pRoot = p[i, root].copy()
            for j in range(opt.nJoints):
                p[i, j] = (p[i, j] - pRoot) / lenPred * h36mSumLen + meta[i, root]
            p[i, 7] = (p[i, 6] + p[i, 8]) / 2
            for j in range(opt.nJoints):
                dis = ((p[i, j] - meta[i, j]) ** 2).sum() ** 0.5
                err += dis / opt.nJoints
    if num3D > 0:
        return err / num3D, num3D
    else:
        return 0, 0
=======
# Eval.py
# TODO: This file providers several functions for model to
#  get the prediction from the output and evaluations
import torch
import numpy as np
from utils.img import Transform


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# Process the scale
def get_ref(dataset, scale):
    if dataset == 'penn-crop':
        return 200 * scale.view(-1) / 1.25


def get_preds(hm):
    """
    :param hm: train_batch x num_joints x w x h
    :return:
    """
    assert len(hm.shape) == 4, 'Input must be a 4-D tensor'
    res = hm.shape[2]
    hm_reshape = hm.reshape(hm.shape[0], hm.shape[1], hm.shape[2] * hm.shape[3]).detach()
    idx = np.argmax(hm_reshape, axis=2)
    preds = torch.zeros((hm.shape[0], hm.shape[1], 2))
    for i in range(hm.shape[0]):
        for j in range(hm.shape[1]):
            preds[i, j, 0], preds[i, j, 1] = idx[i, j] % res, idx[i, j] / res

    return preds


def original_coordinate(pred, center, scale, outputRes):
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            pred[i, j] = torch.from_numpy(Transform(pred[i][j], center[i], scale[i][0], 0, outputRes, True))

    return pred.double()


def error(prePst, gtpts, ref):
    """
    :param   prePst  : N x d x 2
    :param   gtpts   : N x d x 2
    :param   ref     : N x 1
    :return: error, n: 1
    """
    e, n = torch.zeros(gtpts.size(0)).double(), torch.zeros(gtpts.size(0)).double()
    for i in range(gtpts.size(0)):
        for j in range(gtpts.size(1)):
            if gtpts[i,j,0] != 0 and gtpts[i,j,1] != 0:
                n[i] = n[i] + 1
                e[i] = e[i] + torch.dist(gtpts[i,j], prePst[i,j].double()) / ref[i]
    return e.sum(), n.sum()


def calc_dists(prePst, gt, normalize):
    dists = np.zeros((prePst.shape[1], prePst.shape[0]))
    for i in range(prePst.shape[0]):
        for j in range(prePst.shape[1]):
            if gt[i, j, 0] > 0 and gt[i, j, 1] > 0:
                dists[j][i] = torch.dist(gt[i][j], prePst[i][j]) / normalize[i]
            else:
                dists[j][i] = -1
    return dists


def distAccuracy(dist, thr=0.5):
    dist = dist[dist != -1]
    if len(dist) > 0:
        return 1.0*(dist<thr).sum()/len(dist)
    else:
        return -1


# Accuracy for Penn_crop
def coordinates_accuracy(prePst, labels, thr, idxs, ref):
    '''
    :param prePst:
    :param labels:
    :param thr:
    :param idxs:
    :param ref:
    :return:
    '''
    dists = calc_dists(prePst, labels, ref)
    # TODO: import dataset for different configuration
    acc = {}
    avgAcc = 0.0
    badIdxsCount = 0

    if idxs is None:
        for i in range(dists.shape[0]):
            acc[i+1] = distAccuracy(dists[i], thr)
            if acc[i+1] >= 0:
                avgAcc += acc[i+1]
            else:
                badIdxsCount += 1
        if badIdxsCount < dists.shape[0]:
            acc[0] = avgAcc / (dists.shape[0] - badIdxsCount)
        else:
            acc[0] = 0
    else:
        for i in range(len(idxs)):
            acc[i+1] = distAccuracy(dists[idxs[i]], thr)
            if acc[i+1] >= 0:
                avgAcc += acc[i+1]
            else:
                badIdxsCount += 1
        # acc[0] = avgAcc / (len(idxs) - badIdxsCount)
        if badIdxsCount < dists.shape[0]:
            acc[0] = avgAcc / (len(idxs) - badIdxsCount)
        else:
            acc[0] = 0

    return acc[0]



# Computer Accuracy
def accuracy(prePst, labels, ref):
    """
    :param prePst: N x d x 2
    :param labels:  N x d x 2
    :param outputRes: res
    :param ref:     N x 1
    :return: acc
    """
    return coordinates_accuracy(prePst, labels, 0.05, None, ref)


# Accuracy for MPII
def Accuracy(opt, output, target):
    preds = get_preds(output)
    gt = get_preds(target)
    dists = calc_dists(preds, gt, np.ones(preds.shape[0]) * opt.outputRes / 10)
    acc = np.zeros(len(opt.accIdxs))
    avgAcc = 0
    badIdxCount = 0

    for i in range(len(opt.accIdxs)):
        acc[i] = distAccuracy(dists[opt.accIdxs[i]])
        if acc[i] >= 0:
            avgAcc = avgAcc + acc[i]
        else:
            badIdxCount = badIdxCount + 1

    if badIdxCount == len(opt.accIdxs):
        return 0
    else:
        return avgAcc / (len(opt.accIdxs) - badIdxCount)


def MPJPE(opt, output2D, output3D, meta):
    meta = meta.numpy()
    p = np.zeros((output2D.shape[0], opt.nJoints, 3))
    p[:, :, :2] = get_preds(output2D).copy()

    hm = output2D.reshape(output2D.shape[0], output2D.shape[1], opt.outputRes, opt.outputRes)
    for i in range(hm.shape[0]):
        for j in range(hm.shape[1]):
            pX, pY = int(p[i, j, 0]), int(p[i, j, 1])
            scores = hm[i, j, pX, pY]
            if 0 < pX < opt.outputRes - 1 and 0 < pY < opt.outputRes - 1:
                diffY = hm[i, j, pX, pY + 1] - hm[i, j, pX, pY - 1]
                diffX = hm[i, j, pX + 1, pY] - hm[i, j, pX - 1, pY]
                p[i, j, 0] = p[i, j, 0] + 0.25 * (1 if diffX >= 0 else -1)
                p[i, j, 1] = p[i, j, 1] + 0.25 * (1 if diffY >= 0 else -1)
    p = p + 0.5

    p[:, :, 2] = (output3D.copy() + 1) / 2 * opt.outputRes
    h36mSumLen = 4296.99233013
    root = 6
    err = 0
    num3D = 0
    for i in range(p.shape[0]):
        s = meta[i].sum()
        if not (s > - opt.eps and s < opt.eps):
            num3D += 1
            lenPred = 0
            for e in opt.edges:
                lenPred += ((p[i, e[0]] - p[i, e[1]]) ** 2).sum() ** 0.5
            pRoot = p[i, root].copy()
            for j in range(opt.nJoints):
                p[i, j] = (p[i, j] - pRoot) / lenPred * h36mSumLen + meta[i, root]
            p[i, 7] = (p[i, 6] + p[i, 8]) / 2
            for j in range(opt.nJoints):
                dis = ((p[i, j] - meta[i, j]) ** 2).sum() ** 0.5
                err += dis / opt.nJoints
    if num3D > 0:
        return err / num3D, num3D
    else:
        return 0, 0
>>>>>>> 7c234f0346a56d52e13f09f8d7179d3fe92b73d3
