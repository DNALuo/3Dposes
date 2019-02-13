import torch
import torch.nn as nn
import torch.nn.functional as f

from models.layers.Residual import Residual
from .layers.Residual import Residual


# class Hourglass(nn.Module):
#     """
#     Basic Structure for Hourglass
#     n       : number of stacked Hourglass
#     nModules: number of residual modules in Hourglass
#     nFeats  : number of features in Hourglass
#     """
#     def __init__(self, n, nModules, nFeats):
#         super(Hourglass,self).__init__()
#         self.n = n
#         self.nModules = nModules
#         self.nFeats = nFeats
#
#         _up1_, _low1_, _low2_, _low3_ = [], [], [], []
#         # Top => _up1_
#         for j in range(self.nModules):
#             _up1_.append(Residual(self.nFeats, self.nFeats))
#         # Down => _low1_, _low2_, _low3_
#         self.low1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         for j in range(self.nModules):
#             _low1_.append(Residual(self.nFeats, self.nFeats))
#         # Middle low resolution Layers
#         if self.n > 1:
#             self.low2 = Hourglass(n-1,self.nModules, self.nFeats)
#         else:
#             for j in range(self.nModules):
#                 _low2_.append(Residual(self.nFeats,self.nFeats))
#             self.low2_ = nn.ModuleList(_low2_)
#         for j in range(self.nModules):
#             _low3_.append(Residual(self.nFeats, self.nFeats))
#
#         self.up1_ = nn.ModuleList(_up1_)
#         self.low1_ = nn.ModuleList(_low1_)
#         self.low3_ = nn.ModuleList(_low3_)
#
#         # NN upsampling for top-down processing
#         self.up2 = nn.Upsample(scale_factor=2)
#
#     def forward(self, x):
#         up1 = x
#         for j in range(self.nModules):
#             up1 = self.up1_[j](up1)
#
#         low1 = self.low1(x)
#         for j in range(self.nModules):
#             low1 = self.low1_[j](low1)
#
#         if self.n > 1:
#             low2 = self.low2(low1)
#         else:
#             low2 = low1
#             for j in range(self.nModules):
#                 low2 = self.low2_[j](low2)
#
#         low3 = low2
#         for j in range(self.nModules):
#             low3 = self.low3_[j](low3)
#
#         up2 = self.up2(low3)
#
#         return up1 + up2
#
# class HourglassNet(nn.Module):
#     """
#     Hourglass for 2D Pose Estimation.
#     """
#     def __init__(self, nStack, nModules, nFeats, numOutput):
#         super(HourglassNet, self).__init__()
#         self.nStack = nStack
#         self.nModules = nModules
#         self.nFeats = nFeats
#         self.numOutput = numOutput
#
#         self.conv1_ = nn.Conv2d(3, 64, bias=True, kernel_size=7, stride=2, padding=3)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.r1 = Residual(64, 128)
#         self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.r4 = Residual(128, 128)
#         self.r5 = Residual(128, self.nFeats)
#
#         _hourglass, _Residual, _lin_, _tmpOut, _ll_, _tmpOut_ = [], [], [], [], [], []
#         for i in range(self.nStack):
#             # Intermediate supervision process
#             ## Hourglass
#             _hourglass.append(Hourglass(4, self.nModules, nFeats))
#             ## Residual
#             for j in range(self.nModules):
#                 _Residual.append(Residual(self.nFeats, self.nFeats))
#             ## 1×1 conv remaps of heatmaps
#             lin = nn.Sequential(nn.Conv2d(self.nFeats, self.nFeats, bias=True, kernel_size=1, stride=1),
#                                 nn.BatchNorm2d(self.nFeats),
#                                 self.relu)
#             _lin_.append(lin)
#             _tmpOut.append(nn.Conv2d(self.nFeats, self.numOutput,bias=True, kernel_size=1, stride=1))
#             # Except the final part of the Net
#             if i < self.nStack-1:
#                 _ll_.append(nn.Conv2d(self.nFeats,self.nFeats,bias=True, kernel_size=1, stride=1))
#                 _tmpOut_.append(nn.Conv2d(self.numOutput, self.nFeats, bias=True, kernel_size=1, stride=1))
#
#         self.hourglass = nn.ModuleList(_hourglass)
#         self.Residual = nn.ModuleList(_Residual)
#         self.lin_ = nn.ModuleList(_lin_)
#         self.tmpOut = nn.ModuleList(_tmpOut)
#         self.ll_ = nn.ModuleList(_ll_)
#         self._tmpOut_ = nn.ModuleList(_tmpOut_)
#
#     def forward(self, x):
#         x= self.conv1_(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.r1(x)
#         x = self.maxpool(x)
#         x = self.r4(x)
#         x = self.r5(x)
#
#         out =[]
#
#         for i in range(self.nStack):
#             hg = self.hourglass[i](x)
#             ll = hg
#             for j in range(self.nModules):
#                 ll = self.Residual[i*self.nModules + j](ll)
#             ll = self.lin_[i](ll)
#             tmpOut = self.tmpOut[i](ll)
#             out.append(tmpOut)
#             # Except the final part of the Net
#             if i < self.nStack-1:
#                 ll_ = self.ll_[i](ll)
#                 tmpOut_ = self.tmpOut_[i](tmpOut)
#                 # Intermediate supervision process
#                 ## x: splits
#                 ## ll_: features
#                 ## tmpOut_: 1×1 conv remaps of heatmaps
#                 x = x + ll_ + tmpOut_
#
#         return out


# TODO: Edit the CLSTM module to make it work
# apply the LSTM conv on each pixel
class CLSTM(nn.Module):
    """
    Convolutional LSTM for Hourglass.
    * Residual Block(input: numIn x w x h | output: numOut x w x h)
    * CLSTM(input: )
    """

    def __init__(self, inputSize, hiddenSize, numLayers, seqLength, res):
        super(CLSTM, self).__init__()
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers
        self.seqLength = seqLength
        self.res = res

        # torch.nn.LSTM(input_size,hidden_size,num_layers)
        self.lstm = nn.LSTM(self.inputSize, self.hiddenSize, self.numLayers)

    def forward(self, inp):
        # Replicate encoder output
        repDim = list(inp.unsqueeze(1).shape)
        repDim[1] = self.seqLength
        rep = inp.unsqueeze(1).expand(repDim)

        # Merge into one mini-batch
        x = inp.transpose(1, 2).transpose(2, 3)
        x = x.contiguous()
        x = x.view(-1, self.inputSize)
        x = x.view(-1, 1, self.inputSize)
        x = nn.ZeroPad2d((0, 0, 0, 15))(x)
        # self-written
        # x = torch.zeros(inp.shape[0], self.seqLength, inp.shape[1], inp.shape[2], inp.shape[3]).cuda()
        # for i in range(inp.shape[0]):
        #     x[i, 0+self.seqLength*i,] = inp[i,]
        # x = x.view(-1, self.seqLength, self.inputSize)

        # LSTM
        ## features on each pixel, i.o.w., 1 x 1 conv layers
        '''
        Input: seq_len * batch * input_size
            seq_len:        time-steps, number of sequence members
            batch:          number of sequences
            input_size:     non-batch feature size, c x w x h for conv input
        '''
        h, _ = self.lstm(x)
        h = h.contiguous()
        # Split from one mini-batch
        h = h.view(-1, self.res, self.res, self.seqLength, self.hiddenSize)
        h = h.transpose(1, 3).transpose(2, 4)
        # h = h.view(rep.shape)
        # Add residual to encoder output
        out = h + rep
        # Merger output in batch dimension
        out = out.view(-1, self.hiddenSize, self.res, self.res)
        return out


# TODO: Edit the HourglassLSTM module to make it work
class HourglassLSTM(nn.Module):
    """
    One Hourglass with LSTM.
    """

    def __init__(self, nFeats, n, nModules, hiddenSize, numLayers, seqLength):
        super(HourglassLSTM, self).__init__()
        # Parameters
        ## Hyperparameters for Hourglass
        self.nFeats = nFeats
        self.n = n
        self.nModules = nModules
        ## Hyperparameters for LSTM
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers
        self.seqLength = seqLength

        # Network
        self.residual = Residual(self.nFeats, self.nFeats)
        self.clstm1 = CLSTM(self.nFeats, self.hiddenSize, self.numLayers, self.seqLength, 2**(n+2))
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        if self.n > 1:
            self.hglstm = HourglassLSTM(self.nFeats, n-1, self.nModules, self.hiddenSize,
                                        self.numLayers, self.seqLength)
        else:
            self.clstm2 = CLSTM(self.nFeats, self.hiddenSize, self.numLayers, self.seqLength, 2**(n+1))

    def forward(self, inp):
        # Upper Branch
        up1 = self.residual(inp)
        up1 = self.clstm1(up1)
        # Lower Branch
        x = self.maxpool(inp)
        x = self.residual(x)
        if self.n > 1:
            x = self.hglstm(x)
        else:
            x = self.residual(x)
            x = self.clstm2(x)
        x = self.residual(x)
        up2 = f.interpolate(x, scale_factor=2, mode='nearest')

        return up1 + up2


# TODO: Check if the HourglassPrediction Module can work as wanted
class Hourglass2DPrediction(nn.Module):
    def __init__(self, opt):
        super(Hourglass2DPrediction, self).__init__()
        # Hyperparameters for Hourglass from opt
        self.nFeats = opt.nFeats
        self.nModules = opt.nModules
        self.outputRes = opt.outputRes
        self.nJoints = opt.nJoints
        # Hyperparameters for LSTM from opt
        self.seqLength = opt.preSeqLen
        self.hiddenSize = opt.hiddenSize
        self.numLayers = opt.numLayers

        self.conv1 = nn.Conv2d(3, 64, bias=True, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.r1 = Residual(64, 128)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.r2 = Residual(128, 128)
        self.r3 = Residual(128, self.nFeats)
        self.hgLSTM = HourglassLSTM(self.nFeats, 4, self.nModules, self.hiddenSize, self.numLayers, self.seqLength)
        # 1×1 conv remaps of heatmaps
        self.lin = nn.Sequential(nn.Conv2d(self.nFeats, self.nFeats, bias=True, kernel_size=1, stride=1),
                                 nn.BatchNorm2d(self.nFeats),
                                 self.relu)
        # Output heatmaps
        self.conv2 = nn.Conv2d(self.nFeats, self.nJoints, kernel_size=1, stride=1, bias=True)

    def forward(self, inp):
        # Initial processing of the image
        x = self.conv1(inp)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.r1(x)
        x = self.maxpool(x)
        x = self.r2(x)
        x = self.r3(x)

        out = []
        # Forecasting
        x = self.hgLSTM(x)
        # Linear layers to produce first set of predictions
        x = self.lin(x)
        # Output heatmaps
        out = self.conv2(x)
        # Split output in batch dimension
        out = out.view(-1, self.seqLength, self.nJoints, self.outputRes, self.outputRes)
        # out = out.split(1, 1)

        return out