import torch.nn as nn
from .layers.Residual import Residual

class Hourglass(nn.Module):
    """Basic Structure for Hourglass."""
    def __init__(self, n, nModules, nFeats):
        super(Hourglass,self).__init__()
        self.n = n
        self.nModules = nModules
        self.nFeats = nFeats

        _up1_, _low1_, _low2_, _low3_ = [], [], [], []
        # Top => _up1_
        for j in range(self.nModules):
            _up1_.append(Residual(self.nFeats, self.nFeats))
        # Down => _low1_, _low2_, _low3_
        self.low1 = nn.MaxPool2d(kernel_size=2, stride=2)
        for j in range(self.nModules):
            _low1_.append(Residual(self.nFeats, self.nFeats))
        # Middle low resolution Layers
        if self.n > 1:
            self.low2 = Hourglass(n-1,self.nModules, self.nFeats)
        else:
            for j in range(self.nModules):
                _low2_.append(Residual(self.nFeats,self.nFeats))
            self.low2_ = nn.ModuleList(_low2_)
        for j in range(self.nModules):
            _low3_.append(Residual(self.nFeats, self.nFeats))

        self.up1_ = nn.ModuleList(_up1_)
        self.low1_ = nn.ModuleList(_low1_)
        self.low3_ = nn.ModuleList(_low3_)

        # NN upsampling for top-down processing
        self.up2 = nn.Upsample(scale_factor=2)

    def forward(self, x):
        up1 = x
        for j in range(self.nModules):
            up1 = self.up1_[j](up1)

        low1 = self.low1(x)
        for j in range(self.nModules):
            low1 = self.low1_[j](low1)

        if self.n > 1:
            low2 = self.low2(low1)
        else:
            low2 = low1
            for j in range(self.nModules):
                low2 = self.low2_[j](low2)

        low3 = low2
        for j in range(self.nModules):
            low3 = self.low3_[j](low3)

        up2 = self.up2(low3)

        return up1 + up2

class HourglassNet(nn.Module):
    """
    Hourglass for 2D Pose Estimation.
    """
    def __init__(self, nStack, nModules, nFeats, numOutput):
        super(HourglassNet, self).__init__()
        self.nStack = nStack
        self.nModules = nModules
        self.nFeats = nFeats
        self.numOutput = numOutput

        self.conv1_ = nn.Conv2d(3, 64, bias=True, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.r1 = Residual(64, 128)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.r4 = Residual(128, 128)
        self.r5 = Residual(128, self.nFeats)

        _hourglass, _Residual, _lin_, _tmpOut, _ll_, _tmpOut_ = [], [], [], [], [], []
        for i in range(self.nStack):
            # Intermediate supervision process
            ## Hourglass
            _hourglass.append(Hourglass(4, self.nModules, nFeats))
            ## Residual
            for j in range(self.nModules):
                _Residual.append(Residual(self.nFeats, self.nFeats))
            ## 1×1 conv remaps of heatmaps
            lin = nn.Sequential(nn.Conv2d(self.nFeats, self.nFeats, bias=True, kernel_size=1, stride=1),
                                nn.BatchNorm2d(self.nFeats),
                                self.relu)
            _lin_.append(lin)
            _tmpOut.append(nn.Conv2d(self.nFeats, self.numOutput,bias=True, kernel_size=1, stride=1))
            # Except the final part of the Net
            if i < self.nStack-1:
                _ll_.append(nn.Conv2d(self.nFeats,self.nFeats,bias=True, kernel_size=1, stride=1))
                _tmpOut_.append(nn.Conv2d(self.numOutput, self.nFeats, bias=True, kernel_size=1, stride=1))

        self.hourglass = nn.ModuleList(_hourglass)
        self.Residual = nn.ModuleList(_Residual)
        self.lin_ = nn.ModuleList(_lin_)
        self.tmpOut = nn.ModuleList(_tmpOut)
        self.ll_ = nn.ModuleList(_ll_)
        self._tmpOut_ = nn.ModuleList(_tmpOut_)

    def forward(self, x):
        x= self.conv1_(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.r1(x)
        x = self.maxpool(x)
        x = self.r4(x)
        x = self.r5(x)

        out =[]

        for i in range(self.nStack):
            hg = self.hourglass[i](x)
            ll = hg
            for j in range(self.nModules):
                ll = self.Residual[i*self.nModules + j](ll)
            ll = self.lin_[i](ll)
            tmpOut = self.tmpOut[i](ll)
            out.append(tmpOut)
            # Except the final part of the Net
            if i < self.nStack-1:
                ll_ = self.ll_[i](ll)
                tmpOut_ = self.tmpOut_[i](tmpOut)
                # Intermediate supervision process
                ## x: splits
                ## ll_: features
                ## tmpOut_: 1×1 conv remaps of heatmaps
                x = x + ll_ + tmpOut_

        return out

#TODO: Edit the CLSTM module to make it work
class Clstm(nn.Module):
    """
    LSTM for Hourglass.
    """
    def __init__(self, seqLength, hiddenSize, numLayers, inputSize, res):
        super(Clstm, self).__init__()
        self.seqLength = seqLength
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers
        self.inputSize = inputSize
        self.res = res

        # toch.nn.LSTM(input_size,hidden_size,num_layers)
        self.lstm = nn.LSTM(self.inputSize, self.hiddenSize)

    def forward(self, inp):
        # Replicate encoder output
        rep = inp.repeat(self.seqLength, 2)
        # Merge into one mini-batch
        x1_ = inp.Transpose({2, 3}, {3, 4})
        x1 = x1_.View(-1, self.inputSize)
        # LSTM
        x2 = x1.View(-1, 1, self.inputSize)
        x3 = x2.Padding(1, self.seqLength - 1, 1)
        hid, (hn,cn)= self.lstm(x3, (h0,c0))
        h1 = hid.Contiguous()
        # Split from one mini-batch
        h2_ = h1.View(-1, self.res, self.res, self.seqLength, self.hiddenSize)
        h2 = h2_.Transpose({3, 4}, {2, 3}, {4, 5}, {3, 4})
        # Add residual to encoder output
        add = nn.CAddTable({rep,h2})
        # Merger output in batch dimension
        out = add.View(-1, self.hiddenSize, self.res, self.res)
        return out
#TODO: Edit the HourglassLSTM module to make it work
class HourglassLSTM(nn.Module):
    """
    One Hourglass with LSTM.
    """
    def __init__(self, n, nModules, nFeats, seqLength, hiddenSize, numLayers):
        super(HourglassLSTM, self).__init__()
        # Hyperparameters for Hourglass
        self.n = n
        self.nModules = nModules
        self.nFeats = nFeats
        # Hyperparameters for LSTM
        self.seqLength = seqLength
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers

        # Upper Branch
        self.up1 = Residual(self.nFeats, self.nFeats)
        self.clstm1 = Clstm(self.seqLength, self.hiddenSize, self.numLayers, self.nFeats, 2^(n+2))
        # Lower Branch
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.low1 = Residual(nFeats,nFeats)
        if self.n>1:
            self.low2 = HourglassLSTM(n-1, self.nModules, self.nFeats,self.seqLength, self.hiddenSize, self.numLayers)
        else:
            self.low2 = Residual(nFeats,nFeats)
            self.clstm2 = Clstm(self.seqLength, self.hiddenSize, self.numLayers, self.nFeats, 2^(n+1))
        self.low3 = Residual(nFeats, nFeats)
        self.up2 = nn.Upsample(scale_factor=2)

    def forward(self, inp):
        up1 = self.up1(inp)
        up1 = self.clstm1(up1)
        x = self.maxpool1(up1)
        x = self.low1(x)
        if self.n>1:
            x = self.low2(x)
        else:
            x = self.low2(x)
            x = self.clstm2(x)
        x = self.low3(x)
        up2 = self.up2(x)

        return nn.CAddTable({up1,up2})
#TODO: Check if the HourglassPrediction Module can work as wanted
class HourglassPrediction(nn.Module):
    def __init__(self, opt):
        super(HourglassPrediction, self).__init__()
        # Hyperparameters for Hourglass
        self.nModules = opt.nModules
        self.nFeats = opt.nFeats
        # Hyperparameters for LSTM
        self.seqLength = opt.seqLength
        self.hiddenSize = opt.hiddenSize
        self.numLayers = opt.numLayers

        # Net before HourglassLSTM
        self.conv1 = nn.Conv2d(3, 64, bias=True, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.r1 = Residual(64, 128)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.r2 = Residual(128, 128)
        self.r3 = Residual(128, self.nFeats)
        # HourglassLSTM
        self.hgLSTM = HourglassLSTM(4, self.nModules, self.nFeats, self.seqLength, self.hiddenSize, self.numLayers)
        # 1×1 conv remaps of heatmaps
        self.lin = nn.Sequential(nn.Conv2d(self.nFeats, self.nFeats, bias=True, kernel_size=1, stride=1),
                            nn.BatchNorm2d(self.nFeats),
                            self.relu)
        # Output heatmaps
        self.conv2 = nn.Conv2d(self.nFeats, self.nFeats, kernel_size=1,stride=1, bias=True)

    def forward(self, x):
        x= self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.r1(x)
        x = self.maxpool(x)
        x = self.r2(x)
        x = self.r3(x)

        out =[]
        # Forecasting
        ## out = nn.Identity()()

        x = self.hgLSTM(x)
        x = self.lin(x)
        out = self.conv2(x)

        return out