<<<<<<< HEAD
import argparse
import os
import datetime


class Opts:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('-expID', default='train', help='Experiment ID')
        self.parser.add_argument('-task', default='', help='task of the experiment')
        # the number of available GPUs
        ## num_worker = 4 * num_GPU, for the dataloader, considering the memory and IO usage
        self.parser.add_argument('-gpu', default='0', help='-1 for CPU')
        self.parser.add_argument('-num_workers', type=int, default=0, help='num of threads')
        # exp process
        self.parser.add_argument('-test', action='store_true', help='test')
        self.parser.add_argument('-debug', type=int, default=0, help='debug level')
        self.parser.add_argument('-img', default='None', help='path/to/demo/image')
        # training metric
        self.parser.add_argument('-metric', default='acc')
        # dataset
        self.parser.add_argument('-scale',  type=float, default=0.25)
        self.parser.add_argument('-rotate', type=float, default=30)
        self.parser.add_argument('-dataset', default='penn-crop')
        ## maxSeqLen and minSeqLen are both the index of 2
        self.parser.add_argument('-MaxSeqLenIndex', type=int, default=4, help='Max Sequence Length')
        self.parser.add_argument('-MinSeqLenIndex', type=int, default=0, help='Min Sequence Length to train with')
        self.parser.add_argument('-incEpoch', type=int, default=2, help='Increase sequence length after # of epochs')
        # set hourglass
        self.parser.add_argument('-nFeats',   type=int, default=256, help='# features in the hourglass')
        self.parser.add_argument('-nStack',   type=int, default=2,   help='# hourglasses to stack')
        self.parser.add_argument('-nModules', type=int, default=1,   help='# residual modules at each hourglass')# 2 for hourglass 3D
        self.parser.add_argument('-preSeqLen', type=int, default=16, help='predicted sequence length')
        self.parser.add_argument('-hiddenSize', type=int, default=256, help='Hidden Size')
        self.parser.add_argument('-numLayers', type=int, default=1, help='Number of Hidden Layers')
        # set training hyperparameters
        self.parser.add_argument('-LR',           type=float, default=2.5e-4,  help='Learning Rate')#0.001
        self.parser.add_argument('-dropLR',       type=int,   default=1000000, help='drop LR')
        self.parser.add_argument('-nEpochs',      type=int,   default=5,    help='#training epochs')#140
        self.parser.add_argument('-valIntervals', type=int,   default=1,       help='#valid intervel')
        self.parser.add_argument('-trainBatch',   type=int,   default=1,       help='Mini-batch size')#32
        # load trained model
        self.parser.add_argument('-loadModel', default='none', help='Provide full path to a previously trained model')

    def parse(self, args=''):
        if args == '':
            opts = self.parser.parse_args()
        else:
            opts = self.parser.parse_args(args)

        # record the exp time stamp
        exp_time = datetime.datetime.now()
        opts.exp_time = f'{exp_time.year}-{exp_time.month}-{exp_time.day}-{exp_time.hour}-{exp_time.minute}'
        # training strategies
        #opts.eps = 1e-6
        opts.alpha = 0.99
        opts.epsilon = 1e-8
        opts.weightDecay = 0.0
        opts.momentum = 0.0
        #
        opts.root_dir = os.path.join(os.path.dirname(__file__), '..')
        # Paths config for the exp
        opts.exp_dir = os.path.join(opts.root_dir, 'exp')
        ## save path
        opts.save_dir = os.path.join(opts.exp_dir, opts.expID)
        if opts.test:
            opts.expID = opts.expID + 'TEST'
            opts.save_dir = os.path.join(opts.exp_dir, opts.expID)
        ### make sure there is a save directory
        if not os.path.exists(opts.save_dir):
            os.makedirs(opts.save_dir)
        file_opts = os.path.join(opts.save_dir, f'opts_{opts.exp_time}.txt')
        # Set nthreads
        if opts.debug > 0:
            opts.num_workers = 1
        opts.gpu = [int(gpu) for gpu in opts.gpu.split(',')]
        #TODO: all the dataset configurations should be in a separate module
        ## set heatmaps configuration
        opts.hmGauss = 1
        opts.inputRes = 256
        opts.outputRes = 64
        opts.nOutput = 1

        ## File paths
        opts.data_dir = os.path.join(opts.root_dir, 'data')
        # opts.h36mImgSize = 224
        # opts.h36mImgDir = '/mnt/Data/Human3.6M/images/'
        opts.penncropDir = os.path.join(opts.data_dir, 'penn-crop')
        ## Set number of human joints
        opts.num_depth = opts.nOutput if opts.task=='3d' else 0
        ## Set dataset scale and metric
        if opts.scale == -1:
            opts.scale = 0.3 if opts.dataset=='coco' else 0.25
        if opts.rotate == -1:
            opts.rotate = 40 if opts.dataset=='coco' else 30
        # Arguments for opts
        args = dict((name, getattr(opts, name)) for name in dir(opts)
                    if not name.startswith('_'))
        # Save opts
        with open(file_opts, 'wt') as f_o:
            f_o.write('=====Args:\n')
            for k, v in sorted(args.items()):
                f_o.write('  %s: %s\n' % (str(k), str(v)))

        return opts
=======
import argparse
import os
import datetime


class Opts:
  def __init__(self):
    self.parser = argparse.ArgumentParser()

    self.parser.add_argument('-expID', default='train', help='Experiment ID')
    self.parser.add_argument('-task', default='', help='task of the experiment')
    # the number of available GPUs
    ## num_worker = 4 * num_GPU, for the dataloader, considering the memory and IO usage
    self.parser.add_argument('-gpu', default='0', help='-1 for CPU')
    self.parser.add_argument('-num_workers', type=int, default=0, help='num of threads')
    # exp process
    self.parser.add_argument('-test', action='store_true', help='test')
    self.parser.add_argument('-debug', type=int, default=0, help='debug level')
    self.parser.add_argument('-demo', default='', help='path/to/demo/image')
    # training metric
    self.parser.add_argument('-metric', default='acc')
    # dataset
    self.parser.add_argument('-scale',  type=float, default=0.25)
    self.parser.add_argument('-rotate', type=float, default=30)
    self.parser.add_argument('-dataset', default='penn-crop')
    ## maxSeqLen and minSeqLen are both the index of 2
    self.parser.add_argument('-MaxSeqLenIndex', type=int, default=4, help='Max Sequence Length')
    self.parser.add_argument('-MinSeqLenIndex', type=int, default=0, help='Min Sequence Length to train with')
    self.parser.add_argument('-incEpoch', type=int, default=2, help='Increase sequence length after # of epochs')
    # set hourglass
    self.parser.add_argument('-nFeats',   type=int, default=256, help='# features in the hourglass')
    self.parser.add_argument('-nStack',   type=int, default=2,   help='# hourglasses to stack')
    self.parser.add_argument('-nModules', type=int, default=1,   help='# residual modules at each hourglass')# 2 for hourglass 3D
    self.parser.add_argument('-preSeqLen', type=int, default=16, help='predicted sequence length')
    self.parser.add_argument('-hiddenSize', type=int, default=256, help='Hidden Size')
    self.parser.add_argument('-numLayers', type=int, default=1, help='Number of Hidden Layers')
    # set training hyperparameters
    self.parser.add_argument('-LR',           type=float, default=2.5e-4,  help='Learning Rate')#0.001
    self.parser.add_argument('-dropLR',       type=int,   default=1000000, help='drop LR')
    self.parser.add_argument('-nEpochs',      type=int,   default=5,    help='#training epochs')#140
    self.parser.add_argument('-valIntervals', type=int,   default=1,       help='#valid intervel')
    self.parser.add_argument('-trainBatch',   type=int,   default=1,       help='Mini-batch size')#32
    # load trained model
    self.parser.add_argument('-loadModel', default='none', help='Provide full path to a previously trained model')

  def parse(self, args=''):
    if args == '':
      opts = self.parser.parse_args()
    else:
      opts = self.parser.parse_args(args)

    # record the exp time stamp
    exp_time = datetime.datetime.now()
    opts.exp_time = f'{exp_time.year}-{exp_time.month}-{exp_time.day}-{exp_time.hour}-{exp_time.minute}'
    # training strategies
    #opts.eps = 1e-6
    opts.alpha = 0.99
    opts.epsilon = 1e-8
    opts.weightDecay = 0.0
    opts.momentum = 0.0
    #
    opts.root_dir = os.path.join(os.path.dirname(__file__), '..')
    # Paths config for the exp
    opts.exp_dir = os.path.join(opts.root_dir, 'exp')
    ## save path
    opts.save_dir = os.path.join(opts.exp_dir, opts.expID)
    if opts.test:
      opts.expID = opts.expID + 'TEST'
      opts.save_dir = os.path.join(opts.exp_dir, opts.expID)
    ### make sure there is a save directory
    if not os.path.exists(opts.save_dir):
      os.makedirs(opts.save_dir)
    file_opts = os.path.join(opts.save_dir, f'opts_{opts.exp_time}.txt')
    # Set nthreads
    if opts.debug > 0:
      opts.num_workers = 1
    opts.gpu = [int(gpu) for gpu in opts.gpu.split(',')]
    #TODO: all the dataset configurations should be in a separate module
    ## set heatmaps configuration
    opts.hmGauss = 1 #2
    opts.inputRes = 256
    opts.outputRes = 64
    opts.nOutput = 1
    # opts.hmGaussInp = 20
    # opts.shiftPX = 50##
    # opts.disturb = 10##
    ## File paths
    opts.data_dir = os.path.join(opts.root_dir, 'data')
    # opts.h36mImgSize = 224
    # opts.mpiiImgDir = '/mnt/Data/mpii/images/'
    # opts.h36mImgDir = '/mnt/Data/Human3.6M/images/'
    opts.penncropDir = os.path.join(opts.data_dir, 'penn-crop')
    ## Set number of human joints and 3d or not
    opts.nJoints = 13
    ## mpii human 16 joints
    #  0 - r ankle,     1 - r knee,      2 - r hip,    3 - l hip,
    #  4 - l knee,      5 - l ankle,     6 - pelvis,   7 - thorax,
    #  8 - upper neck,  9 - head top,   10 - r wrist, 11 - r elbow,
    # 12 - r shoulder, 13 - l shoulder, 14 - l elbow, 15 - l wrist
    opts.accIdxs = [0, 1, 2, 3, 4, 5, 10, 11, 14, 15]
    opts.shuffleRef = [[0, 5], [1, 4], [2, 3],
                  [10, 15], [11, 14], [12, 13]]
    opts.edges = [[0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5],
             [10, 11], [11, 12], [12, 8], [8, 13], [13, 14], [14, 15],
             [6, 8], [8, 9]]
    opts.num_depth = opts.nOutput if opts.task=='3d' else 0
    ## Set dataset scale and metric
    if opts.scale == -1:
      opts.scale = 0.3 if opts.dataset=='coco' else 0.25
    if opts.rotate == -1:
      opts.rotate = 40 if opts.dataset=='coco' else 30
    # Arguments for opts
    args = dict((name, getattr(opts, name)) for name in dir(opts)
                if not name.startswith('_'))
    # Save opts
    with open(file_opts, 'wt') as f_o:
      f_o.write('=====Args:\n')
      for k, v in sorted(args.items()):
         f_o.write('  %s: %s\n' % (str(k), str(v)))

    return opts
>>>>>>> 7c234f0346a56d52e13f09f8d7179d3fe92b73d3
