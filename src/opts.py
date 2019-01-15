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
    self.parser.add_argument('-num_workers', type=int, default=4, help='num of threads')
    # exp process
    self.parser.add_argument('-test', action='store_true', help='test')
    self.parser.add_argument('-debug', type=int, default=0, help='debug level')
    self.parser.add_argument('-demo', default='', help='path/to/demo/image')
    # training metric
    self.parser.add_argument('-metric', default='acc')
    self.parser.add_argument('-scale',  type=float, default=0.25)
    self.parser.add_argument('-rotate', type=float, default=30)
    self.parser.add_argument('-dataset', default='mpii')
    # set hourglass
    self.parser.add_argument('-nFeats',   type=int, default=256, help='# features in the hourglass')
    self.parser.add_argument('-nStack',   type=int, default=2,   help='# hourglasses to stack')
    self.parser.add_argument('-nModules', type=int, default=2,   help='# residual modules at each hourglass')
    # set training hyperparameters
    self.parser.add_argument('-LR',           type=float, default=2.5e-4,  help='Learning Rate')#0.001
    self.parser.add_argument('-dropLR',       type=int,   default=1000000, help='drop LR')
    self.parser.add_argument('-nEpochs',      type=int,   default=60,      help='#training epochs')#140
    self.parser.add_argument('-valIntervals', type=int,   default=5,       help='#valid intervel')
    self.parser.add_argument('-trainBatch',   type=int,   default=6,       help='Mini-batch size')#32
    # load trained model
    self.parser.add_argument('-loadModel', default='none', help='Provide full path to a previously trained model')

  def parse(self, args=''):
    if args == '':
      opt = self.parser.parse_args()
    else:
      opt = self.parser.parse_args(args)

    # record the exp time stamp
    opt.exp_time = datetime.datetime.now()
    # training strategies
    opt.eps = 1e-6
    opt.momentum = 0.0
    opt.alpha = 0.99
    opt.epsilon = 1e-8
    # set heatmaps configuration
    opt.hmGauss = 1 #2
    opt.inputRes = 256
    opt.outputRes = 64
    ## opt.hmGaussInp = 20
    ## opt.shiftPX = 50##
    ## opt.disturb = 10##
    # file paths
    opt.root_dir = os.path.join(os.path.dirname(__file__),'..')
    opt.data_dir = os.path.join(opt.root_dir,'data')
    opt.h36mImgSize = 224
    opt.mpiiImgDir = '/mnt/Data/mpii/images/'
    opt.h36mImgDir = '/mnt/Data/Human3.6M/images/'

    opt.exp_dir = os.path.join(opt.root_dir,'exp')
    ## save path
    opt.save_dir = os.path.join(opt.exp_dir, opt.expID)
    if opt.test:
      opt.expID = opt.expID + 'TEST'
      opt.save_dir = os.path.join(opt.exp_dir, opt.expID)
    # Set nthreads
    if opt.debug > 0:
      opt.num_workers = 1
    opt.gpu = [int(gpu) for gpu in opt.gpu.split(',')]
    # Set number of human joints and 3d or not
    opt.nJoints = 17 if opt.dataset=='coco' else 16
    # mpii human joints
    #  0 - r ankle,     1 - r knee,      2 - r hip,    3 - l hip,
    #  4 - l knee,      5 - l ankle,     6 - pelvis,   7 - thorax,
    #  8 - upper neck,  9 - head top,   10 - r wrist, 11 - r elbow,
    # 12 - r shoulder, 13 - l shoulder, 14 - l elbow, 15 - l wrist
    opt.accIdxs = [0, 1, 2, 3, 4, 5, 10, 11, 14, 15]
    opt.shuffleRef = [[0, 5], [1, 4], [2, 3],
                  [10, 15], [11, 14], [12, 13]]
    opt.edges = [[0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5],
             [10, 11], [11, 12], [12, 8], [8, 13], [13, 14], [14, 15],
             [6, 8], [8, 9]]
    opt.num_depth = opt.num_output if opt.task=='3d' else 0
    # Set dataset scale and metric
    if opt.scale == -1:
      opt.scale = 0.3 if opt.dataset=='coco' else 0.25
    if opt.rotate == -1:
      opt.rotate = 40 if opt.dataset=='coco' else 30
    # Arguments for opt
    args = dict((name, getattr(opt, name)) for name in dir(opt)
                if not name.startswith('_'))
    #
    if not os.path.exists(opt.save_dir):
      os.makedirs(opt.save_dir)
    file_name = os.path.join(opt.save_dir, f'opt_{opt.exp_time.year}-{opt.exp_time.month}-{opt.exp_time.day}-{opt.exp_time.hour}-{opt.exp_time.minute}.txt')
    with open(file_name, 'wt') as opt_file:
      opt_file.write('==> Args:\n')
      for k, v in sorted(args.items()):
         opt_file.write('  %s: %s\n' % (str(k), str(v)))

    return opt
