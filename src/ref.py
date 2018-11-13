nJoints = 16
#  0 - r ankle,     1 - r knee,      2 - r hip,    3 - l hip, 
#  4 - l knee,      5 - l ankle,     6 - pelvis,   7 - thorax, 
#  8 - upper neck,  9 - head top,   10 - r wrist, 11 - r elbow, 
# 12 - r shoulder, 13 - l shoulder, 14 - l elbow, 15 - l wrist
accIdxs = [0, 1, 2, 3, 4, 5, 10, 11, 14, 15]
shuffleRef = [[0, 5], [1, 4], [2, 3], 
             [10, 15], [11, 14], [12, 13]]
edges = [[0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5], 
         [10, 11], [11, 12], [12, 8], [8, 13], [13, 14], [14, 15], 
         [6, 8], [8, 9]]

h36mImgSize = 224

outputRes = 64
inputRes = 256

eps = 1e-6
    
momentum = 0.0
weightDecay = 0.0
alpha = 0.99
epsilon = 1e-8


scale = 0.25
rotate = 30
hmGauss = 1
hmGaussInp = 20
shiftPX = 50
disturb = 10

expDir = '../exp'
dataDir = '../data'
mpiiImgDir = '/mnt/Data/mpii/images/'
h36mImgDir = '/mnt/Data/Human3.6M/images/'

nThreads = 4
