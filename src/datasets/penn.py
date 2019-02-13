<<<<<<< HEAD
import torch
import torch.utils.data as data
import numpy as np
from h5py import File
import cv2
from utils.utils import Rnd, Flip, ShuffleLR
from utils.img import Crop, DrawGaussian, Transform


# Penn Action Official Joints Info, Menglong
# 0.  head
# 1.  left_shoulder  2.  right_shoulder
# 3.  left_elbow     4.  right_elbow
# 5.  left_wrist     6.  right_wrist
# 7.  left_hip       8.  right_hip
# 9.  left_knee      10. right_knee
# 11. left_ankle     12. right_ankle

# Penn-Crop Joints Info for Dataset, Yuwei
# 0.  head
# 1.  right_shoulder  2.  left_shoulder
# 3.  right_elbow     4.  left_elbow
# 5.  right_wrist     6.  left_wrist
# 7.  right_hip       8.  left_hip
# 9.  right_knee      10. left_knee
# 11. right_ankle     12. left_ankle


class PENN_CROP(data.Dataset):
    def __init__(self, opt, split):

        print(f'==> initializing 2D PENN {split} data.')
        annot = {}
        tags = ['ind2sub', 'part']
        f = File(f'{opt.data_dir}/penn-crop/{split}.h5', 'r')
        for tag in tags:
            annot[tag] = np.asarray(f[tag]).copy()
        f.close()

        print(f"Loaded 2D {split} {len(annot['ind2sub'])} samples")

        self.split = split
        self.opt = opt
        self.annot = annot
        # self.returnMeta = returnMeta
        self.nPhase = opt.preSeqLen
        self.inputRes = opt.inputRes
        self.outputRes = opt.outputRes
        self.seqId = self.annot['ind2sub'][:, 0]
        self.frameId = self.annot['ind2sub'][:, 1]
        self.part = self.annot['part']
        self.nJoints = self.part.shape[1]
        self.skeleton = [[1, 3], [3, 5], [7, 9],  [9, 11],
                         [2, 4], [4, 6], [8, 10], [10, 12],
                         [0, 7], [0, 8], [1, 2]]
        self.data_dir = opt.data_dir

    def getSeq(self, i):
        id = self.seqId[i]
        nFrame = np.sum(self.seqId == id)
        ind = np.linspace(i, i + nFrame - 1, self.nPhase)
        ind = np.round(ind)
        assert (len(ind) == self.nPhase)

        maxind = np.max(np.where(self.seqId == id))
        ind[ind >= maxind] = maxind
        # print('the max index in this video is:',maxind)
        # print(ind)
        return ind

    def getCenterScale(self, im):
        h, w = im.shape[:2]
        x = (w + 1) / 2
        y = (h + 1) / 2
        scale = max(w, h) / 200
        y = y + scale * 15
        scale = scale * 1.25
        center = [x, y]
        center = np.asarray(center)
        return center, scale

    def LoadImage(self, index):
        seqpath = f"{self.data_dir}/penn-crop/frames/{self.seqId[index]:04d}"
        framepath = f"{seqpath}/{self.frameId[index]:06d}.jpg"
        img = cv2.imread(framepath)
        return img

    def GetPartInfo(self, index):
        pts = self.annot['part'][index].copy()
        c = self.annot['center'][index].copy()
        s = self.annot['scale'][index]
        s = s * 200
        return pts, c, s

    def __getitem__(self, index):
        seqIdx = self.getSeq(index)
        """
        input: predSeqLen x 3 x inputRes x inputRes   Input image After Crop and transform
        hmap:  predSeqLen x numJoints x outputRes x outputRes
        gtpts: predSeqLen x numJoints x 2             Joints Positions BEFORE crop and transform
        proj:  predSeqLen x numJoints x 2             Joints Positions AFTER crop and transform
        """
        input = np.zeros((self.nPhase, 3, self.inputRes, self.inputRes))
        hmap = np.zeros((self.nPhase, self.nJoints, self.outputRes, self.outputRes))
        gtpts = np.zeros((self.nPhase, self.nJoints, 2))
        repos, trans, focal, proj = {}, {}, {}, {}
        for i in range(len(seqIdx)):
            sid = seqIdx[i]
            im = self.LoadImage(int(sid))

            if i == 0:
                center, scale = self.getCenterScale(im)
            inp = Crop(im, center, scale, 0, self.inputRes)
            pts = self.part[int(sid)]

            pj = np.zeros(np.shape(pts))
            for j in range(len(pts)):
                if pts[j][0] != 0 and pts[j][1] != 0:
                    pj[j] = Transform(pts[j], center, scale, 0, self.outputRes, False)

            hm = np.zeros((np.shape(pts)[0], self.outputRes, self.outputRes))
            for j in range(len(pts)):
                if pts[j][0] != 0 and pts[j][1] != 0:
                    DrawGaussian(hm[j], np.round(pj[j]), 2)

            inp = inp.transpose(2, 1, 0)
            input[i] = inp
            repos[i] = np.zeros((np.size(1), 3))
            trans[i] = np.zeros(3)
            focal[i] = np.zeros(1)
            hmap[i] = hm
            proj[i] = pj
            gtpts[i] = pts

        if self.split == 'train':
            m1 = np.random.uniform(0.8, 1.2)
            m2 = np.random.uniform(0.8, 1.2)
            m3 = np.random.uniform(0.8, 1.2)
            for i in range(len(input)):
                input[i][:, :, 0] = input[i][:, :, 0] * m1
                np.clip(input[i][:, :, 0], 0, 1, out=input[i][:, :, 0])

                input[i][:, :, 1] = input[i][:, :, 1] * m2
                np.clip(input[i][:, :, 1], 0, 1, out=input[i][:, :, 1])

                input[i][:, :, 2] = input[i][:, :, 2] * m3
                np.clip(input[i][:, :, 2], 0, 1, out=input[i][:, :, 2])

            if np.random.uniform() <= 0.5:
                for i in range(len(input)):
                    input[i] = cv2.flip(input[i], 1)
                    hmap[i] = Flip(ShuffleLR(hmap[i]))
                    proj[i] = ShuffleLR(proj[i])
                    ind = np.where(proj[i] == 0)
                    proj[i][:, 0] = self.outputRes - proj[i][:, 0] + 1
                    if len(ind[0]) != 0:
                        proj[i][ind[0][0]] = 0



        return {'input': input, 'label': hmap, 'gtpts': gtpts, 'center': center, 'scale': scale,'proj':proj}

    def __len__(self):
=======
import torch
import torch.utils.data as data
import numpy as np
from h5py import File
import cv2
from utils.utils import Rnd, Flip, ShuffleLR
from utils.img import Crop, DrawGaussian, Transform
# from src.opts import Opts


class PENN(data.Dataset):
    def __init__(self, opt, split):

        print('==> initializing 2D PENN {} data.'.format(split))
        annot = {}
        tags = ['ind2sub', 'part']
        f = File('{}/penn-crop/{}.h5'.format(opt.data_dir, split), 'r')
        for tag in tags:
            annot[tag] = np.asarray(f[tag]).copy()
        f.close()

        print('Loaded 2D {} {} samples'.format(split, len(annot['ind2sub'])))

        self.split = split
        self.opt = opt
        self.annot = annot
        # self.returnMeta = returnMeta
        self.nPhase = opt.preSeqLen
        self.inputRes = opt.inputRes
        self.nJoints = opt.nJoints
        self.outputRes = opt.outputRes
        self.seqId = self.annot['ind2sub'][:, 0]
        self.frameId = self.annot['ind2sub'][:, 1]
        self.part = self.annot['part']
        self.data_dir = opt.data_dir

    def getSeq(self, i):
        # print('the index is:',i)
        id = self.seqId[i]
        # print('the id is:',id)
        # print('the index in the video:',self.frameId[i])
        nFrame = np.sum(self.seqId == id)
        # print('the num of frame in the id is:',nFrame)
        ind = np.linspace(i, i + nFrame - 1, self.nPhase)
        ind = np.round(ind)
        assert (len(ind) == self.nPhase)

        maxind = np.max(np.where(self.seqId == id))
        ind[ind >= maxind] = maxind
        # print('the max index in this video is:',maxind)
        # print(ind)
        return ind

    def getCenterScale(self, im):
        h, w = im.shape[:2]
        x = (w + 1) / 2
        y = (h + 1) / 2
        scale = max(w, h) / 200
        y = y + scale * 15
        scale = scale * 1.25
        center = [x, y]
        center = np.asarray(center)
        return center, scale

    def LoadImage(self, index):
        seqpath = '{}/penn-crop/frames/{:04d}'.format(self.data_dir, self.seqId[index])
        framepath = '{}/{:06d}.jpg'.format(seqpath, self.frameId[index])
        img = cv2.imread(framepath)
        # print(framepath)
        return img

    def GetPartInfo(self, index):
        pts = self.annot['part'][index].copy()
        c = self.annot['center'][index].copy()
        s = self.annot['scale'][index]
        s = s * 200
        return pts, c, s

    def __getitem__(self, index):
        seqIdx = self.getSeq(index)
        # print('index is:',index)
        input = np.zeros((self.nPhase, 3, self.inputRes, self.inputRes))
        hmap = np.zeros((self.nPhase, self.nJoints, self.outputRes, self.outputRes))
        gtpts = np.zeros((self.nPhase, self.nJoints, 2))
        repos, trans, focal, proj = {}, {}, {}, {}
        # print('len of seqIdx:', len(seqIdx))
        for i in range(len(seqIdx)):
            sid = seqIdx[i]
            im = self.LoadImage(int(sid))

            if i == 0:
                center, scale = self.getCenterScale(im)
            inp = Crop(im, center, scale, 0, self.inputRes)
            pts = self.part[int(sid)]

            pj = np.zeros(np.shape(pts))
            for j in range(len(pts)):
                if pts[j][0] != 0 and pts[j][1] != 0:
                    pj[j] = Transform(pts[j], center, scale, 0, self.outputRes, False)

            hm = np.zeros((np.shape(pts)[0], self.outputRes, self.outputRes))
            for j in range(len(pts)):
                if pts[j][0] != 0 and pts[j][1] != 0:
                    DrawGaussian(hm[j], np.round(pj[j]), 2)

            # print(np.shape(inp))
            inp = inp.transpose(2, 1, 0)
            input[i] = inp
            repos[i] = np.zeros((np.size(1), 3))
            trans[i] = np.zeros(3)
            focal[i] = np.zeros(1)
            hmap[i] = hm
            proj[i] = pj
            gtpts[i] = pts

        if self.split == 'train':
            m1 = np.random.uniform(0.8, 1.2)
            m2 = np.random.uniform(0.8, 1.2)
            m3 = np.random.uniform(0.8, 1.2)
            for i in range(len(input)):
                input[i][:, :, 0] = input[i][:, :, 0] * m1
                np.clip(input[i][:, :, 0], 0, 1, out=input[i][:, :, 0])

                input[i][:, :, 1] = input[i][:, :, 1] * m1
                np.clip(input[i][:, :, 1], 0, 1, out=input[i][:, :, 1])

                input[i][:, :, 2] = input[i][:, :, 2] * m2
                np.clip(input[i][:, :, 2], 0, 1, out=input[i][:, :, 2])

                # np.savetxt('befornor.txt', a)
                # np.savetxt('afternor.txt', input[i][:, :, 0])
        if np.random.uniform() <= 0.5:
            for i in range(len(input)):
                input[i] = cv2.flip(input[i], 1)
                # print('hmap before:',hmap[i][0,:,:])
                # scio.savemat('hmapB.mat',{'hmB':hmap[i]})
                hmap[i] = Flip(ShuffleLR(hmap[i]))
                # scio.savemat('hmapA.mat', {'hmA': hmap[i]})
                # print('before shuffle:', proj[i])
                proj[i] = ShuffleLR(proj[i])
                # print('after shuffle:', proj[i])
                ind = np.where(proj[i] == 0)
                proj[i][:, 0] = self.outputRes - proj[i][:, 0] + 1
                if len(ind[0]) != 0:
                    proj[i][ind[0][0]] = 0

        return {'input': input, 'label': hmap, 'gtpts': gtpts, 'center': center, 'scale': scale}

    def __len__(self):
>>>>>>> 7c234f0346a56d52e13f09f8d7179d3fe92b73d3
        return len(self.annot['ind2sub'])