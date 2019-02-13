import cv2import numpy as npimport matplotlib.pyplot as pltimport matplotlib.image as mpimgclass Visualizer(object):    def __init__(self, nJoints, skeleton, outputRes):        self.nJoints = nJoints        self.skeleton = skeleton        self.outputRes = outputRes        self.imgs = {}        # For 3D visualizer        self.plt = plt        self.fig = self.plt.figure()    def add_img(self, img, imgId='default'):        self.imgs[imgId] = img.copy()    def add_2d_joints_skeleton(self, joints, c, imgId='default'):        """        :param joints: BatchSize x NumJoints x Dim(2)        :param c:        :param imgId:        :return:        """        points = (joints.reshape(self.nJoints, -1)).numpy().astype(np.int32)*4        for j in range(self.nJoints):            if points[j, 0] != 0 and points[j, 1] != 0:                cv2.circle(self.imgs[imgId], (points[j, 0], points[j, 1]), 3, c, -1)        for s in self.skeleton:            if points[s[0], 0] !=0 and points[s[0], 1] !=0 \                    and points[s[1], 0]!= 0 and points[s[1], 1] != 0:                cv2.line(self.imgs[imgId], (points[s[0], 0], points[s[0], 1]),                   (points[s[1], 0], points[s[1], 1]), c, 2)    def show_img(self, pause=False, imgId='default'):        cv2.imshow(f'{imgId}', self.imgs[imgId])        if pause:            cv2.waitKey()    def show_all(self, pause=False):        for i, v in self.imgs.items():            cv2.imshow(f'{i}', v)        if pause:            cv2.waitKey()      def save_img(self, imgId='default', path='../visualize/'):        cv2.imwrite(path + '{}.png'.format(imgId), self.imgs[imgId])        def save_all(self, path='../visualize/'):        for i, v in self.imgs.items():            cv2.imwrite(path + '/{}.png'.format(i), v)