import torch
import numpy as np
import cv2


def GetTransform(center, scale, rot, res):
    h = 200 * scale
    t = np.eye(3)

    # t[0, 0] = res / h
    t[0, 0] = res / h
    t[1, 1] = res / h

    t[0, 2] = res * (- center[0] / h + 0.5)
    t[1, 2] = res * (- center[1] / h + 0.5)

    if rot != 0:
        rot = -rot
        r = np.eye(3)
        ang = rot * np.math.pi / 180
        s = np.math.sin(ang)
        c = np.math.cos(ang)
        r[0, 0] = c
        r[0, 1] = - s
        r[1, 0] = s
        r[1, 1] = c
        t_ = np.eye(3)
        t_[0, 2] = - res / 2
        t_[1, 2] = - res / 2
        t_inv = np.eye(3)
        t_inv[0, 2] = res / 2
        t_inv[1, 2] = res / 2
        t = np.dot(np.dot(np.dot(t_inv, r), t_), t)

    return t


def Transform(pt, center, scale, rot, res, invert=False, round=True):
    pt_ = np.ones(3)
    pt_[0], pt_[1] = pt[0], pt[1]

    t = GetTransform(center, scale, rot, res)
    # print('t is:',t)
    if invert:
        t = np.linalg.inv(t)
    new_point = np.dot(t, pt_)[:2]
    new_point = new_point.astype(np.int32)
    # print('new point before round:',new_point)
    if round == True:
        new_point = np.round(new_point)
    return new_point


# def getTransform3D(center, scale, rot, res):
#     h = 1.0 * scale
#     t = np.eye(4)
#
#     t[0][0] = res / h
#     t[1][1] = res / h
#     t[2][2] = res / h
#
#     t[0][3] = res * (- center[0] / h + 0.5)
#     t[1][3] = res * (- center[1] / h + 0.5)
#
#     if rot != 0:
#         raise Exception('Not Implement')
#
#     return t


# def Transform3D(pt, center, scale, rot, res, invert=False):
#     pt_ = np.ones(4)
#     pt_[0], pt_[1], pt_[2] = pt[0], pt[1], pt[2]
#     # print 'c s r res', center, scale, rot, res
#     t = getTransform3D(center, scale, rot, res)
#     if invert:
#         t = np.linalg.inv(t)
#     # print 't', t
#     # print 'pt_', pt_
#     new_point = np.dot(t, pt_)[:3]
#     # print 'new_point', new_point
#     # if not invert:
#     #  new_point = new_point.astype(np.int32)
#     return new_point


def Crop(img, center, scale, rot, res):
    ul = Transform((1, 1), center, scale, 0, res, True)
    br = Transform((res + 1, res + 1), center, scale, 0, res, True)

    pad = np.linalg.norm(ul - br) / 2 - (br[0] - ul[0]) / 2
    if rot != 0:
        ul = ul - pad
        br = br + pad
    # print(img.shape[2])
    # print('ul and br:',ul,br)
    if img.shape[2] > 2:
        newDim = np.zeros((img.shape[2], br[1] - ul[1], br[0] - ul[0]))
        # print('newDim[0]...:',newDim.shape[0],newDim.shape[1],newDim.shape[2])
        newImg = np.zeros((newDim.shape[1], newDim.shape[2], newDim.shape[0]))
        # newImg = np.array((img.shape[2], br[1] - ul[1], br[0] - ul[0]))
        # print(np.shape(newImg))
        # print('br-ul:',br[1]-ul[1],br[0]-ul[0])
        # newImg = np.array(newImg)
        # print('in if newImg:',np.shape(newImg))
        ht = img.shape[0]
        wd = img.shape[1]
    else:
        newImg = [br[1] - ul[1], br[0] - ul[0]]
        newImg = np.array(newImg)
        ht = img.shape[0]
        wd = img.shape[1]
    # print('ht and wd',ht,wd)
    newX = np.array((max(1, -ul[0] + 2), min(br[0], wd + 1) - ul[0]))
    newY = np.array((max(1, -ul[1] + 2), min(br[1], ht + 1) - ul[1]))
    oldX = np.array((max(1, ul[0]), min(br[0], wd + 1) - 1))
    oldY = np.array((max(1, ul[1]), min(br[1], ht + 1) - 1))
    # print('newDim shape:',np.shape(newDim))
    if newDim.shape[0] > 2:
        # print('Img type:', type(img),np.shape(img))
        # print('newImg type:',type(newImg),np.shape(newImg))
        # print('newX...',newX,newY)
        # print('oldX...',oldX,oldY)
        # print('new shape:',np.shape(newImg[0:newDim.shape[0], 12:236, 38:262]))
        # print('img shape:',np.shape(img[0:newDim.shape[0], 1:225, 1:225]))
        newImg[newY[0]:newY[1], newX[0]:newX[1], 0:newDim.shape[0]] = img[oldY[0]:oldY[1], oldX[0]:oldX[1],
                                                                      0:newDim.shape[0]]
    else:
        newImg[newY[0]:newY[1], newX[0]:newX[1]] = img[oldY[0]:oldY[1], oldX[0]:oldX[1]].copy

    newImg = cv2.resize(newImg, (res, res))
    return newImg


# def Gaussian(sigma):
#     n = sigma * 6 + 1
#     g_inp = np.zeros((n, n))
#     for i in range(n):
#         for j in range(n):
#             g_inp[i, j] = np.exp(-((i - n / 2) ** 2 + (j - n / 2) ** 2) / (2. * sigma * sigma))
#     if sigma == 7:
#         return np.array([0.0529, 0.1197, 0.1954, 0.2301, 0.1954, 0.1197, 0.0529,
#                          0.1197, 0.2709, 0.4421, 0.5205, 0.4421, 0.2709, 0.1197,
#                          0.1954, 0.4421, 0.7214, 0.8494, 0.7214, 0.4421, 0.1954,
#                          0.2301, 0.5205, 0.8494, 1.0000, 0.8494, 0.5205, 0.2301,
#                          0.1954, 0.4421, 0.7214, 0.8494, 0.7214, 0.4421, 0.1954,
#                          0.1197, 0.2709, 0.4421, 0.5205, 0.4421, 0.2709, 0.1197,
#                          0.0529, 0.1197, 0.1954, 0.2301, 0.1954, 0.1197, 0.0529]).reshape(7, 7)
#     elif sigma == n:
#         return g_inp
#     else:
#         raise Exception('Gaussian {} Not Implement'.format(sigma))
#
#
# def DrawGaussian(img, pt, sigma):
#     tmpSize = int(np.math.ceil(3 * sigma))
#     ul = [int(np.math.floor(pt[0] - tmpSize)), int(np.math.floor(pt[1] - tmpSize))]
#     br = [int(np.math.floor(pt[0] + tmpSize)), int(np.math.floor(pt[1] + tmpSize))]
#
#     if ul[0] > img.shape[1] or ul[1] > img.shape[0] or br[0] < 1 or br[1] < 1:
#         return img
#
#     size = 2 * tmpSize + 1
#     g = Gaussian(size)
#     # g = np.random.normal((7,7))
#     # print('gaussian g:',g)
#
#     g_x = [max(1, -ul[0]), min(br[0], img.shape[1]) - max(1, ul[0]) + max(1, -ul[0])]
#     g_y = [max(1, -ul[1]), min(br[1], img.shape[0]) - max(1, ul[1]) + max(1, -ul[1])]
#
#     img_x = [max(1, ul[0]), min(br[0], img.shape[1])]
#     img_y = [max(1, ul[1]), min(br[1], img.shape[0])]
#     # print('g_x g_y',g_x,g_y)
#     # print('img_x img_y',img_x,img_y)
#     img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
#     return img

# def gaussian2D(shape, sigma=1):
#     m, n = [(ss - 1.) / 2. for ss in shape]
#     y, x = np.ogrid[-m:m + 1, -n:n + 1]
#
#     h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
#     h[h < np.finfo(h.dtype).eps * h.max()] = 0
#     return h


# def DrawGaussian(heatmap, center, sigma):
#     tmp_size = sigma * 3
#     mu_x = int(center[0] + 0.5)
#     mu_y = int(center[1] + 0.5)
#     w, h = heatmap.shape[0], heatmap.shape[1]
#     ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
#     br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
#     if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
#         return heatmap
#     size = 2 * tmp_size + 1
#     x = np.arange(0, size, 1, np.float32)
#     y = x[:, np.newaxis]
#     x0 = y0 = size // 2
#     g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
#     g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
#     g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
#     img_x = max(0, ul[0]), min(br[0], h)
#     img_y = max(0, ul[1]), min(br[1], w)
#     try:
#         heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
#             heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
#             g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
#     except:
#         print('center', center)
#         print('gx, gy', g_x, g_y)
#         print('img_x, img_y', img_x, img_y)
#     return heatmap

def Gaussian(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def DrawGaussian(img, pt, sigma):
    tmpSize = int(np.math.ceil(3 * sigma))
    ul = [int(np.math.floor(pt[0] - tmpSize)), int(np.math.floor(pt[1] - tmpSize))]
    br = [int(np.math.floor(pt[0] + tmpSize)), int(np.math.floor(pt[1] + tmpSize))]

    if ul[0] > img.shape[1] or ul[1] > img.shape[0] or br[0] < 1 or br[1] < 1:
        return img

    size = 2 * tmpSize + 1
    g = Gaussian((size, size), sigma)
    # g = np.random.normal((7,7))
    # print('gaussian g:',g)

    g_x = [max(1, -ul[0]), min(br[0], img.shape[1]) - max(1, ul[0]) + max(1, -ul[0])]
    g_y = [max(1, -ul[1]), min(br[1], img.shape[0]) - max(1, ul[1]) + max(1, -ul[1])]

    img_x = [max(1, ul[0]), min(br[0], img.shape[1])]
    img_y = [max(1, ul[1]), min(br[1], img.shape[0])]
    # print('g_x g_y',g_x,g_y)
    # print('img_x img_y',img_x,img_y)
    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img
