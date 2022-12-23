"""
online image augmentation
"""

import cv2
import random
import numpy as np
import math
from PIL import Image
from torch import nn
import copy
from lib.tools import *
from IPython import embed

def crop(img, ori_shape = 160):
    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # print(cX, cY)
    # print(int(cY - ori_shape / 2), int(cY + ori_shape / 2))
    # print(int(cX - ori_shape / 2), int(cX + ori_shape / 2))
    return img[int(cY - ori_shape / 2) : int(cY + ori_shape / 2), \
               int(cX - ori_shape / 2) : int(cX + ori_shape / 2)]

def AugRotate(rgb=None, depth=None, skel=None, max_rotate_degree = 15, ori_shape=160):
    dice = random.random()
    degree = (dice - 0.5) * 2 * max_rotate_degree
    
    RGB_ = rgb[int(ori_shape / 2), int(ori_shape / 2), :]
    img_rot = crop(rotate_bound(rgb, np.copy(degree), (int(RGB_[0]), int(RGB_[1]), int(RGB_[2]))), ori_shape)
    depth_rot = crop(rotate_bound(depth, np.copy(degree), (0, 0, 0)), ori_shape)
    skel_rot = crop(rotate_bound(skel, np.copy(degree), (0, 0, 0)), ori_shape)
    return img_rot, depth_rot, skel_rot
    # return depth_rot

def rotate_bound(image, angle, bordervalue):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    return cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=bordervalue)


def AugHSV(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    return img_hsv

def AugGray(img):
    img_one_channel = cv2.split(img)[0]
    img_gray = cv2.merge((img_one_channel, img_one_channel, img_one_channel))
    return img_gray

# mask 遮挡
class Grid(object):
    def __init__(self, d1, d2, rotate = 1, ratio = 0.5, mode = 1):
        self.d1 = d1
        self.d2 = d2
        self.rotate = rotate
        self.ratio = ratio
        self.mode=mode

    def __call__(self, img, flag=0):
        h, w = img.shape[:2]

        # 1.5 * h, 1.5 * w works fine with the squared images
        # But with rectangular input, the mask might not be able to recover back to the input image shape
        # A square mask with edge length equal to the diagnoal of the input image
        # will be able to cover all the image spot after the rotation. This is also the minimum square.
        hh = math.ceil((math.sqrt(h*h + w*w)))

        d = np.random.randint(self.d1, self.d2)

        # maybe use ceil? but i guess no big difference
        self.l = math.ceil(d*self.ratio)

        mask = np.ones((hh, hh), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)

        for i in range(-1, hh//d+1):
            s = d*i + st_h
            t = s+self.l
            s = max(min(s, hh), 0)
            t = max(min(t, hh), 0)
            mask[s:t,:] *= 0
        for i in range(-1, hh//d+1):
            s = d*i + st_w
            t = s+self.l
            s = max(min(s, hh), 0)
            t = max(min(t, hh), 0)
            mask[:,s:t] *= 0
        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[(hh-h)//2:(hh-h)//2+h, (hh-w)//2:(hh-w)//2+w]

        if self.mode == 1:
            mask = 1 - mask
        if flag == 0:
            mask = np.stack((mask, mask, mask)).transpose(1,2,0)
        img = img * mask
        
        return img

class GridMask(nn.Module):
    def __init__(self, d1, d2, rotate = 1, ratio = 0.5, mode = 1):
        super(GridMask, self).__init__()
        self.rotate = rotate
        self.ratio = ratio
        self.mode = mode
        self.grid = Grid(d1, d2, rotate, ratio, mode)

    def forward(self, x, y, z):
        img_mask = self.grid(x)
        # depth_mask = self.grid(y)
        # skel_mask = self.grid(z, 1)
        # return img_mask, depth_mask, skel_mask

        return img_mask


# 以图像中心进行遮挡
def AugMaskCenter(img, rec_size = 60):
    (h, w, c) = img.shape
    # avg_value = [np.sum(img[:, :, 0]) // (h * w * c),\
    #              np.sum(img[:, :, 1]) // (h * w * c),\
    #              np.sum(img[:, :, 2]) // (h * w * c)]
    (cX, cY) = (w // 2, h // 2)

    # 随机生成几个点
    n = 3
    corners = []
    all = []
    choices = [[-random.randrange(1,2), -random.randrange(1,2)], [random.randrange(1,2), -random.randrange(1,2)], [random.randrange(1,2), random.randrange(1,2)]]
    for i in range(n):

        cornerX = cX + random.randint(40, rec_size) * (choices[i][0])
        cornerY = cY + random.randint(40, rec_size) * (choices[i][1])
        all.append((int(cornerX), int(cornerY)))
    #
    corners.append(all)
    corners = np.array(corners)
    mask = np.ones(img.shape)

    cv2.fillPoly(mask, corners, (0, 0, 0))
    img[mask == 0] = 128 
    return img

def ReplaceCsv(gesture_label):
    csv_file = os.path.join("../xx/xx/",gesture_label, ".csv")   # 通用的
    emg_ori = read_csv(csv_file)
    return emg_ori

if __name__ == "__main__":

    depth = cv2.imread('./3.jpg')
    img = cv2.imread('./1.jpg')
    # img = AugHSV(img)
    # img, depth = AugRotate(img, depth)

    # grid_mask = GridMask(20, 50)
    # img_mask, depth_mask = grid_mask(img, depth)

    img_mc, depth_c = AugMaskCenter(img, depth)


    cv2.imshow('depth', depth_c)
    cv2.imshow('rgb', img_mc)
    cv2.imshow('ori', img)
    cv2.waitKey(0)
    