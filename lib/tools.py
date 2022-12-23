import os
import sys
from path import Path
from IPython import embed
import json
import pandas as pd
import numpy as np
import cv2
import torchvision.transforms as transforms

hand_skel_edge = [[0, 1], [1, 2], [2, 3], [3, 4],
                    [0, 5], [5, 6], [6, 7], [7, 8],
                    [0, 9], [9, 10], [10, 11], [11, 12],
                    [0, 13], [13, 14], [14, 15], [15, 16],
                    [0, 17], [17, 18], [18, 19], [19, 20]]

def read_json(path):
    with open(path, 'r') as file:
        data = json.load(file)
    return data

def write_json(path, result):
    with open(path, 'w') as file:
        json.dump(result, file)

def read_csv(path):
    try:
        data = pd.read_csv(path, header=0)
    except:
        print('dataset not exist')
        return 
    return data

def ensure_dir(path):
    """
    create directories if *path* does not exist
    """
    if not os.path.isdir(path):
        os.makedirs(path)

def emg_mapping(emg_list, fix = 8, len_ = 256):
    emg_list = np.tile(emg_list, (int(np.ceil(fix / emg_list.shape[0])), 1))
    shape = emg_list.shape
    if emg_list.shape[0] > fix:
        diff = emg_list.shape[0] - fix
        emg_list = emg_list[int(0 + diff / 2 ): int(shape[0] - diff / 2), :]
    emg_list = emg_list.transpose()
    emg_map = (np.array(emg_list) / len_ + 0.5) * 255
    emg_map[emg_map < 0] = 0
    return np.array(emg_map, dtype=np.uint8)

def emg_mappingv2(emg_list, fix = 8, len_ = 256):
    emg_list = np.tile(emg_list, (int(np.ceil(fix / emg_list.shape[0])), 1))
    shape = emg_list.shape
    if emg_list.shape[0] > fix:
        diff = emg_list.shape[0] - fix
        emg_list = emg_list[int(0 + diff / 2 ): int(shape[0] - diff / 2), :]
    emg_list = emg_list.transpose()
    emg_list = (np.array(emg_list) / len_ + 0.5)
    emg_list = emg_list.flatten(order="C")
    return emg_list

def depth_mappingv3(depth, base_depth, up_ratio = 0.20, down_ratio = 0.30):
    """
    使用的是手腕的关节深度
    """
    max_val = 1 + up_ratio
    min_val = 1 - down_ratio
    depth = depth / 255.0 * 4096.0 / 10   # 从mm 变成 cm

    depth[depth > base_depth * max_val] = 0
    depth[depth < base_depth * min_val] = 0
    depth[depth != 0] = 1
    depth = np.array(depth, dtype=np.uint8)
    depth = cv2.medianBlur(depth, 3)
    return depth * 255


def depth_mapping(depth, ratio_ = 0.15, size_ = 5, W_ = 5, H_ = 12):
    """
    depth: 深度图
    ratio_: 阈值范围
    size_: 中心范围尺寸
    """
    shape = depth.shape
    max_val = 1 + ratio_
    min_val = 1 - ratio_

    img_center = [int(shape[0] / 2 + H_ + 0.5), int(shape[0] / 2 + W_ + 0.5)]
    depth = depth / 255.0 * 4096.0 / 10   # 从mm 变成 cm

    total_depth = np.sum(depth[img_center[0] - size_ : img_center[0] + size_,
                               img_center[1] - size_ : img_center[1] + size_,
                               0], dtype=np.float)
    mean_depth_val = total_depth / (pow(size_ * 2, 2.0))
    depth[depth > mean_depth_val * max_val] = 0
    depth[depth < mean_depth_val * min_val] = 0
    depth[depth != 0] = 1
    depth = np.array(depth, dtype=np.uint8)
    depth = cv2.medianBlur(depth, 5)
    return depth * 255              # 应该不用再乘以255了

def trans_to_tensor(img):
    transform = transforms.Compose([transforms.ToTensor()])
    img_trans = transform(img)
    return img_trans

def handSkelVis(centerA, centerB, accumulate_vec_map, thre, hand_img_size):
    centerA = centerA.astype(float)
    centerB = centerB.astype(float)
    stride = 1
    crop_size_y = hand_img_size
    crop_size_x = hand_img_size
    grid_y = crop_size_y / stride
    grid_x = crop_size_x / stride
    centerB = centerB / stride
    centerA = centerA / stride

    limb_vec = centerB - centerA  # x,y
    limb_z = 1.0
    norm = np.linalg.norm(limb_vec)
    if norm == 0.0:
        norm = 1.0
    limb_vec_unit = limb_vec / norm

    # To make sure not beyond the border of this two points
    min_x = max(int(round(min(centerA[0], centerB[0]) - thre)), 0)   #round:对数字进行舍入计算
    max_x = min(int(round(max(centerA[0], centerB[0]) + thre)), grid_x)
    min_y = max(int(round(min(centerA[1], centerB[1]) - thre)), 0)
    max_y = min(int(round(max(centerA[1], centerB[1]) + thre)), grid_y)

    range_x = list(range(int(min_x), int(max_x), 1)) 
    range_y = list(range(int(min_y), int(max_y), 1))
    xx, yy = np.meshgrid(range_x, range_y)   # to be a grid
    xx = xx.astype(int)
    yy = yy.astype(int)
    ba_x = xx - centerA[0]  # the vector from (x,y) to centerA
    ba_y = yy - centerA[1]
    limb_width = np.abs(ba_x * limb_vec_unit[1] - ba_y * limb_vec_unit[0])
    mask = limb_width < thre  # mask is 2D
    vec_map = np.copy(accumulate_vec_map) * 0.0
    vec_map[0, yy, xx] = np.repeat(mask[np.newaxis, :, :], 1, axis=0)
    vec_map[0, yy, xx] *= limb_z
    mask = np.logical_or.reduce((np.abs(vec_map[0, :, :]) != 0))
    
    accumulate_vec_map += vec_map
    
    return accumulate_vec_map

def generateHandFeature(hand_numpy, hand_img_size=160, kernel = [3, 3]):
    hand_feature = np.zeros((1, hand_img_size, hand_img_size))
    for i in range(hand_numpy.shape[0] - 1):
        centerA = np.array(hand_numpy[hand_skel_edge[i][0]], dtype=int)
        centerB = np.array(hand_numpy[hand_skel_edge[i][1]], dtype=int)
        hand_feature += handSkelVis(centerA, centerB, hand_feature, 1, hand_img_size)
    
    hand_feature[hand_feature > 1] = 1
    hand_feature[0] = cv2.GaussianBlur(hand_feature[0], kernel, 0)
    hand_feature *= 255

    return hand_feature[0]

def AugGaussianNoise(img, loc=0.0, sigma=0.1, deli = 255.0):
    img_copy = img.copy()
    img_copy = np.array(img_copy / deli, dtype=np.float)
    noise = np.random.normal(loc, sigma, img_copy.shape)    # 正态分布函数
    gaussian_noise = img_copy + noise
    gaussian_noise = np.clip(gaussian_noise, 0, 1)
    gaussian_noise_img = np.uint8(gaussian_noise * 255)
    return gaussian_noise_img


if __name__ == "__main__":

    # xx = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]])
    # shape = xx.shape
    # img_center = [int(shape[0] / 2 + 0.5), int(shape[0] / 2 + 0.5)]
    # size_ = 1
    # total_depth = np.sum(xx[img_center[0] - size_ : img_center[0] + size_,
    #                         img_center[1] - size_ : img_center[1] + size_], dtype=np.float)
    # print(total_depth)

    import random
    xx = random.random()
    print(xx)

    