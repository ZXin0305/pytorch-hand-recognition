from curses.ascii import GS
import sys
sys.path.append('/home/xuchengjun/ZXin/pytorch-hand-recognition')
import cv2
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import random
from path import Path
from lib.tools import *
from IPython import embed
import os.path as osp
from dataset.dataAugmentation import AugGray, AugHSV, AugMaskCenter, AugRotate, GridMask, ReplaceCsv
# from dataAugmentation import AugGray, AugHSV, AugMaskCenter, AugRotate, GridMask, ReplaceCsv
import random
from config.config import cfg

class Dataset(Dataset):
    def __init__(self, cfg, stage, transform=None, with_augmentation=False) -> None:
        super().__init__()
        self.stage = stage 
        assert self.stage in ('train', 'test')

        self.transform = transform
        self.with_augmentation = with_augmentation
        self.cfg = cfg
        self.grid_mask = GridMask(15, 55)

        train_json  = None
        test_json = None

        # --------- get dataset list --------
        if self.stage == "train":
            print(f'loading {self.stage} dataset ..')
            print(f'dataset path -> {self.cfg.DATASET.PATH} ..')
            train_json = read_json(self.cfg.DATASET.PATH)
            data = []
            data += train_json['data']
            self.data_list = data
            # random.shuffle(self.data_list)
        elif self.stage == "test":
            print(f'loading {self.stage} dataset ..')
            print(f'dataset path -> {self.cfg.TEST.PATH} ..')
            test_json = read_json(self.cfg.TEST.PATH)
            data = []
            data += test_json['data']
            self.data_list = data

        print(f"load dataset sucessfully, dataset len is: {len(self.data_list)} ..")
        self.input_shape = self.cfg.INPUT.SHAPE

    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        
        data_info = self.data_list[index]
        rgb_file = data_info[0]
        depth_file = data_info[1]
        skel_file = data_info[2]
        gesture_label = int(data_info[3])
        
        assert rgb_file.endswith(self.cfg.DATASET.SUFFIX[0])
        assert depth_file.endswith(self.cfg.DATASET.SUFFIX[1])
        assert skel_file.endswith(self.cfg.DATASET.SUFFIX[2])
        
        rgb_ori = cv2.imread(osp.join(self.cfg.DATASET.ROOT_PATH, rgb_file), cv2.IMREAD_COLOR)
        depth_ori = cv2.imread(osp.join(self.cfg.DATASET.ROOT_PATH, depth_file))
        skel_ori = read_csv(osp.join(self.cfg.DATASET.ROOT_PATH, skel_file))

        # # process
        skel_numpy = np.array(skel_ori, dtype=np.float)[0:21, 1:]
        """# wrist_depth = np.array(skel_ori, dtype=np.float)[21][1]  # no useful"""
                                                                                                                        
        skel_img = generateHandFeature(skel_numpy)  # --> * 255.0                                                                                               
        
        """
        旋转和遮挡:depth \ RGB都要
        HSV通道分离: RGB
        """
        img = rgb_ori.copy()
        depth = depth_ori.copy()
        skel = skel_img.copy()
        
        if self.with_augmentation:
            probs = [random.random() for i in range(5)]
            # ----------------------------------------------------------------
            # 针对RGB
            flag_1 = False
            flag_2 = False
            flag_3 = False
            flag_4 = False
            if probs[0] > self.cfg.AUGMENTATION_PROB1:
                img = AugHSV(img)
                flag_1 = True
               
            if not flag_1 and probs[1] > self.cfg.AUGMENTATION_PROB2:
                img = AugGray(img)
                
            if probs[2] > self.cfg.AUGMENTATION_PROB3:
                img, depth, skel = AugRotate(img, depth, skel, self.cfg.MAX_ROTATE_DEGREE, self.cfg.INPUT.SHAPE[0])
                # depth = AugRotate(None, depth, None, self.cfg.MAX_ROTATE_DEGREE, self.cfg.INPUT.SHAPE[0])
                flag_2 = True
                
            if not flag_2 and probs[3] > self.cfg.AUGMENTATION_PROB4:
                img = self.grid_mask(img, None, None)
                flag_3 = True
                
            if not flag_3 and not flag_2 and probs[4] > self.cfg.AUGMENTATION_PROB5:
                img = AugMaskCenter(img)
                
                
        #     # -----------------------------------------------------------------
        #     # 针对depth（使用高斯）

            is_aug_data = rgb_file.split("/")[3] == self.cfg.DATASET.AUG_NAME

            if is_aug_data:
                depth = AugGaussianNoise(depth)
                skel = AugGaussianNoise(skel)
        #         # cv2.imshow('skel', skel)
        #         # cv2.imshow('depth', depth)
        #         # cv2.waitKey(0)
        #         # embed()


        if self.transform:
            img = self.transform(img)
        else:
            img = img.transpose((2, 0, 1)).astype(np.float32)
            img = torch.from_numpy(img).float()
        
        # 转torch  使用trans_to_tensor的时候，就别用这个
        # depth = torch.from_numpy(depth).float()

        # 进行归一化
        # depth = depth_mappingv3(depth, wrist_depth)   # 不用这个了..

        depth = trans_to_tensor(depth)[0, :, :]
        depth = depth.unsqueeze(0)
        skel = torch.from_numpy(skel).float()
        skel = skel.unsqueeze(0)
        input1 = torch.cat([img, depth, skel], dim=0)

        # input1 = depth
        
        gesture_label = torch.tensor(int(gesture_label)).float()

        # multimodel 
        return input1, gesture_label

        # for visulization
        # return img, depth, skel

# for MLP training ..
# class DatasetV2(Dataset):
#     def __init__(self, cfg, stage):
#         super(DatasetV2, self).__init__()
#         self.stage = stage 
#         assert self.stage in ('train', 'test')

#         self.cfg = cfg
#         train_json  = None
#         test_json = None

#         # --------- get dataset list --------
#         if self.stage == "train":
#             print(f'loading {self.stage} dataset ..')
#             print(f'dataset path -> {self.cfg.DATASET.PATH} ..')
#             train_json = read_json(self.cfg.DATASET.PATH)
#             data = []
#             data += train_json['data']
#             self.data_list = data
#             # random.shuffle(self.data_list)
#         elif self.stage == "test":
#             print(f'loading {self.stage} dataset ..')
#             print(f'dataset path -> {self.cfg.TEST.PATH} ..')
#             test_json = read_json(self.cfg.TEST.PATH)
#             data = []
#             data += test_json['data']
#             self.data_list = data

#         print(f"load dataset sucessfully, dataset len is: {len(self.data_list)} ..")
    
#     def __len__(self):
#         return len(self.data_list)

#     def __getitem__(self, index):
        
#         data_info = self.data_list[index]
#         rgb_file = data_info[0]
#         depth_file = data_info[1]
#         skel_file = data_info[2]
#         gesture_label = int(data_info[3])
        
#         assert rgb_file.endswith(self.cfg.DATASET.SUFFIX[0])
#         assert depth_file.endswith(self.cfg.DATASET.SUFFIX[1])
#         assert skel_file.endswith(self.cfg.DATASET.SUFFIX[2])
        
#         skel_ori = read_csv(osp.join(self.cfg.DATASET.ROOT_PATH, skel_file))

#         skel_numpy = np.array(skel_ori, dtype=np.float)[0:21, 1:] / 160.0 
#         skel_numpy = skel_numpy.flatten()                                                                                           
#         skel = skel_numpy.copy()

#         skel = torch.from_numpy(skel).float()

#         input1 = skel.unsqueeze(0)
#         gesture_label = torch.tensor(int(gesture_label)).float()
#         return input1, gesture_label

# for recording each class number 
# class Dataset(Dataset):
#     def __init__(self, cfg, stage, transform=None, with_augmentation=True) -> None:
#         super().__init__()
#         self.stage = stage 
#         assert self.stage in ('train', 'test')

#         self.transform = transform
#         self.with_augmentation = with_augmentation
#         self.cfg = cfg
#         self.grid_mask = GridMask(15, 55)

#         train_json  = None
#         test_json = None

#         # --------- get dataset list --------
#         if self.stage == "train":
#             print(f'loading {self.stage} dataset ..')
#             print(f'dataset path -> {self.cfg.DATASET.PATH} ..')
#             train_json = read_json(self.cfg.DATASET.PATH)
#             data = []
#             data += train_json['data']
#             self.data_list = data
#             # random.shuffle(self.data_list)
#         elif self.stage == "test":
#             print(f'loading {self.stage} dataset ..')
#             print(f'dataset path -> {self.cfg.TEST.PATH} ..')
#             test_json = read_json(self.cfg.TEST.PATH)
#             data = []
#             data += test_json['data']
#             self.data_list = data

#         print(f"load dataset sucessfully, dataset len is: {len(self.data_list)} ..")
#         self.input_shape = self.cfg.INPUT.SHgesture_labelAPE

    
#     def __len__(self):
#         return len(self.data_list)

#     def __getitem__(self, index):
        
#         data_info = self.data_list[index]
#         gesture_label = int(data_info[3])                                                                                          
#         return gesture_label

if __name__ == "__main__":
    normalize = transforms.Normalize(mean=cfg.INPUT.MEANS, std=cfg.INPUT.STDS)
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    dataset = Dataset(cfg, 'train', None, True)
    idx_list = [i for i in range(len(dataset))]
    for i in range(len(dataset)):
        data = dataset[i]
        print(data[1].shape)
        # print('working ..')

        cv2.imshow('img', data[0])
        cv2.imshow('depth', data[1])
        cv2.imshow('skel', data[2])
        cv2.imwrite('./res/center/img_' + str(i) + '.jpg', data[0])
        cv2.imwrite('./res/center/depth_' + str(i) + '.jpg', data[1])
        cv2.imwrite('./res/center/skel_' + str(i) + '.jpg', data[2])
        cv2.waitKey(0)