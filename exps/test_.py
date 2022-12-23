import sys
sys.path.append("/home/xuchengjun/ZXin/pytorch-hand-recognition")

import time
import torch.nn as nn
import torch

import numpy as np
from lib.tools import *
from lib.get_net import get_network
from lib.dataloader import get_train_loader, get_test_loader
from path import Path
from IPython import embed
import random
import os
from config.config import cfg


def main():
    hg_class_num = 16
    hg_class_acc = [0] * hg_class_num
    hg_each_num = [0] * hg_class_num
    time_list = []
    model = get_network(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    test_loader = get_test_loader(cfg)

    # 这个记录test中的每个类别的数量
    # for label in test_loader:
    #     hg_each_num[(int)(label)] += 1
    # print(hg_each_num)
        
    # cal average process time
    if Path(cfg.MODEL.TEST_PATH).exists():
        print(f'load weight --> {cfg.MODEL.TEST_PATH} ..')
        state_dict = torch.load(cfg.MODEL.TEST_PATH)
        model.load_state_dict(state_dict)

    model.eval()
    len_data = len(test_loader)
    count = 1
    with torch.no_grad():
        for img, label in test_loader:
            hg_each_num[(int)(label)] += 1
            img = img.to(cfg.MODEL.DEVICE)
            st = time.time()
            output = model(img)
            time_list.append(time.time() - st)
            pre = output.argmax(1).detach().cpu()

            if pre == label:
                hg_class_acc[(int)(label)] += 1
        print('done ..')
        print('acc ->')
        hg_class_acc = np.array(hg_class_acc, np.float) / np.array(hg_each_num, np.float)
        print(hg_class_acc)
        print()
        print('avg process time -> ')
        avg_time = np.sum(np.array(time_list, np.float)) / len_data
        print(avg_time)
    


if __name__ == "__main__":
    main()

