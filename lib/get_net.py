import email
import sys
sys.path.append("/home/xuchengjun/ZXin/pytorch-hand-recognition")
from config.config import cfg
import os
from re import L
import datetime
import numpy as np
import torch
from IPython import embed


def get_network(cfg):
    """
    建立网络模型
    """

    if cfg.MODEL.NAME == "vgg16":
        from models.vgg import vgg16_bn
        net = vgg16_bn()
    elif cfg.MODEL.NAME == "vgg13":
        from models.vgg import vgg13_bn
        net = vgg13_bn()
    elif cfg.MODEL.NAME == "vgg11":
        from models.vgg import vgg11_bn
        net = vgg11_bn()
    elif cfg.MODEL.NAME == "densenet121":
        from models.densenet import densenet121
        net = densenet121()
    elif cfg.MODEL.NAME == "densenet161":
        from models.densenet import densenet161
        net = densenet161()
    elif cfg.MODEL.NAME == "densenet169":
        from models.densenet import densenet169
        net = densenet169()
    elif cfg.MODEL.NAME == "densenet201":
        from models.densenet import densenet201
        net = densenet201()
    elif cfg.MODEL.NAME == "googlenet":
        from models.googlenet import googlenet
        net = googlenet()
    elif cfg.MODEL.NAME == "inceptionv3":
        from models.inceptionv3 import inceptionv3
        net = inceptionv3()
    elif cfg.MODEL.NAME == "inceptionv4":
        from models.inceptionv4 import inceptionv4
        net = inceptionv4()
    elif cfg.MODEL.NAME == "resnet18":
        from models.resnet import resnet18
        net = resnet18()
    elif cfg.MODEL.NAME == "resnet34":
        from models.resnet import resnet34
        net = resnet34()
    elif cfg.MODEL.NAME == "resnet50":
        from models.resnet import resnet50
        net = resnet50(cfg.INPUT.CHANEL_NUM, cfg.OUTPUT.CLASS_NUM)
    elif cfg.MODEL.NAME == "resnet101":
        from models.resnet import resnet101
        net = resnet101(cfg.INPUT.CHANEL_NUM, cfg.OUTPUT.CLASS_NUM)
    elif cfg.MODEL.NAME == "resnet152":
        from models.resnet import resnet152
        net = resnet152()
    elif cfg.MODEL.NAME == "resnet":
        from models.resnet import resnet18
        net = resnet18()
    elif cfg.MODEL.NAME == 'resnext50':
        from models.resnext import resnext50
        net = resnext50()
    elif cfg.MODEL.NAME == 'resnext101':
        from models.resnext import resnext101
        net = resnext101()
    elif cfg.MODEL.NAME == 'resnext152':
        from models.resnext import resnext152
        net = resnext152()
    elif cfg.MODEL.NAME == 'shufflenet':
        from models.shufflenet import shufflenet
        net = shufflenet()
    elif cfg.MODEL.NAME == 'shufflenetv2':
        from models.shufflenetv2 import shufflenetv2
        net = shufflenetv2()
    elif cfg.MODEL.NAME == 'mobilenet':
        from models.mobilenet import mobilenet
        net = mobilenet()
    elif cfg.MODEL.NAME == 'mobilenetv2':
        from models.mobilenetv2 import mobilenetv2
        net = mobilenetv2(cfg.INPUT.CHANEL_NUM, cfg.OUTPUT.CLASS_NUM)
    elif cfg.MODEL.NAME == 'mynet':
        from models.mynet1 import get_net
        net = get_net()
    elif cfg.MODEL.NAME == 'VSGCNN':
        from models.vsgcnn import VSGCNN_model
        net = VSGCNN_model(cfg.OUTPUT.CLASS_NUM, cfg.INPUT.CHANEL_NUM, cfg.OUTPUT.CLASS_NUM)
    elif cfg.MODEL.NAME == 'LSTM':
        from models.LSTM import LSTM_model
        net = LSTM_model(42,42,6,2, 0,11,1,42)
    else:
        print("网络名称输入错误 ..")
        net = None
    
    if cfg.MODEL.USE_GPU and net is not None:
        net = net.cuda()

    return net

if __name__ == "__main__":
    net = get_network(cfg)
    embed()
    pass