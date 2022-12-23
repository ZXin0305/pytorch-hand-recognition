"""_summary_
Myself net for hand gesture classify 
"""
import sys
sys.path.append("/home/xuchengjun/ZXin/pytorch-hand-recognition")
from turtle import forward
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F

from models.resnet import *
from models.inceptionv3 import InceptionV3
from models.mobilenetv2 import MobileNetV2
from models.vsgcnn import VSGCNN

from IPython import embed
import math

class conv_bn_relu(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding,
                 has_bn=True, has_relu=True, efficient=False):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.has_bn = has_bn
        self.has_relu = has_relu
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.efficient = efficient
        
    def forward(self, x):
        def _func_factory(conv, bn, relu, has_bn, has_relu):
            def func(x):
                x = conv(x)
                if has_bn:
                    x = bn(x)
                if has_relu:
                    x = relu(x)
                return x
            return func
        
        func = _func_factory(self.conv, self.bn, self.relu, self.has_bn, self.has_relu)
        
        if self.efficient:
            x = checkpoint(func, x)
        else:
            x = func(x)
        return x
    
class PryBottleNet(nn.Module):
    def __init__(self, in_ch, out_ch, ori_shape, scaled):
        super().__init__()
        self.ori_shape = ori_shape
        self.max_pool = nn.FractionalMaxPool2d(kernel_size=3, output_ratio=(scaled, scaled))
        self.conv1 = conv_bn_relu(out_ch, out_ch, kernel_size=3, stride=1, padding=1, has_bn=True, has_relu=True)
        self.conv2 = conv_bn_relu(out_ch, out_ch, kernel_size=1, stride=1, padding=0, has_bn=True, has_relu=True)
        
    def forward(self, x):
        out = self.max_pool(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = F.interpolate(out, size=self.ori_shape, mode='bilinear', align_corners=True)
        return out
    
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ori_shape, type, cardinality):
        super(ConvBlock, self).__init__()
        self.middle_ch = out_ch // 2
        self.C = cardinality  # the num of prm branch, default is 3
        self.ori_shape = ori_shape
        self.pyramid = list()
        # self.scaled = 2 ** (1 / self.C)  # control the scaled ratio
        self.scaled = 2
        self.type = type

        self.main_branch = nn.Sequential()
        if self.type != 'no_preact':
            self.main_branch.add_module('activation_layer1', nn.BatchNorm2d(in_ch))
            self.main_branch.add_module('activation_layer2', nn.ReLU(inplace=True))

        self.main_branch.add_module('conv1', conv_bn_relu(in_ch, self.middle_ch, kernel_size=1, stride=1, padding=0, has_bn=True, has_relu=True))
        self.main_branch.add_module('conv2', conv_bn_relu(self.middle_ch, self.middle_ch, kernel_size=3, stride=1, padding=1, has_relu=True, has_bn=True))
        self.main_branch.add_module('conv3', conv_bn_relu(self.middle_ch, self.middle_ch, kernel_size=1, stride=1, padding=0, has_bn=True, has_relu=False))

        for i in range(self.C):
            tmp_scaled = 1 / (self.scaled ** (i+1))  # change the scaled to change the feature resolution
            self.pyramid.append(
                PryBottleNet(self.middle_ch, self.middle_ch, self.ori_shape, tmp_scaled)
            )

            setattr(self, 'pry{}'.format(i), self.pyramid[i])

        self.conv_top = conv_bn_relu(in_ch, self.middle_ch, kernel_size=1, stride=1, padding=0, has_bn=True, has_relu=False)  # 这个是在pra各个等级之前的，进行一次卷积
        self.bn = nn.BatchNorm2d(self.middle_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv_bot = conv_bn_relu(self.middle_ch, self.middle_ch, kernel_size=1, stride=1, padding=0, has_bn=True, has_relu=False)  #这个是紧跟着pra输出相加之后的卷积
        self.conv_out = conv_bn_relu(self.middle_ch, out_ch, kernel_size=1, stride=1, padding=0, has_bn=True, has_relu=False)   #这个是再一次进行卷积

    def forward(self, x):
        # 1. main branch
        out_main = self.main_branch(x)  # ori shape in, ori shape out

        # 2. pyramid branch
        # ---------------------------
        out_pry = None
        pyraTable = list()
        conv_top = self.conv_top(x)
        for i in range(self.C):
            out_pry = eval('self.pry' + str(i))(conv_top)   # 这里出来的都是和输入尺寸一样的
            pyraTable.append(out_pry)
            if i != 0:
                out_pry = pyraTable[i] + pyraTable[0]

        out_pry = self.bn(out_pry)        # 在前面使用bn和relu是为了减少方差
        out_pry = self.relu(out_pry)

        out_pry = self.conv_bot(out_pry)  # 金字塔分支进行相加后卷积

        # ------------------------------
        assert out_pry.shape == out_main.shape
        out = out_pry + out_main
        out = self.bn(out)
        out = self.relu(out)
        out = self.conv_out(out)

        return out
    
class SkipLayer(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super(SkipLayer, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.bn = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv = conv_bn_relu(in_ch, out_ch, kernel_size=1, stride=stride, padding=0, has_bn=False, has_relu=False)

    def forward(self, x):
        if self.in_ch == self.out_ch:
            out = x
        else:
            out = self.bn(x)
            out = self.relu(out)
            out = self.conv(out)
        return out
    
class PRM(nn.Module):
    def __init__(self, in_ch, out_ch, ori_shape, cnf, type):
        """
        :param in_ch: 
        :param out_ch: 
        :param cnf:
        """
        cardinality = 4   #这个数？？？？
        super(PRM, self).__init__()
        self.skip_layer = SkipLayer(in_ch, out_ch, stride=1)                      # 主要的原分支
        self.pry_layer = ConvBlock(in_ch, out_ch, ori_shape, type, cardinality)   #　多分辨率分支
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out_skip = self.skip_layer(x) 
        out_pry = self.pry_layer(x)
        assert out_pry.shape == out_skip.shape
        out = out_pry + out_skip
        out = self.bn(out)
        out = self.relu(out)  
        return out 

class MyNet(nn.Module):
    def __init__(self, initial = False):
        super().__init__()

        # inceptionV3
        # self.sub_net1 = InceptionV3(in_ch=4, num_classes=17)
        # self.sub_net2 = InceptionV3(in_ch=1, num_classes=17)

        # self.sub_net1 = InceptionV3(in_ch=5, num_classes=17)
        # self.sub_net2 = InceptionV3(in_ch=5, num_classes=17)
        # self.pool1 = nn.AdaptiveAvgPool2d((1, 1))
        # self.pool2 = nn.AdaptiveAvgPool2d((1, 1))

        # ResNet -- multimodal
        # self.sub_net1 = resnet50(in_ch=4, num_classes=17)
        # self.sub_net2 = resnet50(in_ch=1, num_classes=17)     

        # single modal
        # self.sub_net1 = resnet50(in_ch=1, num_classes=17)
        # self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.head = PRM(5, 32, (160, 160), None, 'no_preact')

        self.sub_net = InceptionV3(in_ch=32, num_classes=16)
        # self.sub_net = VSGCNN(n_classes=16, in_channels=32, num_groups=16)

        # self.sub_net = MobileNetV2(in_ch=32, class_num=11)

        # for emg ......................................
        # self.fc1 = nn.Linear(64, 2048)
        # self.fc2 = nn.Linear(2048, 1024)
        # self.fc3 = nn.Linear(1024, 17)
        if initial:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    m.bias.data.zero_()

    def forward(self, x):
        # x = self.max_pool(x)
        # x_out = self.sub_net1(x)
        # y_out = self.sub_net2(y)
        
        x_out = self.head(x)
        x_out = self.sub_net(x_out)

        # y_out = self.fc1(y)
        # y_out = self.fc2(y_out)
        # y_out = self.fc3(y_out)
        
        # return x_out, y_out

        return x_out

def get_net():
    net = MyNet()
    return net
    
    
if __name__ == "__main__":
    from time import time
    # net = PRM(4, 64, (160, 160), None, 'no_preact')
    net = MyNet().to('cpu')
    # input1 = torch.ones(size=(1, 5, 160 ,160), dtype=torch.float32).to('cpu')
    input2 = torch.ones(size=(1, 5, 160 ,160), dtype=torch.float32).to('cpu')
    # st = time()
    # x_out, y_out = net(input1, input2)
    # et = time()
    # print(et - st)

    x_out = net(input2)
    
