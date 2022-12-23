#!/usr/bin/env python
# Title           :loader.py
# Author          :Venkatraman Narayanan, Bala Murali Manoghar, Vishnu Shashank Dorbala, Aniket Bera, Dinesh Manocha
# Copyright       :"Copyright 2020, Proxemo project"
# Version         :1.0
# License         :"MIT"
# Maintainer      :Venkatraman Narayanan, Bala Murali Manoghar
# Email           :vnarayan@terpmail.umd.edu, bsaisudh@terpmail.umd.edu
#==============================================================================
from collections import OrderedDict

import torch
import torch.nn as nn
import torchsummary as summary

import numpy as np

from IPython import embed


class VSGCNN(nn.Module):
    """VSGCNN emotion classification module."""

    def __init__(self, n_classes, in_channels, num_groups, dropout=0.2, layer_channels=[32, 64, 16]):
        """Constructor

        Args:
            n_classes (int): Num output classes
            in_channels ([type]): Num input channels
            num_groups ([type]): Number of View angle groups
            dropout (float, optional): Dropout rate. Defaults to 0.2.
            layer_channels (list, optional): Number of channels at each stage.
                                             Defaults to [32, 64, 16].
        """
        super().__init__()
        self.in_channels = in_channels
        self.n_groups = num_groups
        self.layer_channels = layer_channels
        self.dropout = dropout
        self.num_classes = n_classes
        self.build_net()

    def _gen_layer_name(self, stage, layer_type, layer_num=''):
        """Generate unique name for layer."""
        name = '_'.join([self.__class__.__name__, 'stage',
                         str(stage), layer_type, str(layer_num)])
        return name

    def build_net(self):
        """Network builder"""
        conv1_0 = nn.Conv2d(self.in_channels,
                            self.layer_channels[0],
                            (3, 3))
        conv1_1 = nn.Conv2d(self.layer_channels[0],
                            self.layer_channels[0]*self.n_groups,
                            (3, 3))
        conv1_2 = nn.Conv2d(self.layer_channels[0]*self.n_groups,
                            self.layer_channels[0]*self.n_groups,
                            (3, 3), groups=self.n_groups)
        bn1 = nn.BatchNorm2d(self.layer_channels[0]*self.n_groups)

        conv2_1 = nn.Conv2d(self.layer_channels[0]*self.n_groups,
                            self.layer_channels[1]*self.n_groups,
                            (3, 3), groups=self.n_groups)

        conv2_2 = nn.Conv2d(self.layer_channels[1]*self.n_groups,
                            self.layer_channels[1]*self.n_groups,
                            (3, 3), groups=self.n_groups)
        bn2 = nn.BatchNorm2d(self.layer_channels[1]*self.n_groups)

        max_pool_1 = nn.MaxPool2d((3, 3), (2, 2))
    
        dropout = nn.Dropout(self.dropout)

        self.conv_stage_1 = nn.Sequential(OrderedDict([
            (self._gen_layer_name(1, 'conv', 0), conv1_0),
            (self._gen_layer_name(1, 'relu', 0), nn.ReLU()),
            (self._gen_layer_name(1, 'conv', 1), conv1_1),
            (self._gen_layer_name(1, 'relu', 1), nn.ReLU()),
            (self._gen_layer_name(1, 'maxpool', 1), max_pool_1),
            (self._gen_layer_name(1, 'conv', 2), conv1_2),
            (self._gen_layer_name(1, 'relu', 2), nn.ReLU()),
            (self._gen_layer_name(1, 'maxpool', 2), max_pool_1),
            (self._gen_layer_name(1, 'bn'), bn1),
            (self._gen_layer_name(1, 'drop'), dropout)
        ]))

        self.conv_stage_2 = nn.Sequential(OrderedDict([
            (self._gen_layer_name(2, 'conv', 1), conv2_1),
            (self._gen_layer_name(2, 'relu', 1), nn.ReLU()),
            (self._gen_layer_name(2, 'conv', 2), conv2_2),
            (self._gen_layer_name(2, 'relu', 2), nn.ReLU()),
            (self._gen_layer_name(2, 'bn'), bn2),
            (self._gen_layer_name(2, 'drop'), dropout)
        ]))

        conv3_1 = nn.Conv2d(self.layer_channels[1]*self.n_groups,
                            self.layer_channels[2]*self.n_groups,
                            (3, 3), groups=self.n_groups)
        conv3_2 = nn.Conv2d(self.layer_channels[2]*self.n_groups,
                            1, (3, 3), stride=(2, 2))

        self.final_layers = nn.Sequential(OrderedDict([
            (self._gen_layer_name(3, 'conv', 1), conv3_1),
            (self._gen_layer_name(3, 'relu', 1), nn.ReLU()),
            (self._gen_layer_name(3, 'conv', 2), conv3_2),
            (self._gen_layer_name(3, 'relu', 2), nn.ReLU())
        ]))
        self.softmax = nn.Softmax(2)
        
        self.fc1 = nn.Linear(15 * 15, 500)
        self.fc2 = nn.Linear(500, 150)
        self.fc3 = nn.Linear(150, self.num_classes)

    def forward(self, input_tensor, apply_sfmax=False):
        """Forward pass

        Args:
            input_tensor (Tensor): Input data
            apply_sfmax (bool, optional): softmax flag. Defaults to False.

        Returns:
            [Tensor]: Forward pass output
        """
        # convert [N, H, W, C] to [N, C, H, W]
        if input_tensor.size(1) != self.in_channels:
            input_tensor = input_tensor.permute(0, 3, 2, 1)

        first_conv = self.conv_stage_1(input_tensor)
        second_conv = self.conv_stage_2(first_conv)
        final_layers = self.final_layers(second_conv)
        final_layers = final_layers.view(input_tensor.shape[0], -1)
        out = self.fc1(final_layers)
        out = self.fc2(out)
        out = self.fc3(out)
        
        return out
    

def VSGCNN_model(n_classes=11, in_channels=5, num_groups=11):
    return VSGCNN(n_classes=n_classes, in_channels=in_channels, num_groups=num_groups)
    

if __name__ == "__main__":
    xx = torch.rand(size=(1, 32, 160, 160))
    vsgcnn = VSGCNN(n_classes=11, in_channels=32, num_groups=11)
    out = vsgcnn(xx)
    
    
