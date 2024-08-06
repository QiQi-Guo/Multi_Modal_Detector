# Copyright (c) Facebook, Inc. and its affiliates.
import math
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np  # ---zhu---#

from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16
from mmdet.models.builder import NECKS


from typing import List, Optional, Dict, Any


__all__ = ["PillarFPN"]


@NECKS.register_module()
class PillarFPN(nn.Module):
    def __init__(self, input_channels=64):
        super().__init__()

        layer_nums = [3, 5, 5]
        # layer_strides = [2, 2, 2]
        layer_strides = [1, 3, 3]
        num_filters = [64, 128, 256]
        num_upsample_filters = [128, 128, 128]
        # upsample_strides = [0.5, 1, 2]
        upsample_strides = [1/3, 1, 3]

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        self.conv_final = nn.Conv2d(384, 256, 1, stride=1, padding=0)
        self.bn_final = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)

        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, spatial_features):

        ups = []
        ret_dict = {}
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(torch.true_divide(spatial_features.shape[2], x.shape[2]))
            ret_dict['spatial_features_%dx' % stride] = x

            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)
        x = self.bn_final(self.conv_final(x))
        x = self.relu(x)
        return x
