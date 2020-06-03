from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2


def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA=True):
    """
    将 (N, num_anchors * box_attrs, H, W) 形式的4D特征图预测转换成 (N, num_anchors * H * W, box_attrs) 形式的3D预测
    :param prediction: yolo层输出,(N,C,H,W)
    :param inp_dim: 网络输入尺寸
    :param anchors: list(tuple),anchor尺寸
    :param num_classes:
    :param CUDA:
    """
    batch_size = prediction.size(0)
    stride = inp_dim // prediction.size(2)  # 从原图到特征图降采样的倍数
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    prediction = prediction.view(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(batch_size, grid_size * grid_size * num_anchors, bbox_attrs)

    anchors = [(a[0] / stride, a[1] / stride) for a in anchors]
