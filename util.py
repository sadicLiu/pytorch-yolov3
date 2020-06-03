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
    grid_size = inp_dim // stride  # feature map尺寸
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    prediction = prediction.view(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(batch_size, grid_size * grid_size * num_anchors, bbox_attrs)

    anchors = [(a[0] / stride, a[1] / stride) for a in anchors]

    # Sigmoid the  centre_X, centre_Y. and object confidencce
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

    # Add the center offsets
    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)
    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)

    prediction[:, :, :2] += x_y_offset

    # log space transform height and the width
    anchors = torch.FloatTensor(anchors)
    if CUDA:
        anchors = anchors.cuda()

    # 这里使用广播机制，处理预测值，得到 b_w, b_h
    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)  # anchors.shape: (1, 13*13*3, 2)
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors  # prediction[:, :, 2:4].shape: (B, 13*13*3, 2)

    # 对class score进行sigmoid
    prediction[:, :, 5: 5 + num_classes] = torch.sigmoid((prediction[:, :, 5: 5 + num_classes]))

    # 预测的box坐标、宽高都是相对于特征图的，将其转换成相对原始图像大小
    prediction[:, :, :4] *= stride

    return prediction
