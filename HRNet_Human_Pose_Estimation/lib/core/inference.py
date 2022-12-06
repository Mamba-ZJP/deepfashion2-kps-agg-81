# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np

from utils.transforms import transform_preds
import pdb


def get_max_preds(cfg, batch_heatmaps, is_deepfashion2=False):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    this is direct output from model: (b, 81, 128, 96) H/W = 4/3 = aspect_ratio
    '''
    
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    
    height, width = batch_heatmaps.shape[2], batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, axis=2) # 每个通道的feature map生成的heatmap中概率最大的像素就为关键点
    maxvals = np.amax(heatmaps_reshaped, axis=2) # max along the axis [0.1, 0.97]

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1)) 

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32) # repeat idx into shape (1, 1, 2)

    # get the coordinate of the specific joint
    preds[:, :, 0] = (preds[:, :, 0]) % width # get original x
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width) # get original y
    # cfg.TEST.MAXVAL
    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2)) # 这里阈值为0.5，只要大于0的都会保留
    # pdb.set_trace()
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask # 坐标乘上0就为0
    # normalize into [0, 1] to learn the relative position

    if not is_deepfashion2:
        preds[..., 0] /= width
        preds[..., 1] /= height
    # pdb.set_trace()
    # print(f'maxpreds: {np.amax(preds)}')
    return preds, maxvals


def get_final_preds(config, batch_heatmaps, center, scale):
    # print(f'get_final_preds')
    coords, maxvals = get_max_preds(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    if config.TEST.POST_PROCESS:
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = batch_heatmaps[n][p]
                px = int(math.floor(coords[n][p][0] + 0.5))
                py = int(math.floor(coords[n][p][1] + 0.5))
                if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                    diff = np.array(
                        [
                            hm[py][px+1] - hm[py][px-1],
                            hm[py+1][px]-hm[py-1][px]
                        ]
                    )
                    coords[n][p] += np.sign(diff) * .25

    preds = coords.copy()

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(
            coords[i], center[i], scale[i], [heatmap_width, heatmap_height]
        )

    return preds, maxvals
