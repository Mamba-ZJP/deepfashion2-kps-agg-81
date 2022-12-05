# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os, pdb, sys, os.path as osp

import numpy as np
import torchvision
import cv2

from core.inference import get_max_preds

import torch

upper_bottom = np.array([63, 65], dtype=np.uint8)
bottom_curve = np.array([78, 80, 79], dtype=np.uint8)
neck_0 = np.array([2, 6], dtype=np.uint8)
neck_1 = np.array([1, 4], dtype=np.uint8)
left_cuff = np.array([11, 12], dtype=np.uint8)
right_cuff = np.array([17, 18], dtype=np.uint8)
# anran_id = np.concatenate([upper_bottom, neck_0, neck_1, bottom_curve, left_cuff, right_cuff], axis=0)


def save_batch_image_inference(cfg, batch_image, batch_joints, maxval, meta, save_dir, nrow=4, padding=0):
    '''
    Draw all the lankmarks firstly
    Args
        batch_image: input of model, tensor: [b, c, h, w] (512, 384)
        maxval: ndarray: []
        batch_joints: output of model, ndarray: [b, 81, 2]
        meta['origin_img']: 
        meta['bbox_pad'] (xywh): [tensor:[b], .. ,] 
        meta['image_id']: 
    '''
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    batch = batch_image.shape[0]
    batch_img_orig = meta['img_orig'].numpy()
    # print('the number above MAXVAL {} is {}'.format(cfg.TEST.MAXVAL, (maxval[k] > cfg.TEST.MAXVAL).sum() ))
    # pdb.set_trace()

    # 看下裤子的点有哪些 71~76 -1
    for k, img in enumerate(batch_img_orig):
        for idx, joint in enumerate(batch_joints[k]):
            if maxval[k, idx] < cfg.infer.thresh: # threshold 
                continue
            cv2.circle(img, (int(joint[0]), int(joint[1])), radius=1, color=[255, 0, 0], thickness=2)
    
        cv2.imwrite(
            osp.join(save_dir, meta['image_id'][k] + '.png'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        )




def save_image_inference(cfg, batch_image, batch_joints, maxval, meta, file_name, nrow=4, padding=0):
    '''
    batch_image: input of model, tensor: [b, c, h, w] (512, 384)
    maxval: ndarray: []
    batch_joints: output of model, ndarray: [b, 81, 2]
    meta['img_orig']: 
    meta['bbox_pad'] (xywh): [tensor:[b], .. ,] 
    '''
    k = 0 # 先取第一个样例做测试
    batch_origin_img = meta['img_orig'].permute(0, 3, 1, 2)
    img = batch_origin_img[k].permute(1, 2, 0).numpy().copy()

    # 单独判断下neck的置信度
    if np.sum(maxval[k, neck_0 - 1]) > np.sum(maxval[k, neck_1 - 1]):
        anran_id = np.concatenate([upper_bottom, neck_0, bottom_curve, left_cuff, right_cuff], axis=0)
    else:
        anran_id = np.concatenate([upper_bottom, neck_1, bottom_curve, left_cuff, right_cuff], axis=0)

    for idx, joint in enumerate(batch_joints[0]):
        if idx not in anran_id - 1 or maxval[k, idx] < cfg.TEST.MAXVAL: # 不在anran的点里面，或者置信度过小
            continue
        cv2.circle(img, (int(joint[0]), int(joint[1])), radius=1, color=[255, 0, 0], thickness=2)
    cv2.imwrite(file_name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


    # grid = torchvision.utils.make_grid(batch_origin_img, nrow, padding, True) 
    # ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    # ndarr = ndarr.copy()
    # nmaps = batch_image.size(0)
    # xmaps = min(nrow, nmaps)
    # ymaps = int(math.ceil(float(nmaps) / xmaps))
    # height = int(batch_origin_img.shape[2] + padding)
    # width = int(batch_origin_img.shape[3] + padding)
    # k = 0
    # for row in range(ymaps):
    #     for col in range(xmaps):
    #         if k >= nmaps:
    #             break
    #         joints = batch_joints[k] # in save gt, torch.unique(joints) => [0.]
    #         # pdb.set_trace()
    #         for idx, joint in enumerate(joints):
    #             joint[0] = col * width + joint[0]
    #             joint[1] = row * height + joint[1]
    #             # if joint_vis[0]: # 只有gt 标注且可见 的点才会画
    #             if idx not in (anran_id - 1): #and idx not in dress:
    #                 continue
    #             cv2.circle(ndarr, (int(joint[0]), int(joint[1])), 2, [255, 0, 0], 2)
    #         k = k + 1
    
    # cv2.imwrite(file_name, cv2.cvtColor(ndarr, cv2.COLOR_RGB2BGR))



def save_batch_image_with_joints(batch_image, batch_joints, batch_joints_vis,
                                 file_name, nrow=8, padding=0):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1], # 这里的joints_vis具体的原理
    }
    '''
    # print(f'{batch_joints_vis[0].shape}, {batch_joints_vis[0]}')
    # make a grid 将所有图片放在一个网格里组成一张新的图片
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True) 
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    skirt_kpt_ids = [76, 77, 78, 79, 80, 62, 63, 64]
    dress = [58 - 1, 59 - 1, 60 - 1, 61 - 1, 62 - 1]

    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            joints = batch_joints[k] # in save gt, torch.unique(joints) => [0.]
            joints_vis = batch_joints_vis[k]
            # pdb.set_trace()
            for idx, (joint, joint_vis) in enumerate(zip(joints, joints_vis)):
                joint[0] = x * width + padding + joint[0]
                joint[1] = y * height + padding + joint[1]
                # if joint_vis[0]: # 只有gt 标注且可见 的点才会画
                if idx not in skirt_kpt_ids: #and idx not in dress:
                    continue
                cv2.circle(ndarr, (int(joint[0]), int(joint[1])), 2, [255, 0, 0], 2)
            k = k + 1
    
    cv2.imwrite(file_name, cv2.cvtColor(ndarr, cv2.COLOR_RGB2BGR))


def save_batch_heatmaps(batch_image, batch_heatmaps, file_name,
                        normalize=True):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_heatmaps: ['batch_size, num_joints, height, width]
    file_name: saved file name
    '''
    if normalize:
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image.add_(-min).div_(max - min + 1e-5)

    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(1)
    heatmap_height = batch_heatmaps.size(2)
    heatmap_width = batch_heatmaps.size(3)

    grid_image = np.zeros((batch_size*heatmap_height,
                           (num_joints+1)*heatmap_width,
                           3),
                          dtype=np.uint8)

    preds, maxvals = get_max_preds(batch_heatmaps.detach().cpu().numpy())

    for i in range(batch_size):
        image = batch_image[i].mul(255)\
                              .clamp(0, 255)\
                              .byte()\
                              .permute(1, 2, 0)\
                              .cpu().numpy()
        heatmaps = batch_heatmaps[i].mul(255)\
                                    .clamp(0, 255)\
                                    .byte()\
                                    .cpu().numpy()

        resized_image = cv2.resize(image,
                                   (int(heatmap_width), int(heatmap_height)))

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)
        for j in range(num_joints):
            cv2.circle(resized_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)
            heatmap = heatmaps[j, :, :]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            masked_image = colored_heatmap*0.7 + resized_image*0.3
            cv2.circle(masked_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)

            width_begin = heatmap_width * (j+1)
            width_end = heatmap_width * (j+2)
            grid_image[height_begin:height_end, width_begin:width_end, :] = \
                masked_image
            # grid_image[height_begin:height_end, width_begin:width_end, :] = \
            #     colored_heatmap*0.7 + resized_image*0.3

        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image

    cv2.imwrite(file_name, grid_image)


def save_debug_images(config, input, meta, target, joints_pred, output,
                      prefix):
    if not config.DEBUG.DEBUG:
        return

    if config.DEBUG.SAVE_BATCH_IMAGES_GT:
        save_batch_image_with_joints(
            input, meta['joints'], meta['joints_vis'],
            '{}_gt.jpg'.format(prefix)
        )
    if config.DEBUG.SAVE_BATCH_IMAGES_PRED:
        save_batch_image_with_joints(
            input, joints_pred, meta['joints_vis'],
            '{}_pred.jpg'.format(prefix)
        )
    if config.DEBUG.SAVE_HEATMAPS_GT:
        save_batch_heatmaps(
            input, target, '{}_hm_gt.jpg'.format(prefix)
        )
    if config.DEBUG.SAVE_HEATMAPS_PRED:
        save_batch_heatmaps(
            input, output, '{}_hm_pred.jpg'.format(prefix)
        )
