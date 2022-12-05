from asyncio.log import logger
from multiprocessing.sharedctypes import Value
import os
from os import path as osp
import sys

import torch
import torchvision
from torchvision import transforms

from PIL import Image
import cv2
from torch.utils.data import Dataset
import numpy as np
import json
import pdb
from utils.transforms import get_affine_transform
from utils.transforms import affine_transform
# from deepfashion2agg81kps import Deepfashion2Agg81KpsDataset


ATR_PARSING = {    
    # with head and hand    
    # 'upper':[1, 2, 3, 4,  11, 16, 17, 14, 15],    
    # #without head    #  
    'upper': [1, 2, 3, 4, 11, 16, 17], # 这些点是 mask的index，只要是属于这个list的，都属于上衣
    'bottom': [5, 6, 8],    
    #   with head and hand    
    # 'upper_bottom':[1, 2, 3, 4, 5, 7, 8,11, 16, 17, 14, 15,6] 
    #    # w/o hand    # 
    'upper_bottom': [4, 5, 7, 16, 17],
    
}

class InferenceDataset(Dataset):
    def __init__(self, cfg):

        img_dir = osp.join(cfg.infer.root, cfg.infer.img_dir)
        mask_dir = osp.join(cfg.infer.root, cfg.infer.mask_dir)
        parse_dir = osp.join(cfg.infer.root, cfg.infer.parse_dir)
        print(f'img_dir is {img_dir}')
        print(f'mask_dir is {mask_dir}')
        print(f'mask_dir is {parse_dir}')
        # img_dir, mask_dir, parse_dir)

        self.imgs_path = self.get_img_path(img_dir)
        self.parses_path = self.get_parse_path(parse_dir)
        self.masks_path = self.get_mask_path(mask_dir)
        print(len(self.imgs_path))
        self.image_width = cfg.MODEL.IMAGE_SIZE[0] # 384
        self.image_height = cfg.MODEL.IMAGE_SIZE[1] # 512
        self.image_size = np.array([self.image_width, self.image_height])
        self.aspect_ratio = self.image_width * 1.0 / self.image_height # 宽高比
        self.pixel_std = 200
        self.transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.num_joints = 81
    
    def get_mask_path(self, root):
        '''
        Get the mask (segmentation or matting) from root dir
        '''
        masks_path = []
        for file in os.listdir(root):
            if file.endswith('png') or file.endswith('jpg'):
                masks_path.append(osp.join(root, file))
        return sorted(masks_path)

    def get_img_path(self, root):
        imgs_path = []
        for file in os.listdir(root):
            if file.endswith('.jpg') or file.endswith('.png'):
                imgs_path.append(osp.join(root, file))
        return sorted(imgs_path)


    def get_parse_path(self, root):
        '''
        Args: the root folder of img and ann
        Return: mask files' path
        '''
        imgs_path, parses_path = [], []
        # pdb.Pdb(nosigint=True).set_trace()
        for file in os.listdir(root):
            if file.endswith('.npy'):
                parses_path.append(osp.join(root, file))
        return sorted(parses_path)


    def __getitem__(self, idx):
        img = Image.open(self.imgs_path[idx]).convert('RGB') # cv2的所有处理代码是不是都需要bgr？
        parse = np.load(self.parses_path[idx], allow_pickle=True)
        mask = cv2.imread(self.masks_path[idx], flags=cv2.IMREAD_UNCHANGED)
        assert len(mask.shape) == 2, "mask's dim is not 2, shape is {}".format(mask.shape)

        data_numpy = np.asarray(img) # (h, w, c)

        if data_numpy is None:
            raise ValueError('Fail to read {}'.format(img))

        mask = self.get_mask(mask)
        # mask = self.get_mask_from_cloth(parse)
        x, y, w, h = self.get_bbox_from_mask(mask)
        c, s = self._xywh2cs(x, y, w, h)

        # pdb.Pdb(nosigint=True).set_trace()
        '''scale the bbox; pad to 3:4; resize(512, 384)'''
        w_scale, h_scale = w * s, h * s

        x1_scale, x2_scale = np.around(c[0] - w_scale/2, 0).astype(np.int32), \
                                np.around(c[0] + w_scale/2, 0).astype(np.int32)
        y1_scale, y2_scale = np.around(c[1] - h_scale/2, 0).astype(np.int32), \
                                np.around(c[1] + h_scale/2, 0).astype(np.int32)
        
        # 所有操作和数据类型尽量用Numpy
        input = data_numpy[y1_scale:y2_scale, x1_scale:x2_scale, :]
        cv2.imwrite('../fuck.png', input)
        input, hw_pad = self.pad_and_resize(input)

        input = self.transform(input)
        # print(input.shape) # (3, 512, 384)
        x1_pad, y1_pad = np.around(c[0] - hw_pad[0]/2, 0).astype(np.int32), \
            np.around(c[1] - hw_pad[1]/2, 0).astype(np.int32)

        meta = {
            'center': c,
            'scale': s,
            'image_id': osp.splitext(self.imgs_path[idx].split('/')[-1])[0],
            'img_orig': data_numpy,
            'bbox_pad': [x1_pad, y1_pad, *hw_pad],
            # 'hw_pad': hw_pad
        }
        return input, meta


    def __len__(self):
        # return 1
        return len(self.imgs_path)


    def pad_and_resize(self, x):
        '''
        pad x to aspect_ratio, and resize it to (512, 384) (h, w)
        Args: x
        Return: x_resized, [w_paded, h_paded]
        '''
        h, w, _ = x.shape
        h_new, w_new = h, w
        if w > self.aspect_ratio * h: # h smaller, pad h
            h_new = h * 1.0 / self.aspect_ratio
            pad_h = round((h_new - h) / 2)
            x = np.pad(
                x, pad_width=((pad_h, pad_h), (0, 0), (0, 0)), 
                mode='constant', constant_values=255
            ) # 补h=>axis=0
        elif w < self.aspect_ratio * h: # w smaller，pad w
            w_new = h * 1.0 * self.aspect_ratio
            pad_w = round((w_new - w) / 2)
            x = np.pad(
                x, pad_width=((0, 0), (pad_w, pad_w), (0, 0)),
                mode='constant', constant_values=255
            )
        x = cv2.resize(
            x, dsize=(self.image_width, self.image_height), interpolation=cv2.INTER_LINEAR
        )
        return x, [w_new, h_new]


    def get_mask(self, ann):
        '''
        get the mask from the mask, this is just for API
        '''
        mask = np.zeros_like(ann).astype(bool)
        mask |= (ann > 0)
        return mask
        

    def get_mask_from_cloth(self, ann):
        '''
        Get the smallest bbox around the person from the cloth
        Args: the segmentation mask 
        Return: the binary mask of the specified category
        '''
        mask = np.zeros_like(ann).astype(bool)
        for idx in ATR_PARSING['upper'] + ATR_PARSING['bottom']:
            mask |= (ann == idx)
        return mask


    def get_bbox_from_mask(self, mask):
        '''
        Args: binary mask (h,w) ndarray
        Return: bbox x,y,h,w
        '''
        y_mask, x_mask = np.where(mask == True)
        x1, x2 = np.amin(x_mask), np.amax(x_mask)
        y1, y2 = np.amin(y_mask), np.amax(y_mask) 
        w, h = x2 - x1, y2 - y1
        return x1, y1, w, h


    def get_center_scale(self, ann):
        '''
        Args: json file of ann
        Return: the center & scale of the ann
        Note: there maybe many ann in one json file
        '''
        # first get x, y, w, h
        pts = []
        for shape in ann['shapes']:
            pt = shape['points']
            pts.append([pt[0][0], pt[0][1], pt[1][0], pt[1][1]]) # (x0, y0, x1, y1)
        
        pts = [[pt[0], pt[1], pt[2] - pt[0], pt[3] - pt[1]] for pt in pts]

        c, s = self._xywh2cs(*pts[0])
        return c, s


    
    

    def _xywh2cs(self, x, y, w, h):
        '''
        How to get scale here?
        according to aspect_ratio: w/h-384/512, scale the smaller edge to match the aspect_ratio
        '''
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        scale = 1.05
        return center, scale