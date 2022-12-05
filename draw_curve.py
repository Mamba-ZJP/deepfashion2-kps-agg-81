import json
from os import path
import os
# from pycocotools.coco import COCO

import pdb
import torch
import numpy as np
import cv2


class CruveDrawer():
    def __init__(self, root) -> None:
        self.root = root
        self.img_file, self.ann_file = self.get_img_ann(root)
        self.num_img = len(self.img_file)
        self.write_dir = 'curve_draw'
        self.check()

    def check(self):
        if not path.exists(self.root):
            raise ValueError('root does not exist!')
        assert len(self.img_file) == len(self.ann_file)
        write_dir = path.join(self.root, self.write_dir)
        if not os.path.exists(write_dir):
            os.mkdir(write_dir)
            print(f'mkdir: {write_dir}')

    def get_img_ann(self, root):
        ann_file, img_file = [], []
        for file in os.listdir(root):
            extension = path.splitext(file)[-1][1:]
            if extension in ['png', 'jpg']:
                img_file.append(file)
            elif extension == 'json':
                ann_file.append(file)
        return sorted(img_file), sorted(ann_file)

    def draw(self):
        # pdb.set_trace()
        for i in range(self.num_img):
            img = cv2.imread(path.join(self.root, self.img_file[i]))
            # pdb.set_trace()
            with open(path.join(self.root, self.ann_file[i]), 'r') as f:
                ann = json.load(f)
            for shape in ann['shapes']:
                pts_list = shape['points']  # 得到点的list
                pts_list = [[round(pt[0]), round(pt[1])] for pt in pts_list]
                for idx, _ in enumerate(pts_list):
                    if idx is not len(pts_list) - 1: # 如果不是
                        cv2.line(img=img, pt1=pts_list[idx], pt2=pts_list[idx + 1], 
                            color=(255, 0, 0), thickness=2)
                    else: 
                        # 如果是最后一个点，那么和第一个点相连
                        cv2.line(img=img, pt1=pts_list[idx], pt2=pts_list[0],
                            color=(255, 0, 0), thickness=2)


            cv2.imwrite(filename=path.join(self.root, self.write_dir, self.img_file[i]), img=img)
    

if __name__ == '__main__':
    root = '/data1/jiapeng/anran_run'
    curve_drawer = CruveDrawer(root)
    curve_drawer.draw()