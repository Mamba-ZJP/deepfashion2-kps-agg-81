import json
import os
from os import path
import numpy as np
import pdb
'''
Note: the coordinate is in the crop image
firstly, we need to get a json template
'''

upper_bottom = np.array([63, 65], dtype=np.uint8)
bottom_curve = np.array([78, 80, 79], dtype=np.uint8)
neck_0 = np.array([2, 6], dtype=np.uint8)
neck_1 = np.array([1, 4], dtype=np.uint8)
left_cuff = np.array([11, 12], dtype=np.uint8)
right_cuff = np.array([17, 18], dtype=np.uint8)
anran_id = np.concatenate([upper_bottom, neck_0, neck_1, bottom_curve, left_cuff, right_cuff], axis=0)


category_to_idx = {
    'upper_bottom': upper_bottom,
    'bottom_curve' : bottom_curve,
    'neck': np.array([6, 1, 2, 4], dtype=np.uint8),
    'left_cuff': left_cuff,
    'right_cuff': right_cuff
}


def check_neck(maxval, k):
    if np.sum(maxval[k, neck_0 - 1]) > np.sum(maxval[k, neck_1 - 1]):
        return 'neck_0'
    else:
        return 'neck_1'


def write_json(cfg, batch_joints, batch_image_id, maxval):
    '''
    batch_joints: (B, 81, 2-(xy))
    batch_image_id: (B),
    '''
    root = '/data1/jiapeng'
    with open(path.join(root, '000000.json'), 'r') as f:
        json_template = json.load(f)
    label_json = json_template
    # pdb.set_trace()
    write_dir = path.join(root, 'anran_run')

    # 循环这个batch的每一张图
    for i, image_id in enumerate(batch_image_id): 
        # 循环json中所有的类别
        for j, shape in enumerate(label_json['shapes']):
            label = shape['label']
            # 特殊判断下neck
            # if label == 'neck': 
            #     label = check_neck(maxval, i)

            pts_list = []
            # 循环这个类别对应的channel
            for channel in category_to_idx[label] - 1:
                # 如果阈值太小，这个点就不放进去
                if maxval[i, channel] < cfg.TEST.MAXVAL and label != 'neck':
                    continue
                pt = batch_joints[i, channel].tolist() # 第i个样本的点
                pts_list.append(pt)
        
            label_json['shapes'][j]['points'] = pts_list # 用一个新的list去覆盖
        
        label_json['imagePath'] = image_id + '.jpg'
        label_json['imageData'] = None

        json_file = path.join(write_dir, image_id + '.json')
        with open(json_file, 'w') as f: # open the file dosen't exist
            json.dump(label_json, f)
        # break