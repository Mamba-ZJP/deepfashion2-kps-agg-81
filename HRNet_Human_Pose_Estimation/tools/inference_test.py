from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from locale import normalize
import os
import sys
import pprint
sys.path.append('./lib')
# print(sys.path)

from core.inference import get_final_preds

import shutil

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import pdb
# import _init_paths
# from lib.core.function import train, validate

from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from core.function import train
from core.function import validate, inference
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
from utils.utils import get_model_summary
from core.inference import *


# from dataset import Deepfashion2Agg81KpsDataset, JointsDataset
import dataset
import models
from PIL import Image
import numpy as np
import cv2
import copy
from pycocotools.coco import COCO


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=False,
                        type=str,
                        default='./experiments/deepfashion2/inference.yaml')

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()

    return args

def get_infer_stuff(cfg):
    infer_dataset = eval('dataset.' + cfg.DATASET.DATASET)(cfg)
    infer_loader = torch.utils.data.DataLoader(
        infer_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )
    return infer_dataset, infer_loader


def main():
    
    args = parse_args()
    
    update_config(cfg, args)
    # logger, final_output_dir, tb_log_dir = create_logger(cfg, args.cfg, 'train')
    
    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(cfg, is_train=False)

    device = torch.device(f'cuda:{cfg.GPUS[0]}') if len(cfg.GPUS) > 0 else 'cpu'
    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).to(device)

    # checkpoint_file = os.path.join(final_output_dir, '81kpscheckpoint.pth')
    checkpoint_file = './output/deepfashion2agg81kps/pose_hrnet/w48_512x384_adam_lr1e-3-agg81kps/81kpscheckpoint.pth'
        
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['state_dict'], strict=True) # load state has problem when strict=True,

    infer_dataset, infer_loader = get_infer_stuff(cfg)

    print('the threshold is {}'.format(cfg.infer.thresh))
    inference(cfg, infer_loader, model)
   

if __name__ == '__main__':
    main()
