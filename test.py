# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models
import torch.multiprocessing as mp
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import make_grid, save_image

import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import time
import os
import argparse
import copy
import datetime
import random
import sys
import json
import glob

### My libs
from core.utils import set_device, postprocess, ZipReader, set_seed
from core.utils import postprocess
from core.dataset import Dataset
from core.model import EdgeGenerator, InpaintGenerator
 

parser = argparse.ArgumentParser(description="PConv")
parser.add_argument("-c", "--config", type=str, required=True)
parser.add_argument("-l", "--level", type=int, required=True)
parser.add_argument("-p", "--port", type=str, default="23451")
args = parser.parse_args()

BATCH_SIZE = 4

def main_worker(gpu, ngpus_per_node, config):
  torch.cuda.set_device(gpu)
  set_seed(config['seed'])

  # Model and version
  edgeG = set_device(EdgeGenerator(use_spectral_norm=True))
  imgG = set_device(InpaintGenerator())
  latest_epoch = open(os.path.join(config['save_dir'], 'latest.ckpt'), 'r').read().splitlines()[-1]
  path = os.path.join(config['save_dir'], 'gen_{}.pth'.format(latest_epoch))
  data = torch.load(path, map_location = lambda storage, loc: set_device(storage)) 
  edgeG.load_state_dict(data['edgeG'])
  imgG.load_state_dict(data['imgG'])
  edgeG.eval() 
  imgG.eval()

  # prepare dataset
  dataset = Dataset(config['data_loader'], debug=False, split='test', level=args.level)
  step = math.ceil(len(dataset) / ngpus_per_node)
  dataset.set_subset(gpu*step, min(gpu*step+step, len(dataset)))
  dataloader = DataLoader(dataset, batch_size= BATCH_SIZE, shuffle=False, num_workers=config['data_loader']['num_workers'], pin_memory=True)

  
  path = os.path.join(config['save_dir'], 'results_{}_level_{}'.format(str(latest_epoch).zfill(5), str(args.level).zfill(2)))
  os.makedirs(path, exist_ok=True)
  # iteration through datasets
  for idx, (images, image_gray, edges, masks, names) in enumerate(dataloader):
    print('[{}] {}/{}: {}  ...'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
      idx, len(dataloader), names[0]))
    images, image_gray, edges, masks = set_device([images, image_gray, edges, masks])
    edges_masked = (edges * (1 - masks))
    gray_masked = (image_gray * (1 - masks)) + masks
    inputs = torch.cat((gray_masked, edges_masked, masks), dim=1)
    pred_edge = edgeG(inputs)                                    # in: [grayscale(1) + edge(1) + mask(1)]
    images_masked = (images * (1 - masks).float()) + masks
    comp_edge = pred_edge*masks + (1-masks)*edges
    inputs = torch.cat((images_masked, comp_edge), dim=1)
    pred_img = imgG(inputs)                                    # in: [rgb(3) + edge(1)]
    orig_imgs = list(postprocess(images))
    comp_imgs = list(postprocess(masks*pred_img+(1-masks)*images))
    for i in range(len(orig_imgs)):
      Image.fromarray(orig_imgs[i]).save(os.path.join(path, '{}_orig.png'.format(names[i].split('.')[0])))
      Image.fromarray(comp_imgs[i]).save(os.path.join(path, '{}_comp.png'.format(names[i].split('.')[0])))
  print('Finish in {}'.format(path))



if __name__ == '__main__':
  ngpus_per_node = torch.cuda.device_count()
  config = json.load(open(args.config))
  config['save_dir'] = os.path.join(config['save_dir'], config['data_loader']['name'])

  print('using {} GPUs for testing ... '.format(ngpus_per_node))
  # setup distributed parallel training environments
  ngpus_per_node = torch.cuda.device_count()
  config['world_size'] = ngpus_per_node
  config['init_method'] = 'tcp://127.0.0.1:'+ args.port 
  config['distributed'] = True
  mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config))
 
