import os
import time
import math
import glob
import shutil
import importlib
import datetime
import numpy as np
from PIL import Image
from math import log10

from functools import partial
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid, save_image
import torch.distributed as dist

from core.dataset import Dataset
from core.utils import set_seed, set_device, Progbar, postprocess
from core.model import EdgeGenerator, InpaintGenerator, Discriminator
from core.loss import AdversarialLoss, PerceptualLoss, StyleLoss
from core import metric as module_metric


class Trainer():
  def __init__(self, config, debug=False):
    self.config = config
    self.epoch = 0
    self.iteration = 0
    if debug:
      self.config['trainer']['save_freq'] = 5
      self.config['trainer']['valid_freq'] = 5

    # setup data set and data loader
    self.train_dataset = Dataset(config['data_loader'], debug=debug, split='train')
    self.valid_dataset = Dataset(config['data_loader'], debug=debug, split='test')
    worker_init_fn = partial(set_seed, base=config['seed'])
    self.train_sampler = None
    self.valid_sampler = None
    if config['distributed']:
      self.train_sampler = DistributedSampler(self.train_dataset, 
        num_replicas=config['world_size'], rank=config['global_rank'])
      self.valid_sampler = DistributedSampler(self.valid_dataset, 
        num_replicas=config['world_size'], rank=config['local_rank'])
    self.train_loader = DataLoader(self.train_dataset, 
      batch_size= config['data_loader']['batch_size'] // config['world_size'],
      shuffle=(self.train_sampler is None), num_workers=config['data_loader']['num_workers'],
      pin_memory=True, sampler=self.train_sampler, worker_init_fn=worker_init_fn)
    self.valid_loader = DataLoader(self.valid_dataset, 
      batch_size= 1, shuffle=None, num_workers=config['data_loader']['num_workers'],
      pin_memory=True, sampler=self.valid_sampler, worker_init_fn=worker_init_fn)

    # set up losses and metrics
    self.adversarial_loss = AdversarialLoss(type=self.config['losses']['GAN_LOSS'])
    self.perceptual_loss = PerceptualLoss()
    self.style_loss = StyleLoss()
    self.l1_loss = nn.L1Loss()
    self.metrics = {met: getattr(module_metric, met) for met in config['metrics']}
    self.dis_writer = None
    self.gen_writer = None
    if self.config['global_rank'] == 0 or (not config['distributed']):
      self.dis_writer = SummaryWriter(os.path.join(config['save_dir'], 'dis'))
      self.gen_writer = SummaryWriter(os.path.join(config['save_dir'], 'gen'))
    self.samples_path = os.path.join(config['save_dir'], 'samples')
    self.results_path = os.path.join(config['save_dir'], 'results')
    
    # other args
    self.log_args = self.config['logger']
    self.train_args = self.config['trainer']

    # Edgegenerator input: [grayscale(1) + edge(1) + mask(1)], discriminator input: (grayscale(1) + edge(1))
    self.edgeG = set_device(EdgeGenerator(use_spectral_norm=True))
    self.edgeD = set_device(Discriminator(in_channels=2, use_sigmoid=config['losses']['GAN_LOSS'] != 'hinge'))
    # Imagegenerator input: [rgb(3) + edge(1)], discriminator input: [rgb(3)]
    self.imgG = set_device(InpaintGenerator())
    self.imgD = set_device(Discriminator(in_channels=3, use_sigmoid=config['losses']['GAN_LOSS'] != 'hinge'))
    self.optimG = torch.optim.Adam(list(self.edgeG.parameters()) + list(self.imgG.parameters()), lr=config['optimizer']['lr'],
      betas=(self.config['optimizer']['beta1'], self.config['optimizer']['beta2']))
    self.optimD = torch.optim.Adam(list(self.edgeD.parameters()) + list(self.imgD.parameters()),
      lr=config['optimizer']['lr'] * config['optimizer']['d2glr'],
      betas=(self.config['optimizer']['beta1'], self.config['optimizer']['beta2']))
    self.load()
    if config['distributed']:
      self.edgeG = DDP(self.edgeG, device_ids=[config['global_rank']], output_device=config['global_rank'], 
                      broadcast_buffers=True, find_unused_parameters=False)
      self.edgeD = DDP(self.edgeD, device_ids=[config['global_rank']], output_device=config['global_rank'], 
                      broadcast_buffers=True, find_unused_parameters=False)
      self.imgG = DDP(self.imgG, device_ids=[config['global_rank']], output_device=config['global_rank'], 
                      broadcast_buffers=True, find_unused_parameters=False)
      self.imgD = DDP(self.imgD, device_ids=[config['global_rank']], output_device=config['global_rank'], 
                      broadcast_buffers=True, find_unused_parameters=False)

  # get current learning rate
  def get_lr(self):
    return self.optimG.param_groups[0]['lr']

  # load netG and netD
  def load(self):
    model_path = self.config['save_dir']
    if os.path.isfile(os.path.join(model_path, 'latest.ckpt')):
      latest_epoch = open(os.path.join(model_path, 'latest.ckpt'), 'r').read().splitlines()[-1]
    else:
      ckpts = [os.path.basename(i).split('.pth')[0] for i in glob.glob(os.path.join(model_path, '*.pth'))]
      ckpts.sort()
      latest_epoch = ckpts[-1] if len(ckpts)>0 else None
    if latest_epoch is not None:
      gen_path = os.path.join(model_path, 'gen_{}.pth'.format(str(latest_epoch).zfill(5)))
      dis_path = os.path.join(model_path, 'dis_{}.pth'.format(str(latest_epoch).zfill(5)))
      if self.config['global_rank'] == 0:
        print('Loading model from {}...'.format(gen_path))
      data = torch.load(gen_path, map_location = lambda storage, loc: set_device(storage)) 
      self.edgeG.load_state_dict(data['edgeG'])
      self.imgG.load_state_dict(data['imgG'])
      data = torch.load(dis_path, map_location = lambda storage, loc: set_device(storage)) 
      self.optimG.load_state_dict(data['optimG'])
      self.optimD.load_state_dict(data['optimD'])
      self.edgeD.load_state_dict(data['edgeD'])
      self.imgD.load_state_dict(data['imgD'])
      self.epoch = data['epoch']
      self.iteration = data['iteration']
    else:
      if self.config['global_rank'] == 0:
        print('Warnning: There is no trained model found. An initialized model will be used.')

  # save parameters every eval_epoch
  def save(self, it):
    if self.config['global_rank'] == 0:
      gen_path = os.path.join(self.config['save_dir'], 'gen_{}.pth'.format(str(it).zfill(5)))
      dis_path = os.path.join(self.config['save_dir'], 'dis_{}.pth'.format(str(it).zfill(5)))
      print('\nsaving model to {} ...'.format(gen_path))
      if isinstance(self.edgeG, torch.nn.DataParallel) or isinstance(self.edgeG, DDP):
        edgeG, edgeD = self.edgeG.module, self.edgeD.module
        imgG, imgD = self.imgG.module, self.imgD.module
      else:
        edgeG, edgeD = self.edgeG, self.edgeD
        imgG, imgD = self.imgG, self.imgD
      torch.save({'edgeG': edgeG.state_dict(),
                  'imgG': imgG.state_dict()}, gen_path)
      torch.save({'epoch': self.epoch, 'iteration': self.iteration,
                  'edgeD': edgeD.state_dict(),
                  'imgD': imgD.state_dict(),
                  'optimG': self.optimG.state_dict(),
                  'optimD': self.optimD.state_dict()}, dis_path)
      os.system('echo {} > {}'.format(str(it).zfill(5), os.path.join(self.config['save_dir'], 'latest.ckpt')))

  def add_summary(self, writer, name, val):
    if writer is not None and self.iteration % 100 == 0:
      writer.add_scalar(name, val, self.iteration)

  # process input and calculate loss every training epoch
  def _train_epoch(self):
    progbar = Progbar(len(self.train_dataset), width=20, stateful_metrics=['epoch', 'iter'])
    mae = 0
    for images, image_gray, edges, masks, names in self.train_loader:
      self.iteration += 1
      end = time.time()
      images, image_gray, edges, masks = set_device([images, image_gray, edges, masks])
      # edge inference
      edges_masked = (edges * (1 - masks))
      gray_masked = (image_gray * (1 - masks)) + masks
      inputs = torch.cat((gray_masked, edges_masked, masks), dim=1)
      pred_edge = self.edgeG(inputs)                                    # in: [grayscale(1) + edge(1) + mask(1)]
      # image reference
      images_masked = (images * (1 - masks).float()) + masks
      comp_edge = pred_edge*masks + (1-masks)*edges
      inputs = torch.cat((images_masked, comp_edge), dim=1)
      pred_img = self.imgG(inputs)                                    # in: [rgb(3) + edge(1)]

      gen_loss = 0
      dis_loss = 0
      # edge discriminator loss
      dis_input_real = torch.cat((image_gray, edges), dim=1)
      dis_input_fake = torch.cat((image_gray, pred_edge.detach()), dim=1)
      dis_real, _ = self.edgeD(dis_input_real)        # in: (grayscale(1) + edge(1))
      dis_fake, _ = self.edgeD(dis_input_fake)        # in: (grayscale(1) + edge(1))
      dis_real_loss = self.adversarial_loss(dis_real, True, True)
      dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
      dis_loss += (dis_real_loss + dis_fake_loss) / 2
      # image discriminator loss
      dis_real, dis_real_feat = self.imgD(images)                    # in: [rgb(3)]
      dis_fake, _ = self.imgD(pred_img.detach())                    # in: [rgb(3)]
      dis_real_loss = self.adversarial_loss(dis_real, True, True)
      dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
      dis_loss += (dis_real_loss + dis_fake_loss) / 2
      self.optimD.zero_grad()
      dis_loss.backward()
      self.optimD.step()

      # generator adversarial loss
      gen_input_fake = torch.cat((image_gray, pred_edge), dim=1)
      gen_fake, gen_fake_feat = self.edgeD(gen_input_fake)        # in: (grayscale(1) + edge(1))
      gen_gan_loss = self.adversarial_loss(gen_fake, True, False)
      gen_loss += gen_gan_loss
      gen_fake, _ = self.imgD(pred_img)                    # in: [rgb(3)]
      gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.config['losses']['INPAINT_ADV_LOSS_WEIGHT']
      gen_loss += gen_gan_loss

      # generator feature matching loss
      gen_fm_loss = 0
      for i in range(len(dis_real_feat)):
        gen_fm_loss += self.l1_loss(gen_fake_feat[i], dis_real_feat[i].detach())
      gen_fm_loss = gen_fm_loss * self.config['losses']['FM_LOSS_WEIGHT']
      gen_loss += gen_fm_loss

      # generator l1 loss
      gen_l1_loss = self.l1_loss(pred_img, images) * self.config['losses']['L1_LOSS_WEIGHT'] / torch.mean(masks)
      gen_loss += gen_l1_loss

      # generator perceptual loss
      gen_content_loss = self.perceptual_loss(pred_img, images)
      gen_content_loss = gen_content_loss * self.config['losses']['CONTENT_LOSS_WEIGHT']
      gen_loss += gen_content_loss

      # generator style loss
      gen_style_loss = self.style_loss(pred_img * masks, images * masks)
      gen_style_loss = gen_style_loss * self.config['losses']['STYLE_LOSS_WEIGHT']
      gen_loss += gen_style_loss
      
      # logs
      new_mae = (torch.mean(torch.abs(images - pred_img)) / torch.mean(masks)).item()
      mae = new_mae if mae == 0 else (new_mae+mae)/2
      speed = images.size(0)/(time.time() - end)*self.config['world_size']
      logs = [("epoch", self.epoch),("iter", self.iteration),("lr", self.get_lr()),
        ('mae', mae), ('samples/s', speed)]
      if self.config['global_rank'] == 0:
        progbar.add(len(images)*self.config['world_size'], values=logs \
          if self.train_args['verbosity'] else [x for x in logs if not x[0].startswith('l_')])

      # saving and evaluating
      if self.iteration % self.train_args['save_freq'] == 0:
        self.save(self.iteration//self.train_args['save_freq'])
      if self.iteration % self.train_args['valid_freq'] == 0:
        self._eval_epoch(self.iteration//self.train_args['save_freq'])
        if self.config['global_rank'] == 0:
          print('[**] Training till {} in Rank {}\n'.format(
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), self.config['global_rank']))
      if self.iteration > self.config['trainer']['iterations']:
        break

  def _eval_epoch(self, it):
    self.valid_sampler.set_epoch(it)
    path = os.path.join(self.config['save_dir'], 'samples_{}'.format(str(it).zfill(5)))
    os.makedirs(path, exist_ok=True)
    if self.config['global_rank'] == 0:
      print('start evaluating ...')
    evaluation_scores = {key: 0 for key,val in self.metrics.items()}
    index = 0
    for images, image_gray, edges, masks, names in self.valid_loader:
      images, image_gray, edges, masks = set_device([images, image_gray, edges, masks])
      with torch.no_grad():
        # edge inference
        edges_masked = (edges * (1 - masks))
        gray_masked = (image_gray * (1 - masks)) + masks
        inputs = torch.cat((gray_masked, edges_masked, masks), dim=1)
        pred_edge = self.edgeG(inputs)                                    # in: [grayscale(1) + edge(1) + mask(1)]
        # image reference
        images_masked = (images * (1 - masks).float()) + masks
        comp_edge = pred_edge*masks + (1-masks)*edges
        inputs = torch.cat((images_masked, comp_edge), dim=1)
        pred_img = self.imgG(inputs)                                    # in: [rgb(3) + edge(1)]
        comp_img = pred_img*masks+(1-masks)*images
      grid_img = make_grid(torch.cat([images, images_masked, edges_masked, pred_edge, pred_img, comp_img], dim=0), nrow=6)
      save_image(grid_img, os.path.join(path, '{}_stack.png'.format(names[0].split('.')[0])))
      orig_img = postprocess(images)
      comp_img = postprocess(comp_img)
      Image.fromarray(orig_img[0]).save(os.path.join(path, '{}_orig.png'.format(names[0].split('.')[0])))
      Image.fromarray(comp_img[0]).save(os.path.join(path, '{}_comp.png'.format(names[0].split('.')[0])))
      for key, val in self.metrics.items():
        evaluation_scores[key] += val(orig_img, comp_img)
      index += 1
    for key, val in evaluation_scores.items():
      tensor = set_device(torch.FloatTensor([val/index]))
      dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
      evaluation_scores[key] = tensor.cpu().item()
    evaluation_message = ' '.join(['{}: {:5f},'.format(key, val/self.config['world_size']) \
                        for key,val in evaluation_scores.items()])
    if self.config['global_rank'] == 0:
      print('[**] Evaluation: {}'.format(evaluation_message))



  def train(self):
    while True:
      self.epoch += 1
      if self.config['distributed']:
        self.train_sampler.set_epoch(self.epoch)
      self._train_epoch()
      if self.iteration > self.config['trainer']['iterations']:
        break
    print('\nEnd training....')
  
