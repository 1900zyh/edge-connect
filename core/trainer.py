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
from core.model import InpaintGenerator, Discriminator

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

    # set up metrics
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

    # set models
    
    # generator input: [grayscale(1) + edge(1) + mask(1)]
    # discriminator input: (grayscale(1) + edge(1))
    generator = EdgeGenerator(use_spectral_norm=True)
    discriminator = Discriminator(in_channels=2, use_sigmoid=config.GAN_LOSS != 'hinge')
    
    # generator input: [rgb(3) + edge(1)]
    # discriminator input: [rgb(3)]
    generator = InpaintGenerator()
    discriminator = Discriminator(in_channels=3, use_sigmoid=config.GAN_LOSS != 'hinge')
        

    self.netG = set_device(InpaintGenerator())
    self.netD = set_device(Discriminator(in_channels=3, use_sigmoid=False))
    self.optimG = optim.Adam(filter(lambda p: p.requires_grad, self.netG.parameters()),
      lr=float(config['optimizer']['lr']), betas=(config['optimizer']['beta1'], config['optimizer']['beta2']))
    self.optimD = optim.Adam(params=self.netD.parameters(),
      lr=float(config['optimizer']['lr']), betas=(config['optimizer']['beta1'], config['optimizer']['beta2']))
    self.load()
    if config['distributed']:
      self.netG = DDP(self.netG, device_ids=[config['global_rank']], output_device=config['global_rank'], 
                      broadcast_buffers=True, find_unused_parameters=False)
      self.netD = DDP(self.netD, device_ids=[config['global_rank']], output_device=config['global_rank'], 
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
      self.netG.load_state_dict(data['netG'])
      data = torch.load(dis_path, map_location = lambda storage, loc: set_device(storage)) 
      self.optimG.load_state_dict(data['optimG'])
      self.optimD.load_state_dict(data['optimD'])
      self.netD.load_state_dict(data['netD'])
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
      if isinstance(self.netG, torch.nn.DataParallel) or isinstance(self.netG, DDP):
        netG, netD = self.netG.module, self.netD.module
      else:
        netG, netD = self.netG, self.netD
      torch.save({'netG': netG.state_dict()}, gen_path)
      torch.save({'epoch': self.epoch, 'iteration': self.iteration,
                  'netD': netD.state_dict(),
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
    for images, masks, names in self.train_loader:
      
        edges_masked = (edges * (1 - masks))
        images_masked = (images * (1 - masks)) + masks
        inputs = torch.cat((images_masked, edges_masked, masks), dim=1)
        outputs = self.generator(inputs)                                    # in: [grayscale(1) + edge(1) + mask(1)]

        # discriminator loss
        dis_input_real = torch.cat((images, edges), dim=1)
        dis_input_fake = torch.cat((images, outputs.detach()), dim=1)
        dis_real, dis_real_feat = self.discriminator(dis_input_real)        # in: (grayscale(1) + edge(1))
        dis_fake, dis_fake_feat = self.discriminator(dis_input_fake)        # in: (grayscale(1) + edge(1))
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2
        # generator adversarial loss
        gen_input_fake = torch.cat((images, outputs), dim=1)
        gen_fake, gen_fake_feat = self.discriminator(gen_input_fake)        # in: (grayscale(1) + edge(1))
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False)
        gen_loss += gen_gan_loss

        # process outputs
        images_masked = (images * (1 - masks).float()) + masks
        inputs = torch.cat((images_masked, edges), dim=1)
        outputs = self.generator(inputs)                                    # in: [rgb(3) + edge(1)]
        gen_loss = 0
        dis_loss = 0


        # discriminator loss
        dis_input_real = images
        dis_input_fake = outputs.detach()
        dis_real, _ = self.discriminator(dis_input_real)                    # in: [rgb(3)]
        dis_fake, _ = self.discriminator(dis_input_fake)                    # in: [rgb(3)]
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2


        # generator adversarial loss
        gen_input_fake = outputs
        gen_fake, _ = self.discriminator(gen_input_fake)                    # in: [rgb(3)]
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.config.INPAINT_ADV_LOSS_WEIGHT
        gen_loss += gen_gan_loss

        # generator l1 loss
        gen_l1_loss = self.l1_loss(outputs, images) * self.config.L1_LOSS_WEIGHT / torch.mean(masks)
        gen_loss += gen_l1_loss

        # generator perceptual loss
        gen_content_loss = self.perceptual_loss(outputs, images)
        gen_content_loss = gen_content_loss * self.config.CONTENT_LOSS_WEIGHT
        gen_loss += gen_content_loss

        # generator style loss
        gen_style_loss = self.style_loss(outputs * masks, images * masks)
        gen_style_loss = gen_style_loss * self.config.STYLE_LOSS_WEIGHT
        gen_loss += gen_style_loss


      # #######################
      # #### old version ######
      # #######################
      self.iteration += 1
      end = time.time()
      images, masks = set_device([images, masks])
      stage1, pred_img = self.netG(images, masks)
      complete_img = (pred_img * masks) + (images * (1. - masks))
      self.add_summary(self.gen_writer, 'lr/LR', self.optimG.param_groups[0]['lr'])
      self.add_summary(self.dis_writer, 'lr/LR', self.optimD.param_groups[0]['lr'])
      
      # discriminator loss
      dis_real = self.netD(images)
      dis_fake = self.netD(complete_img.detach())
      dis_loss = nn.ReLU()(1-dis_real).mean() + nn.ReLU()(1+dis_fake).mean()
      self.add_summary(self.dis_writer, 'loss/adv_loss', dis_loss.item()/2.)
      self.optimD.zero_grad()
      dis_loss.backward()
      self.optimD.step()

      # generator loss
      gen_l1_loss = nn.L1Loss()(pred_img, images) * self.config['losses']['L1_LOSS_WEIGHT'] / torch.mean(masks)
      pyramid_loss = 0
      for res in stage1:
        pyramid_loss +=  nn.L1Loss()(res, F.interpolate(images, size=res.size()[2:4], mode='nearest'))
      pyramid_loss *= self.config['losses']['PYRAMID_LOSS_WEIGHT']

      gen_fake = self.netD(complete_img)
      gen_adv_loss = -gen_fake.mean()
      gen_loss = gen_l1_loss + pyramid_loss + gen_adv_loss * self.config['losses']['ADV_LOSS_WEIGHT']
      self.optimG.zero_grad()
      gen_loss.backward()
      self.optimG.step()
      self.add_summary(self.gen_writer, 'loss/pyramid_loss', pyramid_loss.item())
      self.add_summary(self.gen_writer, 'loss/L1_loss', gen_l1_loss.item())
      self.add_summary(self.gen_writer, 'loss/adv_loss', gen_adv_loss.item())
  
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
    for images, masks, names in self.valid_loader:
      images, masks = set_device([images, masks])
      with torch.no_grad():
        _, prd_img = self.netG(images, masks)
      grid_img = make_grid(torch.cat([(images+1)/2, ((1-masks)*images+1)/2,
        (prd_img+1)/2, ((1-masks)*images+masks*prd_img+1)/2], dim=0), nrow=4)
      save_image(grid_img, os.path.join(path, '{}_stack.png'.format(names[0].split('.')[0])))
      orig_imgs = postprocess(images)
      comp_imgs = postprocess((1-masks)*images+masks*prd_img)
      Image.fromarray(orig_imgs[0]).save(os.path.join(path, '{}_orig.png'.format(names[0].split('.')[0])))
      Image.fromarray(comp_imgs[0]).save(os.path.join(path, '{}_comp.png'.format(names[0].split('.')[0])))
      for key, val in self.metrics.items():
        evaluation_scores[key] += val(orig_imgs, comp_imgs)
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
  
