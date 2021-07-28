import numpy as np
import os
from termcolor import cprint
import math
from sklearn.cluster import KMeans
import torch
import matplotlib
import pdb
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.append('.')
from train import ssltrainer
from model import FeatMatch
from loss import common
from util import misc, metric
from util.command_interface import command_interface
from util.reporter import Reporter
from pathlib import Path
class Get_Scalar:
    def __init__(self, value):
        self.value = value

    def get_value(self, iter):
        return self.value

    def __call__(self, iter):
        return self.value

class FeatMatchTrainer(ssltrainer.SSLTrainer):
    def __init__(self, args, config):
        super().__init__(args, config)
        self.fu, self.pu = [], []
        self.fp, self.yp, self.lp = None, None, None
        self.criterion = getattr(common, self.config['loss']['criterion'])
        self.t_fn = Get_Scalar(args.temperature) #(default: 0.5)
        self.p_fn= Get_Scalar(args.p_cutoff)
        if self.config['loss']['hard_labels'] == "yes":
            self.hard_labels = True
        elif self.config['loss']['hard_labels'] == "no":
            self.hard_labels = False

        self.criterion = getattr(common, self.config['loss']['criterion'])
        self.attr_objs.extend(['fu', 'pu', 'fp', 'yp', 'lp'])
        self.load(args.mode)
        self.mode = args.mode

    def init_model(self):
        model = FeatMatch(backbone=self.config['model']['backbone'],
                          num_classes=self.config['model']['classes'],
                          devices=self.args.devices,
                          num_heads=self.config['model']['num_heads'],
                          amp=self.args.amp,
                          attention = self.config['model']['attention'], #config added
                          d_model = self.config['model']['d_model'],
                          label_prop = self.config['model']['label_prop'],
                          detach = self.config['model']['detach'],
                          scaled = self.config['model']['scaled'],
                          mode = self.args.mode
                          )
        print(f'Use [{self.config["model"]["backbone"]}] model with [{misc.count_n_parameters(model):,}] parameters')
        return model

    def data_mixup(self, xl, prob_xl, xu, prob_xu, alpha=0.75):
        Nl = len(xl)

        x = torch.cat([xl, xu], dim=0)
        prob = torch.cat([prob_xl, prob_xu], dim=0).detach() #(1728, 10)
        idx = torch.randperm(x.shape[0])
        x_, prob_ = x[idx], prob[idx]  #(1728, 10), (1728, 10)
        l = np.random.beta(alpha, alpha)
        l = max(l, 1 - l)
        x = l * x + (1 - l) * x_
        prob = l * prob + (1 - l) * prob_
        prob = prob / prob.sum(dim=1, keepdim=True)

        xl, xu = x[:Nl], x[Nl:]
        probl, probu = prob[:Nl], prob[Nl:]

        return xl, probl, xu, probu

    def train1(self, xl, yl, xu):
        # Forward pass on original data
        bsl, bsu, k, c = len(xl), len(xu), xl.size(1), self.config['model']['classes']
        x = torch.cat([xl, xu], dim=0).reshape(-1, *xl.shape[2:])
        # ((bsl + bsu) x (k+1) , 3, 32, 32) ex. (1728, 3 , 32, 32)
        # print("pretraining_stage_input_shape:", x.shape)
        logits_x = self.model(x)
        # ((bsl + bsu) x (k+1) , 10) ex.(1728, 10)
        # print("pretraining_stage_output_shape:", logits_x.shape)
        logits_x = logits_x.reshape(bsl + bsu, k, c)
        # ((bsl + bsu) x (k+1) , 10) ex.(192, 9, 10)
        # print("pretraining_stage_output_shape:", logits_x.shape)
        logits_xl, logits_xu = logits_x[:bsl], logits_x[bsl:]
        # ex. (64, 9, 10) , (128, 9, 10)

        # Compute pseudo label
        prob_xu_fake = torch.softmax(logits_xu[:, 0].detach(), dim=1)
        # ex. (128, 10)
        #temperature smaller, difference between softmax is bigger
        # temperature bigger, difference between softmax is smaller
        prob_xu_fake = prob_xu_fake ** (1. / self.config['transform']['data_augment']['T'])
        prob_xu_fake = prob_xu_fake / prob_xu_fake.sum(dim=1, keepdim=True) #(128,10)/(128,1)
        #(128,10)
        prob_xu_fake = prob_xu_fake.unsqueeze(1).repeat(1, k, 1)
        #(128, 9, 10)
        # Mixup perturbation
        xu = xu.reshape(-1, *xu.shape[2:]) #(576, 3, 32, 32)
        xl = xl.reshape(-1, *xl.shape[2:]) #(1152, 3, 32, 32)
        prob_xl_gt = torch.zeros(len(xl), c, device=xl.device) #(576, 10)
        prob_xl_gt.scatter_(dim=1, index=yl.unsqueeze(1).repeat(1, k).reshape(-1, 1), value=1.)
        # index (64, 1) -> (64, 9) -> (576, 1). every dim=0,
        xl_mix, probl_mix, xu_mix, probu_mix = self.data_mixup(xl, prob_xl_gt, xu, prob_xu_fake.reshape(-1, c))
        #(in) (576, 3, 32, 32), (576,1), (1152, 3, 32, 32), (1152, 1)

        # Forward pass on mixed data
        Nl = len(xl_mix)
        x_mix = torch.cat([xl_mix, xu_mix], dim=0)
        logits_x_mix = self.model(x_mix)
        logits_xl_mix, logits_xu_mix = logits_x_mix[:Nl], logits_x_mix[Nl:]

        # CLF loss
        loss_pred = self.criterion(None, probl_mix, logits_xl_mix, None)

        # Mixup loss
        loss_con = self.criterion(None, probu_mix, logits_xu_mix, None)

        # Graph loss
        loss_graph = torch.tensor(0.0, device=self.default_device)

        # Total loss
        coeff = self.get_consistency_coeff()
        loss = loss_pred + coeff*self.config['loss']['mix']*loss_con

        # Prediction
        pred_x = torch.softmax(logits_xl[:, 0].detach(), dim=1)

        return pred_x, loss, loss_pred, loss_con, loss_graph

    def train1_wo_mixup(self, xl, yl, xu):
        # Forward pass on original data
        bsl, bsu, k, c = len(xl), len(xu), xl.size(1), self.config['model']['classes']
        x = torch.cat([xl, xu], dim=0).reshape(-1, *xl.shape[2:])
        # ((bsl + bsu) x (k+1) , 3, 32, 32) ex. (1728, 3 , 32, 32)
        logits_x = self.model(x)
        # ((bsl + bsu) x (k+1) , 10) ex.(1728, 10)
        logits_x = logits_x.reshape(bsl + bsu, k, c)
        # ((bsl + bsu) x (k+1) , 10) ex.(192, 9, 10)
        logits_xl, logits_xu = logits_x[:bsl], logits_x[bsl:]

        # ex. (64, 9, 10) , (128, 9, 10)

        # Compute pseudo label
        prob_xu_fake = torch.softmax(logits_xu[:, 0].detach(), dim=1)
        # ex. (128, 10)
        # temperature smaller, difference between softmax is bigger
        # temperature bigger, difference between softmax is smaller
        prob_xu_fake = prob_xu_fake ** (1. / self.config['transform']['data_augment']['T'])
        prob_xu_fake = prob_xu_fake / prob_xu_fake.sum(dim=1, keepdim=True)  # (128,10)/(128,1)
        # (128,10)
        prob_xu_fake = prob_xu_fake.unsqueeze(1).repeat(1, k, 1) #(128, 9, 10)
        prob_xu_fake = prob_xu_fake.reshape(-1, c)
        # (128, 9, 10)
        # Mixup perturbation
        xu = xu.reshape(-1, *xu.shape[2:])  # (576, 3, 32, 32)
        xl = xl.reshape(-1, *xl.shape[2:])  # (1152, 3, 32, 32)
        prob_xl_gt = torch.zeros(len(xl), c, device=xl.device)  # (576, 10)
        prob_xl_gt.scatter_(dim=1, index=yl.unsqueeze(1).repeat(1, k).reshape(-1, 1), value=1.)

        loss_pred = self.criterion(None, prob_xl_gt, logits_xl.reshape(-1,c), None)
        # Con_f loss
        loss_con = self.criterion(None, prob_xu_fake, logits_xu.reshape(-1,c), None)
        # Graph loss
        loss_graph = torch.tensor(0.0, device=self.default_device)

        # Total loss
        coeff = self.get_consistency_coeff()
        loss = loss_pred + coeff * self.config['loss']['mix'] * loss_con

        # Prediction
        pred_x = torch.softmax(logits_xl[:, 0].detach(), dim=1)

        return pred_x, loss, loss_pred, loss_con, loss_graph

    def train2_w_mixup(self, xl, yl, xu):
        bsl, bsu, k, c = len(xl), len(xu), xl.size(1), self.config['model']['classes']
        x = torch.cat([xl, xu], dim=0).reshape(-1, *xl.shape[2:])
        fx = self.model.extract_feature(x)
        logits_x = self.model.cls(fx)
        logits_x = logits_x.reshape(bsl + bsu, k, c)
        logits_xl, logits_xu = logits_x[:bsl], logits_x[bsl:]
        # Compute pseudo label(xfu)
        prob_xu_fake = torch.softmax(logits_xu[:, 0].detach(), dim=1)
        prob_xu_fake = prob_xu_fake ** (1. / self.config['transform']['data_augment']['T'])
        prob_xu_fake = prob_xu_fake / prob_xu_fake.sum(dim=1, keepdim=True)
        prob_xu_fake = prob_xu_fake.unsqueeze(1).repeat(1, k, 1)
        #compute label(xfl)
        prob_xl_gt = torch.zeros(len(xl), c, device=xl.device) #(576, 10)
        prob_xl_gt.scatter_(dim=1, index=yl.unsqueeze(1).repeat(1, k).reshape(-1, 1), value=1.)

        #img_for_mixup
        xu = xu.reshape(-1, *xu.shape[2:]) #(576, 3, 32, 32)
        xl = xl.reshape(-1, *xl.shape[2:]) #(1152, 3, 32, 32)
        pdb.set_trace()

        xl_mix, probl_mix, xu_mix, probu_mix = self.data_mixup(xl, prob_xl_gt, xu, prob_xu_fake.reshape(-1, c))

        # Forward pass on mixed data
        Nl = len(xl_mix)
        x_mix = torch.cat([xl_mix, xu_mix], dim=0)
        prob_mix = torch.cat([xl_mix, xu_mix], dim=0)
        logits_xg_mix, logits_xf_mix, fx, fxg = self.model(x_mix, prob_mix)

        logits_xg_mix = logits_xg_mix.reshape(bsl + bsu, k, c)
        logits_xf_mix = logits_xf_mix.reshape(bsl + bsu, k, c)

        logits_xgl_mix, logits_xgu_mix = logits_xg_mix[:bsl], logits_xg_mix[bsl:]
        logits_xfl_mix, logits_xfu_mix = logits_xf_mix[:bsl], logits_xf_mix[bsl:]

        pdb.set_trace()

        # CLF1 loss
        loss_pred = self.criterion(None, probl_mix, logits_xgl_mix.reshape(-1,c), None)

        # Mixup loss
        loss_con = self.criterion(None, probu_mix, logits_xgu_mix.reshape(-1,c), None)

        # Graph loss
        # loss_graph = self.criterion(None, probu_mix, logits_xfu_mix.reshape(-1,c), None)

        # Total loss
        coeff = self.get_consistency_coeff()
        loss = loss_pred + coeff * (self.config['loss']['mix'] * loss_con + self.config['loss']['graph'] * loss_graph)

        # Prediction
        pred_x = torch.softmax(logits_xgl[:, 0].detach(), dim=1)

        return pred_x, loss, loss_pred, loss_con, loss_graph

    def train2(self, xl, yl, xu):
        bsl, bsu, k, c = len(xl), len(xu), xl.size(1), self.config['model']['classes']
        x = torch.cat([xl, xu], dim=0).reshape(-1, *xl.shape[2:])
        logits_xg, logits_xf, fx, fxg  = self.model(x, self.fp)
        logits_xg = logits_xg.reshape(bsl + bsu, k, c)
        logits_xf = logits_xf.reshape(bsl + bsu, k, c)

        logits_xgl, logits_xgu = logits_xg[:bsl], logits_xg[bsl:]
        logits_xfl, logits_xfu = logits_xf[:bsl], logits_xf[bsl:]

        # Compute pseudo label(g)
        prob_xgu_fake = torch.softmax(logits_xgu[:, 0].detach(), dim=1)
        prob_xgu_fake = prob_xgu_fake ** (1. / self.config['transform']['data_augment']['T'])
        prob_xgu_fake = prob_xgu_fake / prob_xgu_fake.sum(dim=1, keepdim=True)
        prob_xgu_fake = prob_xgu_fake.unsqueeze(1).repeat(1, k, 1)
        # Compute pseudo label(f)
        prob_xfu_fake = torch.softmax(logits_xfu[:, 0].detach(), dim=1)
        prob_xfu_fake = prob_xfu_fake ** (1. / self.config['transform']['data_augment']['T'])
        prob_xfu_fake = prob_xfu_fake / prob_xfu_fake.sum(dim=1, keepdim=True)
        prob_xfu_fake = prob_xfu_fake.unsqueeze(1).repeat(1, k, 1)
        pdb.set_trace()

        # Mixup perturbation
        xu = xu.reshape(-1, *xu.shape[2:])
        xl = xl.reshape(-1, *xl.shape[2:])
        prob_xl_gt = torch.zeros(len(xl), c, device=xl.device)
        prob_xl_gt.scatter_(dim=1, index=yl.unsqueeze(1).repeat(1, k).reshape(-1, 1), value=1.)
        xl_mix, probl_mix, xu_mix, probu_mix = self.data_mixup(xl, prob_xl_gt, xu, prob_xfu_fake.reshape(-1, c))

        # Forward pass on mixed data
        Nl = len(xl_mix)
        x_mix = torch.cat([xl_mix, xu_mix], dim=0)
        logits_xg_mix, logits_xf_mix, _, _, _ = self.model(x_mix, self.fp)
        logits_xgl_mix, logits_xgu_mix = logits_xg_mix[:Nl], logits_xg_mix[Nl:]
        logits_xfl_mix, logits_xfu_mix = logits_xf_mix[:Nl], logits_xf_mix[Nl:]

        # CLF1 loss
        loss_pred = self.criterion(None, probl_mix, logits_xgl_mix, None)

        # Mixup loss
        loss_con = self.criterion(None, probu_mix, logits_xgu_mix, None)

        # Graph loss
        loss_graph = self.criterion(None, probu_mix, logits_xfu_mix, None)

        # Total loss
        coeff = self.get_consistency_coeff()
        loss = loss_pred + coeff * (self.config['loss']['mix'] * loss_con + self.config['loss']['graph'] * loss_graph)

        # Prediction
        pred_x = torch.softmax(logits_xgl[:, 0].detach(), dim=1)

        return pred_x, loss, loss_pred, loss_con, loss_graph

    def train2_wo_mixup(self, xl, yl, xu, T, p_cutoff):
        bsl, bsu, k, c = len(xl), len(xu), xl.size(1), self.config['model']['classes']
        x = torch.cat([xl, xu], dim=0).reshape(-1, *xl.shape[2:])
        logits_xg, logits_xf, fx, fxg = self.model(x)
        logits_xg = logits_xg.reshape(bsl + bsu, k, c)
        logits_xf = logits_xf.reshape(bsl + bsu, k, c)

        logits_xgl, logits_xgu = logits_xg[:bsl], logits_xg[bsl:]
        logits_xfl, logits_xfu = logits_xf[:bsl], logits_xf[bsl:]

        xl = xl.reshape(-1, *xl.shape[2:])
        prob_xl_gt = torch.zeros(len(xl), c, device=xl.device)
        prob_xl_gt.scatter_(dim=1, index=yl.unsqueeze(1).repeat(1, k).reshape(-1, 1), value=1.)
        loss_pred = self.criterion(None, prob_xl_gt, logits_xgl.reshape(-1,c), None)

        # Compute pseudo label(g)
        prob_xgu_weak = torch.softmax(logits_xgu[:, 0].detach(), dim=1)
        # prob_xgu_fake : torch.Size([128, 10])

        # reference: https://github.com/LeeDoYup/FixMatch-pytorch/blob/0f22e7f7c63396e0a0839977ba8101f0d7bf1b04/models/fixmatch/fixmatch_utils.py
        if self.hard_labels:
            max_probs, max_idx = torch.max(prob_xgu_weak, dim=1)  # bu

            mask = max_probs.ge(p_cutoff).float()  # bu

            prob_xgu_fake = torch.zeros(bsu * k, c, device=xl.device)
            # prob_xgu_fake.shape : torch.Size([576, 10])
            prob_xgu_fake.scatter_(dim=1, index=max_idx.unsqueeze(1).repeat(1, k).reshape(-1, 1), value=1.)
            # logits_s(strong_logits), max_idx(pseudo_label)
            loss_con = self.criterion(None, logits_xgu.reshape(-1, c), prob_xgu_fake, mask)
            loss_graph = self.criterion(None, logits_xfu.reshape(-1, c), prob_xgu_fake, mask)
        else:
            # prob_xgu_fake = prob_xgu_fake ** (1. / self.config['transform']['data_augment']['T'])
            prob_xgu_weak = prob_xgu_weak ** (1. / T)
            prob_xgu_weak = prob_xgu_weak / prob_xgu_weak.sum(dim=1, keepdim=True)
            prob_xgu_fake = prob_xgu_weak.unsqueeze(1).repeat(1, k, 1)
            loss_con = self.criterion(None, prob_xgu_fake, logits_xgu.reshape(-1, c), None)
            loss_graph = self.criterion(None, prob_xgu_fake, logits_xfu.reshape(-1, c), None)

        # Compute pseudo label(f)
        # prob_xfu_fake = prob_xfu_fake ** (1. / self.config['transform']['data_augment']['T'])
        # prob_xfu_fake = prob_xfu_fake / prob_xfu_fake.sum(dim=1, keepdim=True)
        # prob_xfu_fake = prob_xfu_fake.unsqueeze(1).repeat(1, k, 1)
        # pdb.set_trace()

        # Mixup perturbation
        # xu = xu.reshape(-1, *xu.shape[2:])
        # Total loss
        coeff = self.get_consistency_coeff()
        loss = loss_pred + coeff * (self.config['loss']['mix'] * loss_con + self.config['loss']['graph'] * loss_graph)

        # Prediction
        pred_xg = torch.softmax(logits_xgl[:, 0].detach(), dim=1)
        pred_xf = torch.softmax(logits_xfl[:, 0].detach(), dim=1)

        return pred_xg, pred_xf, loss, loss_pred, loss_con, loss_graph

    def finetune_wo_mixup(self, xl, yl, xu, T, p_cutoff):
        bsl, bsu, k, c = len(xl), len(xu), xl.size(1), self.config['model']['classes']
        x = torch.cat([xl, xu], dim=0).reshape(-1, *xl.shape[2:])
        logits_xg, logits_xf, fx, fxg = self.model(x)

        logits_xg = logits_xg.reshape(bsl + bsu, k, c)
        logits_xf = logits_xf.reshape(bsl + bsu, k, c)
        # (common) logit.shape : torch.Size([192, 9, 10])

        logits_xgl, logits_xgu = logits_xg[:bsl], logits_xg[bsl:]
        logits_xfl, logits_xfu = logits_xf[:bsl], logits_xf[bsl:]
        # (common) logits_xgl, logits_xfl : torch.Size([64, 9, 10])
        # (common) logits_xgu, logits_xfu : torch.Size([128, 9, 10])

        xl = xl.reshape(-1, *xl.shape[2:])
        prob_xl_gt = torch.zeros(len(xl), c, device=xl.device)
        prob_xl_gt.scatter_(dim=1, index=yl.unsqueeze(1).repeat(1, k).reshape(-1, 1), value=1.)
        # prob_xl_gt.shape : torch.Size([576, 10])
        # CLF1 loss
        loss_pred = self.criterion(None, prob_xl_gt, logits_xgl.reshape(-1,c), None)

        # Compute pseudo label(g)
        prob_xgu_weak = torch.softmax(logits_xgu[:, 0].detach(), dim=1)
        # prob_xgu_fake : torch.Size([128, 10])

        #reference: https://github.com/LeeDoYup/FixMatch-pytorch/blob/0f22e7f7c63396e0a0839977ba8101f0d7bf1b04/models/fixmatch/fixmatch_utils.py
        if self.hard_labels:
            max_probs, max_idx = torch.max(prob_xgu_weak, dim=1) #bu

            mask = max_probs.ge(p_cutoff).float() #bu

            prob_xgu_fake = torch.zeros(bsu*k, c, device=xl.device)
            # prob_xgu_fake.shape : torch.Size([576, 10])
            prob_xgu_fake.scatter_(dim=1, index=max_idx.unsqueeze(1).repeat(1, k).reshape(-1, 1), value=1.)
            #logits_s(strong_logits), max_idx(pseudo_label)
            loss_con = self.criterion(None, logits_xgu.reshape(-1,c), prob_xgu_fake, mask)
            loss_graph = self.criterion(None, logits_xfu.reshape(-1,c), prob_xgu_fake, mask)
        else:
            # prob_xgu_fake = prob_xgu_fake ** (1. / self.config['transform']['data_augment']['T'])
            prob_xgu_weak = prob_xgu_weak ** (1. / T)
            prob_xgu_weak = prob_xgu_weak / prob_xgu_weak.sum(dim=1, keepdim=True)
            prob_xgu_fake = prob_xgu_weak.unsqueeze(1).repeat(1, k, 1)
            loss_con = self.criterion(None, prob_xgu_fake, logits_xgu.reshape(-1,c), None)
            loss_graph = self.criterion(None, prob_xgu_fake, logits_xfu.reshape(-1,c), None)

        # Compute pseudo label(f)
        # prob_xfu_fake = prob_xfu_fake ** (1. / self.config['transform']['data_augment']['T'])
        # prob_xfu_fake = prob_xfu_fake / prob_xfu_fake.sum(dim=1, keepdim=True)
        # prob_xfu_fake = prob_xfu_fake.unsqueeze(1).repeat(1, k, 1)
        # pdb.set_trace()

        # Mixup perturbation
        # xu = xu.reshape(-1, *xu.shape[2:])
        # Total loss
        coeff = self.get_consistency_coeff()
        loss = loss_pred + coeff * (self.config['loss']['mix'] * loss_con + self.config['loss']['graph'] * loss_graph)

        # Prediction
        pred_xg = torch.softmax(logits_xgl[:, 0].detach(), dim=1)
        pred_xf = torch.softmax(logits_xfl[:, 0].detach(), dim=1)

        return pred_xg, pred_xf, loss, loss_pred, loss_con, loss_graph


    def eval1(self, x, y):
        logits_x = self.model(x)

        # Compute pseudo label
        prob_fake = torch.softmax(logits_x.detach(), dim=1)
        prob_fake = prob_fake ** (1. / self.config['transform']['data_augment']['T'])
        prob_fake = prob_fake / prob_fake.sum(dim=1, keepdim=True)

        # Mixup perturbation
        prob_gt = torch.zeros(len(y), self.config['model']['classes'], device=x.device)
        prob_gt.scatter_(dim=1, index=y.unsqueeze(1), value=1.)
        x_mix, prob_mix, _, _ = self.data_mixup(x, prob_gt, x, prob_fake)

        # Forward pass on mixed data
        logits_x_mix = self.model(x_mix)

        # CLF loss and Mixup loss
        loss_con = loss_pred = self.criterion(None, prob_mix, logits_x_mix, None)

        # Graph loss
        loss_graph = torch.tensor(0.0, device=self.default_device)

        # Total loss
        coeff = self.get_consistency_coeff()
        loss = loss_pred + coeff*self.config['loss']['mix']*loss_con

        # Prediction
        pred_x = torch.softmax(logits_x.detach(), dim=1)

        return pred_x, loss, loss_pred, loss_con, loss_graph

    def eval1_wo_mixup(self, x, y):
        logits_x = self.model(x)

        # Compute pseudo label
        prob_fake = torch.softmax(logits_x.detach(), dim=1)
        prob_fake = prob_fake ** (1. / self.config['transform']['data_augment']['T'])
        prob_fake = prob_fake / prob_fake.sum(dim=1, keepdim=True)

        # CLF loss
        loss_con = loss_pred = self.criterion(None, prob_fake, logits_x, None)

        # Graph loss
        loss_graph = torch.tensor(0.0, device=self.default_device)

        # Total loss
        coeff = self.get_consistency_coeff()
        loss = loss_pred + coeff*self.config['loss']['mix']*loss_con

        # Prediction
        pred_x = torch.softmax(logits_x.detach(), dim=1)

        return pred_x, loss, loss_pred, loss_con, loss_graph

    def eval2(self, x, y):
        logits_xg, logits_xf, _, _, _ = self.model(x, self.fp)

        # Compute pseudo label
        prob_fake = torch.softmax(logits_xg.detach(), dim=1)
        prob_fake = prob_fake ** (1. / self.config['transform']['data_augment']['T'])
        prob_fake = prob_fake / prob_fake.sum(dim=1, keepdim=True)

        # Mixup perturbation
        prob_gt = torch.zeros(len(y), self.config['model']['classes'], device=x.device)
        prob_gt.scatter_(dim=1, index=y.unsqueeze(1), value=1.)
        x_mix, prob_mix, _, _ = self.data_mixup(x, prob_gt, x, prob_fake)

        # Forward pass on mixed data
        logits_xg_mix, logits_xf_mix, _, _, _ = self.model(x_mix, self.fp)

        # CLF loss and Mixup loss
        loss_con = loss_pred = self.criterion(None, prob_mix, logits_xg_mix, None)

        # Graph loss
        loss_graph = self.criterion(None, prob_mix, logits_xf_mix, None)

        # Total loss
        coeff = self.get_consistency_coeff()
        loss = loss_pred + coeff*(self.config['loss']['mix']*loss_con + self.config['loss']['graph']*loss_graph)

        # Prediction
        pred_xf = torch.softmax(logits_xf.detach(), dim=1)
        pred_xg = torch.softmax(logits_xg.detach(), dim=1)

        return pred_xg, pred_xf, loss, loss_pred, loss_con, loss_graph

    def eval2_wo_mixup(self, x, y):
        logits_xg, logits_xf, fx, fxg = self.model(x)

        # Compute pseudo label
        prob_fake = torch.softmax(logits_xg.detach(), dim=1)
        prob_fake = prob_fake ** (1. / self.config['transform']['data_augment']['T'])
        prob_fake = prob_fake / prob_fake.sum(dim=1, keepdim=True)


        # Forward pass on mixed data
        loss_con = loss_pred = self.criterion(None, prob_fake, logits_xg, None)

        # Con_f loss
        loss_graph = self.criterion(None, prob_fake, logits_xf, None)

        # Total loss
        coeff = self.get_consistency_coeff()
        loss = loss_pred + coeff*(self.config['loss']['mix']*loss_con + self.config['loss']['graph']*loss_graph)

        # Prediction
        pred_xg = torch.softmax(logits_xg.detach(), dim=1)
        pred_xf = torch.softmax(logits_xf.detach(), dim=1)
        return pred_xg, pred_xf, loss, loss_pred, loss_con, loss_graph

    def forward_train(self, data):
        self.model.train()
        xl = data[0].reshape(-1, *data[0].shape[2:])
        xl = self.Tnorm(xl.to(self.default_device)).reshape(data[0].shape)
        yl = data[1].to(self.default_device)
        xu = data[2].reshape(-1, *data[2].shape[2:])
        xu = self.Tnorm(xu.to(self.default_device)).reshape(data[2].shape)
        #fast debugging

        #hyper_params for update
        T = self.t_fn(self.curr_iter)
        p_cutoff = self.p_fn(self.curr_iter)
        #(train1: T, p_cutoff revision is requiring)
        #for debuging training stage#
        if self.mode != 'finetune':
            if self.config['model']['attention'] == "no":
                self.model.set_mode('pretrain')
                if self.config['model']['mixup'] == 'no':
                    pred_xf, loss, loss_pred, loss_con, loss_graph = self.train1_wo_mixup(xl, yl, xu)
                elif self.config['model']['mixup'] == 'yes':
                    pred_xf, loss, loss_pred, loss_con, loss_graph = self.train1(xl, yl, xu)
                pred_xg = torch.tensor(0.0, device=self.default_device)

            elif self.config['model']['attention'] == "Transformer":
                if self.curr_iter < self.config['train']['pretrain_iters']:
                    self.model.set_mode('pretrain')
                    if self.config['model']['mixup'] =='yes':
                        pred_xf, loss, loss_pred, loss_con, loss_graph = self.train1(xl, yl, xu)
                    elif self.config['model']['mixup'] =='no':
                        pred_xf, loss, loss_pred, loss_con, loss_graph = self.train1_wo_mixup(xl, yl, xu)
                    pred_xg = torch.tensor(0.0, device=self.default_device)
                    #pred_x, (x1_weak_label's softmax)
                    # loss, (total loss)
                    # loss_pred, (log loss for xl_mixup) loss_con, (log loss for xu_mixup),
                    # loss_graph = 0.0
                elif self.curr_iter == self.config['train']['pretrain_iters']:
                    self.model.set_mode('train')
                    if self.config['model']['mixup'] == 'yes':
                        # self.extract_fp()
                        pred_xg, pred_xf, loss, loss_pred, loss_con, loss_graph = self.train2(xl, yl, xu)
                    elif self.config['model']['mixup'] == 'no':
                        pred_xg, pred_xf, loss, loss_pred, loss_con, loss_graph = self.train2_wo_mixup(xl, yl, xu, T, p_cutoff)
                else:
                    self.model.set_mode('train')
                    # if self.curr_iter % self.config['train']['sample_interval'] == 0:
                    if self.config['model']['mixup'] == 'yes':
                        # self.extract_fp()
                        pred_xg, pred_xf, loss, loss_pred, loss_con, loss_graph = self.train2(xl, yl, xu)
                    elif self.config['model']['mixup'] == 'no':
                        print("train2_wo_mixup")
                        pred_xg, pred_xf, loss, loss_pred, loss_con, loss_graph = self. train2_wo_mixup(xl, yl, xu,T, p_cutoff)
        else:
            self.model.set_mode('finetune')
            if self.config['model']['mixup'] == 'no':
                print("finetune_wo_mixup")
                # xl (bs, k, c, h, w)
                # yl (bs)
                # xu (bu, k, c, h, w)
                pred_xg, pred_xf, loss, loss_pred, loss_con, loss_graph = self.finetune_wo_mixup(xl, yl, xu, T,
                                                                                                 p_cutoff)

        results = {
            'y_pred': torch.max(pred_xf, dim=1)[1].detach().cpu().numpy(),
            'y_pred_agg': torch.max(pred_xg, dim=1)[1].detach().cpu().numpy(),
            'y_true': yl.cpu().numpy(),
            'loss': {
                'all': loss.detach().cpu().item(),
                'pred': loss_pred.detach().cpu().item(),
                'con': loss_con.detach().cpu().item(),
                'graph': loss_graph.detach().cpu().item()
            },
            'loss_hyper_params': {
                'temperature': np.array([T]).item(),
                'p_cutoff': np.array([p_cutoff]).item()
            }
        }

        return loss, results

    def forward_finetune(self, data):
        self.model.train()
        xl = data[0].reshape(-1, *data[0].shape[2:])
        pdb.set_trace()
        xl = self.Tnorm(xl.to(self.default_device)).reshape(data[0].shape)

        yl = data[1].to(self.default_device)
        xu = data[2].reshape(-1, *data[2].shape[2:])
        xu = self.Tnorm(xu.to(self.default_device)).reshape(data[2].shape)
        #fast debugging

        #hyper_params for update
        T = self.t_fn(self.curr_iter)
        p_cutoff = self.p_fn(self.curr_iter)
        #(train1: T, p_cutoff revision is requiring)
        #for debuging training stage#
        self.model.set_mode('finetune')
        if self.config['model']['mixup'] == 'no':
            print("finetune_wo_mixup")
            #xl (bs, k, c, h, w)
            #yl (bs)
            #xu (bu, k, c, h, w)
            pred_xg, pred_xf, loss, loss_pred, loss_con, loss_graph = self.finetune_wo_mixup(xl, yl, xu,T, p_cutoff)
        results = {
            'y_pred': torch.max(pred_xf, dim=1)[1].detach().cpu().numpy(),
            'y_pred_agg': torch.max(pred_xg, dim=1)[1].detach().cpu().numpy(),
            'y_true': yl.cpu().numpy(),
            'loss': {
                'all': loss.detach().cpu().item(),
                'pred': loss_pred.detach().cpu().item(),
                'con': loss_con.detach().cpu().item(),
                'graph': loss_graph.detach().cpu().item()
            },
            'loss_hyper_params': {
                'temperature': np.array([T]).item(),
                'p_cutoff': np.array([p_cutoff]).item()
            }
        }

        return loss, results

    def forward_eval(self, data):
        self.model.eval()
        x = self.Tnorm(data[0].to(self.default_device))
        y = data[1].to(self.default_device)
        if self.config['model']['attention'] == "no":
            self.model.set_mode('pretrain')
            pred_xf, loss, loss_pred, loss_con, loss_graph = self.eval1_wo_mixup(x, y)
            pred_xg = torch.tensor(0.0, device=self.default_device)

        elif self.config['model']['attention'] == "Transformer":
            if self.curr_iter < self.config['train']['pretrain_iters']:
                self.model.set_mode('pretrain')
                if self.config['model']['mixup'] =='yes':
                    pred_xf, loss, loss_pred, loss_con, loss_graph = self.eval1(x, y)
                elif self.config['model']['mixup'] =='no':
                    pred_xf, loss, loss_pred, loss_con, loss_graph = self.eval1_wo_mixup(x, y)
                pred_xg = torch.tensor(0.0, device=self.default_device)
            else:
                if self.config['model']['mixup'] == 'yes':
                    pred_xg, pred_xf, loss, loss_pred, loss_con, loss_graph = self.eval2(x,y)
                elif self.config['model']['mixup'] == 'no':
                    pred_xg, pred_xf, loss, loss_pred, loss_con, loss_graph = self.eval2_wo_mixup(x, y)
        results = {
            'y_pred': torch.max(pred_xf, dim=1)[1].detach().cpu().numpy(),
            'y_pred_agg': torch.max(pred_xg, dim=1)[1].detach().cpu().numpy(),
            'y_true': y.cpu().numpy(),
            'loss': {
                'all': loss.detach().cpu().item(),
                'pred': loss_pred.detach().cpu().item(),
                'con': loss_con.detach().cpu().item(),
                'graph': loss_graph.detach().cpu().item()
                    }
                }
        return results


if __name__ == '__main__':
    args, config, save_root = command_interface()

    r = args.rand_seed
    reporter = Reporter(save_root, args)

    for i in range(args.iters):
        args.rand_seed = r + i
        cprint(f'Run iteration [{i+1}/{args.iters}] with random seed [{args.rand_seed}]', attrs=['bold'])

        setattr(args, 'save_root', save_root/f'run{i}')
        if args.mode == 'resume' and not args.save_root.exists():
            args.mode = 'new'
        args.save_root.mkdir(parents=True, exist_ok=True)

        trainer = FeatMatchTrainer(args, config)
        if args.mode != 'test':
            trainer.train()
        # elif args.mode == 'finetune':
        #     trainer.train()
        acc_val, acc_test = trainer.test()
        acc_median = metric.median_acc(os.path.join(args.save_root, 'results.txt'))
        reporter.record(acc_val, acc_test, acc_median)
        with open(args.save_root/'final_result.txt', 'w') as file:
            file.write(f'Val acc: {acc_val*100:.2f} %')
            file.write(f'Test acc: {acc_test*100:.2f} %')
            file.write(f'Median acc: {acc_median*100:.2f} %')

    reporter.report()
