import numpy as np
import os
from termcolor import cprint
import math
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import matplotlib
import pdb

from torch.cuda import device
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
from torch.nn import functional as F

def ce_loss_check(logits_p, prob_p, target_index, target_with_class, mask=None):
    if mask == None:
        mask = torch.ones(len(prob_p), device=prob_p.device).unsqueeze(1)

    print(target_index.shape, target_with_class.shape)
    print(mask.shape)
    print(target_index[0:9], target_with_class[0:9])
    print(mask[0:9])
    temp = F.cross_entropy(logits_p, target_index, reduction='none') * mask.detach()
    temp2 = (target_with_class * -torch.log(F.softmax(logits_p, dim=1))).sum(dim=1) * mask.detach()
    temp_mean = torch.mean(temp)
    temp2_mean = torch.mean(temp2)
    print(temp_mean, temp_mean)

    return temp_mean
class Get_Scalar:
    def __init__(self, value, device):
        self.value = torch.tensor(value, dtype=torch.float32, device=device, requires_grad=True)

    def get_value(self, iter):
        return self.value

    def __call__(self, iter):
        return torch.clamp(self.value, 1e-9, 1.0)

class FeatMatchTrainer(ssltrainer.SSLTrainer):
    def __init__(self, args, config):
        super().__init__(args, config)
        self.fu, self.pu = [], []
        self.fp, self.yp, self.lp = None, None, None
        self.T = args.temperature #(default: 0.5)
        self.p_cutoff= args.p_cutoff
        if self.config['loss']['hard_labels'] == "yes":
            self.hard_labels = True
        elif self.config['loss']['hard_labels'] == "no":
            self.hard_labels = False

        self.criterion = getattr(common, self.config['loss']['criterion'])

        self.hard_ce = getattr(common, 'hard_ce')
        self.attr_objs.extend(['fu', 'pu', 'fp', 'yp', 'lp'])
        self.load(args.mode)
        self.mode = args.mode
        self.end_iter = (self.config['train']['pretrain_iters'] + \
                      2 * self.config['train']['cycle_iters'] + \
                      self.config['train']['end_iters']) - self.config['train']['end_iters']
        print("-------------")
        print(self.end_iter)
        print("-------------")

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
                          mode = self.args.mode,
                          finetune_mode = self.config['model']['finetune_mode'],
                          residual = self.config['model']['residual'],
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

        #Labeled#
        if self.hard_labels:
            target_xl_1D = yl.unsqueeze(1).repeat(1,k).reshape(-1)
            loss_sup = self.hard_ce(logits_xl.reshape(-1,c), target_xl_1D)
        else:
            target_xl_2D = torch.zeros(bsl*k, c, device=xl.device)
            target_xl_2D.scatter_(dim=1, index=yl.unsqueeze(1).repeat(1,k).reshape(-1,1), value=1.)
            loss_sup = self.criterion(None, target_xl_2D, logits_xl.reshape(-1,c), None)

        #Unlabeled#
        # Compute pseudo label
        prob_xu_weak = torch.softmax(logits_xu[:, 0].detach(), dim=1)
        # Generate mask
        max_xu_probs, max_xu_idx = torch.max(prob_xu_weak, dim=1)
        mask_xu = max_xu_idx.ge(self.p_cutoff).float()
        mask_xu = mask_xu.unsqueeze(1).repeat(1, k).reshape(-1)
        # Prediction-logit
        logits_xu = logits_xu.reshape(-1, c)

        if self.hard_labels:
            target_xu_1D = max_xu_idx.unsqueeze(1).repeat(1, k).reshape(-1)
            loss_con_f = self.hard_ce(logits_xu, target_xu_1D, mask_xu)
        else:
            prob_xu_with_T = torch.softmax(prob_xu_weak / self.T, dim=-1)
            prob_xu_with_T = prob_xu_with_T.unsqueeze(1).repeat(1, k, 1).reshape(-1, c)

            loss_con_f = self.criterion(None, prob_xu_with_T, logits_xu.reshape(-1, c), None, mask_xu)
        loss_con_g = torch.tensor(0.0, device=self.default_device)

        # Total loss
        coeff = self.get_consistency_coeff()
        loss = loss_sup + coeff * self.config['loss']['mix'] * loss_con_f

        # Prediction
        pred_xf = torch.softmax(logits_xl[:, 0].detach(), dim=1)
        pred_xg = torch.zeros(len(logits_xl[:,0]), c, device=self.default_device)

        return pred_xg, pred_xf, loss, loss_sup, loss_con_g, loss_con_f

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

    def train2_wo_mixup(self, xl, yl, xu):
        bsl, bsu, k, c = len(xl), len(xu), xl.size(1), self.config['model']['classes']
        x = torch.cat([xl, xu], dim=0).reshape(-1, *xl.shape[2:])
        logits_xg, logits_xf, fx, fxg = self.model(x)
        logits_xg = logits_xg.reshape(bsl + bsu, k, c)
        logits_xf = logits_xf.reshape(bsl + bsu, k, c)

        logits_xgl, logits_xgu = logits_xg[:bsl], logits_xg[bsl:]
        logits_xfl, logits_xfu = logits_xf[:bsl], logits_xf[bsl:]

        #Labeled#
        # [Hard & Soft]

        if self.hard_labels:
            target_xgl_1D = yl.unsqueeze(1).repeat(1, k).reshape(-1)
            loss_sup_g = self.hard_ce(logits_xgl.reshape(-1,c), target_xgl_1D)
        else:
            target_xgl_2D = torch.zeros(bsl*k, c, device=xl.device)
            target_xgl_2D.scatter_(dim=1, index=yl.unsqueeze(1).repeat(1, k).reshape(-1, 1), value=1.)
            loss_sup_g = self.criterion(None, target_xgl_2D, logits_xgl.reshape(-1, c), None)

        #Unlabeled#
        # [Hard & Soft]
        # Compute pseudo label(g)
        prob_xgu_weak = torch.softmax(logits_xgu[:, 0].detach(), dim=1)
        #Generate mask (g)
        max_xgu_probs, max_xgu_idx = torch.max(prob_xgu_weak, dim=1)  # bu
        mask_xgu =max_xgu_idx.ge(self.p_cutoff).float()  # bu
        mask_xgu = mask_xgu.unsqueeze(1).repeat(1, k).reshape(-1)
        #Prediction-logit(g & f)
        logits_xgu = logits_xgu.reshape(-1, c)
        # logits_xfu = logits_xfu.reshape(-1, c)

        # reference: https://github.com/LeeDoYup/FixMatch-pytorch/blob/0f22e7f7c63396e0a0839977ba8101f0d7bf1b04/models/fixmatch/fixmatch_utils.py
        if self.hard_labels:
            #pseudo_labeling
            #target:(bsu*k, c)
            # target_xgu_2D = torch.zeros(bsu*k, c, device=xu.device)
            # target_xgu_2D.scatter_(dim=1, index=max_xgu_idx.unsqueeze(1).repeat(1, k).reshape(-1, 1), value=1.)

            #(target_xgu_1D, mask) : bsu
            target_xgu_1D = max_xgu_idx.unsqueeze(1).repeat(1, k).reshape(-1)
            #(prev)loss_con
            loss_con_g = self.hard_ce(logits_xgu, target_xgu_1D, mask_xgu)
            # (prev) Graph loss
            loss_con_f = torch.tensor(0.0, device=self.default_device)

        else:
            #To check softmax_temperature value#
            # prob_xgu_weak = prob_xgu_weak ** (1. / T)
            # prob_xgu_weak = prob_xgu_weak / prob_xgu_weak.sum(dim=1, keepdim=True)
            prob_xgu_with_T = torch.softmax(prob_xgu_weak/self.T, dim = -1)
            prob_xgu_with_T = prob_xgu_with_T.unsqueeze(1).repeat(1, k, 1).reshape(-1, c)

            loss_con_g = self.criterion(None, prob_xgu_with_T, logits_xgu.reshape(-1, c), None, mask_xgu)
            # loss_con_f = self.criterion(None, prob_xgu_with_T, logits_xfu.reshape(-1, c), None, mask_xgu)
        loss_con_f = torch.tensor(0.0, device=self.default_device)

        # Total loss
        coeff = self.get_consistency_coeff()
        loss = loss_sup_g + coeff * (self.config['loss']['mix'] * loss_con_g + self.config['loss']['graph'] * loss_con_f)

        # Prediction
        pred_xg = torch.softmax(logits_xgl[:, 0].detach(), dim=1)
        pred_xf = torch.softmax(logits_xfl[:, 0].detach(), dim=1)

        return pred_xg, pred_xf, loss, loss_sup_g, loss_con_g, loss_con_f

    def finetune_wo_mixup(self, xl, yl, xu):
        bsl, bsu, k, c = len(xl), len(xu), xl.size(1), self.config['model']['classes']
        x = torch.cat([xl, xu], dim=0).reshape(-1, *xl.shape[2:])
        logits_xg, logits_xf, fx, fxg = self.model(x)
        logits_xg = logits_xg.reshape(bsl + bsu, k, c)
        logits_xf = logits_xf.reshape(bsl + bsu, k, c)

        logits_xgl, logits_xgu = logits_xg[:bsl], logits_xg[bsl:]
        logits_xfl, logits_xfu = logits_xf[:bsl], logits_xf[bsl:]

        # target#
        # target:(bsl*k, c)

        if self.hard_labels:
            target_xgl_1D = yl.unsqueeze(1).repeat(1, k).reshape(-1)
            loss_sup_g = self.hard_ce(logits_xgl.reshape(-1,c), target_xgl_1D)
        else:
            target_xgl_2D = torch.zeros(bsl*k, c, device=xl.device)
            target_xgl_2D.scatter_(dim=1, index=yl.unsqueeze(1).repeat(1, k).reshape(-1, 1), value=1.)
            loss_sup_g = self.criterion(None, target_xgl_2D, logits_xgl.reshape(-1, c), None)

        # [Hard & Soft]
        # Compute pseudo label(g)
        prob_xgu_weak = torch.softmax(logits_xgu[:, 0].detach(), dim=1)
        # Generate mask (g)
        max_xgu_probs, max_xgu_idx = torch.max(prob_xgu_weak, dim=1)  # bu
        mask_xgu = max_xgu_idx.ge(self.p_cutoff).float()  # bu
        mask_xgu = mask_xgu.unsqueeze(1).repeat(1, k).reshape(-1)
        # Prediction-logit(g & f)
        logits_xgu = logits_xgu.reshape(-1, c)
        logits_xfu = logits_xfu.reshape(-1, c)

        # reference: https://github.com/LeeDoYup/FixMatch-pytorch/blob/0f22e7f7c63396e0a0839977ba8101f0d7bf1b04/models/fixmatch/fixmatch_utils.py
        if self.hard_labels:
            # pseudo_labeling
            # target:(bsu*k, c)
            # target_xgu_2D = torch.zeros(bsu*k, c, device=xu.device)
            # target_xgu_2D.scatter_(dim=1, index=max_xgu_idx.unsqueeze(1).repeat(1, k).reshape(-1, 1), value=1.)

            # (target_xgu_1D, mask) : bsu
            target_xgu_1D = max_xgu_idx.unsqueeze(1).repeat(1, k).reshape(-1)
            # (prev)locc_con
            loss_con_g = self.hard_ce(logits_xgu, target_xgu_1D, mask_xgu)
            loss_con_f = self.hard_ce(logits_xfu, target_xgu_1D, mask_xgu)

        else:
            # To check softmax_temperature value#
            # prob_xgu_weak = prob_xgu_weak ** (1. / T)
            # prob_xgu_weak = prob_xgu_weak / prob_xgu_weak.sum(dim=1, keepdim=True)
            prob_xgu_with_T = torch.softmax(prob_xgu_weak / self.T, dim=-1)
            prob_xgu_with_T = prob_xgu_with_T.unsqueeze(1).repeat(1, k, 1).reshape(-1, c)

            loss_con_g = self.criterion(None, prob_xgu_with_T, logits_xgu.reshape(-1, c), None, mask_xgu)
            loss_con_f = self.criterion(None, prob_xgu_with_T, logits_xfu.reshape(-1, c), None, mask_xgu)

        # Total loss
        coeff = self.get_consistency_coeff()
        loss = loss_sup_g + coeff * (
                    self.config['loss']['mix'] * loss_con_g + self.config['loss']['graph'] * loss_con_f)

        # Prediction
        pred_xg = torch.softmax(logits_xgl[:, 0].detach(), dim=1)
        pred_xf = torch.softmax(logits_xfl[:, 0].detach(), dim=1)

        return pred_xg, pred_xf, loss, loss_sup_g, loss_con_g, loss_con_f


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
        # Between (pseudo-label) and (model output)
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
        # encoder clf output, transformer clf output
        # FeatMatch
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

    # caller: trainer.py - def train 406: with amp.autocast
    def forward_train(self, data):
        self.model.train()
        xl = data[0].reshape(-1, *data[0].shape[2:])
        xl = self.Tnorm(xl.to(self.default_device)).reshape(data[0].shape)
        yl = data[1].to(self.default_device)
        xu = data[2].reshape(-1, *data[2].shape[2:])
        xu = self.Tnorm(xu.to(self.default_device)).reshape(data[2].shape)
        # T = torch.clamp(self.T_origin, 1e-9, 1.0)
        # p_cutoff = torch.clamp(self.p_cutoff_origin, 1e-9, 1.0)
        #fast debugging
        self.config['train']['pretrain_iters'] = 20
        self.end_iter = 40
        # #hyper_params for update
        # T = self.t_fn(self.curr_iter)
        # p_cutoff = self.p_fn(self.curr_iter)

        #for debuging training stage#
        #Calculated each stage#
        #Transformer fixed + wo_mixup_fixed
        #(algo_sequence) 1st model_mode setting 2nd train_setting
        if self.curr_iter < self.config['train']['pretrain_iters']:
            self.model.set_mode('pretrain')
            pred_xg, pred_xf, loss, loss_sup, loss_con_g, loss_con_f = self.train1_wo_mixup(xl, yl, xu)
        elif self.curr_iter < self.end_iter:
            self.model.set_mode('finetune')
            #Chagned for no detach
            pred_xg, pred_xf, loss, loss_sup, loss_con_g, loss_con_f = self.finetune_wo_mixup(xl, yl, xu)
        else:
            self.model.set_mode('finetune')
            pred_xg, pred_xf, loss, loss_sup, loss_con_g, loss_con_f = self.finetune_wo_mixup(xl, yl, xu)
        results = {
            'y_pred': torch.max(pred_xf, dim=1)[1].detach().cpu().numpy(),
            'y_pred_agg': torch.max(pred_xg, dim=1)[1].detach().cpu().numpy(),
            'y_true': yl.cpu().numpy(),
            'loss': {
                'all': loss.detach().cpu().item(),
                'sup': loss_sup.detach().cpu().item(),
                'con_g': loss_con_g.detach().cpu().item(),
                'con_f': loss_con_f.detach().cpu().item()
            },
        }

        # if self.mode != 'finetune':
        #     if self.config['model']['attention'] == "no":
        #         self.model.set_mode('pretrain')
        #         if self.config['model']['mixup'] == 'no':
        #             pred_xf, loss, loss_pred, loss_con, loss_graph = self.train1_wo_mixup(xl, yl, xu)
        #         elif self.config['model']['mixup'] == 'yes':
        #             pred_xf, loss, loss_pred, loss_con, loss_graph = self.train1(xl, yl, xu)
        #         pred_xg = torch.tensor(0.0, device=self.default_device)
        #
        #     elif self.config['model']['attention'] == "Transformer":
        #         if self.curr_iter < self.config['train']['pretrain_iters']:
        #             self.model.set_mode('pretrain')
        #             if self.config['model']['mixup'] == 'yes':
        #                 pred_xf, loss, loss_pred, loss_con, loss_graph = self.train1(xl, yl, xu)
        #             elif self.config['model']['mixup'] == 'no':
        #                 pred_xf, loss, loss_pred, loss_con, loss_graph = self.train1_wo_mixup(xl, yl, xu)
        #             pred_xg = torch.tensor(0.0, device=self.default_device)
        #             # pred_x, (x1_weak_label's softmax)
        #             # loss, (total loss)
        #             # loss_pred, (log loss for xl_mixup) loss_con, (log loss for xu_mixup),
        #             # loss_graph = 0.0
        #         elif self.curr_iter == self.config['train']['pretrain_iters']:
        #             self.model.set_mode('train')
        #             if self.config['model']['mixup'] == 'yes':
        #                 # self.extract_fp()
        #                 pred_xg, pred_xf, loss, loss_pred, loss_con, loss_graph = self.train2(xl, yl, xu)
        #             elif self.config['model']['mixup'] == 'no':
        #                 pred_xg, pred_xf, loss, loss_pred, loss_con, loss_graph = self.train2_wo_mixup(xl, yl, xu, T,
        #                                                                                                p_cutoff)
        #         else:
        #             self.model.set_mode('train')
        #             # if self.curr_iter % self.config['train']['sample_interval'] == 0:
        #             if self.config['model']['mixup'] == 'yes':
        #                 # self.extract_fp()
        #                 pred_xg, pred_xf, loss, loss_pred, loss_con, loss_graph = self.train2(xl, yl, xu)
        #             elif self.config['model']['mixup'] == 'no':
        #                 print("train2_wo_mixup")
        #                 pred_xg, pred_xf, loss, loss_pred, loss_con, loss_graph = self.train2_wo_mixup(xl, yl, xu, T,
        #                                                                                                p_cutoff)
        # else:
        #     self.model.set_mode('finetune')
        #     if self.config['model']['mixup'] == 'no':
        #         print("finetune_wo_mixup")
        #         # xl (bs, k, c, h, w)
        #         # yl (bs)
        #         # xu (bu, k, c, h, w)
        #         pred_xg, pred_xf, loss, loss_pred, loss_con, loss_graph = self.finetune_wo_mixup(xl, yl, xu, T,
        #                                                                                          p_cutoff)


        return loss, results

    # Deprecated
    # def forward_finetune(self, data):
    #     self.model.train()
    #     xl = data[0].reshape(-1, *data[0].shape[2:])
    #     xl = self.Tnorm(xl.to(self.default_device)).reshape(data[0].shape)
    #
    #     yl = data[1].to(self.default_device)
    #     xu = data[2].reshape(-1, *data[2].shape[2:])
    #     xu = self.Tnorm(xu.to(self.default_device)).reshape(data[2].shape)
    #     #fast debugging
    #
    #     self.model.set_mode('finetune')
    #     if self.config['model']['mixup'] == 'no':
    #         print("finetune_wo_mixup")
    #         #xl (bs, k, c, h, w)
    #         #yl (bs)
    #         #xu (bu, k, c, h, w)
    #         pred_xg, pred_xf, loss, loss_pred, loss_con, loss_graph = self.finetune_wo_mixup(xl, yl, xu)
    #     results = {
    #         'y_pred': torch.max(pred_xf, dim=1)[1].detach().cpu().numpy(),
    #         'y_pred_agg': torch.max(pred_xg, dim=1)[1].detach().cpu().numpy(),
    #         'y_true': yl.cpu().numpy(),
    #         'loss': {
    #             'all': loss.detach().cpu().item(),
    #             'pred': loss_pred.detach().cpu().item(),
    #             'con': loss_con.detach().cpu().item(),
    #             'graph': loss_graph.detach().cpu().item()
    #         },
    #     }
    #
    #     return loss, results

    def forward_eval(self, data):
        self.model.eval()
        x = self.Tnorm(data[0].to(self.default_device))
        y = data[1].to(self.default_device)
        # if self.config['model']['attention'] == "no":
        #     self.model.set_mode('pretrain')
        #     pred_xf, loss, loss_pred, loss_con, loss_graph = self.eval1_wo_mixup(x, y)
        #     pred_xg = torch.tensor(0.0, device=self.default_device)

        # elif self.config['model']['attention'] == "Transformer":
        #     if self.curr_iter < self.config['train']['pretrain_iters']:
        #         self.model.set_mode('pretrain')
        #         if self.config['model']['mixup'] =='yes':
        #             pred_xf, loss, loss_pred, loss_con, loss_graph = self.eval1(x, y)
        #         elif self.config['model']['mixup'] =='no':
        #             pred_xf, loss, loss_pred, loss_con, loss_graph = self.eval1_wo_mixup(x, y)
        #         # pred_xg: Transformer Output, pred_xf: Encoder Clf Output
        #         pred_xg = torch.tensor(0.0, device=self.default_device)
        #     else:
        #         # FIXME: model??? test mode??? ???????
        #         self.model.set_mode('train')
        #         if self.config['model']['mixup'] == 'yes':
        #             pred_xg, pred_xf, loss, loss_pred, loss_con, loss_graph = self.eval2(x,y)
        #         elif self.config['model']['mixup'] == 'no':
        #             pred_xg, pred_xf, loss, loss_pred, loss_con, loss_graph = self.eval2_wo_mixup(x, y)

        if self.curr_iter > self.config['train']['pretrain_iters']:
            self.model.set_mode('train')
            pred_xg, pred_xf, loss, loss_pred, loss_con, loss_graph = self.eval2_wo_mixup(x, y)
        elif self.curr_iter <= self.config['train']['pretrain_iters']:
            self.model.set_mode('pretrain')
            pred_xf, loss, loss_pred, loss_con, loss_graph = self.eval1_wo_mixup(x, y)
            pred_xg = torch.tensor(0.0, device=self.default_device)
        
        results = {
            'y_pred': torch.max(pred_xf, dim=1)[1].detach().cpu().numpy(),
            'y_pred_agg': torch.max(pred_xg, dim=1)[1].detach().cpu().numpy() if not pred_xg.shape==torch.Size([]) else np.zeros_like(pred_xf[:,0].detach().cpu().numpy()),
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

        acc_val, acc_test, acc_agg_val, acc_agg_test, acc_loaded = trainer.test() # loaded_Acc
        print(f"Val Acc: {acc_val:.4f}, Test Acc: {acc_test:.4f}, Loaded Acc: {acc_loaded:.4f}, Val Agg Acc: {acc_agg_val:.4f}, Test Agg Acc: {acc_agg_test:.4f}")
        # elif args.mode == 'finetune':
        #     trainer.train()
        acc_median = metric.median_acc(os.path.join(args.save_root, 'results.txt'))
        reporter.record(acc_val, acc_test, acc_median)
        with open(args.save_root/'final_result.txt', 'w') as file:
            file.write(f'Val acc: {acc_val*100:.2f} %')
            file.write(f'Test acc: {acc_test*100:.2f} %')
            file.write(f'Median acc: {acc_median*100:.2f} %')

    reporter.report()
