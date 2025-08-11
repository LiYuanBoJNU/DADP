# Copyright (c) 2018-present, Royal Bank of Canada and other authors.
# See the AUTHORS.txt file for a list of contributors.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image


from advertorch.utils import clamp
from advertorch.utils import normalize_by_pnorm
from advertorch.utils import clamp_by_pnorm
from advertorch.utils import is_float_or_torch_tensor
from advertorch.utils import batch_multiply
from advertorch.utils import batch_clamp
from advertorch.utils import replicate_input
from advertorch.utils import batch_l1_proj

from advertorch.attacks.base import Attack
from advertorch.attacks.base import LabelMixin
from advertorch.attacks.utils import rand_init_delta
from advertorch.utils import normalize_by_pnorm

from torchvision import transforms



def random_transforms(img_tensor):
    img_size = img_tensor.shape[2]
    img_transform1 = transforms.Pad(padding=10)
    img_transform2 = transforms.RandomCrop(img_size*0.8)
    img_transform3 = transforms.CenterCrop(img_size*0.8)
    img_transform4 = transforms.RandomRotation(10)
    img_transform5 = transforms.RandomHorizontalFlip()
    img_transform = transforms.RandomChoice([img_transform1, img_transform2,
                                             img_transform3, img_transform4, img_transform5])
    out_tensor = img_transform(img_tensor)
    out_tensor = F.interpolate(out_tensor, size=(img_tensor.shape[2], img_tensor.shape[3]), mode='bilinear', align_corners=False)
    return out_tensor


def perturb_iterative(xvar, yvar, xT, predict, nb_iter, loss_fn,
                      T=20, start_t=20, attack_iter=1, attack_inf_iter=1,
                      repeat_times=1, delta_init=None, vis_full=False, x_ori=None,
                      lam_x=20, lam_z=30, lam_adv=1, lam_lr=1):

    if delta_init is not None:
        delta = delta_init
    else:
        delta = torch.zeros_like(xvar)

    l1_loss = nn.L1Loss()


    indices = list(range(T))[::-1]  # [T-1, T-2, ..., 0]
    # grad_d = torch.zeros_like(delta)
    # optimizer = torch.optim.AdamW([delta], lr=1e-2)

    # alpha = 0.3
    assert start_t <= T  # start_t 5 T 5
    assert (attack_inf_iter + attack_iter) * repeat_times <= start_t  # attack_inf_iter 4 attack_iter 1
    if vis_full:
        x_list = []

    xT_ori = xT.clone()
    xT = xT + torch.randn_like(xT) * 0.01
    xT.requires_grad_(True)
    xvar_ori = xvar.clone()
    xvar = xvar + torch.randn_like(xvar) * 0.01
    xvar.requires_grad_(True)

    optimizer = torch.optim.AdamW([xT, xvar], lr=1e-2)
    for j in range(nb_iter):
        xT_tmp = xT.clone()
        # xT_tmp = xT_tmp.detach()

        # cnt = cnt_skip
        for k in range(repeat_times):
            for ii in range(attack_inf_iter + attack_iter):
                t = torch.tensor(indices[T - start_t + (k * (attack_inf_iter + attack_iter) + ii)] * xT_tmp.size(0), device='cuda').unsqueeze(0)

                if ii < attack_inf_iter:
                    # xT_tmp = xT_tmp.detach()
                    xT_tmp, x = predict(xT_tmp, xvar, T, t, False, False)
                    if vis_full:
                        x_list.append(xT_tmp)
                    continue
                else:
                    xT_tmp.requires_grad_()
                    # outputs, xT_tmp, x = predict(xT_tmp, xvar, T, t, True, False)
                    outputs, xT_tmp, x = predict(random_transforms(xT_tmp), xvar, T, t, True, False)
                    if vis_full:
                        x_list.append(xT_tmp)


                optimizer.zero_grad()

                adv_loss = loss_fn(outputs, yvar)

                loss_lr_x = l1_loss(xT_ori, xT)
                loss_lr_z = l1_loss(xvar_ori, xvar)
                loss_lr = loss_lr_x * lam_x + loss_lr_z * lam_z

                loss = lam_adv * adv_loss + lam_lr * loss_lr


                loss.backward()
                optimizer.step()


                if j % 10 == 0:
                    print(loss.item(), adv_loss.item(), loss_lr_x.item(), loss_lr_z.item())
                    # print(loss.item(), adv_loss.item(), delta_loss1.item(), tv.item(), delta_loss2.item())
                xT_tmp = xT_tmp.detach()


    # x_adv = clamp(xvar + delta, clip_min, clip_max)
    x_adv = xvar.detach()
    xT_adv = xT.detach()
    if vis_full:
        return x_adv, x, x_list
    return x_adv, x_ori, xT_adv


class DADPAttack(Attack, LabelMixin):

    def __init__(
            self, predict, loss_fn=None, nb_iter=40,
            rand_init=True, clip_min=0., clip_max=1.,
            ord=np.inf, targeted=False, T_enc=50, T_atk=20,
            start_t=20, attack_iter=1, attack_inf_iter=1, repeat_times=1, vis_full=False,
            lam_x=20, lam_z=30, lam_adv=1, lam_lr=1):
        """
        Create an instance of the PGDAttack.

        """
        super(DADPAttack, self).__init__(
            predict, loss_fn, clip_min, clip_max)

        self.nb_iter = nb_iter

        self.rand_init = rand_init
        self.ord = ord
        self.targeted = targeted
        self.T_enc = T_enc
        self.T_atk = T_atk
        self.start_t = start_t
        self.attack_iter = attack_iter
        self.attack_inf_iter = attack_inf_iter
        self.repeat_times = repeat_times
        self.vis_full = vis_full
        self.lam_x = lam_x
        self.lam_z = lam_z
        self.lam_adv = lam_adv
        self.lam_lr = lam_lr

        if self.loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss(reduction="sum")


    def perturb(self, x_ori, z, y, x):
        """
        Given examples (x, y), returns their adversarial counterparts with
        an attack length of eps.

        :param x: input tensor.
        :param y: label tensor.
                  - if None and self.targeted=False, compute y as predicted
                    labels.
                  - if self.targeted=True, then y must be the targeted labels.
        :return: tensor containing perturbed inputs.
        """
        
        #x, y = self._verify_and_process_inputs(x, y)
        #print("my pgd")
        delta = torch.zeros_like(z)
        delta = nn.Parameter(delta)
        if self.rand_init:
            rand_init_delta(
                delta, z, self.ord, self.eps, self.clip_min, self.clip_max)
            delta.data = clamp(
                z + delta.data, min=self.clip_min, max=self.clip_max) - z

        rval = perturb_iterative(
            z, y, x, self.predict, nb_iter=self.nb_iter,
            loss_fn=self.loss_fn, delta_init=delta, T=self.T_atk, start_t=self.start_t, attack_iter=self.attack_iter,
            attack_inf_iter=self.attack_inf_iter, repeat_times=self.repeat_times, vis_full=self.vis_full, x_ori=x_ori,
            lam_x=self.lam_x, lam_z=self.lam_z, lam_adv=self.lam_adv, lam_lr=self.lam_lr
        )
        if self.vis_full:
            return rval[0].data, rval[1].data, rval[2]
        else:
            return rval[0].data, rval[1].data, rval[2].data

