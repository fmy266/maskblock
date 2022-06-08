#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# author：fmy
import torch
import torchvision

def nothing(**kwargs):
    pass

class BaseAttack:
    name: "base attack"

    def __init__(self, epsilon: float, step: float = 0.05,
                 iter_num: int = 40, is_targeted: bool = False,
                 ord_: str = "Linf", device=torch.device("cuda:0"),
                 model_preprocessing=nothing, data_preprocessing=lambda x:x,
                 label_preprocessing=lambda x:x, distortion_preprocessing=lambda x:x):
        self.epsilon = epsilon
        self.step = step
        self.iter_num = iter_num
        self.is_targeted = is_targeted
        self.ord_ = ord_
        self.model_preprocessing = model_preprocessing
        self.data_preprocessing = data_preprocessing
        self.label_preprocessing = label_preprocessing
        self.distortion_preprocessing = distortion_preprocessing
        self.device = device
        self.restart = 1
        self.repeat = 1

    def generate_adv_noises(self, data, label, black_model, substitute_model, loss_func, target: int = None):
        distortion = self.distortion_generation(data)
        label = self.label_preprocessing(label)
        
        for _ in range(self.iter_num):
            for repeat in range(self.repeat):
                output = substitute_model(self.data_preprocessing(data) + self.distortion_preprocessing(distortion))
                if self.is_targeted:
                    loss = loss_func(output, torch.full_like(label, target, device=self.device, dtype=torch.long))
                else:
                    loss = -loss_func(output, label)
                loss.backward() 
            self.grad_transform(distortion)
            self.distortion_update(distortion)
            distortion = self.clip(distortion)
        return distortion


    def attack(self, black_model, substitute_model, loss_func, loader, predict_func = lambda x:x, target: int = None):
        if self.is_targeted and target == None:
            raise ValueError("targeted attack must have the value of attack class.")

        num, mis_num = 0, 0
        self.model_preprocessing(model = substitute_model)
        substitute_model = substitute_model.to(self.device)
        substitute_model.eval()
        black_model = black_model.to(self.device)
        black_model.eval()

        for data, label in loader:
            data, label = data.to(self.device), label.to(self.device)
            restart_result = torch.zeros_like(label, device=self.device, dtype=torch.float32)
            for i in range(self.restart):

                distortion = self.generate_adv_noises(data, label, black_model, substitute_model,\
                                                loss_func, target)

                with torch.no_grad():
                    if self.is_targeted:
                        restart_result += (predict_func(black_model(data + distortion)).max(dim=1)[1] == target)
                    else:
                        restart_result += (predict_func(black_model(data + distortion)).max(dim=1)[1] != label)
            with torch.no_grad():
                mis_num += (restart_result != 0).sum().item()
                num += label.size()[0]
        return mis_num / num * 100.

    @torch.no_grad()
    def distortion_generation(self, data):
        return torch.zeros_like(data, device=self.device).requires_grad_(True)

    @torch.no_grad()
    def clip(self, distortion):
        if self.ord_ == "Linf":
            mask = torch.sign(distortion)
            distortion = mask * torch.min(distortion.abs_(),
                                          torch.full_like(distortion, self.epsilon, device=self.device))
        elif self.ord_ == "L2":
            l2_norm = distortion.pow(2).view(distortion.size()[0], -1).sum(dim=1).pow(0.5)
            mask = l2_norm <= self.epsilon  # if norm of tensor bigger than constraint, then scale it into the range
            l2_norm = torch.where(mask, torch.ones_like(l2_norm, device=self.device), l2_norm)
            distortion = distortion / (l2_norm).view(-1, 1, 1, 1)
        elif self.ord_ == "L1":
            l1_norm = distortion.abs().view(distortion.size()[0], -1).sum(dim=1)
            mask = l1_norm <= self.epsilon
            l2_norm = torch.where(mask, torch.ones_like(l1_norm, device=self.device), l1_norm)
            distortion = distortion / (l2_norm).view(-1, 1, 1, 1)
        else:
            raise ValueError("The norm not exists.")
        distortion.requires_grad_(True)
        return distortion

    @torch.no_grad()
    def grad_transform(self, distortion):
        # scale grad to same level for fair comparison
        # distortion.grad = distortion.grad / distortion.grad.abs().max()  # divided by maximum value
        distortion.grad.sign_()

    @torch.no_grad()
    def distortion_update(self, distortion):
        distortion.sub_(self.step * distortion.grad)