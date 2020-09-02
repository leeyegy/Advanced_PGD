from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

class APGD():
    '''
    Notice that if set ODI_num_size = 0, then FGSM_ODI acts as same as FGSM
    '''
    def __init__(self, model, epsilon,PGD_step_size, max_val, min_val,loss,device,max_iter=10,random_start=False):
        self.model = model
        # Maximum perturbation
        self.epsilon = epsilon
        # Movement multiplier per iteration, set it as the same as epsilon by default
        self.PGD_step_size = PGD_step_size
        # Minimum value of the pixels
        self.min_val = min_val
        # Maximum value of the pixels
        self.max_val = max_val
        # loss function
        self.loss = loss
        # device | cpu or gpu
        self.device = device
        # if random start
        self.random_start = random_start
        self.max_iter = max_iter

    def perturb(self,X,y):
        X_adv = Variable(X.data, requires_grad=True)
        if self.random_start:
            random_noise = torch.FloatTensor(*X_adv.shape).uniform_(-self.epsilon, self.epsilon).to(self.device)
            X_adv = Variable(X_adv.data + random_noise, requires_grad=True)

        for i in range(self.max_iter):
            opt = optim.SGD([X_adv], lr=1e-3)
            opt.zero_grad()
            with torch.enable_grad():
                loss = self.loss(self.model(X_adv), y)
            print("iter :{} loss:{}".format(i,loss))
            loss.backward()
            eta = self.PGD_step_size * X_adv.grad.data.sign()
            print(X_adv.grad.data)
            X_adv = Variable(X_adv.data + eta, requires_grad=True)
            eta = torch.clamp(X_adv.data - X.data, -self.epsilon, self.epsilon)
            X_adv = Variable(X.data + eta, requires_grad=True)
            X_adv = Variable(torch.clamp(X_adv, self.min_val, self.max_val), requires_grad=True)
        return X_adv.detach()

