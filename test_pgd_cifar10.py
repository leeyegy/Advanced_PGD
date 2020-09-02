from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
from models.wideresnet import *
from models.resnet import *
from Advanced_PGD import APGD

parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=0.031,type=float,
                    help='perturbation')
parser.add_argument('--num-steps', default=10,type=int,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.007,type=float,
                    help='perturb step size')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')


#
parser.add_argument('--random_start', default=False, action='store_true')

parser.add_argument("--test_model_path",type=str)

# APGD
parser.add_argument("--factor",type=float,default=100000)

args = parser.parse_args()

# settings
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# setup data loader
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
testset = torchvision.datasets.CIFAR10(root='/home/Leeyegy/.torch/datasets/', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

# load net
test_model = ResNet18().to(device)
test_model.load_state_dict(torch.load(args.test_model_path))

# define attacker
adversary = APGD(test_model, epsilon=args.epsilon,PGD_step_size=args.step_size,
                     max_val=1.0, min_val=0.0, loss=nn.CrossEntropyLoss(), device=device,max_iter=args.num_steps,
                     random_start=args.random_start,factor=args.factor)



# eval
def eval_test():
    test_model.eval()
    cln_correct = 0
    adv_correct = 0
    for index,(data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        with torch.no_grad():
            output = test_model(data)

        pred_cln = output.max(1, keepdim=True)[1] % 10
        cln_correct += pred_cln.eq(target.view_as(pred_cln)).sum().item()
        adv_data = adversary.perturb(data,target)

        with torch.no_grad():
            output = test_model(adv_data)
        pred_adv = output.max(1, keepdim=True)[1]

        adv_correct += pred_adv.eq(target.view_as(pred_adv)).sum().item()
        print(adv_correct)
        break
    print('Clean Test: Accuracy: {}/{} ({:.0f}%)'.format(
         cln_correct, len(test_loader.dataset),
        100. * cln_correct / len(test_loader.dataset)))
    print('PGD Test: Accuracy: {}/{} ({:.0f}%)'.format(
        adv_correct, len(test_loader.dataset),
        100. * adv_correct / len(test_loader.dataset)))

if __name__ == "__main__":
    eval_test()
