import os, sys
home_folder = os.path.join(os.getcwd().split('PGRL-main')[0], 'PGRL-main')
sys.path.append(home_folder)

import time

import math
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

from lib.augmentation import image_waug_cifar_freeMatch_orig
from lib.dataLoader import get_dataset, get_dataset2
from lib.models import gen_model, CNN, ResNet18, MyXResNet18
from train import evaluating
debugging_flag = False

############################################################
######################### ResNet18 #########################
############################################################
def conv3x3(in_planes, out_planes, stride=1):
    # 3x3 convolution with padding
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        # print(downsample)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            # print(x.shape)
            residual = self.downsample(residual)

        x += residual
        x = self.relu(x)

        return x

    def input_to_residual(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(residual)
        return residual

    def residual_to_output(self, residual, conv2):
        x = residual + conv2
        x = self.relu(x)

        return x

    def input_to_conv2(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x

    def conv2_to_output(self, x, residual):
        x = self.bn2(x)
        x = residual + x
        x = self.relu(x)
        return x

    def input_to_conv1(self, x):
        x = self.conv1(x)
        return x

    def conv1_to_output(self, x, residual):
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x += residual
        x = self.relu(x)

        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x += residual
        x = self.relu(x)

        return x


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, in_channels=3, fc_in_channel=2048):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=4)
        self.fc = nn.Linear(fc_in_channel * block.expansion, num_classes)

        self.inter_feature = {}
        self.inter_gradient = {}

        self.register_all_hooks()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def get_fm(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)

        return x

    def make_hook(self, name, flag):
        if flag == 'forward':
            def hook(m, input, output):
                self.inter_feature[name] = output

            return hook
        elif flag == 'backward':
            def hook(m, input, output):
                self.inter_gradient[name] = output

            return hook
        else:
            assert False

    def register_all_hooks(self):
        self.conv1.register_forward_hook(self.make_hook("Conv1_Conv1_Conv1_", 'forward'))
        self.layer1[0].conv1.register_forward_hook(self.make_hook("Layer1_0_Conv1_", 'forward'))
        self.layer1[0].conv2.register_forward_hook(self.make_hook("Layer1_0_Conv2_", 'forward'))
        self.layer1[1].conv1.register_forward_hook(self.make_hook("Layer1_1_Conv1_", 'forward'))
        self.layer1[1].conv2.register_forward_hook(self.make_hook("Layer1_1_Conv2_", 'forward'))

        self.layer2[0].conv1.register_forward_hook(self.make_hook("Layer2_0_Conv1_", 'forward'))
        self.layer2[0].downsample.register_forward_hook(self.make_hook("Layer2_0_Downsample_", 'forward'))
        self.layer2[0].conv2.register_forward_hook(self.make_hook("Layer2_0_Conv2_", 'forward'))
        self.layer2[1].conv1.register_forward_hook(self.make_hook("Layer2_1_Conv1_", 'forward'))
        self.layer2[1].conv2.register_forward_hook(self.make_hook("Layer2_1_Conv2_", 'forward'))

        self.layer3[0].conv1.register_forward_hook(self.make_hook("Layer3_0_Conv1_", 'forward'))
        self.layer3[0].downsample.register_forward_hook(self.make_hook("Layer3_0_Downsample_", 'forward'))
        self.layer3[0].conv2.register_forward_hook(self.make_hook("Layer3_0_Conv2_", 'forward'))
        self.layer3[1].conv1.register_forward_hook(self.make_hook("Layer3_1_Conv1_", 'forward'))
        self.layer3[1].conv2.register_forward_hook(self.make_hook("Layer3_1_Conv2_", 'forward'))

        self.layer4[0].conv1.register_forward_hook(self.make_hook("Layer4_0_Conv1_", 'forward'))
        self.layer4[0].downsample.register_forward_hook(self.make_hook("Layer4_0_Downsample_", 'forward'))
        self.layer4[0].conv2.register_forward_hook(self.make_hook("Layer4_0_Conv2_", 'forward'))
        self.layer4[1].conv1.register_forward_hook(self.make_hook("Layer4_1_Conv1_", 'forward'))
        self.layer4[1].conv2.register_forward_hook(self.make_hook("Layer4_1_Conv2_", 'forward'))

    '''def get_all_inner_activation(self, x):
        inner_output_index = [0,2,4,8,10,12,16,18]
        inner_output_list = []
        for i in range(23):
            x = self.classifier[i](x)
            if i in inner_output_index:
                inner_output_list.append(x)
        x = x.view(x.size(0), self.num_classes)
        return x,inner_output_list'''

    def input_to_conv1(self, x):
        x = self.conv1(x)
        return x

    def conv1_to_output(self, x):
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def input_to_layer1_0_residual(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1[0].input_to_residual(x)

        return x

    def layer1_0_residual_to_output(self, residual, conv2):

        x = self.layer1[0].residual_to_output(residual, conv2)
        x = self.layer1[1](x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def input_to_layer1_0_conv2(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1[0].input_to_conv2(x)
        return x

    def layer1_0_conv2_to_output(self, x, residual):
        x = self.layer1[0].conv2_to_output(x, residual)
        x = self.layer1[1](x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def input_to_layer1_0_conv1(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1[0].input_to_conv1(x)
        return x

    def layer1_0_conv1_to_output(self, x, residual):
        x = self.layer1[0].conv1_to_output(x, residual)
        x = self.layer1[1](x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def input_to_layer1_1_residual(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1[0](x)
        x = self.layer1[1].input_to_residual(x)

        return x

    def input_to_layer1_1_conv2(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1[0](x)
        x = self.layer1[1].input_to_conv2(x)
        return x

    def layer1_1_conv2_to_output(self, x, residual):
        x = self.layer1[1].conv2_to_output(x, residual)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def input_to_layer1_1_conv1(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1[0](x)
        x = self.layer1[1].input_to_conv1(x)
        return x

    def layer1_1_conv1_to_output(self, x, residual):
        x = self.layer1[1].conv1_to_output(x, residual)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def input_to_layer2_0_residual(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2[0].input_to_residual(x)

        return x

    def layer2_0_residual_to_output(self, residual, conv2):
        x = self.layer2[0].residual_to_output(residual, conv2)
        x = self.layer2[1](x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def input_to_layer2_0_conv2(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2[0].input_to_conv2(x)
        return x

    def layer2_0_conv2_to_output(self, x, residual):
        x = self.layer2[0].conv2_to_output(x, residual)
        x = self.layer2[1](x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def input_to_layer2_0_conv1(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2[0].input_to_conv1(x)
        return x

    def layer2_0_conv1_to_output(self, x, residual):
        x = self.layer2[0].conv1_to_output(x, residual)
        x = self.layer2[1](x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def input_to_layer2_1_residual(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2[0](x)
        x = self.layer2[1].input_to_residual(x)

        return x

    def input_to_layer2_1_conv2(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2[0](x)
        x = self.layer2[1].input_to_conv2(x)
        return x

    def layer2_1_conv2_to_output(self, x, residual):
        x = self.layer2[1].conv2_to_output(x, residual)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def input_to_layer2_1_conv1(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2[0](x)
        x = self.layer2[1].input_to_conv1(x)
        return x

    def layer2_1_conv1_to_output(self, x, residual):
        x = self.layer2[1].conv1_to_output(x, residual)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


    def input_to_layer3_0_residual(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3[0].input_to_residual(x)

        return x

    def layer3_0_residual_to_output(self, residual, conv2):

        x = self.layer3[0].residual_to_output(residual, conv2)
        x = self.layer3[1](x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def input_to_layer3_0_conv2(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3[0].input_to_conv2(x)
        return x

    def layer3_0_conv2_to_output(self, x, residual):
        x = self.layer3[0].conv2_to_output(x, residual)
        x = self.layer3[1](x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def input_to_layer3_0_conv1(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3[0].input_to_conv1(x)
        return x

    def layer3_0_conv1_to_output(self, x, residual):
        x = self.layer3[0].conv1_to_output(x, residual)
        x = self.layer3[1](x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def input_to_layer3_1_residual(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3[0](x)
        x = self.layer3[1].input_to_residual(x)

        return x

    def input_to_layer3_1_conv2(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3[0](x)
        x = self.layer3[1].input_to_conv2(x)
        return x

    def layer3_1_conv2_to_output(self, x, residual):
        x = self.layer3[1].conv2_to_output(x, residual)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def input_to_layer3_1_conv1(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3[0](x)
        x = self.layer3[1].input_to_conv1(x)
        return x

    def layer3_1_conv1_to_output(self, x, residual):
        x = self.layer3[1].conv1_to_output(x, residual)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


    def input_to_layer4_0_residual(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4[0].input_to_residual(x)

        return x

    def layer4_0_residual_to_output(self, residual, conv2):

        x = self.layer4[0].residual_to_output(residual, conv2)
        x = self.layer4[1](x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def input_to_layer4_0_conv2(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4[0].input_to_conv2(x)
        return x

    def layer4_0_conv2_to_output(self, x, residual):
        x = self.layer4[0].conv2_to_output(x, residual)
        x = self.layer4[1](x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def input_to_layer4_0_conv1(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4[0].input_to_conv1(x)
        return x

    def layer4_0_conv1_to_output(self, x, residual):
        x = self.layer4[0].conv1_to_output(x, residual)
        x = self.layer4[1](x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def input_to_layer4_1_residual(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4[0](x)
        x = self.layer4[1].input_to_residual(x)

        return x

    def input_to_layer4_1_conv2(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4[0](x)
        x = self.layer4[1].input_to_conv2(x)
        return x

    def layer4_1_conv2_to_output(self, x, residual):
        x = self.layer4[1].conv2_to_output(x, residual)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def input_to_layer4_1_conv1(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4[0](x)
        x = self.layer4[1].input_to_conv1(x)
        return x

    def layer4_1_conv1_to_output(self, x, residual):
        x = self.layer4[1].conv1_to_output(x, residual)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
############################################################
######################### ResNet18 #########################
############################################################


def learning_rate_finetuning(optimizer, epoch, opt):
    lr = 0.01
    print('epoch: {}  lr: {:.4f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def learning_rate_unlearning(optimizer, epoch, opt):
    lr = 0.0001
    print('epoch: {}  lr: {:.4f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def compute_loss_value(opt, poisoned_data, model_ascent):
    # Calculate loss value per example
    # Define loss function
    # if opt.cuda:
    #     criterion = nn.CrossEntropyLoss().cuda()
    # else:
    criterion = nn.CrossEntropyLoss().to(opt.device)

    model_ascent.eval()
    losses_record = []

    example_data_loader = DataLoader(dataset=poisoned_data,
                                        batch_size=1,
                                        shuffle=False,
                                        )
    idx = 0
    for batch in tqdm(example_data_loader):
        # if opt.cuda:
        #     img = img.cuda()
        #     target = target.cuda()
        img, target = batch[0], batch[1]
        img, target = img.to(opt.device), target.to(opt.device)
        with torch.no_grad():
            output = model_ascent(img)
            loss = criterion(output, target)
            # print(loss.item())

        losses_record.append(loss.item())
        idx += 1

    losses_idx = np.argsort(np.array(losses_record))   # get the index of examples by loss value in ascending order

    # Show the lowest 10 loss values
    losses_record_arr = np.array(losses_record)
    print('Top ten loss value:', losses_record_arr[losses_idx[:10]])

    return losses_idx, losses_record


def calculate_tpr_fpr(predicted_poisoned_indices, ground_truth_benign_indices, total_indices):
    # Convert lists to sets for faster operations
    predicted_poisoned_set = set(predicted_poisoned_indices)
    ground_truth_benign_set = set(ground_truth_benign_indices)

    # Calculate the number of ground-truth poisoned indices
    ground_truth_poisoned_set = set(range(total_indices)) - ground_truth_benign_set

    # True Positives (TP): correctly identified poisoned indices
    TP = len(predicted_poisoned_set.intersection(ground_truth_poisoned_set))

    # False Positives (FP): incorrectly identified poisoned indices (actually benign)
    FP = len(predicted_poisoned_set.difference(ground_truth_poisoned_set))

    # False Negatives (FN): actually poisoned but not identified by the algorithm
    FN = len(ground_truth_poisoned_set.difference(predicted_poisoned_set))

    # True Negatives (TN): correctly identified benign indices
    TN = len(ground_truth_benign_set.difference(predicted_poisoned_set))

    # Calculate TPR (True Positive Rate) and FPR (False Positive Rate)
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0

    return TPR, FPR


def isolate_data(opt, poisoned_data, losses_idx, losses_record):
    ratio = opt.isolation_ratio

    pd_poisoned_index = losses_idx[0: int(len(losses_idx) * ratio)]
    pd_benign_index = losses_idx[int(len(losses_idx) * ratio):]

    gt_benign_index = poisoned_data.benign_indics
    poison_flag_l = np.ones(len(losses_record))
    poison_flag_l[gt_benign_index] = 0
    auc = roc_auc_score(poison_flag_l, -np.array(losses_record))
    # calculate the tpr and fpr
    tpr, fpr = calculate_tpr_fpr(pd_poisoned_index, gt_benign_index, len(poisoned_data))

    print('TPR, FPR ({:.3f}, {:.3f} auc {:.3f})'.format(tpr, fpr, auc))
    # save the predicted benign index
    np.save('../../poisonDataset/{}/abl_pd_benign_indics_{}_{}_{}_{}_{}_tpr_fpr_poison_{}_{}.pth'.format(
                   opt.poison_type, opt.num_class,
                   opt.poison_or_benign, opt.cover_rate,
                   opt.poison_rate, opt.tuning_epochs, tpr, fpr), pd_benign_index)

    return pd_poisoned_index, pd_benign_index


def train_step(opt, train_loader, model_ascent, optimizer, criterion, epoch, writer):
    model_ascent.train()

    for idx, batch in enumerate(train_loader, start=1):
        # if opt.cuda:
        #     img = img.cuda()
        #     target = target.cuda()
        img, target = batch[0], batch[1]
        img = img.to(opt.device)
        target = target.to(opt.device)
        if opt.gradient_ascent_type == 'LGA':
            output = model_ascent(img)
            loss = criterion(output, target)
            # add Local Gradient Ascent(LGA) loss
            # opt.gamma = 0.05
            # loss_ascent = torch.mean(torch.sign(loss - opt.gamma) * loss)
            loss_ascent = torch.mean(loss)
        elif opt.gradient_ascent_type == 'Flooding':
            output = model_ascent(img)
            # output = student(img)
            loss = criterion(output, target)
            # add flooding loss
            loss_ascent = torch.mean((loss - opt.flooding).abs() + opt.flooding)

        else:
            raise NotImplementedError


        optimizer.zero_grad()
        loss_ascent.backward()
        optimizer.step()
        writer.add_scalar('tra/loss_v', loss_ascent.item(), idx)
        if debugging_flag == True:
            break




def train(model_ascent, copy_poisoned_data_loader, test_clean_loader, test_bad_loader, optimizer, criterion, writer, opt, epoch_acc_asr):
    model_ascent.train()
    # before training test firstly
    acc = evaluating(model_ascent, test_clean_loader, 0, opt.device, writer)
    asr = evaluating(model_ascent, test_bad_loader, 0, opt.device, writer, poison_flag=True)
    epoch_acc_asr.append([0, acc.item(), asr.item()])

    print('----------- Train Initialization --------------')
    for epoch in range(1, opt.tuning_epochs+1): # opt.tuning_epochs

        adjust_learning_rate(optimizer, epoch, opt)

        # train every epoch
        train_step(opt, copy_poisoned_data_loader, model_ascent, optimizer, criterion, epoch + 1, writer)

        # evaluate on testing set
        print('testing the ascended model......')
        if epoch % 5 == 0:
            acc = evaluating(model_ascent, test_clean_loader, epoch, opt.device, writer)
            asr = evaluating(model_ascent, test_bad_loader, epoch, opt.device, writer, poison_flag=True)
            epoch_acc_asr.append([epoch, acc.item(), asr.item()])

        if debugging_flag == True:
            break

    torch.save(model_ascent.state_dict(),
               '../../poisonDataset/{}/abl_train_{}_{}_{}_{}_{}.pth'.format(
                   opt.poison_type, opt.num_class,
                   opt.poison_or_benign,
                   opt.poison_rate, opt.cover_rate, opt.tuning_epochs))

    return model_ascent, epoch_acc_asr


def adjust_learning_rate(optimizer, epoch, opt):
    if epoch < opt.tuning_epochs:
        lr = opt.lr
    else:
        lr = 0.01
    print('epoch: {}  lr: {:.4f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, epoch, is_best, opt):
    if is_best:
        filepath = os.path.join(opt.save, opt.model_name + r'-tuning_epochs{}.tar'.format(epoch))
        torch.save(state, filepath)
    print('[info] Finish saving the model')



def train_step_finetuing(opt, train_loader, model_ascent, optimizer, criterion, epoch, writer):
    model_ascent.train()

    for idx, batch in enumerate(train_loader, start=1):
        img, target = batch[0], batch[1]
        img = img.to(opt.device)
        target = target.to(opt.device)

        output = model_ascent(img)

        loss = torch.mean(criterion(output, target))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # TODO: add writer
        writer.add_scalar('tra/finetune_loss_v', loss.item(), idx)
        if debugging_flag == True:
            break

def train_step_unlearning(opt, train_loader, model_ascent, optimizer, criterion, epoch, writer):
    model_ascent.train()

    for idx, batch in enumerate(train_loader, start=1):
        img, target = batch[0], batch[1]
        img = img.to(opt.device)
        target = target.to(opt.device)

        output = model_ascent(img)

        loss = torch.mean(criterion(output, target))

        optimizer.zero_grad()
        (-loss).backward()  # Gradient ascent training
        optimizer.step()

        # TODO: add writer
        writer.add_scalar('tra/unlearn_loss_v', loss.item(), idx)


def unlearning(model_ascent, poisoned_dl, ts_dl, pts_dl, optimizer, criterion, opt, writer, pd_poison_index, pd_benign_index, epoch_acc_asr):
    poisoned_data_tf = Subset(poisoned_dl.dataset, pd_poison_index)
    isolate_other_data_tf = Subset(poisoned_dl.dataset, pd_benign_index)

    isolate_poisoned_data_loader = DataLoader(dataset=poisoned_data_tf,
                                              batch_size=opt.batch_size,
                                              shuffle=True)

    isolate_other_data_loader = DataLoader(dataset=isolate_other_data_tf,
                                           batch_size=opt.batch_size,
                                           shuffle=True)

    if opt.finetuning_ascent_model == True:
        # this is to improve the clean accuracy of isolation model, you can skip this step
        print('----------- Finetuning isolation model --------------')
        for epoch in range(opt.tuning_epochs + 1, opt.tuning_epochs + opt.finetuning_epochs + 1):
            # learning_rate_finetuning(optimizer, epoch, opt)
            # learning_rate_finetuning(optimizer, epoch, opt)
            train_step_finetuing(opt, isolate_other_data_loader, model_ascent, optimizer, criterion,
                             epoch, writer)
            if epoch % 5 == 0:
                # evaluate on testing set
                print('testing the ascended model......')
                acc = evaluating(model_ascent, ts_dl, epoch, opt.device, writer)
                asr = evaluating(model_ascent, pts_dl, epoch, opt.device, writer, poison_flag=True)
                epoch_acc_asr.append([epoch, acc.item(), asr.item()])

    print('----------- Model unlearning --------------')
    for epoch in range(opt.tuning_epochs + opt.finetuning_epochs + 1, opt.tuning_epochs + opt.finetuning_epochs + 1 + opt.unlearning_epochs):
        learning_rate_unlearning(optimizer, epoch, opt)

        # train stage
        if epoch == 0:
            # test firstly
            evaluating(model_ascent, ts_dl, epoch, opt.device, writer)
            evaluating(model_ascent, pts_dl, epoch, opt.device, writer, poison_flag=True)
        else:
            train_step_unlearning(opt, isolate_poisoned_data_loader, model_ascent, optimizer, criterion, epoch + 1, writer)

        if epoch % 5 == 0:
            # evaluate on testing set
            print('testing the ascended model......')
            acc = evaluating(model_ascent, ts_dl, epoch, opt.device, writer)
            asr = evaluating(model_ascent, pts_dl, epoch, opt.device, writer, poison_flag=True)
            epoch_acc_asr.append([epoch, acc.item(), asr.item()])

    # TODO: save model
    torch.save(model_ascent.state_dict(),
               '../../poisonDataset/{}/{}_unlearning_{}_{}_{}_{}.pth'.format(
                   opt.poison_type, opt.method, opt.num_class,
                   opt.poison_or_benign,
                   opt.poison_rate, opt.tuning_epochs))
    return epoch_acc_asr


def main(opt):
    args_str = '_'.join('{}'.format(value) for _, value in vars(args).items())
    writer = SummaryWriter(comment='{}_args_{}'.format(os.path.basename(__file__), args_str))
    print(args)
    # Load models
    print('----------- Network Initialization --------------')
    # replace by my model
    opt.num_cluster = opt.num_class
    if opt.poison_type == 'ultrasonic':
        model_ascent = resnet18(num_classes=opt.num_class, in_channels=1, fc_in_channel=1536) # Don't use CNN(num_classes=10) since PIPD use this resnet for activation analysis
    # elif opt.poison_type == 'imagenette_pattern':
    #     model_ascent = MyXResNet18(num_classes=opt.num_class)
    else:
        model_ascent = resnet18(num_classes=opt.num_class)

    model_ascent.to(opt.device)

    print('finished model init...')

    # initialize optimizer
    opt.lr, opt.momentum, opt.weight_decay = 0.001, 0.9, 0#1e-4

    optimizer = torch.optim.SGD(model_ascent.parameters(),
                                lr=opt.lr,
                                momentum=opt.momentum,
                                weight_decay=opt.weight_decay,
                                nesterov=True)

    # optimizer = torch.optim.Adam(model_ascent.parameters(), lr=opt.lr, weight_decay=0)

    criterion = nn.CrossEntropyLoss(reduction='none').to(opt.device)
    # TODO: replace by my dataloader
    opt.batch_size, opt.num_workers = 64, 4
    opt.tuning_epochs = 10
    opt.gradient_ascent_type = 'LGA'

    cache_train_ds = '../../poisonDataset/{}/abl_train_{}_{}_{}_{}_{}.pth'.format(
                   opt.poison_type, opt.num_class,
                   opt.poison_or_benign,
                   opt.poison_rate, opt.cover_rate, opt.tuning_epochs)


    transforms = False
    poisoned_data_loader, test_clean_loader, test_bad_loader, train_folder = get_dataset2(opt.poison_type,
                                                                                         opt.poison_or_benign,
                                                                                         opt.poison_rate,
                                                                                         opt.batch_size,
                                                                                         opt.num_class, opt.num_workers,
                                                                                         transforms=transforms,
                                                                                         image_size=32)
    poisoned_data = poisoned_data_loader.dataset
    epoch_acc_asr = []
    # if os.path.exists(cache_train_ds):
    #     model_ascent.load_state_dict(torch.load(cache_train_ds, map_location=opt.device))
    # else:
    start_t = time.time()
    print('----------- Train isolated model -----------')
    model_ascent, epoch_acc_asr = train(model_ascent, poisoned_data_loader, test_clean_loader, test_bad_loader, optimizer, criterion, writer, opt, epoch_acc_asr)

    print('----------- Calculate loss value per example -----------')
    losses_idx, losses_record = compute_loss_value(opt, poisoned_data, model_ascent)

    print('----------- Collect isolation data -----------')
    opt.isolation_ratio = 0.05
    pd_poison_index, pd_benign_index = isolate_data(opt, poisoned_data, losses_idx, losses_record)
    print('poison detection {}'.format(time.time() - start_t))
    start_t = time.time()
    # reinitilze
    if opt.poison_type == 'ultrasonic':
        new_model_ascent = resnet18(num_classes=opt.num_class, in_channels=1, fc_in_channel=1536) # Don't use CNN(num_classes=10) since PIPD use this resnet for activation analysis
    else:
        new_model_ascent = resnet18(num_classes=opt.num_class)

    new_model_ascent.to(opt.device)
    new_optimizer = torch.optim.SGD(new_model_ascent.parameters(),
                                lr=opt.lr,
                                momentum=opt.momentum,
                                weight_decay=opt.weight_decay,
                                nesterov=True)
    print('----------- Backdoor unlearning -----------')
    opt.finetuning_ascent_model, opt.method = True, 'abl'
    opt.finetuning_epochs, opt.unlearning_epochs = 180, 10
    opt.lr_finetuning_init, opt.lr_unlearning_init, opt.lr = 0.1, 5e-4, 0.1
    epoch_acc_asr = unlearning(new_model_ascent, poisoned_data_loader, test_clean_loader, test_bad_loader, new_optimizer, criterion, opt, writer, pd_poison_index, pd_benign_index, epoch_acc_asr)
    if opt.efficiency_analysis:
        np.save('../../poisonDataset/{}/abl_train_epoch_acc.npy'.format(opt.poison_type), np.array(epoch_acc_asr))
    print('unlearning {}'.format(time.time() - start_t))


if __name__ == '__main__':
    import argparse

    def parse_args():
        parser = argparse.ArgumentParser(description='Parse command-line arguments for poisoning and augmentation.')
        parser.add_argument('-t', '--poison_type', required=True, type=str, help='Specify the type of poisoning.')
        parser.add_argument('-class', '--num_class', required=True, type=int, help='The number of classes.')
        parser.add_argument('-pb', '--poison_or_benign', required=True, type=str, help='Specify whether the data is poison or benign.')
        parser.add_argument('-d', '--device', default='cuda:0', type=str, help='The device to use (e.g., "cpu" or "cuda").')
        parser.add_argument('-pr', '--poison_rate', default=0, type=float, help='The rate of poisoning.')
        parser.add_argument('-cr', '--cover_rate', default=0, type=float, help='The rate of cover rate.')
        parser.add_argument('-effana', '--efficiency_analysis', default=False, type=bool,
                            help='Analyse the training efficiency by saving acc at intervals of every five epochs')
        return parser.parse_args()

    args = parse_args()
    main(args)