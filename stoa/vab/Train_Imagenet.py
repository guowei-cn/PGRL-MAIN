from __future__ import print_function

import numpy as np
import torch.backends.cudnn
from torch.cuda.amp import GradScaler
import argparse
import torch.optim as optim
import random
import os

from dataloader_Imagenet import Imagenet12_dataloader
from models.wresnet import *
from models.resnet_cifar import resnet18, resnet34
from functions_ImageNet import *
from models.preact_resnet import *
from models.Conv4 import *
from attention_mix_ImageNet import *

from prepare_data import prepare_data

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--batch_size', default=16, type=int, help='train batchsize')
parser.add_argument('--poisoned_mode', default='all2one', type=str)
parser.add_argument('--posioned_portion', default=0.1, type=int)
parser.add_argument('--trigger_label', default=0, type=int)
parser.add_argument('--trigger_type', default='badnet', type=str)
parser.add_argument('--trigger_path', default='./trigger/ImageNet/ImageNet_badnet.npy', type=str)
parser.add_argument('--lr_clean', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--lr_poison', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--alpha', default=0.5, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=0.5, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.9, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--r', default=0.75, type=float)
parser.add_argument('--num_epochs', default=0, type=int)
parser.add_argument('--dataset', default='ImageNet', type=str)
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=12, type=int)
parser.add_argument('--warm_up', default=6, type=int)
parser.add_argument('--model_name', default='ResNet18', type=str)
parser.add_argument('--data_path', default='./dataset/imagenet12/', type=str, help='path to dataset')
parser.add_argument('--BD_data_path', default='./BD_dataset/imagenet12/', type=str, help='path to dataset')
parser.add_argument('--storage_path', default='./storage/imagenet12/', type=str, help='path to storage')
parser.add_argument('--resume', default=False, type=bool)
parser.add_argument('--resume_net1', default='./storage/imagenet12/epoch100_net1.pth', type=str)
parser.add_argument('--resume_net2', default='./storage/imagenet12/epoch100_net2.pth', type=str)

args = parser.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
torch.backends.cudnn.deterministic = True
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def create_model(model_name,
                 pretrained=False,
                 pretrained_models_path=None,
                 n_classes=12):

    assert model_name in ['WRN-16-1', 'WRN-16-2', 'WRN-40-1', 'WRN-40-2', 'ResNet34', 'WRN-10-2', 'WRN-10-1', 'PreActResNet18', 'ResNet18', 'Conv4']
    if model_name=='WRN-16-1':
        model = WideResNet(depth=16, num_classes=n_classes, widen_factor=1, dropRate=0)
    elif model_name=='WRN-16-2':
        model = WideResNet(depth=16, num_classes=n_classes, widen_factor=2, dropRate=0)
    elif model_name=='WRN-40-1':
        model = WideResNet(depth=40, num_classes=n_classes, widen_factor=1, dropRate=0)
    elif model_name=='WRN-40-2':
        model = WideResNet(depth=40, num_classes=n_classes, widen_factor=2, dropRate=0)
    elif model_name == 'WRN-10-2':
        model = WideResNet(depth=10, num_classes=n_classes, widen_factor=2, dropRate=0)
    elif model_name == 'WRN-10-1':
        model = WideResNet(depth=10, num_classes=n_classes, widen_factor=1, dropRate=0)
    elif model_name=='ResNet34':
        model = resnet34( num_classes=n_classes)
    elif model_name == 'PreActResNet18':
        model = PreActResNet18(num_classes=n_classes)
    elif model_name == 'ResNet18':
        model = resnet18(num_classes=n_classes)
    elif model_name == 'Conv4':
        model = ConvNet(num_class=n_classes)
    else:
        raise NotImplementedError

    return model


print('| Building net')
#claen model
net1 = create_model(args.model_name)
net1 = attention_mix_net(net1, args.num_class)
net1 = nn.DataParallel(net1)
net1.cuda()
#backdoor model
net2 = create_model('ResNet18')
net2 = nn.DataParallel(net2)
net2.cuda()

scalar_net1 = GradScaler()
scalar_net2 = GradScaler()

optimizer1 = optim.SGD(net1.parameters(), lr=args.lr_clean, momentum=0.9, weight_decay=5e-4)
optimizer2 = optim.Adam(net2.parameters(), lr=1e-3, weight_decay=5e-4)
scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer1, [50, 100, 150])

args.storage_path = args.storage_path + args.dataset + '/' + args.trigger_type + '/'
if not os.path.exists(args.storage_path):
    os.makedirs(args.storage_path)
test_log_path = './checkpoint/%s/' % (args.dataset)
if not os.path.exists(test_log_path):
    os.makedirs(test_log_path)
test_log = open(test_log_path + args.trigger_type + '.txt', 'w')


best_acc = 0
asr = 1
flag = 0
threshold = None

start_epoch1 = 0
start_epoch2 = 0
poisoned_idx = []
noisy_idx = []
if args.resume:
    start_epoch1, scalar_net1, _, _, _, _ = load_state(net1, optimizer1, args.resume_net1)
    start_epoch2, scalar_net2, poisoned_idx, noisy_idx, pre_loss, threshold = load_state(net2, optimizer2, args.resume_net2)

#prepare_data(args)# generate poisoned dataset and save to disk

loader = Imagenet12_dataloader(args, args.batch_size, 0)
test_CL_loader1, test_BD_loader1 = loader.run('test_net1')
test_CL_loader2, test_BD_loader2 = loader.run('test_net2')
#eval_loader1 = loader.run('eval_train_net1')
eval_loader2, poisoned_idx, noisy_idx, poisoned_vector = loader.run('eval_train_net2')
warmup_trainloader = loader.run('warmup')


if start_epoch2 == 0:
    print('\nWarmup Net2')
    warmup(args, 0, net2, optimizer2, scalar_net2, warmup_trainloader)
    pre_loss = []

for epoch in range(start_epoch2+1, args.warm_up):
    print('\nWarmup Net2')
    threshold = max(1-0.1*epoch, 0.4)
    prob2, pred2, pre_loss = eval_train(args, net2, pre_loss, eval_loader2, poisoned_vector, epoch-1, threshold)

    labeled_loader, _ = loader.run('train_net2', 1 - pred2, 1 - prob2)
    train_poisoned_model(args, epoch, net2, optimizer2, scalar_net2, labeled_loader)
    print('\nACC:')
    ACC = test(epoch, net1, net2, test_CL_loader1, test_CL_loader2, test_log)
    print('ASR:')
    ASR = test(epoch, net1, net2, test_BD_loader1, test_BD_loader2, test_log)


prob2, pred2, pre_loss = eval_train(args, net2, pre_loss, eval_loader2, poisoned_vector, args.warm_up-1, 0.4)
labeled_credible_trainloader, unlabeled_suspicious_trainloader = loader.run('train_net1', pred2, prob2)

for epoch in range(start_epoch1+1, args.num_epochs+1):
    print('\nTrain Net1')
    if epoch <= 100:
        train_attention_mix(args, epoch, net1, optimizer1, scalar_net1, labeled_credible_trainloader)  # train net1
    else:
        train_attention_mix_mixmatch(args, epoch, net1, net2, optimizer1, scalar_net1, labeled_credible_trainloader, unlabeled_suspicious_trainloader, prob2, pred2, loader)
    scheduler1.step()

    if epoch % 5 == 0 or epoch > 100:
        print('\nTrain Net2')
        labeled_loader, _ = loader.run('train_net2', 1-pred2, 1-prob2)
        train_poisoned_model(args, epoch, net2, optimizer2, scalar_net2, labeled_loader)

        prob2, pred2, pre_loss = eval_train(args, net2, pre_loss, eval_loader2, poisoned_vector, epoch+args.warm_up-1, 0.4)
        labeled_credible_trainloader, unlabeled_suspicious_trainloader = loader.run('train_net1', pred2, prob2)
        save_state(epoch, net2, optimizer2, scalar_net2, poisoned_idx, noisy_idx, pre_loss, threshold, args.storage_path + 'epoch' + str(epoch) + '_net2.pth')

    if epoch % 1 == 0:
        if epoch % 10 == 0:
            save_state(epoch, net1, optimizer1, scalar_net1, poisoned_idx, noisy_idx, None, None, args.storage_path + 'epoch' + str(epoch) + '_net1.pth')
        print('\nACC:')
        ACC = test(epoch, net1, net2, test_CL_loader1, test_CL_loader2, test_log)
        print('ASR:')
        ASR = test(epoch, net1, net2, test_BD_loader1, test_BD_loader2, test_log)
        if ACC[0] > best_acc:
            save_state(epoch, net1, optimizer1, scalar_net1, poisoned_idx, noisy_idx, None, None, args.storage_path + 'best_acc_net1.pth')
            best_acc = ACC[0]
            asr = ASR[0]
        print('Best Acc: %.2f, ASR: %.2f' % (best_acc, asr))



