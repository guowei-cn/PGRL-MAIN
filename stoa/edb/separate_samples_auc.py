import sys
import os

from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import numpy as np
import csv
from PIL import Image

import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from utils import args
from utils.utils import save_checkpoint, progress_bar, normalization
from utils.network import get_network
from utils.dataloader_bd import get_dataloader_train, get_dataloader_test
import matplotlib.pyplot as plt


def calculate_metrics(ground_true_l, prediction_l):
    TP = sum((gt == 1 and pred == 1) for gt, pred in zip(ground_true_l, prediction_l))
    FP = sum((gt == 0 and pred == 1) for gt, pred in zip(ground_true_l, prediction_l))
    TN = sum((gt == 0 and pred == 0) for gt, pred in zip(ground_true_l, prediction_l))
    FN = sum((gt == 1 and pred == 0) for gt, pred in zip(ground_true_l, prediction_l))

    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0

    return TPR, FPR


@torch.no_grad()
def separate_samples(arg, trainloader, model, gamma_low, gamma_high, save_path_prefix='feature_consistency'):
    model.eval()
    clean_samples, poison_samples, suspicious_samples = [], [], []
    ground_true_l, prediction_l, feature_consistency_l = [], [], []

    for i, (inputs, labels, gt_labels, isCleans) in enumerate(trainloader):
        ground_true_l.append(0 if isCleans else 1)  # 1 = poisoned

        if i % 1000 == 0:
            print("Processing samples:", i)

        # Unpack and prepare images
        inputs1, inputs2 = inputs[0], inputs[2]
        img = inputs1.squeeze()
        if arg.dataset == 'ultrasonic':
            img = np.transpose(img, (1, 2, 0)).astype('float32')
        else:
            img = np.transpose((img * 255).cpu().numpy(), (1, 2, 0)).astype('uint8')
        target = labels.squeeze().cpu().numpy()

        # Normalize and move to device
        inputs1 = normalization(arg, inputs1).to(arg.device)
        inputs2 = normalization(arg, inputs2).to(arg.device)
        labels = labels.to(arg.device)
        gt_labels = gt_labels.to(arg.device)

        # Extract features (drop final FC)
        features_backbone = nn.Sequential(*list(model.module.children())[:-1]).to(arg.device)
        f1 = features_backbone(inputs1).view(inputs1.size(0), -1)
        f2 = features_backbone(inputs2).view(inputs2.size(0), -1)

        # Compute consistency
        fc = torch.mean((f1 - f2) ** 2, dim=1)
        feature_consistency_l.append(fc.cpu())

        # Classify by thresholds
        fc_item = fc.item()
        if fc_item <= gamma_low:
            flag = 0
            clean_samples.append((img, target, flag))
        elif fc_item >= gamma_high:
            flag = 2
            poison_samples.append((img, target, flag))
        else:
            flag = 1
            suspicious_samples.append((img, target, flag))

        prediction_l.append(0 if flag == 0 else 1)

    # Convert lists to arrays
    y_true = np.array(ground_true_l)
    all_fc = np.concatenate([x.numpy() if isinstance(x, torch.Tensor) else x for x in feature_consistency_l])

    # Compute ROC AUC
    auc = roc_auc_score(y_true, all_fc)
    print(f'AUC: {auc:.3f}')

    # Save feature consistency scores
    fc_clean = all_fc[y_true == 0]
    fc_poison = all_fc[y_true == 1]
    np.save(f'{save_path_prefix}_clean.npy', fc_clean)
    np.save(f'{save_path_prefix}_poison.npy', fc_poison)
    print(f"Feature consistency scores saved to {save_path_prefix}_clean.npy and {save_path_prefix}_poison.npy")

    # Print average consistency per class
    print(f'Avg consistency — Clean: {fc_clean.mean():.4f}, Poisoned: {fc_poison.mean():.4f}')


def main():
    global arg
    arg = args.get_args()

    # Dataset
    trainloader = get_dataloader_train(arg)

    # Prepare backdoored model, optimizer, scheduler
    model = get_network(arg)
    model = torch.nn.DataParallel(model).cuda()
    checkpoint = torch.load(arg.checkpoint_load)
    print("Continue training...")
    model.load_state_dict(checkpoint['model'])

    # Separate samples
    gamma_low = arg.gamma_low
    gamma_high = arg.gamma_high
    separate_samples(arg, trainloader, model, gamma_low, gamma_high, save_path_prefix='{}'.format(arg.dataset))

if __name__ == '__main__':
    main()
