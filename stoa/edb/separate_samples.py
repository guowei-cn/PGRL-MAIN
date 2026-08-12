import sys
import os
home_folder = os.path.join(os.getcwd().split('ebd')[0], 'ebd')
sys.path.append(home_folder)

home_folder = os.path.join(os.getcwd().split('PGRL-main')[0], 'PGRL-main')
sys.path.append(home_folder)
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


def calculate_metrics(ground_true_l, prediction_l):
    TP = sum((gt == 1 and pred == 1) for gt, pred in zip(ground_true_l, prediction_l))
    FP = sum((gt == 0 and pred == 1) for gt, pred in zip(ground_true_l, prediction_l))
    TN = sum((gt == 0 and pred == 0) for gt, pred in zip(ground_true_l, prediction_l))
    FN = sum((gt == 1 and pred == 0) for gt, pred in zip(ground_true_l, prediction_l))

    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0

    return TPR, FPR

def separate_samples(arg, trainloader, model, gamma_low, gamma_high):
    model.eval()
    clean_samples, poison_samples, suspicious_samples = [], [], []
    ground_true_l, prediction_l, feature_consistency_l = [], [], []
    for i, (inputs, labels, gt_labels, isCleans) in enumerate(trainloader):
        ground_true_l.append(0 if isCleans == True else 1) # poisoned samples are set as 1
        # if i % 1000 == 0:
        #     print("Processing samples:", i)
        inputs1, inputs2 = inputs[0], inputs[2]

        ### Prepare for saved ###
        img = inputs1
        img = img.squeeze()
        target = labels.squeeze()
        if arg.dataset == 'ultrasonic':
            img = np.transpose(img.numpy(), (1, 2, 0)).astype('float32')
        else:
            img = np.transpose((img * 255).cpu().numpy(), (1, 2, 0)).astype('uint8')
        target = target.cpu().numpy()

        inputs1, inputs2 = normalization(arg, inputs1), normalization(arg, inputs2)  # Normalize
        inputs1, inputs2, labels, gt_labels = inputs1.to(arg.device), inputs2.to(arg.device), labels.to(arg.device), gt_labels.to(arg.device)

        ### Features ###
        features_out = list(model.module.children())[:-1] # abandon FC layer
        modelout = nn.Sequential(*features_out).to(arg.device)
        features1, features2 = modelout(inputs1), modelout(inputs2)
        features1, features2 = features1.view(features1.size(0), -1), features2.view(features2.size(0), -1)

        ### Compare consistency ###
        feature_consistency = torch.mean((features1 - features2)**2, dim=1)
        feature_consistency = feature_consistency.detach().cpu()
        # calculate the auc here
        feature_consistency_l.append(feature_consistency)
        ### Separate samples ###
        if feature_consistency.item() <= gamma_low:
            flag = np.array(0)
            clean_samples.append((img, target, flag))
        elif feature_consistency.item() >= gamma_high:
            flag = np.array(2)
            poison_samples.append((img, target, flag))
        else:
            flag = np.array(1)
            suspicious_samples.append((img, target, flag))

        prediction_l.append(1 if flag == 2 else 0)


    tpr, fpr = calculate_metrics(ground_true_l, prediction_l)
    auc = roc_auc_score(np.array(ground_true_l), torch.cat(feature_consistency_l).cpu().detach().numpy())
    print('auc: {:.3f}'.format(auc))
    # draw the histogram of ground_true_l and feature_consistency_l
    print('tpr: {:.3f}, fpr: {:.3f}'.format(tpr, fpr))
    import matplotlib.pyplot as plt
    plt.hist([torch.cat(feature_consistency_l).cpu().detach().numpy()[i] for i in range(len(ground_true_l)) if ground_true_l[i]==0], bins=50, alpha=0.5, label='Clean')
    plt.hist([torch.cat(feature_consistency_l).cpu().detach().numpy()[i] for i in range(len(ground_true_l)) if ground_true_l[i]==1], bins=50, alpha=0.5, label='Poisoned')
    plt.savefig('consistency_histogram_tpr_{:.3f}_fpr_{:.3f}.png'.format(tpr, fpr))

    if arg.dataset == 'ultrasonic':
        dtype = np.dtype([
            ('image', 'float32', (100, 40, 3)),  # specify the image dimensions and type directly
            ('label', 'int64'),
            ('flag', 'int64')
        ])
    else:
        dtype = np.dtype([
            ('image', 'uint8', (32, 32, 3)),  # specify the image dimensions and type directly
            ('label', 'int64'),
            ('flag', 'int64')
        ])
    ### Save samples ###
    folder_path = os.path.join('./saved/separated_samples', 'poison_rate_'+str(arg.poison_rate), arg.dataset, arg.model, arg.trigger_type+'_'+str(arg.clean_ratio)+'_'+str(arg.poison_ratio))
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    data_path_clean = os.path.join(folder_path, 'clean_samples_tpr_{:.3f}_fpr_{:.3f}.npy'.format(tpr, fpr))
    data_path_poison = os.path.join(folder_path, 'poison_samples_tpr_{:.3f}_fpr_{:.3f}.npy'.format(tpr, fpr))
    data_path_suspicious = os.path.join(folder_path, 'suspicious_samples_tpr_{:.3f}_fpr_{:.3f}.npy'.format(tpr, fpr))
    clean_array = np.array(clean_samples, dtype=dtype)
    poison_array = np.array(poison_samples, dtype=dtype)
    suspicious_array = np.array(suspicious_samples, dtype=dtype)

    np.save(data_path_clean, clean_array)
    np.save(data_path_poison, poison_array)
    np.save(data_path_suspicious, suspicious_array)


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
    separate_samples(arg, trainloader, model, gamma_low, gamma_high)

if __name__ == '__main__':
    main()
