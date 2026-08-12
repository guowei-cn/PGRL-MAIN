import os, sys

from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from stoa_dbd.SimCLR.simclr import SimCLRModel

home_folder = os.path.join(os.getcwd().split('DefTimeSeries')[0], 'DefTimeSeries')
sys.path.append(home_folder)


import numpy as np
import umap
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


from lib.dataLoader import get_dataset_info, ToTargetClass

import shutil

import torch

from data.dataset import SelfPoisonDataset, PoisonLabelDataset, PoisonLabelDataset_audio, SelfPoisonDataset_audio
from data.utils import (
    get_loader,
    get_transform,
)
from model.model import SelfModel
from model.utils import (
    get_criterion,
    get_network,
    get_optimizer,
    get_scheduler,
)
from utils.setup import (
    get_saved_dir,
    get_storage_dir,
    load_config,
    set_seed,
)

from utils.trainer.simclr import simclr_train
debugging_flag = False

def main(args):
    args.train_folder, args.target_name = get_dataset_info(args.poison_type, args.poison_or_benign, args.poison_rate)
    args.train_folder = '../poisonDataset/{}/{}'.format(args.poison_type, args.train_folder)
    args.clean_test_folder = '../poisonDataset/{}/Test'.format(args.poison_type)
    args.poison_test_folder = '../poisonDataset/{}/poisonTest'.format(args.poison_type)
    args.config = 'config/defense/simclr/badnets/cifar10_resnet18/example.yaml'
    args.target_class = ToTargetClass(target_name=args.target_name, num_classes=args.num_class,
                                      poison_type=args.poison_type).target_class
    args.resume = False


    config, inner_dir, config_name = load_config(args.config)
    args.saved_dir, args.log_dir = get_saved_dir(
        config, inner_dir, config_name, args.resume
    )
    shutil.copy2(args.config, args.saved_dir)
    args.storage_dir, args.ckpt_dir, _ = get_storage_dir(
        config, inner_dir, config_name, args.resume
    )
    shutil.copy2(args.config, args.storage_dir)

    print("Training on: {}.".format(args.device))
    main_worker(args.device, args, config)


@torch.no_grad()
def get_feature(model, loader):
    model.eval()
    features, labels = [], []
    for batch_idx, batch in enumerate(loader):
        data = batch["img"].to(args.device)
        target = batch["target"].to(args.device)
        feature = model(data) # there is a normalization in the model
        features.append(feature)
        labels.append(target)

    features = torch.cat(features)
    labels = torch.cat(labels)
    ave_features, ave_labels = [], []
    for l in set(labels.tolist()):
        ave_f = torch.mean(features[labels==l], dim=0)
        ave_features.append(ave_f)
        ave_labels.append(torch.tensor(l))

    ave_features, ave_labels = torch.stack(ave_features), torch.stack(ave_labels)

    return ave_features, ave_labels, features, labels


@torch.no_grad()
def normalised_similairty(net, test_data_loader, backdoor_loader, epoch):
    net.eval()
    hide_progress = True
    features, labels = [], []
    # loop test data to predict the label by weighted knn search
    for batch_idx, batch in enumerate(test_data_loader):
        data = batch["img"].to(args.device)
        target = batch["target"].to(args.device)
        feature = net(data)
        # feature: [bsz, dim]
        features.append(feature)
        labels.append(target)

    features = torch.cat(features)
    labels = torch.cat(labels)

    ave_features, ave_labels = [], []
    for l in set(labels.tolist()):
        ave_f = torch.mean(features[labels == l], dim=0)
        ave_features.append(ave_f)
        ave_labels.append(torch.tensor(l).to(ave_f.device))

    ave_features, ave_labels = torch.stack(ave_features), torch.stack(ave_labels)

    p_features, p_labels = [], []
    for batch_idx, batch in enumerate(backdoor_loader):
        data = batch["img"].to(args.device)
        target = batch["target"].to(args.device)
        feature = net(data)
        # feature: [bsz, dim]
        p_features.append(feature)
        p_labels.append(target)

    p_features = torch.cat(p_features)
    p_labels = torch.cat(p_labels)
    assert len(set(p_labels.tolist())) == 1
    target_class = p_labels[0]
    ave_p_features, ave_p_labels = [], []
    for l in set(p_labels.tolist()):
        ave_f = torch.mean(p_features[p_labels == l], dim=0)
        ave_p_features.append(ave_f)
        ave_p_labels.append(torch.tensor(l).to(ave_f.device))

    ave_p_features, ave_p_labels = torch.stack(ave_p_features), torch.stack(ave_p_labels)
    # calculate the similairty
    cos_sim = torch.nn.functional.cosine_similarity(ave_p_features[ave_p_labels == target_class],
                                                    ave_features[ave_labels == target_class])
    ave_sim = 0
    for l in set(ave_labels.tolist()):
        ave_sim += torch.nn.functional.cosine_similarity(ave_p_features[ave_p_labels == target_class],
                                                         ave_features[ave_labels == l])

    ave_sim /= len(set(ave_labels.tolist()))

    normalised_sim = cos_sim / ave_sim

    # print('test/normalised_sim: {} at {}'.format(normalised_sim.item(), epoch))

    return normalised_sim

def majority_vote(sim_labels):
    # Count occurrences of each label in each row
    counts = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=sim_labels)
    return torch.tensor(counts)


def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # feature: [bsz, dim]
    # feature_bank: [dim, total_num]
    # feature_labels: [total_num]

    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # sim_matrix: [bsz, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)

    # sim_labels: [bsz, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    my_pred_labels = majority_vote(sim_labels.cpu().detach().numpy())
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # one_hot_label: [bsz*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [bsz, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return my_pred_labels.to(pred_labels.device)


def vis_distribution(feature_bank, target_bank, test_feature_bank, test_target_bank, poison_feature_bank, poison_target_bank, target_class):
    target_feature_bank = feature_bank
    test_feature_bank = test_feature_bank[test_target_bank == target_class][1:2]
    poison_feature_bank = poison_feature_bank[poison_target_bank == target_class][1:2]

    # Concatenate the feature banks
    feature = np.concatenate([target_feature_bank, test_feature_bank, poison_feature_bank])

    # Apply t-SNE to reduce dimensionality to 2
    umap_reducer = umap.UMAP(n_components=2, metric='cosine', n_neighbors=200, min_dist=0.99)
    feature_2d = umap_reducer.fit_transform(feature)

    # pca = PCA(n_components=2, random_state=42)
    # feature_2d = pca.fit_transform(feature)

    # Split the reduced features into target, test, and poison
    target_feature_2d = feature_2d[:target_feature_bank.shape[0]]
    test_feature_bank_2d = feature_2d[target_feature_bank.shape[0]:target_feature_bank.shape[0] + test_feature_bank.shape[0]]
    poison_feature_bank_2d = feature_2d[target_feature_bank.shape[0] +  + test_feature_bank.shape[0]:]

    # Plot function for convex hull and scatter
    def plot_hull_for_target(feature_2d, label, color):
        # Plot points for target class
        plt.scatter(feature_2d[:, 0], feature_2d[:, 1], label=f'{label} Points', alpha=0.5, color=color)
        #
        # # Compute and plot convex hull for target data
        # if feature_2d.shape[0] > 2:  # ConvexHull needs at least 3 points
        #     hull = ConvexHull(feature_2d)
        #     for simplex in hull.simplices:
        #         plt.plot(feature_2d[simplex, 0], feature_2d[simplex, 1], color=color)
        #     plt.fill(feature_2d[hull.vertices, 0], feature_2d[hull.vertices, 1], color=color, alpha=0.2,
        #              label=f'{label} Hull Area')

    # Plot function for scatter points only (test and poison data)
    def plot_scatter(feature_2d, label, color):
        plt.scatter(feature_2d[:, 0], feature_2d[:, 1], label=f'{label} Points', alpha=0.5, color=color)

    # Plot the target, test, and poison data with convex hull only for target data
    plt.figure(figsize=(10, 8))

    # Plot the convex hull for target data
    colors = plt.cm.get_cmap('tab10', 10)
    for i in range(10):
        plot_hull_for_target(target_feature_2d[target_bank==i], label='Target {}'.format(i), color=colors(i))
    # plot_hull_for_target(target_feature_2d, label='Target {}'.format(target_class), color='blue')

    # plot_hull_for_target(target_feature_2d, label='Target', color='blue')
    # Plot the scatter points for test data (no convex hull)
    plot_scatter(test_feature_bank_2d, label='Test', color='green')

    # Plot the scatter points for poison data (no convex hull)
    plot_scatter(poison_feature_bank_2d, label='Poison', color='red')

    # Annotate test_feature_bank_2d
    for i in range(test_feature_bank_2d.shape[0]):
        plt.annotate('Test', xy=test_feature_bank_2d[i], xytext=(-20, 20), textcoords='offset points',
                     arrowprops=dict(arrowstyle='->', color='green'))

    # Annotate poison_feature_bank_2d
    for i in range(poison_feature_bank_2d.shape[0]):
        plt.annotate('Poison', xy=poison_feature_bank_2d[i], xytext=(-20, -20), textcoords='offset points',
                     arrowprops=dict(arrowstyle='->', color='red'))

    # Set title and show legend
    plt.title("t-SNE Visualization with Convex Hull for Target and Point Cloud for Test and Poison")
    plt.legend()
    plt.savefig(
        'dis.png'
    )


@torch.no_grad()
def knn_monitor_fre(net, memory_data_loader, test_data_loader, epoch, device, k=200, t=0.1, hide_progress=True,
                     classes=-1, subset=False, backdoor_loader=None, target_class=None):

    net.eval()

    total_top1, total_top5, total_num, feature_bank, feature_labels = 0.0, 0.0, 0, [], []
    # generate feature bank
    for batch_idx, batch in enumerate(memory_data_loader):
        data = batch["img"].to(device)
        if len(data.shape) == 3:
            data = data.unsqueeze(1)
        target = batch["target"].to(device)
        feature = net(data) # there is a normalization in the model
        feature_bank.append(feature)
        feature_labels.append(target)

    # feature_bank: [dim, total num]
    feature_bank = torch.cat(feature_bank, dim=0)
    # feature_labels: [total num]
    feature_labels = torch.cat(feature_labels, dim=0)


    # loop test data to predict the label by weighted knn search
    test_feature_bank, test_target_bank = [], []
    for batch_idx, batch in enumerate(test_data_loader):
        data = batch["img"].to(device)
        if len(data.shape) == 3:
            data = data.unsqueeze(1)
        target = batch["target"].to(device)
        feature = net(data)
        test_feature_bank.append(feature)
        test_target_bank.append(target)
        # feature: [bsz, dim]
        pred_labels = knn_predict(feature, feature_bank.t().contiguous(), feature_labels.t().contiguous(), classes, k, t)

        total_num += data.size(0)
        total_top1 += (pred_labels == target).float().sum().item()

    test_feature_bank = torch.cat(test_feature_bank)
    test_target_bank = torch.cat(test_target_bank)
    # frequency test data

        # if args.threatmodel == 'single-class' or args.threatmodel == 'single-poison':
    if backdoor_loader is not None:

        backdoor_top1, backdoor_num = 0.0, 0
        poison_feature_bank, poison_target_bank = [], []
        for batch_idx, batch in enumerate(backdoor_loader):
            data = batch["img"].to(device)
            if len(data.shape) == 3:
                data = data.unsqueeze(1)
            target = batch["target"].to(device)

            feature = net(data)
            poison_feature_bank.append(feature)
            poison_target_bank.append(target)
            # feature: [bsz, dim]
            pred_labels = knn_predict(feature, feature_bank.t().contiguous(), feature_labels.t().contiguous(), classes, k, t)

            backdoor_num += data.size(0)
            backdoor_top1 += (pred_labels == target).float().sum().item()

        poison_feature_bank = torch.cat(poison_feature_bank)
        poison_target_bank = torch.cat(poison_target_bank)
        vis_distribution(feature_bank.cpu().detach().numpy(), feature_labels.cpu().detach().numpy(),
                         test_feature_bank.cpu().detach().numpy(), test_target_bank.cpu().detach().numpy(),
                         poison_feature_bank.cpu().detach().numpy(),
                         poison_target_bank.cpu().detach().numpy(), target_class=target_class)

        return total_top1 / total_num * 100, backdoor_top1 / backdoor_num * 100

    return total_top1 / total_num * 100


def main_worker(device, args, config):
    writer = SummaryWriter()

    set_seed(**config["seed"])

    if args.poison_type == 'ultrasonic':
        test_transform = None
        # benign training dataset
        clean_tra_data = PoisonLabelDataset_audio(args.clean_test_folder.replace('Test', 'Train'), args.poison_type, test_transform, args.num_class)
        # benign test dataset
        clean_test_data = PoisonLabelDataset_audio(args.clean_test_folder, args.poison_type, test_transform, args.num_class)
        # Poison test dataset
        poison_test_data = PoisonLabelDataset_audio(args.poison_test_folder, args.poison_type, test_transform,
                                                    args.num_class, args.target_class)
        # Poison training dataset
        self_poison_train_data = SelfPoisonDataset_audio(args.train_folder, args.poison_type, args.num_class)
    else:
        pre_transform = get_transform(config["transform"]["pre"])
        aug_primary_transform = get_transform(config["transform"]["aug"]["primary"])
        aug_remaining_transform = get_transform(config["transform"]["aug"]["remaining"])
        aug_transform = {
            "pre": pre_transform,
            "primary": aug_primary_transform,
            "remaining": aug_remaining_transform,
        }
        # test dataset and poisoned test dataset
        test_primary_transform = get_transform(config["transform"]["test"]["primary"])
        test_remaining_transform = get_transform(config["transform"]["test"]["remaining"])
        test_transform = {
            "pre": pre_transform,
            "primary": test_primary_transform,
            "remaining": test_remaining_transform,
        }
        # benign training dataset
        clean_tra_data = PoisonLabelDataset(args.clean_test_folder.replace('Test', 'Train'), args.poison_type, test_transform)
        # benign test dataset
        clean_test_data = PoisonLabelDataset(args.clean_test_folder, args.poison_type, test_transform)
        # Poison test dataset
        poison_test_data = PoisonLabelDataset(args.poison_test_folder, args.poison_type, test_transform, args.target_class)
        self_poison_train_data = SelfPoisonDataset(args.train_folder, args.poison_type, aug_transform)

    memory_loader = DataLoader(clean_tra_data, batch_size=128, shuffle=False, num_workers=4)

    clean_test_loader = DataLoader(clean_test_data, batch_size=128, shuffle=False, num_workers=4)
    poison_test_loader = DataLoader(poison_test_data, batch_size=128, shuffle=False, num_workers=4)
    self_poison_train_loader = get_loader(
        self_poison_train_data, config["loader"], shuffle=True
    )

    backbone = get_network(config["network"], args.poison_type)
    self_model = SelfModel(backbone)

    if os.path.exists('../poisonDataset/{}/cacheDBD_simCLR_{}_{}.pth'.format(args.poison_type, args.poison_or_benign, args.poison_rate)):
        self_model.load_state_dict(torch.load('../poisonDataset/{}/cacheDBD_simCLR_{}_{}.pth'.format(args.poison_type, args.poison_or_benign, args.poison_rate), map_location='cpu'))
        self_model = self_model.to(device)


    criterion = get_criterion(config["criterion"])
    criterion = criterion.to(device)
    optimizer = get_optimizer(self_model, config["optimizer"])
    scheduler = get_scheduler(optimizer, config["lr_scheduler"])

    best_back_acc = -1
    epoch = 0
    knn_acc, back_acc = knn_monitor_fre(self_model,
                                             memory_loader, clean_test_loader, epoch, args.device,
                                             classes=args.num_class,
                                             subset=False,
                                             backdoor_loader=poison_test_loader, target_class=args.target_class
                                             )
    # # calculate the normalised similarity
    # normalised_sim = normalised_similairty(self_model, clean_test_loader, poison_test_loader, epoch)
    print(
        '[{}-epoch] knn acc: {:.3f} | back acc: {:.3f} '.format(
            epoch + 1,
            knn_acc, back_acc))


if __name__ == "__main__":
    import argparse


    def parse_args():
        parser = argparse.ArgumentParser(description='Parse command-line arguments for poisoning and augmentation.')
        parser.add_argument('-t', '--poison_type', required=True, type=str, help='Specify the type of poisoning.')
        parser.add_argument('-class', '--num_class', required=True, type=int, help='The number of classes.')
        parser.add_argument('-pb', '--poison_or_benign', required=True, type=str, help='Specify whether the data is poison or benign.')
        parser.add_argument('-d', '--device', default='cuda:0', type=str, help='The device to use (e.g., "cpu" or "cuda").')

        parser.add_argument('-pr', '--poison_rate', default=0, type=float, help='The rate of poisoning.')

        return parser.parse_args()

    args = parse_args()
    main(args)

