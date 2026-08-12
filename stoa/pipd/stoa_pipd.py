import os, sys
home_folder = os.path.join(os.getcwd().split('PGRL-main')[0], 'PGRL-main')
sys.path.append(home_folder)

import time
import math
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
import jenkspy
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from lib.dataLoader import get_dataset2
from stoa.abl.stoa_abl import resnet18, unlearning, calculate_tpr_fpr


import numpy as np
import torch
debugging_flag = False

def analyze_neuros(model, num_classes, trainloader_no_shuffle, args,
                   sure_clean=None, last_poison_preset=None, last_clean_preset=None, conv=None):
    model.eval()
    loader = trainloader_no_shuffle
    num_for_detect_biased = 0.01

    conv_list = [
        "Layer2_0_Conv1_",
        "Layer2_0_Downsample_", "Layer2_0_Conv2_", "Layer2_1_Conv1_", "Layer2_1_Conv2_",
        "Layer3_0_Conv1_", "Layer3_0_Downsample_", "Layer3_0_Conv2_", "Layer3_1_Conv1_", "Layer3_1_Conv2_",
        "Layer4_0_Conv1_", "Layer4_0_Downsample_",
        "Layer4_0_Conv2_",
        "Layer4_1_Conv1_",
        "Layer4_1_Conv2_"
    ]
    channel_num_list = [
        128,
        128, 128, 128, 128,
        256, 256, 256, 256, 256,
        512, 512,
        512,
        512,
        512
    ]
    if conv is not None:
        conv_list = conv_list[conv:]
        channel_num_list = channel_num_list[conv:]

    imgs_high_activation_times = np.zeros([len(loader.dataset)])
    conv_activation_all_list = []

    # initalize the conv_activation_all_list
    for i in range(len(conv_list)):
        channel_num = channel_num_list[i]
        conv_activation_all_list.append(np.zeros([len(loader.dataset), channel_num]))

    output_all = np.zeros([len(loader.dataset), num_classes])

    counter = 0

    poison_preset_indices = last_poison_preset
    clean_preset_indices = last_clean_preset
    if sure_clean is not None:
        clean_preset_indices = np.concatenate((clean_preset_indices, np.array(sure_clean)), axis=0)
    clean_preset_indices = list(set(clean_preset_indices))

    # generate activation maps
    for batch in loader:
        data, target = batch[0], batch[1]
        data, target = Variable(data.to(args.device)), Variable(target.to(args.device))
        output = model(data)
        pred = F.softmax(output)
        batch_size = data.shape[0]
        output_all[counter:counter + batch_size] = pred.cpu().detach().numpy()
        for i in range(len(conv_list)):
            conv_name = conv_list[i]
            if "FC" in conv_name:
                conv_activation_all_list[i][counter:counter + batch_size] = model.inter_feature[
                    conv_name].cpu().detach().numpy()
            else:
                conv_activation_all_list[i][counter:counter + batch_size] = model.inter_feature[conv_name].max(
                    -1).values.max(-1).values.cpu().detach().numpy()
        counter = counter + batch_size

    # initialize the diff_class_channel_numpy_list
    diff_class_channel_numpy_list = []
    for i in range(len(conv_list)):
        channel_num = channel_num_list[i]
        diff_class_channel_numpy_list.append(np.zeros([channel_num]))

    strong_output_indexs = poison_preset_indices
    weak_output_indexs = clean_preset_indices

    for i in range(len(conv_list)):
        channel_num = channel_num_list[i]
        for j in range(channel_num):
            strong_output_conv_activation_max = conv_activation_all_list[i][strong_output_indexs, j].max()
            statistics_for_base_distribution = np.mean(
                conv_activation_all_list[i][weak_output_indexs, j]) + 3 * np.std(
                conv_activation_all_list[i][weak_output_indexs, j])
            diff = strong_output_conv_activation_max - statistics_for_base_distribution
            diff_class_channel_numpy_list[i][j] = diff

        diff_channel_numpy = diff_class_channel_numpy_list[i]
        channel_sorted = diff_channel_numpy.argsort()[::-1]

        if num_for_detect_biased == -1:
            top_channel_calculate_biased_imgs = channel_sorted[:1]
        else:
            top_channel_calculate_biased_imgs = channel_sorted[:math.ceil(num_for_detect_biased * channel_num)]

        for j in top_channel_calculate_biased_imgs:
            imgs_high_activation_times[:] = imgs_high_activation_times[:] + conv_activation_all_list[i][:, j]

    breaks = jenkspy.jenks_breaks(imgs_high_activation_times, n_classes=2)
    print(breaks)
    poison_sample_index = np.where(imgs_high_activation_times >= breaks[1])[0].tolist()
    print("number of detected Trojan samples:", len(poison_sample_index))
    return poison_sample_index, poison_preset_indices, clean_preset_indices, imgs_high_activation_times


def check_file_with_prefix(directory, prefix):
    # List all files in the specified directory
    files = os.listdir(directory)

    # Find files that start with the specified prefix
    matching_files = [f for f in files if f.startswith(prefix)]

    if matching_files:
        # If there are matching files, print the full path of the first one found
        full_path = os.path.join(directory, matching_files[0])
        print(f"Found file: {full_path}")
        return full_path
    else:
        # If no matching file is found, print the message and exit
        print(f"No file found with prefix '{prefix}' in directory '{directory}'")
        sys.exit(1)  # Exit with status code 1 indicating an error


def main(opt):
    args_str = '_'.join('{}'.format(value) for _, value in vars(args).items())
    writer = SummaryWriter(comment='{}_args_{}'.format(os.path.basename(__file__), args_str))
    print(args)
    # TODO: replace by our dataloader
    opt.batch_size, opt.num_workers = 64, 0
    transforms = False
    # poisoned_data_loader, test_clean_loader, test_bad_loader, train_folder = get_dataset2(opt.poison_type,
    #                                                                                      opt.poison_or_benign,
    #                                                                                      opt.poison_rate,
    #                                                                                      opt.cover_rate,
    #                                                                                     opt.batch_size,
    #                                                                                      opt.num_class, opt.num_workers,
    #                                                                                      transforms=transforms)
    poisoned_data_loader, test_clean_loader, test_bad_loader, train_folder = get_dataset2(
        opt.poison_type, opt.poison_or_benign, opt.poison_rate, opt.batch_size, opt.num_class, opt.num_workers,
        transforms=False, image_size=32)

    gt_benign_indices = poisoned_data_loader.dataset.benign_indics
    gt_poison_indices = np.array(list(set(range(len(poisoned_data_loader.dataset))) - set(gt_benign_indices)))
    trainloader_no_shuffle = DataLoader(dataset=poisoned_data_loader.dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    if opt.poison_type == 'ultrasonic':
        model = resnet18(num_classes=opt.num_class, in_channels=1, fc_in_channel=1536) # Don't use CNN(num_classes=10) since PIPD use this resnet for activation analysis
    elif opt.poison_type == 'imagenette':
        model = resnet18(num_classes=opt.num_class, fc_in_channel=25088)
    else:
        model = resnet18(num_classes=opt.num_class)
    # # load the trained model and poisoned indices
    opt.tuning_epochs = 10
    model_path = '/storageA/david_projects/PGRL-main/poisonDataset/{}/abl_train_{}_{}_{}_{}_{}.pth'.format(
                   opt.poison_type, opt.num_class,
                   opt.poison_or_benign,
                   opt.poison_rate, opt.cover_rate, opt.tuning_epochs)
    # TODO: check whether model_path exist or not
    print(model_path)
    if not os.path.isfile(model_path):
        print(f"No file found at {model_path}")
        sys.exit(1)  # Exit with status code 1 indicating an error
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.to(opt.device)
    # load benign_indics
    benign_indices_path = '/storageA/david_projects/PGRL-main/poisonDataset/{}/abl_pd_benign_indics_{}_{}_{}_{}_{}'.format(
        opt.poison_type, opt.num_class,
        opt.poison_or_benign, opt.cover_rate,
        opt.poison_rate, opt.tuning_epochs)
    # TODO: check whether the folder includes benign indices
    # get the file with prefix of benign_indices_path
    directory = os.path.dirname(benign_indices_path)
    prefix = os.path.basename(benign_indices_path)

    benign_indices_path = check_file_with_prefix(directory, prefix)

    clean_preset = np.load(benign_indices_path)
    poison_preset = np.array(list(set(range(len(poisoned_data_loader.dataset))) - set(clean_preset)))
    sure_clean = None
    start_t = time.time()
    print('strart the time check')
    for round_id in range(1, 5 + 1):
        poison_sample_index, poison_preset, clean_preset, imgs_high_activation_times = analyze_neuros(
            model,
            10,
            trainloader_no_shuffle,
            args,
            sure_clean=sure_clean,
            last_poison_preset=poison_preset,
            last_clean_preset=clean_preset)

        final_poison_decision = np.setdiff1d(poison_sample_index, clean_preset)
        tp = np.intersect1d(gt_poison_indices, poison_sample_index)
        print(f'tp: {len(tp)}, iso total: {len(poison_sample_index)}, all: {len(gt_poison_indices)}')
        hk = np.arange(0, len(poisoned_data_loader.dataset))
        sure_clean = np.setdiff1d(hk, final_poison_decision)

        if debugging_flag == True:
            break

    print('act time {}'.format(time.time() - start_t))
    # exit()
    # TODO: calculate the tpr and fpr
    tpr, fpr = calculate_tpr_fpr(final_poison_decision, gt_benign_indices, len(poisoned_data_loader.dataset))
    print(tpr, fpr)
    # save the predicted benign index
    np.save('/storageA/david_projects/PGRL-main/poisonDataset/{}/pipd_pd_benign_indics_{}_{}_{}_{}_tpr_fpr_poison_{}_{}.pth'.format(
                   opt.poison_type, opt.num_class,
                   opt.poison_or_benign,
                   opt.poison_rate, opt.tuning_epochs, tpr, fpr), sure_clean)
    #
    # opt.finetuning_ascent_model, opt.method = True, 'pipd'
    # # initialize optimizer
    # opt.lr, opt.momentum, opt.weight_decay = 0.1, 0.9, 1e-4
    # optimizer = torch.optim.SGD(model.parameters(),
    #                             lr=opt.lr,
    #                             momentum=opt.momentum,
    #                             weight_decay=opt.weight_decay,
    #                             nesterov=True)
    # criterion = nn.CrossEntropyLoss().to(opt.device)
    # pd_poison_index, pd_benign_index = list(set(range(len(poisoned_data_loader.dataset))) - set(sure_clean)), list(set(sure_clean))
    # # unlearning
    # epoch_acc_asr = []
    #
    # opt.finetuning_ascent_model, opt.method = True, 'abl'
    # opt.finetuning_epochs, opt.unlearning_epochs = 180, 10
    # opt.lr_finetuning_init, opt.unlearning_lr, opt.lr = 0.1, 5e-4, 0.1
    # epoch_acc_asr = unlearning(model, poisoned_data_loader, test_clean_loader, test_bad_loader, optimizer, criterion, opt,
    #            writer, pd_poison_index, pd_benign_index, epoch_acc_asr)
    # if opt.efficiency_analysis:
    #     np.save('/storageA/david_projects/PGRL-main/poisonDataset/{}/pipd_train_epoch_acc.npy'.format(opt.poison_type), np.array(epoch_acc_asr))


if __name__ == '__main__':
    import argparse

    def parse_args():
        parser = argparse.ArgumentParser(description='Parse command-line arguments for poisoning and augmentation.')
        parser.add_argument('-t', '--poison_type', required=True, type=str, help='Specify the type of poisoning.')
        parser.add_argument('-class', '--num_class', required=True, type=int, help='The number of classes.')
        parser.add_argument('-pb', '--poison_or_benign', required=True, type=str,
                            help='Specify whether the data is poison or benign.')
        parser.add_argument('-d', '--device', default='cuda:0', type=str,
                            help='The device to use (e.g., "cpu" or "cuda").')
        parser.add_argument('-pr', '--poison_rate', default=0, type=float, help='The rate of poisoning.')
        parser.add_argument('-cr', '--cover_rate', default=0, type=float, help='The rate of poisoning.')
        parser.add_argument('-effana', '--efficiency_analysis', default=False, type=bool,
                            help='Analyse the training efficiency by saving acc at intervals of every five epochs')
        return parser.parse_args()

    args = parse_args()
    main(args)