import argparse
import os

classes_10 = ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes']


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--device', type=str, default='cuda:0', help='cuda, cpu')
    parser.add_argument('--checkpoint_load', type=str, default=None)
    parser.add_argument('--checkpoint_save', type=str, default=None)
    parser.add_argument('--log', type=str, default=None)
    parser.add_argument("--data_root", type=str, default='./dataset/')

    parser.add_argument('--dataset', type=str, default='cifar10', help='cifar10, cifar100, imagenet, freq, freq_meg_2000, pattern, wanet')
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--input_height", type=int, default=None)
    parser.add_argument("--input_width", type=int, default=None)
    parser.add_argument("--input_channel", type=int, default=None)

    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument("--num_workers", type=float, default=0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')
    parser.add_argument('--poison_rate', type=float, default=0.1) # decides how many training samples are poisoned
    parser.add_argument('--cover_rate', type=float, default=0.0) # \alpha_p

    parser.add_argument('--clean_rate', type=float, default=1.0) # decides how many clean training samples are provided in some defense methods
    parser.add_argument('--target_type', type=str, default='all2one', help='all2one, all2all, cleanLabel') 
    parser.add_argument('--target_label', type=str, default=0)
    parser.add_argument('--trigger_type', type=str, default='gridTrigger', help='gridTrigger, squareTrigger, trojanTrigger, signalTrigger, kittyTrigger, sigTrigger, fourCornerTrigger')

    # Others
    parser.add_argument('--model', type=str, default='resnet18')

    parser.add_argument('--gamma_low', type=float, default=None, help='<=gamma_low is clean') # \gamma_c
    parser.add_argument('--gamma_high', type=float, default=None, help='>=gamma_high is poisoned') # \gamma_p
    parser.add_argument('--clean_ratio', type=float, default=0.20, help='ratio of clean data') # \alpha_c
    parser.add_argument('--poison_ratio', type=float, default=0.05, help='ratio of poisoned data') # \alpha_p

    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--schedule', type=int, nargs='+', default=[100, 150], help='Decrease learning rate at these epochs.')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')

    parser.add_argument('--trans1', type=str, default='rotate') # the first data augmentation
    parser.add_argument('--trans2', type=str, default='affine') # the second data augmentation
    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    arg = parser.parse_args()

    # Set image class and size
    if arg.dataset == "cifar10" or arg.dataset == 'adaptivecifar10' or arg.dataset == 'blto' or 'freq' in arg.dataset \
            or 'pattern' in arg.dataset or 'wanet' in arg.dataset or 'imagenette' in arg.dataset:
        arg.num_classes = 10
        arg.input_height = 32
        arg.input_width = 32
        arg.input_channel = 3
        arg.target_label = int(arg.target_label)
    elif arg.dataset == 'ultrasonic':
        arg.num_classes = 10
        arg.input_height = 100
        arg.input_width = 40
        arg.input_channel = 1
        arg.target_label = classes_10.index(arg.target_label)
    elif arg.dataset == "cifar100":
        arg.num_classes = 100
        arg.input_height = 32
        arg.input_width = 32
        arg.input_channel = 3
    elif arg.dataset == "imagenet":
        arg.num_classes = 200
        arg.input_height = 224
        arg.input_width = 224
        arg.input_channel = 3
    else:
        raise Exception("Invalid Dataset")

    arg.data_root = arg.data_root + arg.dataset    
    if not os.path.isdir(arg.data_root):
        os.makedirs(arg.data_root)
    print(arg)
    return arg
