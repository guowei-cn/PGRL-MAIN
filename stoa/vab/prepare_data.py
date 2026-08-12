import os.path

from torchvision import datasets
from bd_transforms import BadNet, Blend, SIG, WaNet, WaNet_noisy, Benign
import numpy as np
import PIL.Image as Image
import shutil

def load_init_data(dataset_path):
    train_data = datasets.ImageFolder(root=dataset_path + 'train')
    test_data = datasets.ImageFolder(root=dataset_path + 'val')
    return train_data, test_data

def get_poison_idx(data, targets, mode, trigger_label, trigger_type, portion, poisoned_idx, noisy_idx):
    if mode == 'train':
        print("## generate training data")
        if trigger_type == 'badnet' or trigger_type == 'blended' or trigger_type == 'badnet_':
            poisoned_idx = np.random.permutation(len(data))[0: int(len(data) * portion)]
        elif trigger_type == 'SIG':
            index_target = np.where(np.array(targets) == trigger_label)[0]
            np.random.shuffle(index_target)
            poisoned_idx = index_target[0: int(len(index_target) * portion)]
        elif trigger_type == 'WaNet':
            index = np.random.permutation(len(data))
            noisy_ratio = 0.2
            poisoned_idx = index[: int(len(data) * portion)]
            noisy_idx = index[int(len(data) * portion): int(len(data)* (portion + noisy_ratio))]

    elif mode == 'ACC test':
        print("## generate Acc testing data")
        poisoned_idx = np.array([])

    elif mode == 'ASR test':
        print("## generate ASR testing data")
        index_not_target = np.where(np.array(targets) != trigger_label)[0]
        new_data = [data[i] for i in index_not_target]
        new_targets = [targets[i] for i in index_not_target]
        poisoned_idx = range(0, len(new_data))  # np.random.permutation(len(self.data))[0: int(len(self.data) * portion)]
        data = new_data
        targets = np.array(new_targets).astype(np.int64)

    print("Injecting Over: %d Bad Imgs, %d Clean Imgs (%.2f)" % (len(poisoned_idx), len(data) - len(poisoned_idx), portion))
    return poisoned_idx, noisy_idx, data, targets

def prepare_data(args):
    train_data, test_data = load_init_data(dataset_path=args.data_path)
    class_num = len(train_data.classes)
    trigger_type = args.trigger_type
    trigger_path = args.trigger_path

    if trigger_type == 'badnet':
        bd_transform = BadNet(trigger_path)
    elif trigger_type == 'blended':
        bd_transform = Blend(trigger_path)
    elif trigger_type == 'SIG':
        bd_transform = SIG(trigger_path)
    elif trigger_type == 'WaNet':
        bd_transform = WaNet(trigger_path)
        bd_transform_noisy = WaNet_noisy(trigger_path)
    elif trigger_type == 'Benign':
        bd_transform = Benign()

    #----------------------------------------train data--------------------------------------
    imgs = train_data.imgs
    data = []
    targets = []
    for i in range(len(imgs)):
        data.append(imgs[i][0])
        targets.append(imgs[i][1])
    targets = np.array(targets).astype(np.int64)
    poisoned_idx, noisy_idx, data, targets = get_poison_idx(data, targets, 'train', args.trigger_label, trigger_type, args.posioned_portion, [], [])
    #targets_ori = copy.deepcopy(targets)

    if args.poisoned_mode == 'all2one':
        targets[poisoned_idx] = args.trigger_label
    elif args.poisoned_mode == 'all2all':
        targets[poisoned_idx] = (targets[poisoned_idx] + 1) % class_num

    if os.path.exists(os.path.join(args.BD_data_path, trigger_type, 'train')):
        shutil.rmtree(os.path.join(args.BD_data_path, trigger_type, 'train'))
    os.makedirs(os.path.join(args.BD_data_path, trigger_type, 'train'))
    file = open(os.path.join(args.BD_data_path, trigger_type, 'train', 'info.txt'), 'w')

    for idx, path in enumerate(data):
        img = Image.open(path).convert('RGB').resize((224, 224))
        img = np.array(img)
        poisoned = 0
        if idx in poisoned_idx:
            img = bd_transform(img)
            poisoned = 1
        # elif idx in noisy_idx:
        #     img = bd_transform_noisy(img)
        img = Image.fromarray(img)
        paths = path.split('/')
        paths[4] = str(targets[idx])
        saved_path = os.path.join(args.BD_data_path, trigger_type, '/'.join(paths[3:]))
        if not os.path.exists('/'.join(saved_path.split('/')[:-1])):
            os.mkdir('/'.join(saved_path.split('/')[:-1]))
        img.save(saved_path, quality=100, subsampling=0)
        file.write(' '.join([saved_path, str(poisoned), str(targets[idx])]) + '\n')
    file.close()

    #----------------------------------------Acc test data--------------------------------------
    imgs = test_data.imgs
    data = []
    targets = []
    for i in range(len(imgs)):
        data.append(imgs[i][0])
        targets.append(imgs[i][1])
    targets = np.array(targets).astype(np.int64)
    #poisoned_idx, noisy_idx, data, targets = get_poison_idx(data, targets, 'train', args.trigger_label, trigger_type, args.posioned_portion, [], [])
    if os.path.exists(os.path.join(args.BD_data_path, trigger_type, 'ACC_test')):
        shutil.rmtree(os.path.join(args.BD_data_path, trigger_type, 'ACC_test'))
    os.makedirs(os.path.join(args.BD_data_path, trigger_type, 'ACC_test'))
    file = open(os.path.join(args.BD_data_path, trigger_type, 'ACC_test', 'info.txt'), 'w')

    for idx, path in enumerate(data):
        img = Image.open(path).convert('RGB').resize((224, 224))
        img = np.array(img)
        # poisoned = 0
        # if idx in poisoned_idx:
        #     img = bd_transform(img)
        #     poisoned = 1
        img = Image.fromarray(img)
        paths = path.split('/')
        paths[3] = 'ACC_test'
        paths[4] = str(targets[idx])
        saved_path = os.path.join(args.BD_data_path, trigger_type, '/'.join(paths[3:]))
        if not os.path.exists('/'.join(saved_path.split('/')[:-1])):
            os.mkdir('/'.join(saved_path.split('/')[:-1]))
        img.save(saved_path, quality=100, subsampling=0)
        file.write(' '.join([saved_path, str(0), str(targets[idx])]) + '\n')
    file.close()

    # ----------------------------------------ASR test data--------------------------------------
    imgs = test_data.imgs
    data = []
    targets = []
    for i in range(len(imgs)):
        data.append(imgs[i][0])
        targets.append(imgs[i][1])
    targets = np.array(targets).astype(np.int64)
    poisoned_idx, noisy_idx, data, targets = get_poison_idx(data, targets, 'ASR test', args.trigger_label, trigger_type, args.posioned_portion, [], [])

    if args.poisoned_mode == 'all2one':
        targets[poisoned_idx] = args.trigger_label
    elif args.poisoned_mode == 'all2all':
        targets[poisoned_idx] = (targets[poisoned_idx] + 1) % class_num

    if os.path.exists(os.path.join(args.BD_data_path, trigger_type, 'ASR_test')):
        shutil.rmtree(os.path.join(args.BD_data_path, trigger_type, 'ASR_test'))
    os.makedirs(os.path.join(args.BD_data_path, trigger_type, 'ASR_test'))
    file = open(os.path.join(args.BD_data_path, trigger_type, 'ASR_test', 'info.txt'), 'w')

    for idx, path in enumerate(data):
        img = Image.open(path).convert('RGB').resize((224, 224))
        img = np.array(img)
        # poisoned = 0
        # if idx in poisoned_idx:
        #     img = bd_transform(img)
        #     poisoned = 1
        img = bd_transform(img)
        img = Image.fromarray(img)
        paths = path.split('/')
        paths[3] = 'ASR_test'
        paths[4] = str(targets[idx])
        saved_path = os.path.join(args.BD_data_path, trigger_type, '/'.join(paths[3:]))
        if not os.path.exists('/'.join(saved_path.split('/')[:-1])):
            os.mkdir('/'.join(saved_path.split('/')[:-1]))
        img.save(saved_path, quality=100, subsampling=0)
        file.write(' '.join([saved_path, str(1), str(targets[idx])]) + '\n')
    file.close()


