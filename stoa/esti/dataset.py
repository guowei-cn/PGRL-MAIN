import copy
import os
import torch
from PIL import ImageFilter, Image
import random
from torch.utils.data import DataLoader, Subset,ConcatDataset
from torchvision import transforms
from collections import defaultdict
from io import BytesIO
from pathlib import Path
import utils
import numpy as np




class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_folder, file_name, dataset_name,split_time=None, label_flipping=None, target=None):
        bin_data_path = data_folder
        index_file = file_name
        self.bin_data_path = bin_data_path
        self.images = []
        self.labels = []
        self.num_classes = get_num_classes(dataset_name,split_time)
        self.image_names = []
        index_file_path = os.path.join(bin_data_path, f'{index_file}_index.txt')
        with open(index_file_path, 'r') as f:
            lines = f.readlines()
        bin_file_path = os.path.join(bin_data_path, f'{index_file}.bin')
        with open(bin_file_path, 'rb') as bin_file:
            bin_data = bin_file.read()
        for line in lines:
            image_name, label, offset, size = line.strip().split(',')
            offset, size = int(offset), int(size)
            label = int(label)
            if label_flipping=='poison' and 'test' != split_time:
                label = int(target)
            elif label_flipping=='poison' and 'test' == split_time:
                if int(label) == int(target):
                    continue
                label = int(target)
            else:
                label = label
            image_data = bin_data[offset:offset + size]
            image = Image.open(BytesIO(image_data)).convert('RGB')
            self.image_names.append(image_name)
            self.images.append(image)
            self.labels.append(label)
    def __getitem__(self, index):
        return self.image_names[index], self.labels[index], self.images[index],
    def __len__(self):
        return len(self.images)

    def get_names(self):
        data_names = set()
        for idx in range(self.__len__()):
            image_name, _, _ = self.__getitem__(idx)
            data_names.add(image_name)
        return data_names

class SubsetFromIdx(torch.utils.data.Dataset):
    def __init__(self,image_names, dataset, transform,num_classes,label_flipping=None,target=None):
        if(image_names == None):
            self.data = copy.deepcopy(dataset)
        else:
            self.data = [copy.deepcopy(dataset[i]) for i in range(len(dataset)) if dataset[i][0] in image_names]
        self.transform = transform
        self.num_classes = num_classes
        self.image_names = []
        self.labels = []
        self.images = []
        for image_name, label,image in self.data:
            if (label_flipping=='poison'):
                flipped_label = int(target)
            elif (label_flipping=='random'):
                flipped_label = utils.generate_random_except(0, self.num_classes-1, int(label))
            elif (label_flipping=='trap'):
                flipped_label = self.num_classes-1
            else:
                flipped_label = label
            self.image_names.append(image_name)
            self.labels.append(int(flipped_label))
            self.images.append(image)
    def __getitem__(self, index):
        image_name = self.image_names[index]
        label = self.labels[index]
        image = self.images[index]
        if self.transform:
            image = self.transform(image)

        return image_name,image, label
    def __len__(self):
        return len(self.image_names)

class SelfDataset(torch.utils.data.Dataset):
    def __init__(self, train_data, transform):
        super(SelfDataset, self).__init__()
        x, y = [], []
        for i in range(len(train_data)):
            image_name, label, image = train_data[i]
            x.append(image)
            y.append(label)
        self.dataset = list(zip(x,y))
        self.data = x
        self.targets = y
        self.transform = transform
        # self.prefetch = args.prefetch
        # if self.prefetch:
        #     norm = get_dataset_normalization(args.dataset)
        #     self.mean, self.std = norm.mean, norm.std
    def __getitem__(self, index):
        if isinstance(self.data[index], str):
            with open(self.data[index], "rb") as f:
                img = np.array(Image.open(f).convert("RGB"))
        else:
            img = self.data[index]
        target = self.targets[index]
        img1 = self.bd_first_augment(img)
        img2 = self.bd_first_augment(img)
        item = {
            "img1": img1,
            "img2": img2,
            "target": target,
        }
        return item
    def __len__(self):
        return len(self.data)
    def bd_first_augment(self, img):
        img = np.array(img)
        img = Image.fromarray(np.uint8(img))
        img = self.transform(img)
        # if self.prefetch:
        #     img = np.rollaxis(np.array(img, dtype=np.uint8), 2)
        #     img = torch.from_numpy(img)
        return img



def get_train_dataset(clean_pool_image_paths,poison_pool_image_paths,train_data,split_time,dataset_name,):
    if split_time == 'CTMv1':
        train_dataset = SubsetFromIdx(
            clean_pool_image_paths,
            train_data,
            get_start_transforms(
                dataset_name=dataset_name,
            ),
            num_classes=get_num_classes(dataset_name,split_time),
        )
    elif 'PTMv1' == split_time or 'Split_Clean' == split_time:
        train_dataset = SubsetFromIdx(
            poison_pool_image_paths,
            train_data,
            get_split_transforms(
                dataset_name=dataset_name,
                is_enhance=False
            ),
            num_classes=get_num_classes(dataset_name,split_time),
        )
    elif 'CTM' in split_time:
        train_dataset = SubsetFromIdx(
            clean_pool_image_paths,
            train_data,
            get_split_transforms(
                dataset_name=dataset_name,
                is_enhance=True
        ),
            num_classes=get_num_classes(dataset_name,split_time),
        )
    elif ('PTM' in split_time or 'TrapPre' in split_time) and (split_time != 'Split_Clean' and split_time != 'PTMv1'):
        adding_dataset = SubsetFromIdx(
            clean_pool_image_paths,
            train_data,
            get_random_crop_transform(
                dataset_name=dataset_name,
            ),
            num_classes=get_num_classes(dataset_name,split_time),
            label_flipping='random'
        )
        poison_train_dataset = get_avgExtract_subSet(adding_dataset,
                              int(0.7 * len(adding_dataset) / get_num_classes(dataset_name, split_time)))
        train_dataset = ConcatDataset(
            [
                SubsetFromIdx(
                    poison_pool_image_paths,
                    train_data,
                    get_split_transforms(
                        dataset_name=dataset_name,
                        is_enhance=False
                    ),
                    num_classes=get_num_classes(dataset_name,split_time),
                ),
                poison_train_dataset
            ]
        )
    elif 'TrapModel' in split_time:
        if split_time == 'TrapModelvDeplo':
            clean_transform = get_train_transforms(
                dataset_name=dataset_name,
                is_enhance=True
            )
            poison_transform = get_train_transforms(
                dataset_name=dataset_name,
                is_enhance=False
            )
        elif('TrapModel' in split_time ):
            #clean_transform = get_split_transforms(
            #    dataset_name=dataset_name,
            #    is_enhance=False#False
            #)
            
            clean_transform = get_train_transforms(
                dataset_name=dataset_name,
                is_enhance=True
            )#for onlyES
            poison_transform=get_split_transforms(
                dataset_name=dataset_name,
                is_enhance=False#False
            )
        clean_train_data = SubsetFromIdx(
            clean_pool_image_paths,
            train_data,
            clean_transform,
            num_classes=get_num_classes(dataset_name,split_time),
        )
        poison_train_data = SubsetFromIdx(
            poison_pool_image_paths,
            train_data,
            poison_transform,
            num_classes=get_num_classes(dataset_name,split_time),
            label_flipping='trap'
        )
        train_dataset = ConcatDataset([clean_train_data, poison_train_data])
    elif('TrapClean' == split_time):
        train_dataset = SubsetFromIdx(
            clean_pool_image_paths,
            train_data,
            get_random_crop_transform(
                dataset_name=dataset_name,
            ),
            num_classes=get_num_classes(dataset_name,split_time),
        )
    return train_dataset
def get_test_dataset(clean_test_data,poison_test_data,dataset_name):
    clean_test_dataset = SubsetFromIdx(
        None,
        clean_test_data,
        transform = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            getNormal(dataset_name=dataset_name)
        ]),
        num_classes=get_num_classes(dataset_name),
    )
    poison_test_dataset = SubsetFromIdx(
        None,
        poison_test_data,
        transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            getNormal(dataset_name=dataset_name)
        ]),
        num_classes=get_num_classes(dataset_name),
    )
    return clean_test_dataset,poison_test_dataset

def get_avgExtract_subSetImageNames(
        dataset=None,
        samples_per_class=None,
):
    class_image_name = defaultdict(list)
    for  idx, (image_name,label,_,)  in enumerate(dataset):
        class_image_name[label].append(image_name)
    subset_images = []
    for key in class_image_name.keys():
        subset_images.extend(random.sample(class_image_name[key],samples_per_class))
    return subset_images

def get_avgExtract_subSet(
        dataset=None,
        samples_per_class=None,
):
    class_indices = defaultdict(list)
    for idx, (_,_,label) in enumerate(dataset):
        class_indices[label].append(idx)
    subset_indices = []
    for key in class_indices.keys():
        subset_indices.extend(random.sample(class_indices[key],samples_per_class))
    subset = Subset(dataset, subset_indices)
    return subset

def get_num_classes(dataset_name,split_time=None):
    if(dataset_name in ['cifar10', 'imagenette']):
        num_classes = 10
    elif(dataset_name=='gtsrb'):
        num_classes = 43
    else:
        num_classes = -1
    if split_time!=None and ('TrapModel' in split_time or 'CTM' in split_time):
    #if split_time!=None and ('TrapModel' in split_time):
        num_classes = num_classes +0
    return num_classes

def get_train_transforms(dataset_name,is_enhance = False):
    if(not is_enhance):
        transform = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            getNormal(dataset_name=dataset_name)
        ])
    elif(is_enhance):
        if(dataset_name in ['cifar10', 'imagenette']):
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                transforms.RandomRotation(10),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                getNormal(dataset_name=dataset_name)

            ])
        elif(dataset_name == 'gtsrb'):
            transform = transforms.Compose([
                transforms.Resize((32,32)),
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                getNormal(dataset_name=dataset_name)
            ])
    return transform

def get_random_crop_transform(dataset_name):
    if(dataset_name in ['cifar10', 'imagenette']):
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.ToTensor(),
            getNormal(dataset_name=dataset_name)
        ])
    elif(dataset_name == 'gtsrb'):
        transform = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.ToTensor(),
            getNormal(dataset_name=dataset_name)
        ])
    return transform

def get_split_transforms(dataset_name,is_enhance = False):
    if(not is_enhance):
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            getNormal(dataset_name=dataset_name)
        ])
    elif(is_enhance):
        if(dataset_name in ['cifar10', 'imagenette']):
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.RandomRotation(degrees=15),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                transforms.RandomResizedCrop(size=(32, 32), scale=(0.8, 1.0)),
                transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), shear=(-10, 10)),
                transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
                transforms.ToTensor(),
                getNormal(dataset_name=dataset_name)
            ])
        elif(dataset_name == 'gtsrb'):
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.RandomResizedCrop(size=(32, 32), scale=(0.8, 1.0)),
                transforms.RandomAffine(degrees=10, translate=(0.0, 0.1), shear=(-10, 10)),
                transforms.RandomPerspective(distortion_scale=0.1),
                transforms.ToTensor(),
                getNormal(dataset_name=dataset_name)
            ])
    return transform

def get_start_transforms(dataset_name):
    if(dataset_name in ['cifar10', 'imagenette']):
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            transforms.RandomResizedCrop(size=(32, 32), scale=(0.8, 1.0)),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), shear=(-10, 10)),
            transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
            transforms.ToTensor(),
            getNormal(dataset_name=dataset_name)

        ])
    elif(dataset_name == 'gtsrb'):
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomRotation(degrees=10),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.RandomResizedCrop(size=(32, 32), scale=(0.8, 1.0)),
            transforms.RandomAffine(degrees=10, translate=(0.0, 0.1), shear=(-10, 10)),
            transforms.RandomPerspective(distortion_scale=0.1),
            transforms.ToTensor(),
            getNormal(dataset_name=dataset_name)
        ])
    return transform

def getNormal(dataset_name):
    if (dataset_name in ['cifar10', 'imagenette']):
        Normal = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    elif (dataset_name == 'gtsrb'):
        Normal = transforms.Normalize((0, 0, 0), (1, 1, 1))
    else :
        Normal = None
    return Normal

def get_dataset_normalization(dataset_name):
    # idea : given name, return the default normalization of images in the dataset
    if dataset_name in ["cifar10", "imagenette"]:
        # from wanet
        dataset_normalization = (transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]))
    elif dataset_name == 'cifar100':
        '''get from https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151'''
        dataset_normalization = (transforms.Normalize([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762]))
    elif dataset_name == "mnist":
        dataset_normalization = (transforms.Normalize([0.5], [0.5]))
    elif dataset_name == 'tiny':
        dataset_normalization = (transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]))
    elif dataset_name == "gtsrb" or dataset_name == "celeba":
        dataset_normalization = transforms.Normalize([0, 0, 0], [1, 1, 1])
    elif dataset_name == 'imagenet':
        dataset_normalization = (
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        )
    else:
        raise Exception("Invalid Dataset")
    return dataset_normalization


def get_transform_self(dataset_name, train=True):
    # idea : given name, return the final implememnt transforms for the dataset during self-supervised learning
    if(dataset_name in ['cifar10', 'imagenette']):
        input_height = 32
        input_width = 32
    elif(dataset_name == 'gtsrb'):
        input_height = 32
        input_width = 32
    elif(dataset_name == 'tiny'):
        input_height = 64
        input_width = 64
    transforms_list = []
    transforms_list.append(transforms.Resize((input_height, input_width)))
    if train:

        transforms_list.append(
            transforms.RandomResizedCrop(size=(input_height, input_width), scale=(0.2, 1.0), ratio=(0.75, 1.3333),
                                         interpolation=3))
        transforms_list.append(transforms.RandomHorizontalFlip(p=0.5))
        transforms_list.append(transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter(brightness=[0.6, 1.4],
                                                                                                  contrast=[0.6, 1.4],
                                                                                                  saturation=[0.6, 1.4],
                                                                                                  hue=[-0.1, 0.1])]),
                                                      p=0.8))
        transforms_list.append(transforms.RandomGrayscale(p=0.2))
        transforms_list.append(transforms.RandomApply([GaussianBlur(sigma=[0.1, 2.0])], p=0.5))

    transforms_list.append(transforms.ToTensor())
    transforms_list.append(get_dataset_normalization(dataset_name))
    return transforms.Compose(transforms_list)

class GaussianBlur(object):
    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))

        return x


# following is my dataset builder
# the dataset_folder should include:
# data_path/
# ├── trainSet_clean_list.bin
# ├── trainSet_clean_list_index.txt
# ├── trainSet_poisoned_list.bin
# ├── trainSet_poisoned_list_index.txt
# ├── testSetClear_labels.bin
# ├── testSetClear_labels_index.txt
# ├── testSetPoisoned_poisoned_list.bin
# └── testSetPoisoned_poisoned_list_index.txt
# There are two types of files: *.txt and *.bin
# For txt file, a list of (image_name,label,offset,size), 
# where image_name: string ID (e.g. 000123.png)
# label: integer class (for CIFAR-10: 0–9)
# offset: byte offset into the corresponding .bin
# size: number of bytes for that image
# For bin file, [bytes of image 0][bytes of image 1][bytes of image 2]...[bytes of image N-1]

