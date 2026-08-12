import models
import os
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import dataset
from torch.utils.data import  DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch


self_supervised_model_name = f'model_ID_supervised_tranformed.pth.tar'


def train(device, dataset_name, train_dataset,num_workers, model_config,split_time,
                                            save_model=True,save_folder = None,self_supervised_model_path = None):
    global self_supervised_model_name
    model_name = model_config.model_name
    batch_size = model_config.batch_size
    epoch_num = model_config.epoch_num
    lr = model_config.lr
    weight_decay = model_config.weight_decay
    os.makedirs(save_folder, exist_ok=True)
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True
    )
    if("CTMv1" in split_time):

        model = models.get_model(model_name=model_name,
                                           num_classes=dataset.get_num_classes(dataset_name, split_time))
        model.to(device)

    elif 'CTM' not in split_time and 'TrapModel' not in split_time:
        model = models.get_model(model_name=model_name, num_classes=dataset.get_num_classes(dataset_name, split_time))
        model.to(device)
    elif("CTMvEnd" in split_time):
        model = models.get_model(model_name=model_name,
                                 num_classes=dataset.get_num_classes(dataset_name, split_time))
        model.to(device)
        model_path = os.path.join(self_supervised_model_path, f'model_ID_CTMv1.pth.tar')
        model.load_state_dict(torch.load(model_path))

    elif("TrapModelvDeplo" in split_time):

        model = models.get_model(model_name=model_name,
                                 num_classes=dataset.get_num_classes(dataset_name, split_time))
        model.to(device)
        model_path = os.path.join(self_supervised_model_path, 'model_ID.pth.tar')
        model.load_state_dict(torch.load(model_path))
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(reduction='none')
    scheduler = CosineAnnealingLR(optimizer, T_max=100)

    progress_bar = tqdm(range(epoch_num))
    for epoch in progress_bar:
        model.train()
        total_loss = 0
        step = 0
        for image_name, inputs, labels in train_data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            tmp_loss = criterion(outputs, labels)
            step += 1
            loss = tmp_loss.mean()
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
        scheduler.step()
    progress_bar.close()
    if (save_model):
        torch.save(model.state_dict(),
                   os.path.join(save_folder, f'model_ID_' + split_time + '.pth.tar'))
        return model


def trap_train(device,dataset_name, train_dataset,num_workers,clean_train_dataset, model_config,split_time,
                                            save_model=True,save_folder = None,self_supervised_model_path = None):
    global self_supervised_model_name
    model_name = model_config.model_name
    batch_size = model_config.batch_size
    epoch_num = model_config.epoch_num
    lr = model_config.lr
    weight_decay = model_config.weight_decay
    os.makedirs(save_folder, exist_ok=True)
    train_data_loader_all = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True
    )
    train_data_loader_clean = DataLoader(
        clean_train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True
    )
    if("TrapModelv1" == split_time):


        model = models.get_model(model_name=model_name,
                                 num_classes=dataset.get_num_classes(dataset_name, split_time))
        model.to(device)
        model_path = os.path.join(self_supervised_model_path, 'model_ID_CTMvEnd.pth.tar')
        model.load_state_dict(torch.load(model_path))

    else:
        model = models.get_model(model_name=model_name,
                                 num_classes=dataset.get_num_classes(dataset_name, split_time))
        model.to(device)
        model_path = os.path.join(self_supervised_model_path, 'model_ID.pth.tar')
        model.load_state_dict(torch.load(model_path))

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(reduction='none')
    scheduler = CosineAnnealingLR(optimizer, T_max=100)
    if(dataset_name in ['cifar10', 'imagenette']):
        staget2_start_epoch = 0
    elif(dataset_name == 'gtsrb'):
        staget2_start_epoch = 0
    if(split_time =='TrapModelv1'):
        epoch_num = 100
    progress_bar = tqdm(range(epoch_num))
    for epoch in progress_bar:
        model.train()
        total_loss = 0
        step = 0
        if(epoch+1>=staget2_start_epoch):
            train_data_loader = train_data_loader_all
        else:
            train_data_loader = train_data_loader_clean
        for image_name, inputs, labels in train_data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            tmp_loss = criterion(outputs, labels)
            step += 1
            loss = tmp_loss.mean()
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
        scheduler.step()
    progress_bar.close()
    if (save_model):
        torch.save(model.state_dict(),
                   os.path.join(save_folder, f'model_ID.pth.tar'))
        return model



