# VaB

This a Pytorch implementation of our paper ["The Victim and The Beneficiary: Exploiting a Poisoned Model to Train a Clean Model on Poisoned Data"](http://arxiv.org/abs/2404.11265), ICCV23, **Oral**, **Best paper candidate**.



## Setup

### Environments

Please install the required packages according to requirement.txt

### Datasets

Download corresponding datasets and extract them to 'dataset'

1. Original CIFAR-10 will be automatically downloaded during training. You can download modified data in [Google Drive](https://drive.google.com/drive/folders/1KzUcys85Y9eYlWXFzxKSYW7UPzhcNbjr?usp=sharing) to implement "CL" and "Dynamic" attacks.
2. The benign and poisoned ImageNet subset can be downloaded from [Google Drive](https://drive.google.com/file/d/1Qu3s5BjqhBQcc-ku863PsZ-5Ml_0yTZU/view?usp=drive_link).

## Usage

Run the following script to train a clean model on the poisoned data.

```shell
python Train_cifar10.py --trigger_type badnet --trigger_label 0 --trigger_path ./trigger/cifar10/cifar_1.png --posioned_portion 0.1 --model_name ResNet18
```
For the ImageNet subset, you can directly use the downloaded poisoned dataset or generate them by calling prepare_data.py.

```shell
python Train_Imagenet.py --trigger_type badnet --trigger_label 0 --trigger_path ./trigger/ImageNet/ImageNet_badnet.npy --posioned_portion 0.1 --model_name ResNet18 --BD_data_path (directory containing poisoned datasets)
```

Please modify the attack settings as you want.

 
