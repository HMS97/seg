#!/usr/bin/env python
# coding: utf-8

# In[28]:
from PIL import Image
import cv2
from path import Path
import collections
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset
import torchvision.transforms.functional as tf
import segmentation_models_pytorch as smp
from utils.datasets import SlippyMapTilesConcatenation
from utils.loss import CrossEntropyLoss2d, mIoULoss2d, FocalLoss2d, LovaszLoss2d
from utils.transforms import (
    JointCompose,
    JointTransform,
    JointRandomHorizontalFlip,
    JointRandomRotation,
    ConvertImageMode,
    ImageToTensor,
    MaskToTensor,
)

from torchvision.transforms import Resize, CenterCrop, Normalize
from utils.metrics import Metrics
from models.segnet.segnet import segnet
from models.unet.unet import UNet
import datetime
import random
import os
import tqdm
import json
import argparse
from logsetting import  get_log
device = 'cuda'
path = './alldataset'


def get_dataset_loaders( workers,    batch_size = 4  ):
    target_size = 512

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    transform = JointCompose(
        [
#             JointTransform(ConvertImageMode("RGB"), ConvertImageMode("P")),
            JointTransform(Resize(target_size, Image.BILINEAR), Resize(target_size, Image.NEAREST)),
            JointTransform(CenterCrop(target_size), CenterCrop(target_size)),
            JointRandomHorizontalFlip(0.5),
            JointRandomRotation(0.5, 90),
            JointRandomRotation(0.5, 90),
            JointRandomRotation(0.5, 90),
            JointTransform(ImageToTensor(), MaskToTensor()),
            JointTransform(Normalize(mean=mean, std=std), None),
        ]
    )

    train_dataset = SlippyMapTilesConcatenation(
        os.path.join(path, "training", "images"), os.path.join(path, "training", "labels"), transform,debug = False
    )

    val_dataset = SlippyMapTilesConcatenation(
        os.path.join(path, "validation", "images"), os.path.join(path, "validation", "labels"), transform,debug = False
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=workers)

    return train_loader, val_loader


def train(loader, num_classes, device, net, optimizer, criterion):
    num_samples = 0
    running_loss = 0

    metrics = Metrics(range(num_classes))

    net.train()
    for images, masks  in tqdm.tqdm(loader):
        images = images.to(device)
        masks = masks.to(device)

        assert images.size()[2:] == masks.size()[1:], "resolutions for images and masks are in sync"

        num_samples += int(images.size(0))

        optimizer.zero_grad()
        outputs = net(images)

        assert outputs.size()[2:] == masks.size()[1:], "resolutions for predictions and masks are in sync"
        assert outputs.size()[1] == num_classes, "classes for predictions and dataset are in sync"

        loss = criterion(outputs, masks)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        for mask, output in zip(masks, outputs):
            prediction = output.detach()
            metrics.add(mask, prediction)

    assert num_samples > 0, "dataset contains training images and labels"

    return {
        "loss": running_loss / num_samples,
        "miou": metrics.get_miou(),
        "fg_iou": metrics.get_fg_iou(),
        "mcc": metrics.get_mcc(),
    }

def validate(loader, num_classes, device, net, criterion):
    num_samples = 0
    running_loss = 0

    metrics = Metrics(range(num_classes))

    net.eval()

    for images, masks,  in tqdm.tqdm(loader):
        images = images.to(device)
        masks = masks.to(device)

        assert images.size()[2:] == masks.size()[1:], "resolutions for images and masks are in sync"

        num_samples += int(images.size(0))

        outputs = net(images)

        assert outputs.size()[2:] == masks.size()[1:], "resolutions for predictions and masks are in sync"
        assert outputs.size()[1] == num_classes, "classes for predictions and dataset are in sync"

        loss = criterion(outputs, masks)

        running_loss += loss.item()

        for mask, output in zip(masks, outputs):
            metrics.add(mask, output)

    assert num_samples > 0, "dataset contains validation images and labels"

    return {
        "loss": running_loss / num_samples,
        "miou": metrics.get_miou(),
        "fg_iou": metrics.get_fg_iou(),
        "mcc": metrics.get_mcc(),
    }






if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epoch', nargs='?', type=int, default=50,
                    help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=4,
                    help='Batch Size')
    parser.add_argument('--swa_start', nargs='?', type=int, default=1)
    parser.add_argument('--l_rate', nargs='?', type=float, default=5e-3, 
                    help='Learning Rate')
    parser.add_argument('--model',nargs='?',type=str,default='unet')
    parser.add_argument('--swa',nargs='?',type=bool,default=True)

    arg = parser.parse_args()


  


    num_classes = 16
    model_name = arg.model
    learning_rate = arg.l_rate
    num_epochs = arg.n_epoch
    batch_size = arg.batch_size


    history = collections.defaultdict(list)
    model_dict = {
                'unet':UNet( num_classes = num_classes).train().to(device),
                'segnet':segnet(  n_classes = num_classes ).train().to(device),
                'pspnet':smp.PSPNet(classes= num_classes ).train().to(device),
                }

    net = model_dict[model_name]
    if torch.cuda.device_count() > 1:
        print("using multi gpu")
        net = torch.nn.DataParallel(net,device_ids = [0, 1, 2, 3])
    else:
        print('using one gpu')

    # if True:
    #     print("The ckp has been loaded sucessfully ")
    #     net = torch.load("./model/unet_2019-07-23.pth") # load the pretrained model
    criterion = FocalLoss2d().to(device)
    train_loader, val_loader = get_dataset_loaders(5, batch_size)
    opt = torch.optim.SGD(net.parameters(), lr=learning_rate)
    today=str(datetime.date.today())
    # logger = get_log(model_name + today +'_log.txt')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5,eta_min=4e-08)



    for epoch in range(num_epochs):
        logger.info("Epoch: {}/{}".format(epoch + 1, num_epochs))
        scheduler.step()
        train_hist = train(train_loader, num_classes, device, net, opt, criterion)
        print( 'loss',train_hist["loss"],
                'miou',train_hist["miou"],
                'fg_iou',train_hist["fg_iou"],
                'mcc',train_hist["mcc"] )

 
        for k, v in train_hist.items():
            history["train " + k].append(v)

        val_hist = validate(val_loader, num_classes, device, net, criterion)
        print('loss',val_hist["loss"],
                'miou',val_hist["miou"],
                'fg_iou',val_hist["fg_iou"],
                'mcc',val_hist["mcc"])

        for k, v in val_hist.items():
            history["val " + k].append(v)


        checkpoint = 'model/{}_{}.pth'.format(model_name,today)
        torch.save(net,checkpoint)
    json = json.dumps(history)
    f = open("model/{}_{}.json".format(model_name,today),"w")
    f.write(json)
    f.close()
