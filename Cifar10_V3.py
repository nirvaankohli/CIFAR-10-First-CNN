#!/usr/bin/env python3
import os
import argparse
from multiprocessing import freeze_support

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, models
from tqdm import tqdm

# ─── Model ────────────────────────────────────────────────────────────────────

def resnet18_cifar(
        
        num_classes=10

        ):

    model = models.resnet18(

        pretrained=False, 

        num_classes=num_classes

        )
    
    # adapt for 32×32 inputs

    model.conv1 = nn.Conv2d(

        3, 

        64, 

        kernel_size=3, 

        stride=1, 

        padding=1, 

        bias=False

        )
    
    model.maxpool = nn.Identity()

    return model

# ─── Data loaders ────────────────────────────────────────────────────────────
def get_data_loaders(
        
        data_dir, 

        batch_size=128, 

        num_workers=4, 

        device=None

        ):

    mean = (

        0.4914, 

        0.4822, 

        0.4465

            )
    
    std  = (

        0.2023, 

        0.1994, 

        0.2010

        )
    
    pin = device is not None and device.type == "cuda"

    train_tf = transforms.Compose([

        transforms.RandomCrop(32, padding=4),

        transforms.RandomHorizontalFlip(),

        transforms.RandAugment(num_ops=2, magnitude=9),

        transforms.ToTensor(),

        transforms.Normalize(mean, std),

        transforms.RandomErasing(p=0.2, scale=(0.02,0.33), ratio=(0.3,3.3)),
    ])

    test_tf = transforms.Compose([

        transforms.ToTensor(),

        transforms.Normalize(

            mean, 
            
            std

            ),

    ])

    train_ds = datasets.CIFAR10(

        data_dir, 
        
        train=True, 
        
        download=True, 
        
        transform=train_tf
        
        )
    
    test_ds  = datasets.CIFAR10(
        
        data_dir, 
        
        train=False, 
        
        download=True, 
        
        transform=test_tf
        
        )

    train_loader = DataLoader(

        train_ds, 
        
        batch_size=batch_size, 
        
        shuffle=True,

        num_workers=num_workers, 

        pin_memory=pin, 

        persistent_workers=True

        )
    
    test_loader  = DataLoader(

        test_ds,  
        
        batch_size=batch_size, 
        
        shuffle=False,

        num_workers=num_workers, 

        pin_memory=pin, 

        persistent_workers=True

        )
    
    return train_loader, test_loader

# ─── MixUp ───────────────────────────────────────────────────────────────────

def mixup_data(x, y, alpha=0.2):

    if alpha > 0:

        lam = np.random.beta(

            alpha, 
            
            alpha

            )

    else:

        lam = 1.0

    batch_size = x.size(0)

    idx = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]

    return mixed_x, y_a, y_b, lam

# ─── CutMix ──────────────────────────────────────────────────────────────────

def rand_bbox(size, lam):

    W, H = size[2], size[3]

    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix_data(
        x, 
        
        y, 
        
        alpha=1.0):

    if alpha > 0:

        lam = np.random.beta(alpha, alpha)

    else:

        lam = 1.0
        
    batch_size = x.size(0)
    idx = torch.randperm(batch_size, device=x.device)

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[idx, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
    
    y_a, y_b = y, y[idx]
    return x, y_a, y_b, lam

# ─── Training & Validation ───────────────────────────────────────────────────
def train_one_epoch(
        
        model, 
        loader, 
        optimizer, 
        criterion, 
        device, 
        scaler,
        epoch, 
        total_epochs, 
        mixup_alpha, 
        cutmix_alpha, 
        cutmix_prob
        
        ):
    
    model.train()

    running_loss = 0.0
    running_correct = 0.0

    total = 0
    pbar = tqdm(loader, desc=f"Train [{epoch}/{total_epochs}]", ncols=120)
    
    for imgs, labels in pbar:

        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        # choose MixUp or CutMix
        if np.random.rand() < cutmix_prob:

            imgs, y_a, y_b, lam = cutmix_data(imgs, labels, cutmix_alpha)
        else:
            imgs, y_a, y_b, lam = mixup_data(imgs, labels, mixup_alpha)

        with autocast():
            
            outputs = model(imgs)
            loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        preds = outputs.argmax(dim=1)
        running_loss += loss.item() * imgs.size(0)
        running_correct += lam * (preds==y_a).sum().item() + (1-lam)*(preds==y_b).sum().item()
        total += labels.size(0)

        pbar.set_postfix({
            "loss": f"{running_loss/total:.4f}",
            "acc":  f"{100*running_correct/total:.2f}%"
        })

    return running_loss/total, running_correct/total

@torch.no_grad()
def validate(model, loader, criterion, device, epoch, total_epochs):
    model.eval()
    val_loss = 0.0
    val_correct = 0
    total = 0
    pbar = tqdm(loader, desc=f"Valid [{epoch}/{total_epochs}]", ncols=120)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        with autocast():
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        preds = outputs.argmax(dim=1)
        val_loss += loss.item() * imgs.size(0)
        val_correct += (preds==labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({
            "loss": f"{val_loss/total:.4f}",
            "acc":  f"{100*val_correct/total:.2f}%"
        })

    return val_loss/total, val_correct/total

# ─── Main ────────────────────────────────────────────────────────────────────
def main():

    parser = argparse.ArgumentParser(

        description="CIFAR-10 ResNet-18 Advanced Training"
        
        )
    
    parser.add_argument(

        '--data-dir',  

        type=str,   

        default='./data'
        
        )
    
    parser.add_argument(

        '--batch-size',   
        
        type=int,   
        
        default=128
        
        )
    
    parser.add_argument(
        
        '--epochs',       
        
        type=int,   
        
        default=50
        
        )
    
    parser.add_argument(

        '--lr',      

        type=float, 

        default=0.1

        )
    
    parser.add_argument(
        
        '--weight-decay', 
        
        type=float, 
        
        default=5e-4
        
        )
    
    parser.add_argument(
        
         '--num-workers',  
         
         type=int,   
         
         default=4
         
         )
    
    parser.add_argument(
        
        '--mixup-alpha',  
        
        type=float, 
        
        default=0.2
        
        )
    
    parser.add_argument(

        '--cutmix-alpha', 

        type=float, 

        default=1.0
        
        )
    
    parser.add_argument(
        '--cutmix-prob',  
        type=float, 
        default=0.5
        
        )
    parser.add_argument('--swa-start',    type=int,   default=40)
    
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_data_loaders(

        args.data_dir, args.batch_size, args.num_workers, device
    )

    model     = resnet18_cifar(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    scaler    = GradScaler()

    # SWA setup
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=args.lr)
    swa_start = args.swa_start

    # Logging
    writer = SummaryWriter()
    metrics_file = open('V3metrics.csv','w')
    metrics_file.write('Epoch,TrainLoss,TrainAcc,ValLoss,ValAcc\n')

    best_val_acc = 0.0
    for epoch in range(1, args.epochs+1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            scaler, epoch, args.epochs,
            args.mixup_alpha, args.cutmix_alpha, args.cutmix_prob
        )
        val_loss, val_acc = validate(
            model, test_loader, criterion, device, epoch, args.epochs
        )

        # LR scheduling & SWA
        if epoch > swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        # Save metrics
        metrics_file.write(f"{epoch},{train_loss:.4f},{train_acc:.4f},{val_loss:.4f},{val_acc:.4f}\n")
        metrics_file.flush()
        writer.add_scalars('Loss',     {'train':train_loss, 'val':val_loss}, epoch)
        writer.add_scalars('Accuracy', {'train':train_acc,  'val':val_acc}, epoch)

        # Checkpoint best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "modelV3.pth.pth")

        print(f"Epoch {epoch:02d}: Train L={train_loss:.4f}, A={train_acc*100:.2f}% | "
              f"Val   L={val_loss:.4f}, A={val_acc*100:.2f}%")

    # finalize SWA
    update_bn(train_loader, swa_model)
    torch.save(swa_model.module.state_dict(), "swa_modelV3.pth")

    metrics_file.close()
    writer.close()

    print(f"Training complete. Best val acc: {best_val_acc*100:.2f}%")
    print("Models saved: best_resnet18_cifar.pth & swa_modelV3.pth")

if __name__ == "__main__":

    freeze_support()
    main()
