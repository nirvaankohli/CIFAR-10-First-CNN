import os
import argparse
from multiprocessing import freeze_support

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, models
from tqdm import tqdm

# ─── CIFAR-10–adapted ResNet-18 ────────────────────────────────────────────────
def resnet18_cifar(num_classes=10):

    model = models.resnet18(pretrained=False, num_classes=num_classes)

    # Nirvaan Note Network(The Triple N): adapt first conv to 32×32 inputs

    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    # Nirvaan Note Network(The Triple N): Adapt the maxpool to CIFAR-10 based on input size

    model.maxpool = nn.Identity()
    return model

# ─── data loaders with stronger augmentation & persistent workers ─────────────
def get_data_loaders(data_dir, batch_size=128, num_workers=4, device=None):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)
    pin = device is not None and device.type == "cuda"

    # Nirvaan Note Network(The Triple N): add RandAugment & RandomErasing for stronger augmentation
    
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),

        transforms.RandomHorizontalFlip(),

        transforms.RandAugment(num_ops=2, magnitude=9),

        transforms.ToTensor(),

        transforms.Normalize(mean, std),

        transforms.RandomErasing(p=0.2, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
    ])

    # Nirvaan Note Network(The Triple N): Above is one of the most important parts of the code, it is a strong augmentation pipeline for CIFAR-10

    # Nirvaan Note Network(The Triple N): test set only normalized
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Nirvaan Note Network(The Triple N): Downloading/ loading CIFAR-10 dataset

    train_ds = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_tf)
    test_ds  = datasets.CIFAR10(data_dir, 
                                train=False, 
                                download=True, transform=test_tf)

    train_loader = DataLoader(

        train_ds, 
        batch_size=batch_size, 
        shuffle=True,

        num_workers=num_workers, pin_memory=pin, persistent_workers=True

    )
    test_loader = DataLoader(

        test_ds, batch_size=batch_size, shuffle=False,

        num_workers=num_workers, pin_memory=pin, persistent_workers=True

    )

    return train_loader, test_loader

# ─── mixup helper ───────────────────────────────────────────────────────────────
def mixup_data(x, y, alpha=0.2):
    # Nirvaan Note Network(The Triple N): Perform MixUp on the batch
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# ─── single-epoch training with AMP & MixUp ────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device, scaler, epoch, total_epochs, mixup_alpha):
    model.train()
    running_loss = 0.0
    running_correct = 0.0
    total = 0
    pbar = tqdm(loader, desc=f"Train [{epoch}/{total_epochs}]", ncols=120)

    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        # Nirvaan Note Network(The Triple N): apply MixUp augmentation
        imgs, labels_a, labels_b, lam = mixup_data(imgs, labels, mixup_alpha)

        with autocast():
            outputs = model(imgs)
            # Nirvaan Note Network(The Triple N): compute mixed loss
            loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        preds = outputs.argmax(dim=1)
        running_loss += loss.item() * imgs.size(0)
        
        # Nirvaan Note Network(The Triple N): Weighted correct for MixUp

        running_correct += lam * (preds == labels_a).sum().item() + (1 - lam) * (preds == labels_b).sum().item()
        total += labels.size(0)

        pbar.set_postfix({
            "loss": f"{running_loss/total:.4f}",
            "acc":  f"{100*running_correct/total:.2f}%"
        })

    return running_loss/total, running_correct/total

# ─── validation ───────────────────────────────────────────────────────────────
@torch.no_grad()
def validate(model, loader, criterion, device, epoch, total_epochs):

    # Nirvaan Note Network(The Triple N): Set model to evaluation mode

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
        val_correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({
            "loss": f"{val_loss/total:.4f}",
            "acc":  f"{100*val_correct/total:.2f}%"
        })

    return val_loss/total, val_correct/total

# ─── main training loop ────────────────────────────────────────────────────────
def main():
    # Nirvaan Note Network(The Triple N): parse command-line arguments for flexibility
    parser = argparse.ArgumentParser(description="CIFAR-10 ResNet-18 Training w/ MixUp & RandAugment")
    parser.add_argument('--data-dir',     type=str,   default='./data')
    parser.add_argument('--batch-size',   type=int,   default=128)
    parser.add_argument('--epochs',       type=int,   default=20)
    parser.add_argument('--lr',           type=float, default=0.05)
    parser.add_argument('--max-lr',       type=float, default=0.2)
    parser.add_argument('--momentum',     type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--num-workers',  type=int,   default=4)
    parser.add_argument('--patience',     type=int,   default=5)
    parser.add_argument('--mixup-alpha',  type=float, default=0.2)
    args = parser.parse_args()

    # Nirvaan Note Network(The Triple N): reproducibility and performance
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Nirvaan Note Network(The Triple N): prepare data loaders
    train_loader, test_loader = get_data_loaders(
        args.data_dir, 
        args.batch_size, 
        args.num_workers, 
        device
    )

    # Nirvaan Note Network(The Triple N): model, loss with label smoothing, optimizer
    model     = resnet18_cifar(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    # Nirvaan Note Network(The Triple N): OneCycleLR for warm-up plus annealing

    scheduler = OneCycleLR(

        optimizer,

        max_lr=args.max_lr,

        steps_per_epoch=len(train_loader),

        epochs=args.epochs,

        pct_start=0.1,

        div_factor=10.0,

        final_div_factor=1e4,

        anneal_strategy='cos'
    )

    # Nirvaan Note Network(The Triple N): Mixed-precision scaler
    scaler = GradScaler()

    # Nirvaan Note Network(The Triple N): CSV logging (open once)
    metrics_file = open('CIFAR_ResNet18_V2.csv', 'w')
    metrics_file.write('Epoch,Train Loss,Train Accuracy,Validation Loss,Validation Accuracy\n')

    # Nirvaan Note Network(The Triple N): TensorBoard for richer logging
    writer = SummaryWriter()9

    # Nirvaan Note Network(The Triple N): early stopping setup
    best_val_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, 
            train_loader, 
            optimizer, 
            criterion,
            device, 
            scaler, 
            epoch, 
            args.epochs, 
            args.mixup_alpha
        )
        val_loss, val_acc = validate(
            model, 
            test_loader, 
            criterion,
            device, epoch, args.epochs
        )

        scheduler.step()

        # Nirvaan Note Network(The Triple N): append metrics to CSV
        metrics_file.write(f"{epoch},{train_loss:.4f},{train_acc:.4f},{val_loss:.4f},{val_acc:.4f}\n")
        metrics_file.flush()

        # Nirvaan Note Network(The Triple N): log to TensorBoard
        writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
        writer.add_scalars('Accuracy', {'train': train_acc, 'val': val_acc}, epoch)

        print(
            f"Epoch {epoch:02d}: "
            f"Train L={train_loss:.4f}, A={train_acc*100:5.2f}% | "
            f"Val   L={val_loss:.4f}, A={val_acc*100:5.2f}%"
        )

        # ─── early stopping & checkpointing ───────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_resnet18_cifar.pth")
            # Nirvaan Note Network(The Triple N): saved new best model
        else:
            epochs_no_improve += 1
            # Nirvaan Note Network(The Triple N): no improvement this epoch(plateu)

        if epochs_no_improve >= args.patience:
            print(f"No improvement for {args.patience} epochs. Stopping early.")
            break

    # Nirvaan Note Network(The Triple N): clean up and final save
    metrics_file.close()
    writer.close()
    print(f"Training complete. Best validation accuracy: {best_val_acc*100:.2f}%")
    print("Best model saved as best_resnet18_cifar.pth")

if __name__ == "__main__":
    freeze_support()
    main()
