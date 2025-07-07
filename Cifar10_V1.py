import os
from multiprocessing import freeze_support

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

# ─── Mixed-precision context manager ───────────────────────────────────────────
from torch.cuda.amp import autocast, GradScaler

# ─── CIFAR-10–adapted ResNet-18 ────────────────────────────────────────────────
def resnet18_cifar(num_classes=10):
    model = models.resnet18(pretrained=False, num_classes=num_classes)
    # adapt first conv to 32×32 inputs
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model

# ─── data loaders with lightweight augmentation & conditional pin_memory ─────
def get_data_loaders(batch_size=128, num_workers=4, device=None):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)
    pin = device is not None and device.type == "cuda"

    # Nirvaan Note Network(The Triple N): CIFAR-10 images are 32x32, so we use RandomCrop with padding of 4, Random horizontal flip, Convert the image to a Tensor, Normalize using the mean/std above

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Nirvaan Note Network(The Triple N): For the test set, we only normalize the images without any augmentation because we want to evaluate the model on the original images, not change anything.

    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Nirvaan Note Network(The Triple N) extract CIFAR-10 dataset & return DataLoaders for training and testing. The DataLoaders will use the transformations defined above, shuffle, split into batches, and they will be set to pin memory if a CUDA device is available.

    train_ds = datasets.CIFAR10("./data", train=True,  download=True, transform=train_tf)
    test_ds  = datasets.CIFAR10("./data", train=False, download=True, transform=test_tf)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin
    )

    return train_loader, test_loader

# ─── single-epoch training with AMP ────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device, scaler, epoch, total_epochs):

    # Nirvaan Note Network(The Triple N): The model is set to training mode, and we initialize running loss, running correct predictions, and total samples to zero. We also create a progress bar using tqdm to visualize the training process(tqdm).

    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0
    pbar = tqdm(loader, desc=f"Train [{epoch}/{total_epochs}]", ncols=120)

    # Nirvaan Note Network(The Triple N): We fetch training images and labels from the DataLoader, move them to the specified device (GPU or CPU), and zero out the gradients in the optimizer. We then use the autocast context manager to enable mixed-precision training.

    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        # Nirvaan Note Network(The Triple N): The section below speeds up the forward pass and loss computation by automatically choosing whether each operation should run in half-precision (FP16) or full-precision (FP32).

        with autocast():
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        # Nirvaan Note Network(The Triple N): Scales the loss before backprop, applies the scaled gradients, and adjusts the scaling factor for the next iteration.

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Nirvaan Note Network(The Triple N): For each image, we compute the most probable classification of a image by taking the index of the maximum value in the output tensor. We then update the running loss and correct predictions, and finally update the total number of samples processed.

        preds = outputs.argmax(dim=1)
        running_loss += loss.item() * imgs.size(0)
        running_correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Nirvaan Note Network(The Triple N): We update the progress bar with the current loss and accuracy.

        pbar.set_postfix({
            "loss": f"{running_loss/total:.4f}",
            "acc":  f"{100*running_correct/total:.2f}%"
        })

    return running_loss/total, running_correct/total

# ─── validation ───────────────────────────────────────────────────────────────
@torch.no_grad()
def validate(model, loader, criterion, device, epoch, total_epochs):
    # Nirvaan Note Network(The Triple N): The model is set to evaluation mode, and we initialize validation loss, correct predictions, and total samples to zero. We also create a progress bar using tqdm to visualize the validation process(tqdm).

    model.eval()
    val_loss = 0.0
    val_correct = 0
    total = 0
    pbar = tqdm(loader, desc=f"Valid [{epoch}/{total_epochs}]", ncols=120)

    # Nirvaan Note Network(The Triple N): We fetch validation images and labels from the DataLoader, move them to the specified device (GPU or CPU), and use the autocast context manager to enable mixed-precision inference.

    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        with autocast():
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        # Nirvaan Note Network(The Triple N): For each image, we compute the most probable classification of a image by taking the index of the maximum value in the output tensor. We then update the validation loss and correct predictions, and finally update the total number of samples processed.

        preds = outputs.argmax(dim=1)
        val_loss += loss.item() * imgs.size(0)
        val_correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Nirvaan Note Network(The Triple N): We update the progress bar with the current loss and accuracy.

        pbar.set_postfix({
            "loss": f"{val_loss/total:.4f}",
            "acc":  f"{100*val_correct/total:.2f}%"
        })

    # Nirvaan Note Network(The Triple N): We return the average validation loss and accuracy.

    return val_loss/total, val_correct/total

# ─── main training loop ────────────────────────────────────────────────────────
def main():

    # Nirvaan Note Network(The Triple N): Initialize the CSV file.

    import pandas as pd

    columns = ['Epoch', 'Train Loss', 'Train Accuracy', 'Validation Loss', 'Validation Accuracy']

    df_empty = pd.DataFrame(columns=columns)
    df_empty.to_csv('CIFAR_ResNet18_V1.csv', index=False)
        

    # Nirvaan Note Network(The Triple N): We set the random seed for reproducibility, enable cuDNN benchmarking for performance, and determine the device to use (GPU or CPU).

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Nirvaan Note Network(The Triple N): We create the data loaders for training and testing using the get_data_loaders function defined above. We specify the batch size, number of workers for data loading, and the device.

    train_loader, test_loader = get_data_loaders(batch_size=128, num_workers=4, device=device)

    # Nirvaan Note Network(The Triple N): We initialize the ResNet-18 model adapted for CIFAR-10, the loss function with label smoothing, the optimizer (SGD), and the learning rate scheduler (CosineAnnealingLR). We also create a GradScaler for mixed-precision training.

    model = resnet18_cifar(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-4)

    # Nirvaan Note Network(The Triple N): We set the model to training mode, initialize the GradScaler for mixed-precision training, and define the number of epochs for training.

    scaler = GradScaler()
    num_epochs = 20

    # Nirvaan Note Network(The Triple N): We cycle through the training & validation process for the specified number of epochs. In each epoch, we train the model on the training set and validate it on the test set. We also update the learning rate scheduler after each epoch.

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch( 
            model, 
            train_loader, 
            optimizer, 
            criterion, 
            device, 
            scaler, 
            epoch, 
            num_epochs
        )
        val_loss, val_acc = validate(
            model, 
            test_loader, 
            criterion, 
            device, 
            epoch, 
            num_epochs
        )
        scheduler.step()

        # Nirvaan Note Network(The Triple N): We print the training & validation loss and accuracy for each epoch. Additionally, we save the results to a CSV file for later analysis.
        
        import pandas as pd

        df = pd.DataFrame({
            'Epoch': [epoch] ,
            
            'Train Loss': [train_loss] ,

            'Train Accuracy': [train_acc] ,
            
            'Validation Loss': [val_loss] ,
            'Validation Accuracy': [val_acc]
        }).to_csv('CIFAR_ResNet18.csv', mode='a', header=False, index=False)

        print(
            f"\n Epoch {epoch:02d}: "

            f"Train L={train_loss:.4f}, A={train_acc*100:5.2f}% | "

            f"Val   L={val_loss:.4f}, A={val_acc*100:5.2f}%"

        )
    
    # Nirvaan Note Network(The Triple N): After training, we save the model. This allows us to load the model later for inference.

    torch.save(model.state_dict(), "resnet18_cifar.pth")
    print("Model saved to resnet18_cifar.pth")

if __name__ == "__main__":
    freeze_support()
    main()
