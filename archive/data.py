import torch
from torch.utils.data import DataLoader
import os
import glob
from torchvision import datasets, transforms

def get_dataloaders(batch_size=256):
    """
    Downloads and loads the CIFAR-10 dataset.
    Returns train_loader and test_loader.
    """
    data_root = './data'
    download_data = True
    
    # Kaggle Workaround: If you attach a dataset in Kaggle, it goes to /kaggle/input/
    if os.path.exists("/kaggle/input"):
        kaggle_matches = glob.glob("/kaggle/input/**/cifar-10-batches-py", recursive=True)
        if kaggle_matches:
            data_root = os.path.dirname(kaggle_matches[0])
            download_data = False
            print(f"Found mounted Kaggle dataset! Skipping slow download. Using: {data_root}")

    # Data Augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),
        transforms.RandomErasing(p=0.25)
    ])
    
    # Standard transform for testing
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616))
    ])

    train_dataset = datasets.CIFAR10(root=data_root, train=True, download=download_data, transform=train_transform)
    test_dataset = datasets.CIFAR10(root=data_root, train=False, download=download_data, transform=test_transform)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4
    )

    return train_loader, test_loader
