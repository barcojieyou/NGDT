"""
Data loading utilities
"""
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
from typing import Tuple, Optional


def get_dataset(dataset_name: str, batch_size: int, n_train: Optional[int] = None,
                val_split: float = 0.1, **kwargs) -> Tuple[DataLoader, DataLoader, int]:
    """Get dataset and create data loaders"""
    
    if dataset_name == 'mnist':
        return _get_mnist(batch_size, n_train, val_split, **kwargs)
    elif dataset_name == 'cifar10':
        return _get_cifar10(batch_size, n_train, val_split, **kwargs)
    elif dataset_name == 'cifar100':
        return _get_cifar100(batch_size, n_train, val_split, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def _get_mnist(batch_size: int, n_train: Optional[int], val_split: float, **kwargs) -> Tuple[DataLoader, DataLoader, int]:
    """Get MNIST dataset"""
    num_classes = 10
    
    # Transformations
    transform_train = transforms.Compose([
        transforms.RandomCrop(28, padding=4),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load datasets
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    # Split training set if needed
    if val_split > 0:
        val_size = int(len(train_dataset) * val_split)
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    else:
        val_dataset = test_dataset
    
    # Limit training samples if specified
    if n_train is not None and n_train < len(train_dataset):
        indices = np.random.choice(len(train_dataset), n_train, replace=False)
        train_dataset = Subset(train_dataset, indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, num_classes


def _get_cifar10(batch_size: int, n_train: Optional[int], val_split: float, **kwargs) -> Tuple[DataLoader, DataLoader, int]:
    """Get CIFAR-10 dataset"""
    num_classes = 10
    
    # Transformations
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    
    # Load datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    # Split training set if needed
    if val_split > 0:
        val_size = int(len(train_dataset) * val_split)
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    else:
        val_dataset = test_dataset
    
    # Limit training samples if specified
    if n_train is not None and n_train < len(train_dataset):
        indices = np.random.choice(len(train_dataset), n_train, replace=False)
        train_dataset = Subset(train_dataset, indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, num_classes


def _get_cifar100(batch_size: int, n_train: Optional[int], val_split: float, **kwargs) -> Tuple[DataLoader, DataLoader, int]:
    """Get CIFAR-100 dataset"""
    num_classes = 100
    
    # Transformations
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    # Load datasets
    train_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    # Split training set if needed
    if val_split > 0:
        val_size = int(len(train_dataset) * val_split)
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    else:
        val_dataset = test_dataset
    
    # Limit training samples if specified
    if n_train is not None and n_train < len(train_dataset):
        indices = np.random.choice(len(train_dataset), n_train, replace=False)
        train_dataset = Subset(train_dataset, indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, num_classes


def get_sample_distribution(dataloader: DataLoader) -> dict:
    """Get distribution of samples in dataloader"""
    class_counts = {}
    
    for _, targets in dataloader:
        targets_np = targets.numpy()
        unique, counts = np.unique(targets_np, return_counts=True)
        
        for cls, count in zip(unique, counts):
            if cls not in class_counts:
                class_counts[cls] = 0
            class_counts[cls] += count
    
    total_samples = sum(class_counts.values())
    class_distribution = {cls: count / total_samples for cls, count in class_counts.items()}
    
    return {
        'class_counts': class_counts,
        'class_distribution': class_distribution,
        'total_samples': total_samples,
        'num_classes': len(class_counts)
    }
