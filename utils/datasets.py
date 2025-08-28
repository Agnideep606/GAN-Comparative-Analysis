import torch
from torchvision import datasets, transforms

def get_dataset(name, data_dir="./data", train=True, download=True):
    """
    Loads MNIST, CIFAR-10, FashionMNIST, or SVHN datasets.
    """
    if name.lower() == "mnist":
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        return datasets.MNIST(root=data_dir, train=train, transform=transform, download=download)

    elif name.lower() == "fmnist":
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        return datasets.FashionMNIST(root=data_dir, train=train, transform=transform, download=download)

    elif name.lower() == "cifar10":
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        return datasets.CIFAR10(root=data_dir, train=train, transform=transform, download=download)

    elif name.lower() == "svhn":
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        split = 'train' if train else 'test'
        return datasets.SVHN(root=data_dir, split=split, transform=transform, download=download)

    else:
        raise ValueError(f"Dataset {name} not supported.")

def get_dataloader(name, batch_size=128, train=True, shuffle=True, num_workers=4):
    dataset = get_dataset(name, train=train)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
