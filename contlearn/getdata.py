import random
import torch
from torchvision import datasets

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


def getMNIST(batch_size:int):

    mnist_train = datasets.MNIST(
        "../data", 
        train=True, 
        download=True, 
        transform=transforms.ToTensor(),
    )

    mnist_test = datasets.MNIST(
        "../data", 
        train=False, 
        download=True, 
        transform=transforms.ToTensor(),
    )

    train_loader = DataLoader(mnist_train, batch_size = batch_size, shuffle=True)
    test_loader = DataLoader(mnist_test, batch_size = batch_size, shuffle=False)

    return train_loader, test_loader

def getFashionMNIST(batch_size:int):

    mnist_train = datasets.FashionMNIST(
        "../data", 
        train=True, 
        download=True, 
        transform=transforms.ToTensor(),
    )

    mnist_test = datasets.FashionMNIST(
        "../data", 
        train=False, 
        download=True, 
        transform=transforms.ToTensor(),
    )

    train_loader = DataLoader(mnist_train, batch_size = batch_size, shuffle=True)
    test_loader = DataLoader(mnist_test, batch_size = batch_size, shuffle=False)

    return train_loader, test_loader