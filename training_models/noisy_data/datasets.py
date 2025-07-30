import numpy as np 
import torchvision.transforms as transforms
from .cifar import CIFAR10, CIFAR100



train_cifar10_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

test_cifar10_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
train_cifar100_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
test_cifar100_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

def input_dataset(dataset, noise_type, noise_path, is_human):
    if dataset == 'cifar10':
        train_dataset = CIFAR10(root='./data/',
                                download=True,  
                                train=True, 
                                transform = train_cifar10_transform,
                                noise_type = noise_type,
                                noise_path = noise_path, is_human=is_human
                           )
        test_dataset = CIFAR10(root='./data/',
                                download=False,  
                                train=False, 
                                transform = test_cifar10_transform,
                                noise_type=noise_type
                          )
        num_classes = 10
        num_training_samples = 50000
    elif dataset == 'cifar100':
        train_dataset = CIFAR100(root='./data/',
                                download=True,  
                                train=True, 
                                transform=train_cifar100_transform,
                                noise_type=noise_type,
                                noise_path = noise_path, is_human=is_human
                            )
        test_dataset = CIFAR100(root='./data/',
                                download=False,  
                                train=False, 
                                transform=test_cifar100_transform,
                                noise_type=noise_type
                            )
        num_classes = 100
        num_training_samples = 50000
    return train_dataset, test_dataset, num_classes, num_training_samples








