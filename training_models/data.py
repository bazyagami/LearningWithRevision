import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.utils.data as data
import torchvision.models as models
from imbalance_cifar import IMBALANCECIFAR100, IMBALANCECIFAR10
from medmnist import NoduleMNIST3D, INFO, Evaluator
import medmnist

def load_cifar100(long_tail, batch_size=128):
    cls_num_list = None
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if long_tail:
        trainset = IMBALANCECIFAR100(root='./data', imb_type="exp", imb_factor=0.01, rand_number=0, train=True, download=True, transform=transform)
        cls_num_list = trainset.get_cls_num_list()

    else: 
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, cls_num_list

def load_cifar10(long_tail, batch_size=128):
    cls_num_list = None
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if long_tail:
        trainset = IMBALANCECIFAR10(root='./data', imb_type="exp", imb_factor=0.01, rand_number=0, train=True, download=True, transform=transform)
        cls_num_list = trainset.get_cls_num_list()

    else: 
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, cls_num_list

def load_mnist(batch_size=128):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
    ])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def load_imagenet(batch_size=16):
    print("Performing transformations")
    # transform = transforms.Compose([transforms.Resize((224,224))
    #     ,transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # transform = models.EfficientNet_B0_Weights.IMAGENET1K_V1.transforms
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
    print("Transformations done, extracting the trainset")
    trainset = torchvision.datasets.ImageNet(root='E:\ImageNet', split="train", transform=transform)
    valset = torchvision.datasets.ImageNet(root='E:\ImageNet', split='val', transform=transform)
    print("loading the dataset")
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(valset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader
    ##TODO: download imagenet

def load_cityscapes(data_dir="D:\LearningWithRevision\mmsegmentation\data\cityscapes", batch_size=8):
    transform = transforms.Compose([
        transforms.Resize((512, 1024)),  # Resize to a common size
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    target_transform = transforms.Compose([
        transforms.Resize((512, 1024)),  
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t.squeeze().long())
    ])

    train_dataset = torchvision.datasets.Cityscapes(
        root=data_dir,
        split="train",
        mode="fine",  # Use "fine" annotations
        target_type="semantic",
        transform=transform,
        target_transform=target_transform
    )

    test_dataset = torchvision.datasets.Cityscapes(
        root=data_dir,
        split="test",
        mode="fine",  # Use "fine" annotations
        target_type="semantic",
        transform=transform,
        target_transform=target_transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    return train_loader, test_loader

def load_medmnist3D(batch_size=128):
    data_flag = "organmnist3d"
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])
    train_dataset = DataClass(split='train', download=True, size=64)
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dataset = DataClass(split="test", download=True, size=64)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    return train_loader, test_loader
