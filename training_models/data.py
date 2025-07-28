import os
import pandas as pd
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.utils.data as data
import torchvision.models as models
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_file_from_google_drive, download_url, extract_archive
from imbalance_cifar import IMBALANCECIFAR100, IMBALANCECIFAR10
from medmnist import NoduleMNIST3D, INFO, Evaluator
import medmnist
import numpy as np


class Cub2011(VisionDataset):
    """`CUB-200-2011 <http://www.vision.caltech.edu/visipedia/CUB-200-2011.html>`_ Dataset.

        Args:
            root (string): Root directory of the dataset.
            train (bool, optional): If True, creates dataset from training set, otherwise
               creates from test set.
            transform (callable, optional): A function/transform that  takes in an PIL image
               and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
               target and transforms it.
            download (bool, optional): If true, downloads the dataset from the internet and
               puts it in root directory. If dataset is already downloaded, it is not
               downloaded again.
    """
    base_folder = 'CUB_200_2011/images'
    # url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    file_id = '1hbzc_P1FuxMkcabkgn9ZKinBwW683j45'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(Cub2011, self).__init__(root, transform=transform, target_transform=target_transform)

        self.loader = default_loader
        self.train = train
        if download:
            self._download()


    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        class_names = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'classes.txt'),
                                  sep=' ', names=['class_name'], usecols=[1])
        self.class_names = class_names['class_name'].to_list()
        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_file_from_google_drive(self.file_id, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class Aircraft(VisionDataset):
    """`FGVC-Aircraft <http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/>`_ Dataset.

    Args:
        root (string): Root directory of the dataset.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        class_type (string, optional): choose from ('variant', 'family', 'manufacturer').
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    url = 'http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz'
    class_types = ('variant', 'family', 'manufacturer')
    splits = ('train', 'val', 'trainval', 'test')
    img_folder = os.path.join('fgvc-aircraft-2013b', 'data', 'images')

    def __init__(self, root, train=True, class_type='variant', transform=None,
                 target_transform=None, download=False):
        super(Aircraft, self).__init__(root, transform=transform, target_transform=target_transform)
        split = 'trainval' if train else 'test'
        if split not in self.splits:
            raise ValueError('Split "{}" not found. Valid splits are: {}'.format(
                split, ', '.join(self.splits),
            ))
        if class_type not in self.class_types:
            raise ValueError('Class type "{}" not found. Valid class types are: {}'.format(
                class_type, ', '.join(self.class_types),
            ))

        self.class_type = class_type
        self.split = split
        self.classes_file = os.path.join(self.root, 'fgvc-aircraft-2013b', 'data',
                                         'images_%s_%s.txt' % (self.class_type, self.split))

        if download:
            self.download()

        (image_ids, targets, classes, class_to_idx) = self.find_classes()
        samples = self.make_dataset(image_ids, targets)

        self.loader = default_loader

        self.samples = samples
        self.classes = classes
        self.class_to_idx = class_to_idx

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.samples)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.img_folder)) and \
               os.path.exists(self.classes_file)

    def download(self):
        if self._check_exists():
            return

        # prepare to download data to PARENT_DIR/fgvc-aircraft-2013.tar.gz
        print('Downloading %s...' % self.url)
        tar_name = self.url.rpartition('/')[-1]
        download_url(self.url, root=self.root, filename=tar_name)
        tar_path = os.path.join(self.root, tar_name)
        print('Extracting %s...' % tar_path)
        extract_archive(tar_path)
        print('Done!')

    def find_classes(self):
        # read classes file, separating out image IDs and class names
        image_ids = []
        targets = []
        with open(self.classes_file, 'r') as f:
            for line in f:
                split_line = line.split(' ')
                image_ids.append(split_line[0])
                targets.append(' '.join(split_line[1:]))

        # index class names
        classes = np.unique(targets)
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        targets = [class_to_idx[c] for c in targets]

        return image_ids, targets, classes, class_to_idx

    def make_dataset(self, image_ids, targets):
        assert (len(image_ids) == len(targets))
        images = []
        for i in range(len(image_ids)):
            item = (os.path.join(self.root, self.img_folder,
                                 '%s.jpg' % image_ids[i]), targets[i])
            images.append(item)
        return images


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
    return train_loader, test_loader, cls_num_list, len(trainset)

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
    return train_loader, test_loader, cls_num_list, len(trainset)


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
    return train_loader, test_loader, len(trainset)


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
    trainset = torchvision.datasets.ImageNet(root='E:\\ImageNet', split="train", transform=transform)
    valset = torchvision.datasets.ImageNet(root='E:\\ImageNet', split='val', transform=transform)
    print("loading the dataset")
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(valset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader, len(trainset)

    ##TODO: download imagenet

def load_cityscapes(data_dir="D:\\LearningWithRevision\\mmsegmentation\\data\\cityscapes", batch_size=8):
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

    return train_loader, test_loader, len(train_dataset)


def load_cub2011(batch_size=128, root='./data', download=True):
    """加载 CUB-200-2011 数据集
    
    Args:
        batch_size (int): 批次大小
        root (str): 数据根目录
        download (bool): 是否自动下载数据集
    
    Returns:
        tuple: (train_loader, test_loader, len(trainset))
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    trainset = Cub2011(root=root, train=True, transform=transform, download=download)
    testset = Cub2011(root=root, train=False, transform=transform, download=download)
    
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, len(trainset)


def load_aircraft(batch_size=128, class_type='variant', root='./data', download=True):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    trainset = Aircraft(root=root, train=True, class_type=class_type, transform=transform, download=download)
    testset = Aircraft(root=root, train=False, class_type=class_type, transform=transform, download=download)
    
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, len(trainset)



class Flowers102(VisionDataset):
    """`Oxford 102 Category Flower Dataset <https://www.robots.ox.ac.uk/~vgg/data/flowers/102/>`_ Dataset.

    Args:
        root (string): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"train"`` (default), ``"val"``, or ``"test"``.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    
    _download_url_prefix = 'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/'
    _file_dict = {
        'image': ('102flowers.tgz', '52808999861908f626f3c1f4e79d11fa'),
        'label': ('imagelabels.mat', 'e0620be6f572b9609742df49c70aed4d'),
        'setid': ('setid.mat', 'a5357ecc9cb78c4bef273ce3793fc85c')
    }
    _splits_map = {'train': 'trnid', 'val': 'valid', 'test': 'tstid'}

    def __init__(self, root, split='train', transform=None, target_transform=None, download=False):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self._split = split
        self._base_folder = os.path.join(self.root, 'flowers-102')
        self._images_folder = os.path.join(self._base_folder, 'jpg')

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted. You can use download=True to download it')

        from scipy.io import loadmat
        import random


        set_ids = loadmat(os.path.join(self._base_folder, 'setid.mat'), squeeze_me=True)
        image_ids = set_ids[self._splits_map[self._split]].tolist()

        labels = loadmat(os.path.join(self._base_folder, 'imagelabels.mat'), squeeze_me=True)
        image_id_to_label = dict(enumerate(labels['labels'].tolist(), 1))

        self._labels = []
        self._image_files = []
        for image_id in image_ids:
            self._labels.append(image_id_to_label[image_id] - 1)  # 从1开始索引转为从0开始
            self._image_files.append(os.path.join(self._images_folder, f'image_{image_id:05d}.jpg'))

        self.loader = default_loader


    def __len__(self):
        return len(self._image_files)

    def __getitem__(self, idx):
        image_file, label = self._image_files[idx], self._labels[idx]
        image = self.loader(image_file)

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def extra_repr(self):
        return f"split={self._split}"

    def _check_integrity(self):
        if not os.path.isdir(self._images_folder):
            return False

        for id in ['label', 'setid']:
            filename, md5 = self._file_dict[id]
            fpath = os.path.join(self._base_folder, filename)
            if not os.path.isfile(fpath):
                return False

        return True

    def _download(self):
        if self._check_integrity():
            return
        
        import tarfile
        from torchvision.datasets.utils import check_integrity

        for id in ['image', 'label', 'setid']:
            filename, md5 = self._file_dict[id]
            url = self._download_url_prefix + filename
            fpath = os.path.join(self._base_folder, filename)
            
            os.makedirs(self._base_folder, exist_ok=True)
            
            if not check_integrity(fpath, md5):
                print(f'Downloading {url}...')
                download_url(url, root=self._base_folder, filename=filename, md5=md5)

        with tarfile.open(os.path.join(self._base_folder, '102flowers.tgz')) as tar:
            tar.extractall(self._base_folder)

        print('Done!')


def load_flowers(batch_size=128, split='train', root='./data', download=True):

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    

    original_trainset = Flowers102(root=root, split='train', transform=train_transform, download=download)
    original_valset = Flowers102(root=root, split='val', transform=train_transform, download=download)
    testset = Flowers102(root=root, split='test', transform=test_transform, download=download)
    

    from torch.utils.data import ConcatDataset
    combined_trainset = ConcatDataset([original_trainset, original_valset])

    val_dataset_for_eval = Flowers102(root=root, split='val', transform=test_transform, download=download, use_custom_split=use_custom_split)

    train_loader = DataLoader(combined_trainset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset_for_eval, batch_size=batch_size, shuffle=False)  # 验证时不使用数据增强
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, len(combined_trainset)

