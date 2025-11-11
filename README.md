# Progressive Data Dropout: An Embarrassingly Simple Approach to Faster Train

Link to paper : https://arxiv.org/pdf/2505.22342

### Setup
Install python3+.
Install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

Then run:
```
make env
conda activate pdd
pip install -e .
```

### Datasets Supported
The 'data.py' file has the dataloaders for CIFAR100, CIFAR10, MNIST and IMAGENET. 

### Additional datasets
Note: `make clean` can remove all the extraneous pieces from running training.

For longtail classification, the dataset is created by the imbalance_cifar.py file. 

Running the baseline (example: model, save_path, dataset and batch size can be changed accordingly): 
```
python .\main.py --model mobilenet_v2 --mode baseline --epoch 30 --save_path cifar10_results/mobilenet_v2 --dataset cifar10 --batch_size 32 --task classification 
```

Running the Difficulty-Based Progressive Dropout (DBPD):
```
python .\main.py --model mobilenet_v2 --mode train_with_revision --epoch 30 --save_path cifar10_results/mobilenet_v2 --dataset cifar10 --batch_size 32 --start_revision 29 --task classification --threshold 0.3
```

Running the Scheduled Match Random Dropout (SMRD) : 
```
python .\main.py --model mobilenet_v2 --mode train_with_random --epoch 30 --save_path cifar10_results/mobilenet_v2 --dataset cifar10 --batch_size 32 --start_revision 29 --task classification --threshold 0.3
```

Running the Scalar Random Dropout (SRD): 
```
python .\main.py --model mobilenet_v2 --mode train_with_percentage --epoch 30 --save_path cifar10_results/mobilenet_v2 --dataset cifar10 --batch_size 32 --start_revision 29 --task classification --threshold 0.3
```

If you wish to change the percentage parameter, head to the train_with_random function and change the decay parameter.

Note: The current codebase provides a recipe for single GPU training, multi-GPU training codes will be released soon.
