import argparse
import torch
from model import resnet18
from data import load_cifar100
from baseline import train_baseline
from selective_gradient import train_selective

def main():
    parser = argparse.ArgumentParser(description="Train ResNet on CIFAR-100")
    parser.add_argument("--mode", type=str, choices=["baseline", "selective_gradient"], required=True,
                        help="Choose training mode: 'baseline' or 'selective_gradient'")
    parser.add_argument("--epoch", type=int, required=False,
                        help="Number of epochs to train for")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, _ = load_cifar100()
    model = resnet18(num_classes=100)

    if args.mode == "baseline":
        print("Training in baseline mode...")
        train_baseline(model, train_loader, device, args.epoch)
    elif args.mode == "selective_gradient":
        print("Training with selective gradient updates...")
        train_selective(model, train_loader, device, args.epoch)

if __name__ == "__main__":
    main()
