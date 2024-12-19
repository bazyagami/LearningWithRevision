import argparse
import torch
from model import get_pretrained_resnet
from data import load_cifar100
from pretrained import train_pretrained_baseline, train_pretrained_selective

def main():
    parser = argparse.ArgumentParser(description="Train ResNet on CIFAR-100")
    parser.add_argument("--mode", type=str, choices=["baseline", "selective_gradient"], required=True,
                        help="Choose training mode: 'baseline' or 'selective_gradient'")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained ResNet on ImageNet")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, _ = load_cifar100()

    if args.pretrained:
        print("Using pretrained ResNet...")
        model = get_pretrained_resnet(num_classes=100)
        if args.mode == "baseline":
            print("Training pretrained model in baseline mode...")
            train_pretrained_baseline(model, train_loader, device)
        elif args.mode == "selective_gradient":
            print("Training pretrained model with selective gradient updates...")
            train_pretrained_selective(model, train_loader, device)

if __name__ == "__main__":
    main()
