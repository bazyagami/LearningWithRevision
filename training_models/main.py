import argparse
import torch
from model import resnet18, efficientnet_b0
from model_zoo import mobilenet_v2, mobilenet_v3, resnet18, resnet34, resnet50, resnet101, vit_b_16, efficientnet_b0, efficientformer
from data import load_cifar100, load_mnist
from baseline import train_baseline
from selective_gradient import train_selective, train_selective_epoch, train_with_revision
from test import test_model

def main():
    parser = argparse.ArgumentParser(description="Train ResNet on CIFAR-100")
    parser.add_argument("--mode", type=str, choices=["baseline", "selective_gradient", "selective_epoch", "train_with_revision"], required=True,
                        help="Choose training mode: 'baseline' or 'selective_gradient'")
    parser.add_argument("--epoch", type=int, required=False, default=10,
                        help="Number of epochs to train for")
    parser.add_argument("--model", type=str, choices=["resnet18", "resnet34", "resnet50", "resnet101", "efficientnet_b0", "mobilenet_v2", "mobilenet_v3", "vit_b_16", "efficientformer"], required=True,
                        help="Choose the model: 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'mobilenet_v2', mobilenet_v3 or 'efficientnet_b0'")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained versions")
    parser.add_argument("--save_path", type=str, help="to save graphs")
    parser.add_argument("--threshold", type=float, help="threshold to remove samples")
    parser.add_argument("--epoch_threshold", type=int, help="threshold to reintroduce correct samples in epoch")
    parser.add_argument("--dataset", type=str, help="CIFAR or MNIST")
    parser.add_argument("--batch_size", type=int, help="32,64,128 etc.")
    parser.add_argument("--start_revision", type=int, help="Start revision after the given epoch")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained = False
    if args.dataset == "mnist":
        num_classes = 10
        train_loader, test_loader = load_mnist()
    elif args.dataset == "cifar":
        if args.batch_size:
            train_loader, test_loader = load_cifar100(args.batch_size)
        else:
            train_loader, test_loader = load_cifar100()
        num_classes = 100

    if args.pretrained:
        pretrained = True

    ###Models From Scratch###
    if args.model == "resnet18":
        # model = resnet18(num_classes=100)
        model = resnet18(num_classes, pretrained)
    elif args.model == "efficientnet_b0":
        # model = efficientnet_b0(num_classes=100)
        model = efficientnet_b0(num_classes, pretrained)

    ###PyTorch Models###
    elif args.model == "mobilenet_v2":
        model = mobilenet_v2(num_classes, pretrained)
    elif args.model == "mobilenet_v3":
        model = mobilenet_v3(num_classes, pretrained)
    elif args.model == "resnet34":
        model = resnet34(num_classes, pretrained)
    elif args.model == "resnet50":
        model = resnet50(num_classes, pretrained)
    elif args.model == "resnet101":
        model = resnet101(num_classes, pretrained)
    elif args.model == "vit_b_16":
        model = vit_b_16(num_classes, pretrained)
    elif args.model == "efficientformer":
        model = efficientformer(num_classes, pretrained)

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if args.pretrained:
        args.model = args.model + "_" + "pretrained" + "_" + str(args.threshold)
    else:
        args.model = args.model + "_" + str(args.threshold)

    if args.mode == "baseline":
        args.model = args.model + "_" + "baseline"

    if args.mode == "baseline":
        print("Training in baseline mode...")
        trained_model = train_baseline(args.model, model, train_loader, test_loader, device, args.epoch, args.save_path)
    elif args.mode == "selective_gradient":
        print("Training with selective gradient updates...")
        trained_model = train_selective(args.model, model, train_loader, test_loader, device, args.epoch, args.save_path, args.threshold)
    elif args.mode == "selective_epoch":
        print(f"Reintroducing correct examples and training...")
        trained_model = train_selective_epoch(args.model,model,train_loader,
                                              device,args.epoch,args.save_path,args.threshold)
    elif args.mode == "train_with_revision":
        print(f"Training {args.mode}, will start revision after {args.start_revision}")
        trained_model = train_with_revision(args.model, model, train_loader, test_loader, device, args.epoch, args.save_path, args.threshold, args.start_revision)

    test_accuracy = test_model(trained_model, test_loader, device)
    print("Model accuracy:", test_accuracy)

if __name__ == "__main__":
    main()
