import argparse
import torch
from model import resnet18, efficientnet_b0
from model_zoo import ModelZoo
from data import load_cifar100, load_mnist, load_imagenet, load_cityscapes, load_cifar10, load_medmnist3D
from baseline import train_baseline
from selective_gradient import TrainRevision
from test import test_model
from longtail_train import train_baseline_longtail, train_with_revision_longtail


def main():
    parser = argparse.ArgumentParser(description="Train ResNet on CIFAR-100")
    parser.add_argument("--mode", type=str, choices=["baseline", "selective_gradient", "selective_epoch", "train_with_revision", "train_with_samples", "train_with_revision_3d", "train_with_random", "train_with_inv_lin", "train_with_log", "train_with_percentage"], required=True,
                        help="Choose training mode: 'baseline' or 'selective_gradient'")
    parser.add_argument("--epoch", type=int, required=False, default=10,
                        help="Number of epochs to train for")
    parser.add_argument("--task", type=str, required=True, default="classification",
                        help="segmentation or classification")
    parser.add_argument("--model", type=str, choices=["resnet18", "resnet_3d", "resnet34", "resnet50", "resnet101", "efficientnet_b0","efficientnet_b7", "efficientnet_b4", "mobilenet_v2", "mobilenet_v3", "vit_b_16", "mae_vit_b_16", "efficientformer", "segformer_b2"], required=True,
                        help="Choose the model: 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'mobilenet_v2', 'mobilenet_v3', 'efficientnet_b0', 'vit_b_16', 'mae_vit_b_16'")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained versions (applies to torchvision models, not MAE)")
    parser.add_argument("--mae_checkpoint", type=str, default=None, help="Path to MAE pretrained checkpoint file (used with --model mae_vit_b_16)")
    parser.add_argument("--save_path", type=str, help="to save graphs")
    parser.add_argument("--threshold", type=float, help="threshold to remove samples")
    parser.add_argument("--epoch_threshold", type=int, help="threshold to reintroduce correct samples in epoch")
    parser.add_argument("--dataset", type=str, help="CIFAR or MNIST")
    parser.add_argument("--batch_size", type=int, help="32,64,128 etc.")
    parser.add_argument("--start_revision", type=int, help="Start revision after the given epoch")
    parser.add_argument("--long_tail", action="store_true", help="LongTail CIFAR100 or native version")
    parser.add_argument("--ldam", action="store_true", help="Use LDAM-DRW method for long tail classification")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained = False
    if args.dataset == "mnist":
        num_classes = 10
        train_loader, test_loader = load_mnist()
    elif args.dataset == "cifar":
        if args.batch_size:
            train_loader, test_loader, cls_num_list, data_size = load_cifar100(args.long_tail, args.batch_size)
        else:
            train_loader, test_loader, cls_num_list, data_size = load_cifar100(args.long_tail)

        num_classes = 100
    elif args.dataset == "cifar10":
        train_loader, test_loader, cls_num_list, data_size = load_cifar10(args.long_tail, args.batch_size)
        num_classes = 10
    elif args.dataset == "imagenet":
        num_classes = 1000
        train_loader, test_loader, data_size = load_imagenet(args.batch_size)
    elif args.dataset == "cityscapes":
        num_classes = 19
        train_loader, test_loader = load_cityscapes()
    elif args.dataset == "organ_medmnist3d":
        num_classes = 11
        train_loader, test_loader, data_size = load_medmnist3D(args.batch_size)

    if args.pretrained:
        pretrained = True
    
    if args.task == "classification":
        mz = ModelZoo(num_classes, pretrained)
    elif args.task == "segmentation":
        mz = ModelZoo(num_classes, pretrained)

    ###Models From Scratch###
    if args.model == "resnet18":
        # model = resnet18(num_classes=100)
        model = mz.resnet18()
    elif args.model == "efficientnet_b0":
        # model = efficientnet_b0(num_classes=100)
        model = mz.efficientnet_b0()

    ###PyTorch Models###
    elif args.model == "mobilenet_v2":
        model = mz.mobilenet_v2()
    elif args.model == "mobilenet_v3":
        model = mz.mobilenet_v3()
    elif args.model == "resnet34":
        model = mz.resnet34()
    elif args.model == "resnet50":
        model = mz.resnet50()
    elif args.model == "resnet101":
        model = mz.resnet101()
    elif args.model == "vit_b_16":
        model = mz.vit_b_16()
    elif args.model == "mae_vit_b_16":
        if not args.mae_checkpoint:
            parser.error("--mae_checkpoint is required when using --model mae_vit_b_16")
        # The 'pretrained' flag for ModelZoo is not directly used by mae_vit_b_16,
        # as it loads weights from the checkpoint_path.
        # However, ModelZoo still needs to be initialized.
        # We can pass False for pretrained here, or adjust ModelZoo if needed.
        # For simplicity, let's assume ModelZoo's pretrained flag is for its other models.
        model = mz.mae_vit_b_16(checkpoint_path=args.mae_checkpoint)
    elif args.model == "efficientformer":
        model = mz.efficientformer()
    elif args.model == "efficientnet_b7":
        model = mz.efficientnet_b7()
    elif args.model == "efficientnet_b4":
        model = mz.efficientnet_b4()
    elif args.model == "segformer_b2":
        # model = mz.segformer_b2()
        # model = mz.mmseg_model()
        # model = mz.lraspp_mobilenet_v3_large()
        model = mz.segformer()
    elif args.model == "resnet_3d":
        model = mz.resnet18_3d()

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if args.pretrained:
        args.model = args.model + "_" + "pretrained" + "_" + str(args.threshold)
    else:
        args.model = args.model + "_" + str(args.threshold)

    if args.mode == "baseline":
        args.model = args.model + "_" + "baseline"

    if args.long_tail and args.ldam:
        if args.mode == "baseline":
            trained_model = train_baseline_longtail(args.model, model, train_loader, test_loader, device, args.epoch, args.save_path, cls_num_list)
        elif args.mode == "train_with_revision":
            trained_model = train_with_revision_longtail(args.model, model, train_loader, test_loader, device, args.epoch, args.save_path, args.threshold, args.start_revision, args.task, cls_num_list)

    else: 
        if args.mode == "baseline":
            print("Training in baseline mode...")
            trained_model = train_baseline(args.model, model, train_loader, test_loader, device, args.epoch, args.save_path)
        elif args.mode == "selective_gradient":
            train_revision = TrainRevision(args.model, model, train_loader, test_loader, device, args.epoch, args.save_path, args.threshold)
            print("Training with selective gradient updates...")
            trained_model = train_revision.train_selective()
        elif args.mode == "selective_epoch":
            train_revision = TrainRevision(args.model, model, train_loader, test_loader, device, args.epoch, args.save_path, args.threshold)
            print(f"Reintroducing correct examples and training...")
            trained_model = train_revision.train_selective_epoch()
        elif args.mode == "train_with_revision":
            train_revision = TrainRevision(args.model, model, train_loader, test_loader, device, args.epoch, args.save_path, args.threshold)
            print(f"Training {args.mode}, will start revision after {args.start_revision}")
            trained_model, num_step = train_revision.train_with_revision(args.start_revision, args.task)
            print("Number of steps : ", num_step)
        elif args.mode == "train_with_random":
            train_revision = TrainRevision(args.model, model, train_loader, test_loader, device, args.epoch, args.save_path, args.threshold)
            print(f"Training {args.mode}, will start revision after {args.start_revision}")
            trained_model, num_step = train_revision.train_with_random(args.start_revision, args.task)
            print("Number of steps : ", num_step)
        elif args.mode == "train_with_revision_3d":
            train_revision = TrainRevision(args.model, model, train_loader, test_loader, device, args.epoch, args.save_path, args.threshold)
            print(f"Training {args.mode}, will start revision after {args.start_revision}")
            trained_model, num_step = train_revision.train_with_revision_3d(args.start_revision, args.task)
            print("Number of steps : ", num_step)
        elif args.mode == "train_with_percentage":
            train_revision = TrainRevision(args.model, model, train_loader, test_loader, device, args.epoch, args.save_path, args.threshold)
            print(f"Training {args.mode}, will start revision after {args.start_revision}")
            trained_model, num_step = train_revision.train_with_percentage(args.start_revision, args.task)
            print("Number of steps : ", num_step)
        elif args.mode == "train_with_inv_lin":
            train_revision = TrainRevision(args.model, model, train_loader, test_loader, device, args.epoch, args.save_path, args.threshold)
            print(f"Training {args.mode}, will start revision after {args.start_revision}")
            trained_model, num_step = train_revision.train_with_inverse_linear(args.start_revision, data_size)
            print("Number of steps : ", num_step)
        elif args.mode == "train_with_log":
            train_revision = TrainRevision(args.model, model, train_loader, test_loader, device, args.epoch, args.save_path, args.threshold)
            print(f"Training {args.mode}, will start revision after {args.start_revision}")
            trained_model, num_step = train_revision.train_with_log(args.start_revision, data_size)
            print("Number of steps : ", num_step)
    
    eff_epoch = int(num_step/data_size)

    print("Effective Epochs: ", eff_epoch)
    torch.save(trained_model, "trained_model.pth")
    
if __name__ == "__main__":
    main()
