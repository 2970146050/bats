import argparse
import os
import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader

try:
    import torchattacks
    from autoattack import AutoAttack
except ImportError:
    print("错误: 请先安装攻击库: pip install torchattacks autoattack")
    exit()

from models import *

# 与 utils.py 保持严格对齐：不使用标准归一化，保持 [0, 1] 像素空间
CIFAR_MEAN = [0.0, 0.0, 0.0]
CIFAR_STD = [1.0, 1.0, 1.0]


def get_args():
    parser = argparse.ArgumentParser(description='CIFAR-100 Robustness Evaluation for Paper')
    parser.add_argument('--model', default='ResNet18', type=str, help='Model architecture')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the saved model checkpoint')
    parser.add_argument('--data-dir', default='./data/cifar100', type=str, help='Dataset directory')
    parser.add_argument('--batch-size', default=100, type=int)
    parser.add_argument('--n-examples', default=10000, type=int, help='CIFAR100 测试集共10000样本')
    parser.add_argument('--seed', default=1, type=int)
    return parser.parse_args()


class NormalizedModel(nn.Module):
    """
    包装模型：确保评估时的输入预处理与训练时 (utils.py) 完全一致。
    """

    def __init__(self, model):
        super(NormalizedModel, self).__init__()
        self.model = model
        self.register_buffer('mean', torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(CIFAR_STD).view(1, 3, 1, 1))

    def forward(self, x):
        x_norm = (x - self.mean) / self.std
        return self.model(x_norm)


def run_torchattack(attacker, loader, name, device):
    correct = 0
    total = 0
    start = time.time()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        adv_images = attacker(images, labels)
        with torch.no_grad():
            outputs = attacker.model(adv_images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"{name:<15} | {acc:>6.2f}% \t| {time.time() - start:.1f}s")
    return acc


def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)

    # 1. 准备数据 - CIFAR100
    print(f"Loading CIFAR-100 test dataset from {args.data_dir}...")
    testset = CIFAR100(root=args.data_dir, train=False, download=True, transform=transforms.ToTensor())
    if args.n_examples < len(testset):
        indices = torch.randperm(len(testset))[:args.n_examples]
        testset = torch.utils.data.Subset(testset, indices)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 2. 加载模型架构 - 严格指定 100 类
    print(f"Loading {args.model} model for CIFAR-100...")
    try:
        if args.model == "WideResNet":
            base_model = WideResNet(num_classes=100)
        else:
            base_model = eval(args.model)(num_classes=100)
    except Exception as e:
        print(f"警告: 模型初始化失败或不支持 num_classes=100，尝试默认加载。错误: {e}")
        base_model = eval(args.model)()

    # 3. 加载权重 (兼容 DataParallel)
    checkpoint = torch.load(args.model_path, map_location=device)
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('state_dict', checkpoint.get('model_state_dict', checkpoint))
    else:
        state_dict = checkpoint

    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    base_model.load_state_dict(new_state_dict)

    # 使用 NormalizedModel 包装
    model = NormalizedModel(base_model).to(device)
    model.eval()

    print(f"\nEvaluating Model: {os.path.basename(args.model_path)}")
    print("=" * 55)
    print(f"{'Attack Method':<15} | {'Robust Acc':<9} | {'Time Usage'}")
    print("-" * 55)

    # 标准对抗扰动设置
    eps = 8 / 255.
    alpha = 2 / 255.

    # 字典用于最终美观地汇总打印结果
    results_summary = {}

    # 1. 干净样本准确率 (Clean)
    correct = 0
    total = 0
    start_time = time.time()
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    clean_acc = 100 * correct / total
    print(f"{'Clean':<15} | {clean_acc:>6.2f}% \t| {time.time() - start_time:.1f}s")
    results_summary['Clean'] = clean_acc

    # # 2. FGSM 评估 (检测灾难性过拟合)
    # fgsm = torchattacks.FGSM(model, eps=eps)
    # results_summary['FGSM'] = run_torchattack(fgsm, test_loader, "FGSM", device)
    #
    # # 3. 标准多步 PGD 攻击
    # pgd20 = torchattacks.PGD(model, eps=eps, alpha=alpha, steps=20, random_start=True)
    # results_summary['PGD-20'] = run_torchattack(pgd20, test_loader, "PGD-20", device)
    #
    # pgd50 = torchattacks.PGD(model, eps=eps, alpha=alpha, steps=50, random_start=True)
    # results_summary['PGD-50'] = run_torchattack(pgd50, test_loader, "PGD-50", device)

    # 4. C&W (使用目前学术界最强的 APGD-CW 替代)
    # 4. APGD-DLR (最强 Margin 损失攻击，作为 C&W 的升级替代)
    apgd_dlr = torchattacks.APGD(model, eps=eps, norm='Linf', loss='dlr', steps=100, n_restarts=1)
    results_summary['APGD-DLR'] = run_torchattack(apgd_dlr, test_loader, "APGD-DLR", device)

    print("-" * 55)

    # 5. AutoAttack (黄金标准)
    print("\nRunning AutoAttack (Standard Evaluation)...")
    adversary = AutoAttack(model, norm='Linf', eps=eps, version='standard', device=device)

    all_x, all_y = [], []
    for x, y in test_loader:
        all_x.append(x)
        all_y.append(y)
    all_x = torch.cat(all_x).to(device)
    all_y = torch.cat(all_y).to(device)

    with torch.no_grad():
        aa_dict = adversary.run_standard_evaluation(all_x, all_y, bs=args.batch_size, return_labels=False)

    # 打印最终结果汇总
    print("\n" + "=" * 55)
    print("Final Results Summary (CIFAR-100)")
    print("-" * 55)
    for atk, acc in results_summary.items():
        print(f"{atk:<15} : {acc:.2f}%")
    print("=" * 55)


if __name__ == "__main__":
    main()