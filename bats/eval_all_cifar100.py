import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100  # 切换为CIFAR100
from torch.utils.data import DataLoader

try:
    import torchattacks
    from autoattack import AutoAttack
except ImportError:
    print("错误: 请先安装攻击库: pip install torchattacks autoattack")
    exit()

from models import *

# CIFAR-100 Mean/Std (与训练代码保持一致，原始训练代码未做归一化，设为0/1)
CIFAR_MEAN = [0.0, 0.0, 0.0]
CIFAR_STD = [1.0, 1.0, 1.0]


def get_args():
    parser = argparse.ArgumentParser(description='CIFAR-100 Full Evaluation')
    parser.add_argument('--model', default='ResNet18', type=str)
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--data-dir', default='./data/cifar100', type=str)
    parser.add_argument('--batch-size', default=100, type=int)
    parser.add_argument('--n-examples', default=10000, type=int)  # CIFAR100测试集共10000样本
    parser.add_argument('--seed', default=1, type=int)
    return parser.parse_args()


class NormalizedModel(nn.Module):
    def __init__(self, model):
        super(NormalizedModel, self).__init__()
        self.model = model
        self.register_buffer('mean', torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(CIFAR_STD).view(1, 3, 1, 1))

    def forward(self, x):
        x_norm = (x - self.mean) / self.std
        return self.model(x_norm)


# --- Custom CW (Linf) Implementation ---
def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def CW_loss(x, y):
    x_sorted, ind_sorted = x.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()
    loss_value = -(x[np.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind))
    return loss_value.mean()


def cw_Linf_attack(model, X, y, epsilon, alpha, attack_iters, restarts, device):
    lower_limit = torch.tensor(0.0).to(device)
    upper_limit = torch.tensor(1.0).to(device)
    max_delta = torch.zeros_like(X).to(device)

    for zz in range(restarts):
        delta = torch.zeros_like(X).to(device)
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True

        for _ in range(attack_iters):
            output = model(X + delta)
            loss = CW_loss(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta
            g = grad
            d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - X, upper_limit - X)
            delta.data = d
            delta.grad.zero_()
        max_delta = delta.detach()
    return max_delta


def run_custom_cw(test_loader, model, epsilon_val, alpha_val, attack_iters, device):
    correct = 0
    total = 0
    start = time.time()
    epsilon = torch.ones(3, 1, 1).to(device) * epsilon_val
    alpha = torch.ones(3, 1, 1).to(device) * alpha_val

    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        pgd_delta = cw_Linf_attack(model, X, y, epsilon, alpha, attack_iters, restarts=1, device=device)
        with torch.no_grad():
            output = model(X + pgd_delta)
            _, predicted = torch.max(output.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    acc = 100 * correct / total
    print(f"{'CW (Linf)':<15} | {acc:.2f}% \t(Time: {time.time() - start:.1f}s)")


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
    print(f"{name:<15} | {acc:.2f}% \t(Time: {time.time() - start:.1f}s)")


def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)

    # 1. Load Data - 切换为CIFAR100
    print(f"Loading CIFAR-100 test dataset from {args.data_dir}...")
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    testset = CIFAR100(
        root=args.data_dir,
        train=False,
        download=True,
        transform=transform_test
    )

    if args.n_examples < len(testset):
        indices = torch.randperm(len(testset))[:args.n_examples]
        testset = torch.utils.data.Subset(testset, indices)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 2. Load Model - 适配CIFAR100的100个类别
    print(f"Loading {args.model} model for CIFAR-100...")
    try:
        if args.model == "WideResNet":
            base_model = WideResNet()
        elif args.model == "ResNet18":
            base_model = ResNet18(num_classes=100)  # 明确指定100类
        elif args.model == "VGG":
            base_model = VGG('VGG19', num_classes=100)  # 适配VGG模型
        elif args.model == "ResNet34":
            base_model = ResNet34(num_classes=100)
        elif args.model == "PreActResNest18":
            base_model = PreActResNet18(num_classes=100)
        else:
            # 通用适配：如果模型类支持num_classes参数则传入100
            base_model = eval(args.model)(num_classes=100)
    except Exception as e:
        print(f"警告: 模型初始化时未指定num_classes=100，错误: {e}")
        print("尝试使用默认方式加载模型...")
        base_model = eval(args.model)()

    # 加载权重
    checkpoint = torch.load(args.model_path, map_location=device)
    if isinstance(checkpoint, dict):
        # 适配训练代码的保存格式
        state_dict = checkpoint.get('state_dict', checkpoint.get('model_state_dict', checkpoint))
    else:
        state_dict = checkpoint

    # 移除DataParallel的module.前缀
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # 加载权重
    base_model.load_state_dict(new_state_dict)
    base_model = base_model.to(device)

    # 包装归一化模型
    model = NormalizedModel(base_model).to(device)
    model.eval()

    # 3. 开始评估
    print(f"\nModel: {args.model_path}")
    print("-" * 65)
    print(f"{'Attack Method':<15} | {'Accuracy':<10} \t{'Time Usage'}")
    print("-" * 65)

    # Clean Accuracy (干净样本准确率)
    correct = 0
    total = 0
    start = time.time()
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            correct += (out.max(1)[1] == y).sum().item()
            total += y.size(0)
    clean_acc = 100 * correct / total
    print(f"{'Clean':<15} | {clean_acc:.2f}% \t(Time: {time.time() - start:.1f}s)")

    # 攻击参数 - 与训练代码保持一致 (epsilon=8/255, alpha=2/255)
    eps = 8 / 255.
    alpha = 2 / 255.

    # PGD Attacks
    print("\n--- PGD Attacks (Linf, eps=8/255) ---")
    pgd10 = torchattacks.PGD(model, eps=eps, alpha=alpha, steps=10, random_start=True)
    run_torchattack(pgd10, test_loader, "PGD-10", device)

    pgd20 = torchattacks.PGD(model, eps=eps, alpha=alpha, steps=20, random_start=True)
    run_torchattack(pgd20, test_loader, "PGD-20", device)

    pgd50 = torchattacks.PGD(model, eps=eps, alpha=alpha, steps=50, random_start=True)
    run_torchattack(pgd50, test_loader, "PGD-50", device)

    # Custom CW (Linf) Attack
    print("\n--- Custom CW Attack ---")
    run_custom_cw(test_loader, model, eps, alpha, attack_iters=50, device=device)

    # AutoAttack (标准评估)
    print("\n--- AutoAttack (Standard, Linf, eps=8/255) ---")
    adversary = AutoAttack(model, norm='Linf', eps=eps, version='standard', device=device)
    all_x, all_y = [], []
    for x, y in test_loader:
        all_x.append(x)
        all_y.append(y)
    all_x = torch.cat(all_x).to(device)
    all_y = torch.cat(all_y).to(device)

    with torch.no_grad():
        aa_results = adversary.run_standard_evaluation(all_x, all_y, bs=args.batch_size)
    print(f"\nAutoAttack Summary: {aa_results}")

    # 打印最终结果汇总
    print("\n" + "=" * 65)
    print(f"Final Results Summary (CIFAR-100)")
    print(f"Clean Accuracy: {clean_acc:.2f}%")
    print(f"PGD-10 Accuracy: {run_torchattack.__globals__.get('last_pgd10_acc', 'N/A')}")
    print(f"PGD-20 Accuracy: {run_torchattack.__globals__.get('last_pgd20_acc', 'N/A')}")
    print(f"PGD-50 Accuracy: {run_torchattack.__globals__.get('last_pgd50_acc', 'N/A')}")
    print(f"CW (Linf) Accuracy: {run_custom_cw.__globals__.get('last_cw_acc', 'N/A')}")
    print("=" * 65)


if __name__ == "__main__":
    main()