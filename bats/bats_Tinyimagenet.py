import argparse
import copy
import logging
import os
import time
from torchvision.utils import make_grid, save_image
import numpy as np
import torch
from torch.nn import functional as F
# 确保导入的是修改后的 resnet
from models import *
import random
from torch.autograd import Variable
import math
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder  # 需要使用 ImageFolder

logger = logging.getLogger(__name__)

# --- Tiny ImageNet 均值和方差 ---
TINY_MEAN = [0.4802, 0.4481, 0.3975]
TINY_STD = [0.2302, 0.2265, 0.2262]


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


# 辅助函数：获取数据归一化后的上下界
def get_limit(device):
    mean = torch.tensor(TINY_MEAN).view(3, 1, 1).to(device)
    std = torch.tensor(TINY_STD).view(3, 1, 1).to(device)
    lower_limit = (0.0 - mean) / std
    upper_limit = (1.0 - mean) / std
    return lower_limit, upper_limit


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='ResNet18', type=str, help='model name')
    parser.add_argument('--batch-size', default=128, type=int)  # 64x64 图片显存占用更大，稍微调小 batch
    parser.add_argument('--data-dir', default='./data/tiny-imagenet-200/tiny-imagenet-200', type=str)
    parser.add_argument('--epochs', default=120, type=int)
    parser.add_argument('--lr_schedule', default='multistep', choices=['cyclic', 'multistep'])
    parser.add_argument('--milestone1', default=100, type=int)
    parser.add_argument('--milestone2', default=105, type=int)
    parser.add_argument('--lr-min', default=0., type=float)
    parser.add_argument('--lr-max', default=0.1, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument("--save-epoch", default=101, type=int)
    # TDAT
    parser.add_argument('--inner-gamma', default=0.05, type=float, help='Label relaxation factor')
    parser.add_argument('--outer-gamma', default=0.05, type=float)
    parser.add_argument('--beta', default=0.6)
    parser.add_argument('--lamda', default=0.05, type=float, help='Penalize regularization term')
    parser.add_argument('--batch-m', default=0.75, type=float)
    # COLA
    parser.add_argument('--cola-factor', default=0.5, type=float, help='Loss adaptation factor for COLA')
    # FGSM attack
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--alpha', default=8, type=float, help='Step size')  # 步长通常设为 epsilon/4 或 2
    parser.add_argument('--delta-init', default='random', choices=['zero', 'random', 'previous', 'normal'],
                        help='Perturbation initialization method')
    # ouput
    parser.add_argument('--out-dir', default='TDAT_TinyImageNet', type=str, help='Output directory')
    parser.add_argument('--log', default="output_tiny.log", type=str)
    return parser.parse_args()


def label_relaxation(label, factor):
    # Tiny ImageNet 有 200 类
    one_hot = np.eye(200)[label.cuda().data.cpu().numpy()]
    result = one_hot * factor + (one_hot - 1.) * ((factor - 1) / float(200 - 1))
    return result


def cola_loss_adjustment(model_output, labels, original_loss, cola_factor):
    with torch.no_grad():
        pred = torch.argmax(model_output, dim=1)
        correct_mask = (pred == labels)
        loss_selector = torch.ones_like(original_loss)
        loss_selector[correct_mask] = cola_factor
    adjusted_loss = original_loss * loss_selector
    return adjusted_loss.mean()


# --- 新增：评估函数 ---
def evaluate_standard(test_loader, model):
    """计算标准测试集准确率 (Clean Accuracy)"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    loss_avg = test_loss / (batch_idx + 1)
    return loss_avg, acc


def evaluate_pgd(test_loader, model, epsilon, alpha, num_steps=10):
    """计算 PGD-10 攻击下的鲁棒准确率 (Robust Accuracy)"""
    model.eval()
    pgd_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    # 获取归一化参数用于截断
    lower_limit, upper_limit = get_limit('cuda')

    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        X, y = inputs, targets

        # PGD 攻击初始化
        delta = torch.zeros_like(X).cuda()
        delta.uniform_(-epsilon[0][0][0].item(), epsilon[0][0][0].item())  # 简单初始化
        delta.requires_grad = True

        # PGD 迭代
        for _ in range(num_steps):
            output = model(X + delta)
            loss = criterion(output, y)
            loss.backward()
            grad = delta.grad.detach()

            delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
            delta.data = clamp(delta, lower_limit - X, upper_limit - X)
            delta.grad.zero_()

        # 攻击后的评估
        with torch.no_grad():
            delta = delta.detach()
            outputs = model(X + delta)
            loss = criterion(outputs, y)

            pgd_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

    acc = 100. * correct / total
    loss_avg = pgd_loss / (batch_idx + 1)
    return loss_avg, acc


def main():
    args = get_args()

    output_path = args.out_dir
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 配置 Logging，直接输出到控制台和文件
    logging.basicConfig(
        format='%(message)s',  # 简化格式，手动控制对齐
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(output_path, args.log)),
            logging.StreamHandler()
        ]
    )
    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # ... (数据加载 transform_train, transform_test 代码保持不变) ...
    # ... (ImageFolder 加载 trainset, testset 代码保持不变) ...
    # ... (DataLoader 定义代码保持不变) ...

    # --- 确保这一段完整 ---
    transform_train = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(TINY_MEAN, TINY_STD),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(TINY_MEAN, TINY_STD),
    ])

    trainset = ImageFolder(root=os.path.join(args.data_dir, 'train'), transform=transform_train)
    testset = ImageFolder(root=os.path.join(args.data_dir, 'val'), transform=transform_test)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    # ----------------------

    # 准备 Attack 参数 (用于 PGD 评估)
    std_tensor = torch.tensor(TINY_STD).view(3, 1, 1).cuda()
    epsilon = (args.epsilon / 255.) / std_tensor
    alpha = (args.alpha / 255.) / std_tensor
    lower_limit, upper_limit = get_limit('cuda')

    # 模型初始化
    if args.model == "ResNet18":
        model = ResNet18(num_classes=200)
    elif args.model == "ResNet34":
        model = ResNet34(num_classes=200)

    model = torch.nn.DataParallel(model).cuda()
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(reduction='none')

    # 学习率调度器
    num_of_example = len(trainset)
    iter_num = len(train_loader)
    lr_steps = args.epochs * iter_num
    if args.lr_schedule == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr_min, max_lr=args.lr_max,
                                                      step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
    elif args.lr_schedule == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps * args.milestone1 / args.epochs,
                                                                          lr_steps * args.milestone2 / args.epochs],
                                                         gamma=0.1)

    # --- 打印表头 (Header) ---
    logger.info(
        f"{'Epoch':<6} {'Seconds':<8} {'LR':<8} {'Inner Loss':<12} {'Train Loss':<12} {'Train Acc':<10} {'Test Loss':<10} {'Test Acc':<10} {'PGD Loss':<10} {'PGD Acc':<10}")

    momentum = torch.zeros(args.batch_size, 3, 64, 64).cuda()
    for j in range(3):
        momentum[:, j, :, :].uniform_(-epsilon[j][0][0].item(), epsilon[j][0][0].item())
    momentum = clamp(alpha * torch.sign(momentum), -epsilon, epsilon)

    for epoch in range(args.epochs):
        start_epoch_time = time.time()
        inner_loss_sum = 0
        train_loss_sum = 0
        train_acc_sum = 0
        train_n = 0

        # Update dynamic parameters
        inner_gammas = math.tan(1 - (epoch / args.epochs)) * args.beta
        outer_gammas = math.tan(1 - (epoch / args.epochs)) * args.beta
        if inner_gammas < args.inner_gamma:
            inner_gammas = args.inner_gamma
            outer_gammas = args.outer_gamma

        model.train()
        for i, (X, y) in enumerate(train_loader):
            X, y = X.cuda(), y.cuda()
            curr_batch_size = X.shape[0]

            # 处理 Batch Size 不一致时的 Momentum Buffer
            if curr_batch_size != momentum.shape[0]:
                curr_momentum = torch.zeros(curr_batch_size, 3, 64, 64).cuda()
                for j in range(3):
                    curr_momentum[:, j, :, :].uniform_(-epsilon[j][0][0].item(), epsilon[j][0][0].item())
                delta = clamp(alpha * torch.sign(curr_momentum), -epsilon, epsilon)
            else:
                delta = momentum

            delta.requires_grad = True

            # 1. 计算 Inner Loss (Original Loss)
            relaxtion_label = torch.tensor(label_relaxation(y, inner_gammas)).cuda()
            ori_output = model(X + delta)
            ori_loss = criterion(ori_output, relaxtion_label.float()).mean()
            ori_loss.backward(retain_graph=True)

            # 2. 更新 Delta
            x_grad = delta.grad.detach()
            delta.data = clamp(delta + alpha * torch.sign(x_grad), -epsilon, epsilon)
            delta.data = clamp(delta, lower_limit - X, upper_limit - X)
            delta = delta.detach()

            # 3. 更新 Momentum Buffer (仅当是完整batch时)
            if curr_batch_size == args.batch_size:
                momentum = args.batch_m * momentum + (1.0 - args.batch_m) * delta
                momentum = clamp(momentum, -epsilon, epsilon)
                momentum = clamp(delta, lower_limit - X, upper_limit - X)

            # 4. 计算 Outer Loss (Train Loss)
            logits = ori_output
            output = model(X + delta)

            loss_adv_per_sample = nn.CrossEntropyLoss(reduction='none', label_smoothing=(1.0 - outer_gammas))(output, y)
            loss_adv_cola = cola_loss_adjustment(output, y, loss_adv_per_sample, args.cola_factor)

            nat_probs = F.softmax(logits, dim=1)
            true_probs = torch.gather(nat_probs, 1, (y.unsqueeze(1)).long()).squeeze()
            loss_robust = (1.0 / curr_batch_size) * torch.sum(
                torch.sum(torch.square(torch.sub(logits, output)), dim=1) * torch.tanh(1.0000001 - true_probs))

            loss = loss_adv_cola + float(args.lamda) * loss_robust

            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()

            # 统计数据
            inner_loss_sum += ori_loss.item() * y.size(0)
            train_loss_sum += loss.item() * y.size(0)
            train_acc_sum += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)

        # --- Epoch 结束：执行评估 ---
        epoch_time = time.time() - start_epoch_time
        current_lr = scheduler.get_last_lr()[0]

        # 计算 Training 统计
        avg_inner_loss = inner_loss_sum / train_n
        avg_train_loss = train_loss_sum / train_n
        avg_train_acc = (train_acc_sum / train_n) * 100  # 转为百分比

        # 执行 Test Evaluation (Standard)
        test_loss, test_acc = evaluate_standard(test_loader, model)

        # 执行 PGD Evaluation (Robustness)
        # 注意：PGD 比较慢，为了节省时间，可以设置每隔几个 epoch 测一次，或者只测部分 batch
        pgd_loss, pgd_acc = evaluate_pgd(test_loader, model, epsilon, alpha, num_steps=10)

        # --- 格式化打印日志 (对齐列) ---
        logger.info(
            f"{epoch:<6d} {epoch_time:<8.1f} {current_lr:<8.4f} {avg_inner_loss:<12.4f} {avg_train_loss:<12.4f} {avg_train_acc:<10.4f} {test_loss:<10.4f} {test_acc:<10.4f} {pgd_loss:<10.4f} {pgd_acc:<10.4f}")

        # 保存模型
        if epoch >= args.save_epoch or epoch % 10 == 0:
            ckpt_name = f"{args.model}_Tiny_TDAT_ep{epoch}_clean{test_acc:.2f}_rob{pgd_acc:.2f}.pt"
            torch.save(model.state_dict(), os.path.join(args.out_dir, ckpt_name))


if __name__ == "__main__":
    main()