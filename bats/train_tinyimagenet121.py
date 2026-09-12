import argparse
import copy
import logging
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms
from models import *  # 确保 models/resnet.py 是你提供的修改版
import random
from utils import *  # 确保 utils.py 中的 mu/std 是 0/1，或者在此处修正
import math

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='ResNet18', type=str, help='model name')
    parser.add_argument('--batch-size', default=128, type=int)
    # 修改默认路径为 Tiny-ImageNet 路径
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
    parser.add_argument("--save-epoch", default=100, type=int)
    # TDAT
    parser.add_argument('--inner-gamma', default=0.05, type=float, help='Label relaxation factor')
    parser.add_argument('--outer-gamma', default=0.05, type=float)
    parser.add_argument('--beta', default=0.6)
    parser.add_argument('--lamda', default=0.025, type=float, help='Penalize regularization term')
    parser.add_argument('--batch-m', default=0.75, type=float)
    # COLA
    parser.add_argument('--cola-factor', default=0.5, type=float, help='Loss adaptation factor for COLA')
    # FGSM attack
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--alpha', default=8, type=float, help='Step size')
    parser.add_argument('--delta-init', default='random', choices=['zero', 'random', 'previous', 'normal'],
                        help='Perturbation initialization method')
    # ouput
    parser.add_argument('--out-dir', default='TDAT_TinyImageNet_COLA', type=str, help='Output directory')
    parser.add_argument('--log', default="output_tiny_cola.log", type=str)
    return parser.parse_args()


# 修改：增加 num_classes 参数，默认为 200
def label_relaxation(label, factor, num_classes=200):
    one_hot = np.eye(num_classes)[label.cuda().data.cpu().numpy()]
    result = one_hot * factor + (one_hot - 1.) * ((factor - 1) / float(num_classes - 1))
    return result


def cola_loss_adjustment(model_output, labels, original_loss, cola_factor):
    """
    COLA: Catastrophic Overfitting aware Loss Adaptation
    """
    with torch.no_grad():
        pred = torch.argmax(model_output, dim=1)
        correct_mask = (pred == labels)
        loss_selector = torch.ones_like(original_loss)
        loss_selector[correct_mask] = cola_factor

    adjusted_loss = original_loss * loss_selector
    return adjusted_loss.mean()


def main():
    args = get_args()

    output_path = args.out_dir
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    logfile = os.path.join(output_path, args.log)

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=logfile)
    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # === 修改 1：Tiny-ImageNet 数据加载 ===
    # 注意：这里不使用 Normalize，因为 utils.py 里的 std=1, mean=0。
    # 如果 utils.py 变了，这里也要变。
    transform_train = transforms.Compose([
        transforms.RandomCrop(64, padding=8),  # 64x64
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')

    # 检查路径是否存在
    if not os.path.exists(train_dir):
        raise ValueError(f"Train dir not found: {train_dir}. Please ensure structure is ./data/train/class_xxx/")

    trainset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform_train)
    testset = torchvision.datasets.ImageFolder(root=val_dir, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Dataset Loaded. Classes: {len(trainset.classes)}")  # 应为 200

    # 计算 epsilon (依赖 utils.py 中的全局 std，假设为 1)
    epsilon = (args.epsilon / 255.) / std
    alpha = (args.alpha / 255.) / std

    # === 修改 2：模型初始化 ===
    if args.model == "ResNet18":
        model = ResNet18(num_classes=200)  # 200类
    elif args.model == "ResNet34":
        model = ResNet34(num_classes=200)
    # ... 其他模型类似处理
    else:
        # Fallback for generic models
        print(f"Warning: Model {args.model} might default to 10 classes. Check implementation.")
        model = ResNet18(num_classes=200)

    model = torch.nn.DataParallel(model)
    model = model.cuda()
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)

    criterion = nn.CrossEntropyLoss(reduction='none')

    # 计算 iter_num
    num_of_example = len(trainset)
    batch_size = args.batch_size
    iter_num = len(train_loader)

    lr_steps = args.epochs * iter_num
    if args.lr_schedule == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr_min, max_lr=args.lr_max,
                                                      step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
    elif args.lr_schedule == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps * args.milestone1 / args.epochs,
                                                                          lr_steps * args.milestone2 / args.epochs],
                                                         gamma=0.1)

    # Training
    logger.info(
        'Epoch \t Seconds \t LR \t Inner Loss \t Train Loss \t Train Acc \t Test Loss \t Test Acc \t PGD Loss \t PGD Acc')

    epoch_clean_list = []
    epoch_pgd_list = []

    # === 修改 3：Momentum 初始化尺寸改为 64x64 ===
    # momentum batch initialization
    momentum = torch.zeros(batch_size, 3, 64, 64).cuda()
    # 使用 epsilon 的第一个值进行初始化范围设定
    eps_val = epsilon[0, 0, 0].item()
    for j in range(3):
        momentum[:, j, :, :].uniform_(-eps_val, eps_val)

    momentum = clamp(alpha * torch.sign(momentum), -epsilon, epsilon)

    # 这里的 lower_limit, upper_limit 来自 utils.py (基于 mean=0, std=1)
    # 只要数据没做 Normalize，0-1 范围是适用的

    for epoch in range(args.epochs):

        start_epoch_time = time.time()
        inner_loss = 0
        train_loss = 0
        train_acc = 0
        train_n = 0

        # dynamic label relaxtion
        inner_gammas = math.tan(1 - (epoch / args.epochs)) * args.beta
        outer_gammas = math.tan(1 - (epoch / args.epochs)) * args.beta
        if inner_gammas < args.inner_gamma:
            inner_gammas = args.inner_gamma
            outer_gammas = args.outer_gamma

        for _, (X, y) in enumerate(train_loader):

            X = X.cuda()
            y = y.cuda()
            curr_batch_size = X.shape[0]

            # 处理最后一个 batch 大小不一致的问题
            if curr_batch_size != momentum.shape[0]:
                batch_momentum = momentum[:curr_batch_size].clone().detach()
            else:
                batch_momentum = momentum

            # 建议：去掉 "if X.shape[0] == args.batch_size" 限制，让最后一个 batch 也能训练
            # 如果不想改逻辑，保留即可。这里为了兼容性，我做了简单的适配:

            delta = batch_momentum.clone()
            delta.requires_grad = True

            # 传入 200 类
            relaxtion_label = torch.tensor(label_relaxation(y, inner_gammas, num_classes=200)).cuda()

            ori_output = model(X + delta)

            ori_loss_per_sample = criterion(ori_output, relaxtion_label.float())
            ori_loss = ori_loss_per_sample.mean()

            ori_loss.backward(retain_graph=True)
            x_grad = delta.grad.detach()

            delta.data = clamp(delta + alpha * torch.sign(x_grad), -epsilon, epsilon)
            delta.data = clamp(delta, lower_limit - X, upper_limit - X)
            delta = delta.detach()

            # update momentum
            # 注意：如果 curr_batch < args.batch_size，这里更新 global momentum 会有 shape 问题
            # 简单处理：只更新当前 batch 对应的 momentum
            if curr_batch_size == args.batch_size:
                momentum = args.batch_m * momentum + (1.0 - args.batch_m) * delta
                momentum = clamp(momentum, -epsilon, epsilon)
                momentum = clamp(momentum, lower_limit - X, upper_limit - X)

            logits = ori_output
            output = model(X + delta)

            loss_adv_per_sample = nn.CrossEntropyLoss(reduction='none', label_smoothing=(1.0 - outer_gammas))(
                output, y)

            # COLA Adjustment
            loss_adv_cola = cola_loss_adjustment(output, y, loss_adv_per_sample, args.cola_factor)

            nat_probs = F.softmax(logits, dim=1)
            true_probs = torch.gather(nat_probs, 1, (y.unsqueeze(1)).long()).squeeze()
            loss_robust = (1.0 / curr_batch_size) * torch.sum(
                torch.sum(torch.square(torch.sub(logits, output)), dim=1) * torch.tanh(1.0000001 - true_probs))

            loss = loss_adv_cola + float(args.lamda) * loss_robust
            opt.zero_grad()
            loss.backward()

            opt.step()
            inner_loss += ori_loss.item() * y.size(0)
            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)
            scheduler.step()

        epoch_time = time.time()
        lr = scheduler.get_last_lr()[0]

        if args.model == "ResNet18":
            model_test = ResNet18(num_classes=200).cuda()
        elif args.model == "ResNet34":
            model_test = ResNet34(num_classes=200).cuda()
        # ... Add other models
        else:
            model_test = ResNet18(num_classes=200).cuda()

        model_test = torch.nn.DataParallel(model_test)
        model_test.load_state_dict(model.state_dict())
        model_test.float()
        model_test.eval()

        # 调用 utils 中的评估函数
        # 注意：evaluate_pgd 默认使用 epsilon=(8/255)/std。由于我们 std=1，这正是我们需要的。
        pgd_loss, pgd_acc = evaluate_pgd(test_loader, model_test, 10, 1)
        test_loss, test_acc = evaluate_standard(test_loader, model_test)
        epoch_clean_list.append(test_acc)
        epoch_pgd_list.append(pgd_acc)

        if train_n == 0: train_n = 1  # Prevent division by zero

        logger.info('%d \t %.1f \t \t %.4f \t %.4f \t \t %.4f \t %.4f \t %.4f \t \t %.4f \t %.4f \t %.4f',
                    epoch, epoch_time - start_epoch_time, lr, inner_loss / train_n, train_loss / train_n,
                    train_acc / train_n, test_loss, test_acc, pgd_loss, pgd_acc)

        ckpt_name = args.model + "_Tiny_TDAT_COLA_robustAcc_" + str(pgd_acc) + "_clean_acc_" + str(test_acc) + ".pt"
        if epoch >= args.save_epoch:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_test.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'loss': train_loss / train_n
            }, os.path.join(args.out_dir, ckpt_name))

    logger.info(epoch_clean_list)
    logger.info(epoch_pgd_list)


if __name__ == "__main__":
    main()