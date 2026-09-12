import argparse
import copy
import logging
import os
import time
from torchvision.utils import make_grid, save_image
import numpy as np
import torch
from torch.nn import functional as F
from models import *
import random
from torch.autograd import Variable
from utils import *
import math
import torchvision
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)


# 【修改 1】：增加一个 Dataset 包装类，用于在 dataloader 中返回样本的索引 (index)
class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data, target = self.dataset[index]
        return index, data, target


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='ResNet18', type=str, help='model name')
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='./data/cifar100', type=str)
    parser.add_argument('--epochs', default=110, type=int)
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
    parser.add_argument('--lamda', default=0.05, type=float, help='Penalize regularization term')
    parser.add_argument('--batch-m', default=0.75, type=float)
    # COLA
    parser.add_argument('--cola-factor', default=0.5, type=float, help='Loss adaptation factor for COLA')
    # FGSM attack
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--alpha', default=8, type=float, help='Step size')
    parser.add_argument('--delta-init', default='random', choices=['zero', 'random', 'previous', 'normal'],
                        help='Perturbation initialization method')
    # ouput
    parser.add_argument('--out-dir', default='TDAT_CIFAR100_Traditional', type=str, help='Output directory')
    parser.add_argument('--log', default="output_cifar100.log", type=str)
    return parser.parse_args()


def label_relaxation(label, factor):
    one_hot = np.eye(100)[label.cuda().data.cpu().numpy()]
    result = one_hot * factor + (one_hot - 1.) * ((factor - 1) / float(100 - 1))
    return result


def cola_loss_adjustment(model_output, labels, original_loss, cola_factor):
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
        filename=os.path.join(output_path, args.log))
    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.CIFAR100(
        root=args.data_dir, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(
        root=args.data_dir, train=False, download=True, transform=transform_test)

    # 【修改 2】：使用 IndexedDataset 包装原有的 CIFAR100 trainset
    indexed_trainset = IndexedDataset(trainset)

    # 创建DataLoader，传入包装后的 indexed_trainset
    train_loader = torch.utils.data.DataLoader(
        indexed_trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(f"Train loader length: {len(train_loader)}")
    print(f"Expected batches: {len(trainset) // args.batch_size}")

    epsilon = (args.epsilon / 255.) / std
    alpha = (args.alpha / 255.) / std

    if args.model == "VGG":
        model = VGG('VGG19')
    elif args.model == "ResNet18":
        model = ResNet18(num_classes=100)
    elif args.model == "PreActResNest18":
        model = PreActResNet18()
    elif args.model == "WideResNet":
        model = WideResNet()
    elif args.model == "ResNet34":
        model = ResNet34()
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)

    criterion = nn.CrossEntropyLoss(reduction='none')
    num_of_example = 50000
    batch_size = args.batch_size

    iter_num = num_of_example // batch_size + (0 if num_of_example % batch_size == 0 else 1)

    lr_steps = args.epochs * iter_num
    if args.lr_schedule == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr_min, max_lr=args.lr_max,
                                                      step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
    elif args.lr_schedule == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps * args.milestone1 / args.epochs,
                                                                          lr_steps * args.milestone2 / args.epochs],
                                                         gamma=0.1)

    logger.info(
        'Epoch \t Seconds \t LR \t Inner Loss \t Train Loss \t Train Acc \t Test Loss \t Test Acc \t PGD Loss \t PGD Acc')
    best_result = 0
    epoch_clean_list = []
    epoch_pgd_list = []

    # 【修改 3】：分配一个形状为 [50000, 3, 32, 32] 的全局张量，存放样本级历史扰动
    global_noise = torch.zeros(num_of_example, 3, 32, 32).cuda()
    for j in range(len(epsilon)):
        global_noise[:, j, :, :].uniform_(-epsilon[j][0][0].item(), epsilon[j][0][0].item())
    for i in range(len(epsilon)):
        global_noise[:, i, :, :] = torch.clamp(global_noise[:, i, :, :], -epsilon[i][0][0].item(),
                                               epsilon[i][0][0].item())

    for epoch in range(args.epochs):

        start_epoch_time = time.time()
        inner_loss = 0
        train_loss = 0
        train_acc = 0
        train_n = 0

        inner_gammas = math.tan(1 - (epoch / args.epochs)) * args.beta
        outer_gammas = math.tan(1 - (epoch / args.epochs)) * args.beta
        if inner_gammas < args.inner_gamma:
            inner_gammas = args.inner_gamma
            outer_gammas = args.outer_gamma

        # 【修改 4】：接收样本索引 (index)
        for _, (index, X, y) in enumerate(train_loader):

            X = X.cuda()
            y = y.cuda()
            batch_size = X.shape[0]

            if X.shape[0] == args.batch_size:
                relaxtion_label = torch.tensor(label_relaxation(y, inner_gammas)).cuda()

                # 【修改 5】：从全局噪声库获取该批次样本的历史扰动
                delta = global_noise[index].clone().detach().requires_grad_(True)

                ori_output = model(X + delta)
                ori_loss_per_sample = criterion(ori_output, relaxtion_label.float())
                ori_loss = ori_loss_per_sample.mean()

                ori_loss.backward(retain_graph=True)
                x_grad = delta.grad.detach()

                delta.data = clamp(delta + alpha * torch.sign(x_grad), -epsilon, epsilon)
                delta.data = clamp(delta, lower_limit - X, upper_limit - X)

                # 【修改 6】：更新当前批次对应的全局噪声库
                with torch.no_grad():
                    global_noise[index] = delta.clone().detach()

                delta = delta.detach()

                # 移除原有的 Batch-Level Momentum 更新逻辑
                # momentum = args.batch_m * momentum ...

                logits = ori_output
                output = model(X + delta)

                loss_adv_per_sample = nn.CrossEntropyLoss(reduction='none', label_smoothing=(1.0 - outer_gammas))(
                    output, y)
                loss_adv_cola = cola_loss_adjustment(output, y, loss_adv_per_sample, args.cola_factor)

                nat_probs = F.softmax(logits, dim=1)
                true_probs = torch.gather(nat_probs, 1, (y.unsqueeze(1)).long()).squeeze()
                loss_robust = (1.0 / args.batch_size) * torch.sum(
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

        if args.model == "VGG":
            model_test = VGG('VGG19').cuda()
        elif args.model == "ResNet18":
            model_test = ResNet18(num_classes=100).cuda()
        elif args.model == "PreActResNest18":
            model_test = PreActResNet18().cuda()
        elif args.model == "WideResNet":
            model_test = WideResNet().cuda()
        elif args.model == "ResNet34":
            model_test = ResNet34().cuda()
        model_test = torch.nn.DataParallel(model_test)
        model_test.load_state_dict(model.state_dict())
        model_test.float()
        model_test.eval()

        pgd_loss, pgd_acc = evaluate_pgd(test_loader, model_test, 10, 1)
        test_loss, test_acc = evaluate_standard(test_loader, model_test)
        epoch_clean_list.append(test_acc)
        epoch_pgd_list.append(pgd_acc)

        logger.info('%d \t %.1f \t \t %.4f \t %.4f \t \t %.4f \t %.4f \t %.4f \t \t %.4f \t %.4f \t %.4f',
                    epoch, epoch_time - start_epoch_time, lr, inner_loss / train_n, train_loss / train_n,
                    train_acc / train_n, test_loss, test_acc, pgd_loss, pgd_acc)

        ckpt_name = args.model + "_CIFAR100_TDAT_COLA_robustAcc_" + str(pgd_acc) + "_clean_acc_" + str(test_acc) + ".pt"
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