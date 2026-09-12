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

logger = logging.getLogger(__name__)


# 【修改 1】：增加一个 Dataset 包装类，用于在 dataloader 中返回样本的索引 (index)
# 传统方法必须知道当前是哪个样本，才能去全局字典里取对应的历史扰动
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
    parser.add_argument('--data-dir', default='./data', type=str)
    parser.add_argument('--epochs', default=110, type=int)
    parser.add_argument('--lr_schedule', default='multistep', choices=['cyclic', 'multistep'])
    parser.add_argument('--milestone1', default=100, type=int)
    parser.add_argument('--milestone2', default=105, type=int)
    parser.add_argument('--lr-min', default=0., type=float)
    parser.add_argument('--lr-max', default=0.1, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument("--save-epoch", default=111, type=int)
    # TDAT
    parser.add_argument('--inner-gamma', default=0.15, type=float, help='Label relaxation factor')
    parser.add_argument('--outer-gamma', default=0.15, type=float)
    parser.add_argument('--beta', default=0.6)
    parser.add_argument('--lamda', default=0.65, type=float, help='Penalize regularization term')
    # COLA
    parser.add_argument('--cola-factor', default=0.5, type=float, help='Loss adaptation factor for COLA')
    # FGSM attack
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--alpha', default=8, type=float, help='Step size')
    # ouput
    parser.add_argument('--out-dir', default='TDAT_Traditional', type=str, help='Output directory')
    parser.add_argument('--log', default="output.log", type=str)
    return parser.parse_args()


def label_relaxation(label, factor):
    one_hot = np.eye(10)[label.cuda().data.cpu().numpy()]
    result = one_hot * factor + (one_hot - 1.) * ((factor - 1) / float(10 - 1))
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

    train_loader, test_loader = cifar10_get_loaders(args.data_dir, args.batch_size)

    # 【修改 2】：使用 IndexedDataset 包装原有的 train_dataset
    indexed_train_dataset = IndexedDataset(train_loader.dataset)
    num_workers = train_loader.num_workers if hasattr(train_loader, 'num_workers') else 2
    indexed_train_loader = torch.utils.data.DataLoader(
        indexed_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)

    epsilon = (args.epsilon / 255.) / std
    alpha = (args.alpha / 255.) / std

    if args.model == "VGG":
        model = VGG('VGG19')
    elif args.model == "ResNet18":
        model = ResNet18(num_classes=10)
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

    # 【修改 3】：传统样本级初始化 (Sample-level Initialization)
    # 为整个训练集(50000个样本)分配显存空间来存储每个样本的历史扰动
    global_noise = torch.zeros(num_of_example, 3, 32, 32).cuda()
    for j in range(len(epsilon)):
        global_noise[:, j, :, :].uniform_(-epsilon[j][0][0].item(), epsilon[j][0][0].item())
    # 限制在epsilon范围内
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

        # 【修改 4】：遍历 Dataloader 时接收 index
        for _, (index, X, y) in enumerate(indexed_train_loader):
            X = X.cuda()
            y = y.cuda()
            batch_size = X.shape[0]

            if batch_size == args.batch_size:
                relaxtion_label = torch.tensor(label_relaxation(y, inner_gammas)).cuda()

                # 【修改 5】：根据当前 batch 的样本索引，从全局噪声库中提取对应的历史扰动
                delta = global_noise[index].clone().detach().requires_grad_(True)

                ori_output = model(X + delta)
                ori_loss_per_sample = criterion(ori_output, relaxtion_label.float())
                ori_loss = ori_loss_per_sample.mean()

                ori_loss.backward(retain_graph=True)
                x_grad = delta.grad.detach()

                delta.data = clamp(delta + alpha * torch.sign(x_grad), -epsilon, epsilon)
                delta.data = clamp(delta, lower_limit - X, upper_limit - X)

                # 【修改 6】：更新全局噪声库中对应样本的扰动状态
                with torch.no_grad():
                    global_noise[index] = delta.clone().detach()

                delta = delta.detach()

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
            model_test = ResNet18(num_classes=10).cuda()
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


if __name__ == "__main__":
    main()