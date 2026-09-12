import argparse
import os
import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# 引入必要的攻击库
try:
    import torchattacks
    from autoattack import AutoAttack
except ImportError:
    print("请先安装攻击库: pip install torchattacks autoattack")
    exit()

# 导入你的模型定义
# 假设 models 文件夹在当前目录下
from models import *  # --- Tiny ImageNet 均值和方差 (与训练代码一致) ---

TINY_MEAN = [0.4802, 0.4481, 0.3975]
TINY_STD = [0.2302, 0.2265, 0.2262]


def get_args():
    parser = argparse.ArgumentParser(description='Tiny ImageNet Evaluation Script')
    parser.add_argument('--model', default='ResNet18', type=str, help='Model architecture')
    parser.add_argument('--model-path', type=str, required=True, help='Path to .pt checkpoint')
    parser.add_argument('--data-dir', default='./data/tiny-imagenet-200/tiny-imagenet-200', type=str)
    parser.add_argument('--batch-size', default=100, type=int, help='Batch size for evaluation')
    parser.add_argument('--n-examples', default=10000, type=int,
                        help='Number of examples to evaluate (default all 10k)')
    parser.add_argument('--seed', default=1, type=int)
    return parser.parse_args()


# --- 辅助类：将归一化集成到模型中 ---
# 这样攻击算法生成的样本可以直接在 [0,1] 范围内截断，无需手动处理 Normalize
class NormalizedModel(nn.Module):
    def __init__(self, model):
        super(NormalizedModel, self).__init__()
        self.model = model
        self.register_buffer('mean', torch.tensor(TINY_MEAN).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(TINY_STD).view(1, 3, 1, 1))

    def forward(self, x):
        # x is assumed to be in [0, 1]
        x_norm = (x - self.mean) / self.std
        return self.model(x_norm)


def main():
    args = get_args()

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)

    print(f"Loading data from {args.data_dir}...")

    # 注意：这里只做 ToTensor，不做 Normalize，Normalize 移入模型内部
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    val_dir = os.path.join(args.data_dir, 'val')
    testset = ImageFolder(root=val_dir, transform=transform_test)

    # 如果想为了速度只测部分数据，可以使用 Subset
    if args.n_examples < len(testset):
        indices = torch.randperm(len(testset))[:args.n_examples]
        testset = torch.utils.data.Subset(testset, indices)
        print(f"Subset selected: evaluating on {args.n_examples} images.")

    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # --- 加载模型 ---
    print(f"Loading model: {args.model} from {args.model_path}")
    if args.model == "ResNet18":
        base_model = ResNet18(num_classes=200)
    elif args.model == "ResNet34":
        base_model = ResNet34(num_classes=200)
    else:
        raise ValueError("Unsupported model")

    # 加载权重 (处理 DataParallel 的 module. 前缀)
    checkpoint = torch.load(args.model_path, map_location=device)
    # 如果 checkpoint 保存的是 state_dict
    state_dict = checkpoint if isinstance(checkpoint, dict) else checkpoint.state_dict()

    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "")  # 去掉 module.
        new_state_dict[name] = v

    base_model.load_state_dict(new_state_dict)

    # 包装归一化层
    model = NormalizedModel(base_model).to(device)
    model.eval()

    print("Model loaded successfully. Starting evaluation...")
    print("-" * 60)
    print(f"{'Attack':<15} | {'Accuracy (%)':<10}")
    print("-" * 60)

    # 1. Clean Accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    clean_acc = 100 * correct / total
    print(f"{'Clean':<15} | {clean_acc:.2f}")

    # 定义攻击参数
    epsilon = 8 / 255.
    alpha = 2 / 255.

    # 2. PGD-10
    pgd10 = torchattacks.PGD(model, eps=epsilon, alpha=alpha, steps=10, random_start=True)
    run_torchattack(pgd10, test_loader, "PGD-10", device)

    # 3. PGD-20
    pgd20 = torchattacks.PGD(model, eps=epsilon, alpha=alpha, steps=20, random_start=True)
    run_torchattack(pgd20, test_loader, "PGD-20", device)

    # 4. PGD-50
    pgd50 = torchattacks.PGD(model, eps=epsilon, alpha=alpha, steps=50, random_start=True)
    run_torchattack(pgd50, test_loader, "PGD-50", device)

    # 5. CW (L2)
    # 注意: CW 攻击非常慢，且通常基于 L2 范数。这里使用 torchattacks 的 CW
    # c 是 confidence, steps 是迭代次数。标准配置比较耗时。
    cw_attack = torchattacks.CW(model, c=1, kappa=0, steps=100, lr=0.01)
    run_torchattack(cw_attack, test_loader, "CW (L2)", device)

    # 6. AutoAttack (Standard)
    # 包含 APGD-CE, APGD-DLR, FAB, Square
    print("\nRunning AutoAttack (Standard)... This usually takes the longest.")
    adversary = AutoAttack(model, norm='Linf', eps=epsilon, version='standard', device=device)

    # AutoAttack 需要一次性传入所有数据，或者使用它的 run_standard_evaluation 接口
    # 为了方便，我们手动通过 loader 收集数据（注意显存）
    # 如果显存不足，AutoAttack 会自动分批处理，但需要传入完整的 tensor

    l_acc = []
    # 由于 AutoAttack 比较重，我们按 batch 运行并手动统计，或者直接用它的 loader 接口
    # 这里使用简单的 batch 循环调用（虽然 AutoAttack 内部有优化，但这样兼容性最好）

    aa_correct = 0
    aa_total = 0

    # AutoAttack 本身比较慢，使用库自带的 run_standard_evaluation
    # 需要先从 loader 中提取出所有 x 和 y (如果内存够的话)

    all_x = []
    all_y = []
    for x, y in test_loader:
        all_x.append(x)
        all_y.append(y)

    all_x = torch.cat(all_x)
    all_y = torch.cat(all_y)

    # 运行 AutoAttack
    # bs=args.batch_size 确保显存不溢出
    dict_adv = adversary.run_standard_evaluation(all_x, all_y, bs=args.batch_size)
    # AutoAttack 会自动打印结果


def run_torchattack(attacker, loader, name, device):
    correct = 0
    total = 0
    start = time.time()

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        adv_images = attacker(images, labels)

        with torch.no_grad():
            outputs = model_forward(attacker.model, adv_images)  # attacker.model 是被攻击的模型
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    duration = time.time() - start
    print(f"{name:<15} | {acc:.2f} \t(Time: {duration:.1f}s)")


# 辅助函数，确保调用的是内部模型
def model_forward(model, x):
    return model(x)


if __name__ == "__main__":
    main()
# import os

# import torch
# from torch.utils.data import DataLoader
# from torchattacks import PGD

# from models import ResNet18
# from utils.dataset import TinyImageNet200, test_transform
# from utils.logger import Logger
# from utils.set_seed import set_seed

# def load_model_weights(model, model_path, device):
#     """安全地加载模型权重，处理DataParallel的情况"""
#     checkpoint = torch.load(model_path, map_location=device)

#     # 打印检查点的键以便调试
#     print(f"检查点键: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else '非字典对象'}")

#     if isinstance(checkpoint, dict):
#         # 尝试不同的键名
#         for key in ['model_state_dict', 'state_dict', 'model']:
#             if key in checkpoint:
#                 state_dict = checkpoint[key]
#                 # 处理DataParallel的情况
#                 new_state_dict = {}
#                 for k, v in state_dict.items():
#                     if k.startswith('module.'):
#                         # 去除'module.'前缀
#                         new_state_dict[k[7:]] = v
#                     else:
#                         new_state_dict[k] = v
#                 model.load_state_dict(new_state_dict)
#                 print(f"从键 '{key}' 加载模型权重，已处理DataParallel前缀")
#                 return

#         # 如果没有找到特定键，检查是否直接是状态字典
#         state_dict = checkpoint
#         # 检查是否包含'module.'前缀
#         has_module_prefix = any(k.startswith('module.') for k in state_dict.keys())
#         if has_module_prefix:
#             new_state_dict = {}
#             for k, v in state_dict.items():
#                 if k.startswith('module.'):
#                     new_state_dict[k[7:]] = v
#                 else:
#                     new_state_dict[k] = v
#             model.load_state_dict(new_state_dict)
#             print("从直接状态字典加载模型权重，已处理DataParallel前缀")
#             return
#         else:
#             # 直接加载
#             model.load_state_dict(state_dict)
#             print("从直接状态字典加载模型权重")
#             return
#     else:
#         # 如果不是字典，假设是直接的状态字典
#         state_dict = checkpoint
#         # 检查是否包含'module.'前缀
#         has_module_prefix = any(k.startswith('module.') for k in state_dict.keys())
#         if has_module_prefix:
#             new_state_dict = {}
#             for k, v in state_dict.items():
#                 if k.startswith('module.'):
#                     new_state_dict[k[7:]] = v
#                 else:
#                     new_state_dict[k] = v
#             model.load_state_dict(new_state_dict)
#             print("从直接对象加载模型权重，已处理DataParallel前缀")
#         else:
#             model.load_state_dict(state_dict)
#             print("从直接对象加载模型权重")

#     print(f"成功从 {model_path} 加载模型权重")


# test_set = TinyImageNet200('./data/tiny-imagenet-200', train=False, download=True,
#                            transform=test_transform)
# test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=4)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# base_dir = './log_tiny_imagenet/'
# method_names = ['fgsm_uap']

# for method_name in method_names:
#     method_path = os.path.join(base_dir, method_name)

#     for time_path in os.listdir(method_path):
#         time_path = os.path.join(method_path, time_path)
#         if 'seed' not in time_path:
#             continue

#         if os.path.exists(os.path.join(time_path, method_name + '_best_pgd_test.log')):
#             continue

#         set_seed(0)
#         logger = Logger(os.path.join(time_path, method_name + '_best_pgd_test.log'))
#         model_path = os.path.join(time_path, 'best.pth')

#         model = ResNet18(num_classes=200).to(device)
#         load_model_weights(model, model_path, device)
#         model.eval()

#         attacker = PGD(model, eps=8.0 / 255, alpha=2.0 / 255, steps=10)
#         clean_correct, correct, total = 0, 0, 0
#         for images, labels in test_loader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             _, pre = torch.max(outputs.data, 1)
#             clean_correct += (pre == labels).sum().item()

#             adv_images = attacker(images, labels)
#             outputs = model(adv_images)
#             _, pre = torch.max(outputs.data, 1)
#             correct += (pre == labels).sum().item()
#             total += labels.size(0)
#         logger.log('clean accuracy: {} %'.format(100 * clean_correct / total))
#         logger.log('pgd-10 accuracy: {} %'.format(100 * correct / total))

#         attacker = PGD(model, eps=8.0 / 255, alpha=2.0 / 255, steps=20)
#         correct, total = 0, 0
#         for images, labels in test_loader:
#             images, labels = images.to(device), labels.to(device)
#             adv_images = attacker(images, labels)
#             outputs = model(adv_images)
#             _, pre = torch.max(outputs.data, 1)
#             correct += (pre == labels).sum().item()
#             total += labels.size(0)
#         logger.log('pgd-20 accuracy: {} %'.format(100 * correct / total))

#         attacker = PGD(model, eps=8.0 / 255, alpha=2.0 / 255, steps=50)
#         correct, total = 0, 0
#         for images, labels in test_loader:
#             images, labels = images.to(device), labels.to(device)
#             adv_images = attacker(images, labels)
#             outputs = model(adv_images)
#             _, pre = torch.max(outputs.data, 1)
#             correct += (pre == labels).sum().item()
#             total += labels.size(0)
#         logger.log('pgd-50 accuracy: {} %'.format(100 * correct / total))

#         output_path = os.path.join(time_path, 'output.log')
#         if os.path.exists(output_path):
#             with open(output_path, 'r') as f:
#                 lines = f.readlines()
#                 flag = False
#                 for line in lines:
#                     line = line.strip()
#                     if line == 'total_training_time:':
#                         flag = True
#                     if flag and line != '':
#                         logger.log(line)

#         logger.new_line()

#         if os.path.exists(os.path.join(time_path, method_name + '_last_pgd_test.log')):
#             continue

#         set_seed(0)
#         logger = Logger(os.path.join(time_path, method_name + '_last_pgd_test.log'))
#         model_path = os.path.join(time_path, 'last.pth')

#         model = ResNet18(num_classes=200).to(device)
#         load_model_weights(model, model_path, device)
#         model.eval()

#         attacker = PGD(model, eps=8.0 / 255, alpha=2.0 / 255, steps=10)
#         clean_correct, correct, total = 0, 0, 0
#         for images, labels in test_loader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             _, pre = torch.max(outputs.data, 1)
#             clean_correct += (pre == labels).sum().item()

#             adv_images = attacker(images, labels)
#             outputs = model(adv_images)
#             _, pre = torch.max(outputs.data, 1)
#             correct += (pre == labels).sum().item()
#             total += labels.size(0)
#         logger.log('clean accuracy: {} %'.format(100 * clean_correct / total))
#         logger.log('pgd-10 accuracy: {} %'.format(100 * correct / total))

#         attacker = PGD(model, eps=8.0 / 255, alpha=2.0 / 255, steps=20)
#         correct, total = 0, 0
#         for images, labels in test_loader:
#             images, labels = images.to(device), labels.to(device)
#             adv_images = attacker(images, labels)
#             outputs = model(adv_images)
#             _, pre = torch.max(outputs.data, 1)
#             correct += (pre == labels).sum().item()
#             total += labels.size(0)
#         logger.log('pgd-20 accuracy: {} %'.format(100 * correct / total))

#         attacker = PGD(model, eps=8.0 / 255, alpha=2.0 / 255, steps=50)
#         correct, total = 0, 0
#         for images, labels in test_loader:
#             images, labels = images.to(device), labels.to(device)
#             adv_images = attacker(images, labels)
#             outputs = model(adv_images)
#             _, pre = torch.max(outputs.data, 1)
#             correct += (pre == labels).sum().item()
#             total += labels.size(0)
#         logger.log('pgd-50 accuracy: {} %'.format(100 * correct / total))

#         output_path = os.path.join(time_path, 'output.log')
#         if os.path.exists(output_path):
#             with open(output_path, 'r') as f:
#                 lines = f.readlines()
#                 flag = False
#                 for line in lines:
#                     line = line.strip()
#                     if line == 'total_training_time:':
#                         flag = True
#                     if flag and line != '':
#                         logger.log(line)

#         logger.new_line()