from torchvision import datasets, transforms

# 自动下载并加载数据集（首次运行时下载）
train_dataset = datasets.CIFAR100(
    root='./data',          # 存储路径
    train=True,             # True为训练集，False为测试集
    download=True,          # 自动下载
    transform=transforms.ToTensor()  # 转换为张量
)
test_dataset = datasets.CIFAR100(
    root='./data',
    train=False,
    download=True,
    transform=transforms.ToTensor()
)