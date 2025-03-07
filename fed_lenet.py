import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
from utils import label_to_onehot, cross_entropy_for_onehot
from models.vision import weights_init, LeNet,LeNetMnist
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import argparse

# python fed_lenet.py --dataset CIFAR100 --type sample



parser = argparse.ArgumentParser(description='Deep Leakage from Gradients.')
parser.add_argument('--dataset', type=str, default="MNIST",
                    help='dataset to do the experiment')
parser.add_argument('--type', type=str, default="sample",
                    help='unlearning type')
args = parser.parse_args()


num_classes=100
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
client_batch_size = 128
# 定义客户端数量和遗忘参数
CLIENT_NUM = 4
FORGOTTEN_CLIENT_IDX = 3  # 要遗忘的客户端索引
FORGET_SIZE = 1000        # 固定遗忘样本数
FORGOTTEN_CLASS = 1 #遗忘类别时要遗忘的类别是什么.默认为1

"""
模型定义
"""
net = LeNet(num_classes).to("cuda")
compress_rate = 1.0
torch.manual_seed(1234)
net.apply(weights_init)
criterion = cross_entropy_for_onehot


def federated_train(global_model, client_loaders, criterion, num_rounds=10, num_local_epochs=1, lr=0.001):
    """联邦训练函数(FedAvg)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_model.to(device)
    
    for round in range(num_rounds):
        print(f"Communication Round {round+1}/{num_rounds}")
        client_models = []
        
        # 训练所有客户端
        for client_id, loader in enumerate(client_loaders):
            # 克隆全局模型
            local_model = LeNet(num_classes).cuda()
            local_model.load_state_dict(global_model.state_dict())
            local_model.to(device)
            optimizer = optim.Adam(local_model.parameters(), lr=lr)
            
            # 本地训练
            local_model.train()
            for _ in range(num_local_epochs):
                for images, labels in loader:
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = local_model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
            
            # 保存客户端模型参数
            client_models.append(local_model.state_dict())
        
        # 参数平均（FedAvg）
        global_dict = global_model.state_dict()
        for key in global_dict.keys():
            global_dict[key] = torch.stack(
                [client_models[i][key].float() for i in range(len(client_models))], 0
            ).mean(0)
        global_model.load_state_dict(global_dict)
    
    return global_model

# 验证固定遗忘样本
def verify_fixed_samples():
    # 第一次运行获取样本特征
    first_run_samples = []
    for batch in forgotten_loader:
        first_run_samples.append(batch[0].sum().item())
    first_sum = sum(first_run_samples)
    
    # 第二次运行应该完全相同
    second_run_samples = []
    for batch in forgotten_loader:
        second_run_samples.append(batch[0].sum().item())
    
    assert np.allclose(first_sum, sum(second_run_samples)), "样本不固定!"
    print("验证通过：遗忘数据集样本保持固定")





"""
数据加载
cifar10:train:50000;test:10000; class=10
mnist:train:60000;test:10000;  class=10
cifar100:train:50000;test:10000; class=100
"""
if args.dataset == "CIFAR10":
    transform = transforms.Compose([transforms.ToTensor()])
    dst_train = datasets.CIFAR10(root="~/.torch", train=True, download=True, transform=transform)
elif args.dataset == "MNIST":
    transform = transforms.Compose([transforms.ToTensor(),])
    dst_train = datasets.MNIST("~/.torch", download=True, train=True, transform=transform)
elif args.dataset == "CIFAR100":
    transform = transforms.Compose([transforms.ToTensor(),])
    dst_train = datasets.CIFAR100("~/.torch", download=True, train=True, transform=transform)


if args.type == "sample":
    # 固定划分客户端数据（使用确定性的随机划分）
    client_datasets = torch.utils.data.random_split(
        dst_train,
        [len(dst_train)//CLIENT_NUM]*CLIENT_NUM,
        generator=torch.Generator().manual_seed(SEED)  # 固定划分随机种子
    )

    # 获取目标客户端原始数据索引
    target_dataset = client_datasets[FORGOTTEN_CLIENT_IDX]
    original_indices = target_dataset.indices.copy()  # 原始索引列表

    # 确定性地选择前N个样本作为遗忘集
    fixed_forgotten_indices = sorted(original_indices)[:FORGET_SIZE]  # 按原始顺序取前1000

    # 更新客户端数据集划分
    remaining_indices = list(set(original_indices) - set(fixed_forgotten_indices))
    client_datasets[FORGOTTEN_CLIENT_IDX] = Subset(dst_train, remaining_indices)

    # 创建遗忘数据集加载器
    forgotten_dataset = Subset(dst_train, fixed_forgotten_indices)
    forgotten_loader = DataLoader(
        forgotten_dataset, 
        batch_size=128, 
        shuffle=False
    )
    # 创建客户端加载器（包含更新后的数据集）
    client_loaders = [
        DataLoader(
            ds, 
            batch_size=128, 
            shuffle=True,  # 训练时保持shuffle但随机种子固定
            generator=torch.Generator().manual_seed(SEED))
        for ds in client_datasets ]
    
elif args.type == "client":
    # 固定划分客户端数据
    client_datasets = torch.utils.data.random_split(
        dst_train,
        [len(dst_train)//CLIENT_NUM]*CLIENT_NUM,
        generator=torch.Generator().manual_seed(SEED)
    )

    # 获取目标客户端原始数据
    target_dataset = client_datasets[FORGOTTEN_CLIENT_IDX]
    original_indices = target_dataset.indices.copy()


    # 将整个客户端的数据作为遗忘集
    fixed_forgotten_indices = original_indices
        # 更新客户端数据集为空
    client_datasets[FORGOTTEN_CLIENT_IDX] = Subset(dst_train, [])
        
    remaining_indices = list(set(original_indices) - set(fixed_forgotten_indices))
    # 创建遗忘数据集加载器
    forgotten_dataset = Subset(dst_train, fixed_forgotten_indices)
    forgotten_loader = DataLoader(
        forgotten_dataset, 
        batch_size=128, 
        shuffle=False
    )

    # 创建客户端加载器
    client_loaders = [
    DataLoader(
        ds,
        batch_size=128,
        shuffle=(len(ds) > 0),  # Shuffle only if dataset has samples
        generator=torch.Generator().manual_seed(SEED) if len(ds) > 0 else None
    ) for ds in client_datasets
]

elif args.type == "class":
    # 固定划分客户端数据
    client_datasets = torch.utils.data.random_split(
        dst_train,
        [len(dst_train)//CLIENT_NUM]*CLIENT_NUM,
        generator=torch.Generator().manual_seed(SEED)
    )

    # 获取目标客户端数据索引
    target_dataset = client_datasets[FORGOTTEN_CLIENT_IDX]
    original_indices = target_dataset.indices.copy()

    # 收集目标类别的样本索引
    forgotten_indices = []
    for idx in original_indices:
        _, label = dst_train[idx]  # 假设数据格式为（数据，标签）
        if label == FORGOTTEN_CLASS:
            forgotten_indices.append(idx)
    
    # 确定性地排序索引
    fixed_forgotten_indices = sorted(forgotten_indices)

    # 更新客户端数据集（移除目标类别）
    remaining_indices = list(set(original_indices) - set(fixed_forgotten_indices))
    client_datasets[FORGOTTEN_CLIENT_IDX] = Subset(dst_train, remaining_indices)

    # 创建遗忘数据集加载器
    forgotten_dataset = Subset(dst_train, fixed_forgotten_indices)
    forgotten_loader = DataLoader(
        forgotten_dataset, 
        batch_size=128, 
        shuffle=False
    )

    # 客户端加载器
    client_loaders = [
        DataLoader(
            ds, 
            batch_size=128, 
            shuffle=True,
            generator=torch.Generator().manual_seed(SEED)
        ) for ds in client_datasets
    ]


# if args.type == "client":

"""
开始训练
"""

print("联邦训练完整模型...")
# 初始化全局模型
full_net = LeNet(num_classes).cuda()
criterion = nn.CrossEntropyLoss()
global_round = 20

full_net = federated_train(
    full_net,
    client_loaders,  # 包含调整后的客户端3数据
    criterion,
    num_rounds=global_round,
    num_local_epochs=1,
    lr=0.001
)

# 未学习模型训练（从原始客户端加载器重建）
# 需要重新加载原始客户端数据（排除遗忘样本）
modified_client_loaders = [
    DataLoader(
        ds if idx != FORGOTTEN_CLIENT_IDX else Subset(ds.dataset, remaining_indices),
        batch_size=client_batch_size,
        shuffle=(len(ds) > 0)
    )
    for idx, ds in enumerate(client_datasets)
]

unlearned_net = LeNet(num_classes).cuda()
print("federated unlearning training...")
unlearned_net = federated_train(
    unlearned_net,
    modified_client_loaders,  # 使用排除遗忘样本的加载器
    criterion,
    num_rounds=global_round,
    num_local_epochs=1,
    lr=0.001
)

# 保存模型
torch.save(full_net.state_dict(), f"{args.dataset}_{args.type}_federated_full_round_20_partial.pth")
torch.save(unlearned_net.state_dict(), f"{args.dataset}_{args.type}_federated_unlearned_round_20_partial.pth")