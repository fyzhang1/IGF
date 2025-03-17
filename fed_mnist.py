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

# python fed_mnist.py --dataset MNIST --type sample --unlearning retrain

parser = argparse.ArgumentParser(description='Deep Leakage from Gradients.')
parser.add_argument('--dataset', type=str, default="CIFAR10",
                    help='dataset to do the experiment:CIFAR10,CIFAR100')
parser.add_argument('--type', type=str, default="sample",
                    help='unlearning data type:sample,class,client')
parser.add_argument('--unlearning', type=str, default="CIFAR10",
                    help='unlearning method:retrain,efficient')
args = parser.parse_args()


if args.dataset in ["FashionMNIST", "MNIST"]:
    image_size = 784
    num_classes = 10
elif args.dataset.startswith("CIFAR10"):
    image_size = 3 * 32 * 32
    num_classes = 10
elif args.dataset.startswith("CIFAR100"):
    image_size = 3 * 32 * 32
    num_classes = 100
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
net = LeNetMnist(num_classes).to("cuda")
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
            local_model = LeNetMnist(num_classes).cuda()
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


def federated_unlearning(global_model, forgotten_loader, remaining_client_loaders, 
                        criterion, num_unlearn_rounds=3, num_finetune_rounds=5,
                        unlearn_lr=0.1, finetune_lr=0.001):
    """
    param remaining_client_loaders: 剩余客户端的数据加载器（排除被遗忘的客户端）
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_model.to(device)
    
    # 客户端本地梯度上升遗忘
    print("=== 客户端本地遗忘 ===")
    client_models = []
    for client_id, loader in enumerate(remaining_client_loaders):
        # 克隆全局模型到本地
        local_model = LeNetMnist(num_classes).to(device)
        local_model.load_state_dict(global_model.state_dict())
        
        # 本地执行梯度上升（仅对需要遗忘的客户端）
        if client_id == FORGOTTEN_CLIENT_IDX: 
            print(f"客户端 {client_id} 执行遗忘...")
            optimizer = optim.SGD(local_model.parameters(), lr=unlearn_lr)
            
            for _ in range(num_unlearn_rounds):
                for images, labels in forgotten_loader:
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = local_model(images)
                    loss = criterion(outputs, labels)
                    loss += 0.01 * sum(p.pow(2.0).sum() for p in local_model.parameters())  # L2 正则化
                    loss.backward()
                    
                    # 梯度反转
                    for param in local_model.parameters():
                        if param.grad is not None:
                            param.grad.data = -param.grad.data
                    
                    optimizer.step()
        
        # 保存本地模型
        client_models.append(local_model.state_dict())
    
    # 聚合
    print("=== 安全聚合 ===")
    global_dict = global_model.state_dict()
    for key in global_dict.keys():
        # 仅聚合剩余客户端模型
        valid_clients = [client_models[i] for i in range(len(client_models)) 
                        if i != FORGOTTEN_CLIENT_IDX]
        global_dict[key] = torch.stack(
            [client[key].float() for client in valid_clients], 0
        ).mean(0)
    global_model.load_state_dict(global_dict)
    
    # 微调
    if num_finetune_rounds > 0:
        print("=== 联邦微调 ===")
        global_model = federated_train(
            global_model,
            remaining_client_loaders,  # 确保不包含被遗忘客户端
            criterion,
            num_rounds=num_finetune_rounds,
            num_local_epochs=1,
            lr=finetune_lr
        )
    
    return global_model

def single_federated_unlearning(global_model, forgotten_loader, remaining_client_loaders, 
                        criterion, num_unlearn_rounds=1, num_finetune_rounds=5,
                        unlearn_lr=0.01, finetune_lr=0.001, m=1, b=128):
    """
    使用单梯度遗忘方法，仅遗忘客户端计算梯度
    m: 微调 epoch 数
    b: 微调 batch size
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_model.to(device)
    
    print("=== 客户端本地单梯度遗忘 ===")
    client_models = []
    
    # 仅遗忘客户端计算梯度
    grad_sum = None
    for client_id, loader in enumerate(remaining_client_loaders):
        local_model = LeNetMnist(num_classes).to(device)
        local_model.load_state_dict(global_model.state_dict())
        
        if client_id == FORGOTTEN_CLIENT_IDX:
            print(f"客户端 {client_id} 计算遗忘样本梯度...")
            local_model.eval()  # 不训练，仅计算梯度
            
            for images, labels in forgotten_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = local_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                
                # 累加参数梯度
                if grad_sum is None:
                    grad_sum = [p.grad.data.clone() for p in local_model.parameters()]
                else:
                    for g1, g2 in zip(grad_sum, local_model.parameters()):
                        g1.add_(g2.grad.data)
                
                # 清零梯度以便下次计算
                local_model.zero_grad()
        
        client_models.append(local_model.state_dict())
    
    print("=== 服务器应用梯度 ===")
    # 仅遗忘客户端提供梯度
    if grad_sum is not None:
        # 平均梯度（按样本数）
        num_samples = len(forgotten_loader.dataset)
        for g in grad_sum:
            g.div_(num_samples)  # 平均到每个样本
        
        # 应用公式 (13): ∇_u = (m/b) * ∇L
        with torch.no_grad():
            for param, grad in zip(global_model.parameters(), grad_sum):
                grad_adjusted = (m / b) * grad * unlearn_lr  # unlearn_lr 作为 η
                param.data.add_(grad_adjusted)
    
    print("=== 安全聚合 ===")
    global_dict = global_model.state_dict()
    valid_clients = [client_models[i] for i in range(len(client_models)) 
                    if i != FORGOTTEN_CLIENT_IDX]
    for key in global_dict.keys():
        global_dict[key] = torch.stack(
            [client[key].float() for client in valid_clients], 0
        ).mean(0)
    global_model.load_state_dict(global_dict)
    
    # 可选微调
    if num_finetune_rounds > 0:
        print("=== 联邦微调 ===")
        global_model = federated_train(
            global_model,
            remaining_client_loaders,
            criterion,
            num_rounds=num_finetune_rounds,
            num_local_epochs=1,
            lr=finetune_lr
        )
    
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
if args.unlearning == "retrain":
    print("联邦训练完整模型...")
    # 初始化全局模型
    full_net = LeNetMnist(num_classes).cuda()
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

    unlearned_net = LeNetMnist(num_classes).cuda()
    print("federated unlearning training...")
    unlearned_net = federated_train(
        unlearned_net,
        modified_client_loaders,  # 使用排除遗忘样本的加载器
        criterion,
        num_rounds=global_round,
        num_local_epochs=1,
        lr=0.001
    )
elif args.unlearning == "efficient":
    print("联邦训练完整模型...")
    full_net = LeNetMnist(num_classes).cuda()
    criterion = nn.CrossEntropyLoss()
    global_round = 20
    client_batch_size =128

    # modified_client_loaders是除遗忘样本外的数据集
    modified_client_loaders = [
        DataLoader(
            ds if idx != FORGOTTEN_CLIENT_IDX else Subset(ds.dataset, remaining_indices),
            batch_size=client_batch_size,
            shuffle=(len(ds) > 0)
        )
        for idx, ds in enumerate(client_datasets)
    ]

    # 完整联邦训练（包含遗忘客户端的数据）
    full_net = federated_train(
        full_net,
        client_loaders,
        criterion,
        num_rounds=global_round,
        num_local_epochs=1,
        lr=0.001
    )

    # 执行近似遗忘
    print("执行近似遗忘...")
    unlearned_net = federated_unlearning(full_net, forgotten_loader, modified_client_loaders, 
                            criterion, num_unlearn_rounds=3, num_finetune_rounds=0,
                            unlearn_lr=0.001, finetune_lr=0.001)


elif args.unlearning == "single_efficient":
    print("联邦训练完整模型...")
    full_net = LeNetMnist(num_classes).cuda()
    criterion = nn.CrossEntropyLoss()  # 修正损失函数
    global_round = 20
    client_batch_size = 128

    modified_client_loaders = [
        DataLoader(
            ds if idx != FORGOTTEN_CLIENT_IDX else Subset(ds.dataset, remaining_indices),
            batch_size=client_batch_size,
            shuffle=(len(ds) > 0)
        )
        for idx, ds in enumerate(client_datasets)
    ]

    full_net = federated_train(
        full_net,
        client_loaders,
        criterion,
        num_rounds=global_round,
        num_local_epochs=1,
        lr=0.001
    )

    print("执行单梯度遗忘...")
    unlearned_net = single_federated_unlearning(full_net, forgotten_loader, modified_client_loaders, 
                                        criterion, num_unlearn_rounds=20, num_finetune_rounds=0,
                                        unlearn_lr=0.01, finetune_lr=0.001, m=1, b=128)
    



# 保存模型
torch.save(full_net.state_dict(), f"/home/ecs-user/fgi/federated_weight/Lenet/{args.dataset}_{args.type}_{args.unlearning}_federated_full_round_20_partial.pth")
torch.save(unlearned_net.state_dict(), f"/home/ecs-user/fgi/federated_weight/Lenet/{args.dataset}_{args.type}_{args.unlearning}_federated_unlearned_round_20_partial.pth")