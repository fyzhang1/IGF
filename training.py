import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
from utils_com.utils import label_to_onehot, cross_entropy_for_onehot
from models.resnet import resnet20,mnist_resnet20
from models.vision import weights_init, LeNet,LeNetMnist
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import argparse
from utils_com.federated import federated_train, federated_train_opt, federated_train_proximal





# python training.py --model mnist_resnet20 --dataset MNIST --type sample --unlearning retrain --aggregation fedavg


parser = argparse.ArgumentParser(description='Deep Leakage from Gradients.')
parser.add_argument('--model', type=str, default="lenet",
                    help='lenet,lenetmnist,resnet20,mnist_resnet20,')
parser.add_argument('--dataset', type=str, default="CIFAR10",
                    help='dataset to do the experiment:CIFAR10,CIFAR100')
parser.add_argument('--type', type=str, default="sample",
                    help='unlearning data type:sample,class,client')
parser.add_argument('--unlearning', type=str, default="retrain",
                    help='unlearning method:retrain,efficient')
parser.add_argument('--aggregation', type=str, default="fedavg",
                    help='fedavg,fedprox,fedopt')
args = parser.parse_args()


if args.dataset in ["FashionMNIST", "MNIST"]:
    image_size = 784
    num_classes = 10
    input_channels = 1
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
if args.model == "lenet":
    net = LeNet(num_classes).to("cuda")
    compress_rate = 1.0
    torch.manual_seed(1234)
    net.apply(weights_init)
    criterion = cross_entropy_for_onehot
    g_model = LeNet(num_classes).to("cuda")
elif args.model == "lenetmnist":
    net = LeNetMnist(input_channels=1,num_classes=10).to("cuda")
    compress_rate = 1.0
    torch.manual_seed(1234)
    net.apply(weights_init)
    criterion = cross_entropy_for_onehot
    g_model = LeNetMnist(input_channels=1,num_classes=10).to("cuda")
elif args.model == "resnet20":
    net = resnet20(num_classes).to("cuda")
    g_model = resnet20(num_classes).to("cuda")
    compress_rate = 1.0
    torch.manual_seed(1234)
    net.apply(weights_init)
    criterion = cross_entropy_for_onehot
elif args.model == "mnist_resnet20":
    net = mnist_resnet20(num_classes).to("cuda")
    g_model = mnist_resnet20(num_classes).to("cuda")
    compress_rate = 1.0
    torch.manual_seed(1234)
    net.apply(weights_init)
    criterion = cross_entropy_for_onehot
    

def efficient_federated_unlearning(global_model, forgotten_loader, remaining_client_loaders, 
                        criterion, num_unlearn_rounds=3, num_finetune_rounds=5,
                        unlearn_lr=0.1, finetune_lr=0.001,epsilon=0.1):
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
        local_model = g_model.to(device)
        local_model.load_state_dict(global_model.state_dict())
        local_model.train()
        
        # 本地执行梯度上升（仅对需要遗忘的客户端）
        if client_id == FORGOTTEN_CLIENT_IDX: 
            print(f"客户端 {client_id} 执行遗忘...")
            optimizer = optim.SGD(local_model.parameters(), lr=unlearn_lr)
            
            for epoch in range(num_unlearn_rounds):
                epoch_loss = 0
                for images, labels in forgotten_loader:
                    images, labels = images.to(device), labels.to(device)
                    
                    optimizer.zero_grad()
                    outputs = local_model(images)
                    loss = criterion(outputs, labels)
                    loss = -loss

                    
                    loss.backward()
                    epoch_loss += loss.item()
                    optimizer.step()
                    
                    # 梯度反转
                    with torch.no_grad():
                        for param, ref_param in zip(local_model.parameters(), global_model.parameters()):
                            # Compute difference from reference model
                            diff = param - ref_param
                            norm = torch.norm(diff)
                            if norm > epsilon:
                                # Project back to the L2 ball boundary
                                param.data = ref_param + (diff / norm) * epsilon
                    
                    optimizer.step()
                    print(f"Client {client_id} - Epoch {epoch+1}/{num_unlearn_rounds} - Loss: {epoch_loss / len(loader):.4f}")

        
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
            g_model,
            remaining_client_loaders,  # 确保不包含被遗忘客户端
            criterion,
            num_rounds=num_finetune_rounds,
            num_local_epochs=1,
            lr=finetune_lr
        )
    
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
        local_model = g_model.to(device)
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
            g_model,
            remaining_client_loaders,  # 确保不包含被遗忘客户端
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
    full_net = g_model.cuda()
    criterion = nn.CrossEntropyLoss()
    global_round = 20

    if args.aggregation == "fedavg":
        full_net = federated_train(
            full_net,
            g_model,
            client_loaders,  # 包含调整后的客户端3数据
            criterion,
            num_rounds=global_round,
            num_local_epochs=1,
            lr=0.001,
            num_classes=num_classes
        )
    elif args.aggregation == "fedprox":
        full_net = federated_train_proximal(
            full_net, 
            g_model,
            client_loaders, 
            criterion, 
            num_rounds=global_round,
            num_local_epochs=1, 
            lr=0.001, 
            mu=0.01, 
            num_classes=num_classes)
    
    elif args.aggregation == "fedopt":
        full_net = federated_train_opt(
            full_net, 
            g_model,
            client_loaders, 
            criterion, 
            num_rounds=global_round, 
            num_local_epochs=1, 
            lr=0.001, 
            server_lr=0.1, 
            server_momentum=0.9, 
            num_classes=num_classes)


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

    unlearned_net = g_model.cuda()
    print("federated unlearning training...")


    if args.aggregation == "fedavg":
        unlearned_net = federated_train(
            unlearned_net,
            g_model,
            modified_client_loaders,  # 使用排除遗忘样本的加载器
            criterion,
            num_rounds=global_round,
            num_local_epochs=1,
            lr=0.001
        )
    elif args.aggregation == "fedprox":
        unlearned_net = federated_train_proximal(
            unlearned_net, 
            modified_client_loaders, 
            criterion, 
            num_rounds=global_round,
            num_local_epochs=1, 
            lr=0.001, 
            mu=0.01, 
            num_classes=num_classes)
    
    elif args.aggregation == "fedopt":
        unlearned_net = federated_train_opt(
            unlearned_net, 
            modified_client_loaders, 
            criterion, 
            num_rounds=global_round, 
            num_local_epochs=1, 
            lr=0.001, 
            server_lr=0.1, 
            server_momentum=0.9, 
            num_classes=num_classes)





elif args.unlearning == "efficient":
    print("联邦训练完整模型...")
    full_net = g_model.cuda()
    criterion = nn.CrossEntropyLoss()
    global_round = 10
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
        g_model,
        client_loaders,
        criterion,
        num_rounds=global_round,
        num_local_epochs=1,
        lr=0.001,
        num_classes=num_classes
    )

    # 执行近似遗忘
    print("执行近似遗忘...")
    unlearned_net = efficient_federated_unlearning(full_net, forgotten_loader, modified_client_loaders, 
                            criterion, num_unlearn_rounds=10, num_finetune_rounds=10,
                            unlearn_lr=0.001, finetune_lr=0.001)
    



# 保存模型
torch.save(full_net.state_dict(), f"/home/ecs-user/fgi/federated_weight/{args.model}/{args.dataset}_{args.type}_{args.unlearning}_{args.aggregation}_federated_full_round_20_partial.pth")
torch.save(unlearned_net.state_dict(), f"/home/ecs-user/fgi/federated_weight/{args.model}/{args.dataset}_{args.type}_{args.unlearning}_{args.aggregation}_federated_unlearned_round_20_partial.pth")