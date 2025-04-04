import torch
from models.vision import weights_init, LeNet,LeNetMnist
import torch.optim as optim
import copy

def federated_train(global_model, g_model, client_loaders, criterion, num_rounds=10, num_local_epochs=1, lr=0.001, num_classes=10):
    """联邦训练函数(FedAvg)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_model.to(device)
    
    for round in range(num_rounds):
        print(f"Communication Round {round+1}/{num_rounds}")
        client_models = []
        
        # 训练所有客户端
        for client_id, loader in enumerate(client_loaders):
            # 克隆全局模型
            local_model = g_model.cuda()
            local_model.load_state_dict(global_model.state_dict())
            local_model.to(device)
            optimizer = optim.Adam(local_model.parameters(), lr=lr)
            
            # 本地训练
            local_model.train()
            for epoch in range(num_local_epochs):
                epoch_loss = 0
                for images, labels in loader:
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = local_model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    epoch_loss += loss.item()
                    optimizer.step()

                print(f"Client {client_id} - Epoch {epoch+1}/{num_local_epochs} - Loss: {epoch_loss / len(loader):.4f}")

            
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



def federated_train_proximal(global_model, g_model, client_loaders, criterion, num_rounds=10, num_local_epochs=1, lr=0.001, mu=0.01, num_classes=10):
    """FedProx: 带有近端项的联邦训练"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_model.to(device)
    
    for round in range(num_rounds):
        print(f"Communication Round {round+1}/{num_rounds}")
        client_models = []
        
        # 获取全局模型参数
        global_params = copy.deepcopy(global_model.state_dict())
        
        # 训练所有客户端
        for client_id, loader in enumerate(client_loaders):
            # 克隆全局模型
            local_model = g_model.cuda()
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
                    
                    # 原始任务损失
                    loss = criterion(outputs, labels)
                    
                    # 添加近端项（FedProx）
                    proximal_term = 0.0
                    for w, w_t in zip(local_model.parameters(), global_model.parameters()):
                        proximal_term += (w - w_t).norm(2)**2
                    
                    # 总损失 = 任务损失 + mu * 近端项
                    loss += (mu / 2) * proximal_term
                    
                    loss.backward()
                    optimizer.step()
            
            # 保存客户端模型参数
            client_models.append(local_model.state_dict())
        
        # 参数平均（与FedAvg相同）
        global_dict = global_model.state_dict()
        for key in global_dict.keys():
            global_dict[key] = torch.stack(
                [client_models[i][key].float() for i in range(len(client_models))], 0
            ).mean(0)
        global_model.load_state_dict(global_dict)
    
    return global_model


def federated_train_opt(global_model, g_model, client_loaders, criterion, num_rounds=10, num_local_epochs=1, lr=0.001, 
                         server_lr=0.1, server_momentum=0.9, num_classes=10):
    """FedOpt: 带有服务器优化器的联邦学习"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_model.to(device)
    
    # 服务器优化器状态
    server_momentum_buffer = {}
    for key in global_model.state_dict().keys():
        server_momentum_buffer[key] = torch.zeros_like(global_model.state_dict()[key])
    
    for round in range(num_rounds):
        print(f"Communication Round {round+1}/{num_rounds}")
        client_models = []
        sample_counts = []
        
        # 获取全局模型参数
        global_params = copy.deepcopy(global_model.state_dict())
        
        # 训练所有客户端
        for client_id, loader in enumerate(client_loaders):
            # 克隆全局模型
            local_model = g_model.cuda()
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
            
            # 保存客户端模型参数和样本数量
            client_models.append(local_model.state_dict())
            sample_counts.append(len(loader.dataset))
        
        # 计算加权平均模型更新
        pseudo_gradient = {}
        total_samples = sum(sample_counts)
        for key in global_params.keys():
            # 计算加权平均的伪梯度
            pseudo_gradient[key] = torch.zeros_like(global_params[key])
            for client_id, model_state in enumerate(client_models):
                weight = sample_counts[client_id] / total_samples
                pseudo_gradient[key] += weight * (global_params[key] - model_state[key])
        
        # 使用服务器端动量SGD更新全局模型
        global_dict = global_model.state_dict()
        for key in global_dict.keys():
            # 更新动量缓冲区
            server_momentum_buffer[key] = server_momentum * server_momentum_buffer[key] + pseudo_gradient[key]
            # 更新全局模型
            global_dict[key] = global_params[key] - server_lr * server_momentum_buffer[key]
        
        global_model.load_state_dict(global_dict)
    
    return global_model