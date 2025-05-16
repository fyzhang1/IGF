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

CLIENT_NUM = 4
FORGOTTEN_CLIENT_IDX = 3  
FORGET_SIZE = 1000      
FORGOTTEN_CLASS = 1 


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_model.to(device)
    

    print("=== local client unlearning ===")
    client_models = []
    for client_id, loader in enumerate(remaining_client_loaders):
        local_model = g_model.to(device)
        local_model.load_state_dict(global_model.state_dict())
        local_model.train()
        
        if client_id == FORGOTTEN_CLIENT_IDX: 
            print(f"Client {client_id} Unlearning...")
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

        
        # save
        client_models.append(local_model.state_dict())
    
    print("=== aggregation ===")
    global_dict = global_model.state_dict()
    for key in global_dict.keys():
        valid_clients = [client_models[i] for i in range(len(client_models)) 
                        if i != FORGOTTEN_CLIENT_IDX]
        global_dict[key] = torch.stack(
            [client[key].float() for client in valid_clients], 0
        ).mean(0)
    global_model.load_state_dict(global_dict)
    
    # (opt)
    if num_finetune_rounds > 0:
        print("=== federared fine-tne ===")
        global_model = federated_train(
            global_model,
            g_model,
            remaining_client_loaders,
            criterion,
            num_rounds=num_finetune_rounds,
            num_local_epochs=1,
            lr=finetune_lr
        )
    
    return global_model


def federated_unlearning(global_model, forgotten_loader, remaining_client_loaders, 
                        criterion, num_unlearn_rounds=3, num_finetune_rounds=5,
                        unlearn_lr=0.1, finetune_lr=0.001):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_model.to(device)
    
    print("=== local client unlearning ===")
    client_models = []
    for client_id, loader in enumerate(remaining_client_loaders):
        local_model = g_model.to(device)
        local_model.load_state_dict(global_model.state_dict())
        
        if client_id == FORGOTTEN_CLIENT_IDX: 
            print(f"Client {client_id} Unlearning...")
            optimizer = optim.SGD(local_model.parameters(), lr=unlearn_lr)
            
            for _ in range(num_unlearn_rounds):
                for images, labels in forgotten_loader:
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = local_model(images)
                    loss = criterion(outputs, labels)
                    loss += 0.01 * sum(p.pow(2.0).sum() for p in local_model.parameters())  # L2
                    loss.backward()
                    
                    for param in local_model.parameters():
                        if param.grad is not None:
                            param.grad.data = -param.grad.data
                    
                    optimizer.step()
        
        client_models.append(local_model.state_dict())
    
    print("=== aggregation ===")
    global_dict = global_model.state_dict()
    for key in global_dict.keys():
        valid_clients = [client_models[i] for i in range(len(client_models)) 
                        if i != FORGOTTEN_CLIENT_IDX]
        global_dict[key] = torch.stack(
            [client[key].float() for client in valid_clients], 0
        ).mean(0)
    global_model.load_state_dict(global_dict)
    
    if num_finetune_rounds > 0:
        global_model = federated_train(
            global_model,
            g_model,
            remaining_client_loaders,
            criterion,
            num_rounds=num_finetune_rounds,
            num_local_epochs=1,
            lr=finetune_lr
        )
    
    return global_model


def verify_fixed_samples():
    first_run_samples = []
    for batch in forgotten_loader:
        first_run_samples.append(batch[0].sum().item())
    first_sum = sum(first_run_samples)
    
    second_run_samples = []
    for batch in forgotten_loader:
        second_run_samples.append(batch[0].sum().item())
    
    assert np.allclose(first_sum, sum(second_run_samples)), "error"
    print("sample right")


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
    client_datasets = torch.utils.data.random_split(
        dst_train,
        [len(dst_train)//CLIENT_NUM]*CLIENT_NUM,
        generator=torch.Generator().manual_seed(SEED)
    )

    target_dataset = client_datasets[FORGOTTEN_CLIENT_IDX]
    original_indices = target_dataset.indices.copy()

    fixed_forgotten_indices = sorted(original_indices)[:FORGET_SIZE]

    remaining_indices = list(set(original_indices) - set(fixed_forgotten_indices))
    client_datasets[FORGOTTEN_CLIENT_IDX] = Subset(dst_train, remaining_indices)

    forgotten_dataset = Subset(dst_train, fixed_forgotten_indices)
    forgotten_loader = DataLoader(
        forgotten_dataset, 
        batch_size=128, 
        shuffle=False
    )

    client_loaders = [
        DataLoader(
            ds, 
            batch_size=128, 
            shuffle=True,
            generator=torch.Generator().manual_seed(SEED))
        for ds in client_datasets ]
    
elif args.type == "client":
    client_datasets = torch.utils.data.random_split(
        dst_train,
        [len(dst_train)//CLIENT_NUM]*CLIENT_NUM,
        generator=torch.Generator().manual_seed(SEED)
    )


    target_dataset = client_datasets[FORGOTTEN_CLIENT_IDX]
    original_indices = target_dataset.indices.copy()

    fixed_forgotten_indices = original_indices
    client_datasets[FORGOTTEN_CLIENT_IDX] = Subset(dst_train, [])
        
    remaining_indices = list(set(original_indices) - set(fixed_forgotten_indices))
    forgotten_dataset = Subset(dst_train, fixed_forgotten_indices)
    forgotten_loader = DataLoader(
        forgotten_dataset, 
        batch_size=128, 
        shuffle=False
    )

    client_loaders = [
    DataLoader(
        ds,
        batch_size=128,
        shuffle=(len(ds) > 0),  # Shuffle only if dataset has samples
        generator=torch.Generator().manual_seed(SEED) if len(ds) > 0 else None
    ) for ds in client_datasets
]

elif args.type == "class":
    client_datasets = torch.utils.data.random_split(
        dst_train,
        [len(dst_train)//CLIENT_NUM]*CLIENT_NUM,
        generator=torch.Generator().manual_seed(SEED)
    )

    target_dataset = client_datasets[FORGOTTEN_CLIENT_IDX]
    original_indices = target_dataset.indices.copy()

    forgotten_indices = []
    for idx in original_indices:
        _, label = dst_train[idx]
        if label == FORGOTTEN_CLASS:
            forgotten_indices.append(idx)
    
    fixed_forgotten_indices = sorted(forgotten_indices)

    remaining_indices = list(set(original_indices) - set(fixed_forgotten_indices))
    client_datasets[FORGOTTEN_CLIENT_IDX] = Subset(dst_train, remaining_indices)

    forgotten_dataset = Subset(dst_train, fixed_forgotten_indices)
    forgotten_loader = DataLoader(
        forgotten_dataset, 
        batch_size=128, 
        shuffle=False
    )

    client_loaders = [
        DataLoader(
            ds, 
            batch_size=128, 
            shuffle=True,
            generator=torch.Generator().manual_seed(SEED)
        ) for ds in client_datasets
    ]


if args.unlearning == "retrain":
    print("federated learning retrain...")
    full_net = g_model.cuda()
    criterion = nn.CrossEntropyLoss()
    global_round = 20

    if args.aggregation == "fedavg":
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
            modified_client_loaders,
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
    full_net = g_model.cuda()
    criterion = nn.CrossEntropyLoss()
    global_round = 10
    client_batch_size =128

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
        g_model,
        client_loaders,
        criterion,
        num_rounds=global_round,
        num_local_epochs=1,
        lr=0.001,
        num_classes=num_classes
    )

    print("approximate Federated Unlearning...")
    unlearned_net = efficient_federated_unlearning(full_net, forgotten_loader, modified_client_loaders, 
                            criterion, num_unlearn_rounds=10, num_finetune_rounds=10,
                            unlearn_lr=0.001, finetune_lr=0.001)
    


torch.save(full_net.state_dict(), f"./fgi/federated_weight/{args.model}/{args.dataset}_{args.type}_{args.unlearning}_{args.aggregation}_federated_full_round_20_partial.pth")
torch.save(unlearned_net.state_dict(), f"./fgi/federated_weight/{args.model}/{args.dataset}_{args.type}_{args.unlearning}_{args.aggregation}_federated_unlearned_round_20_partial.pth")