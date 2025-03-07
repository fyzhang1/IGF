import argparse
import numpy as np
from tqdm import tqdm
import math
from copy import deepcopy
import os 
os.environ['KMP_WARNINGS'] = '0'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import models, datasets, transforms
from scipy.optimize import linear_sum_assignment

from utils import label_to_onehot, cross_entropy_for_onehot
from models.vision import LeNetMnist, weights_init, LeNet
from models.resnet import resnet20
from logger import set_logger
import lpips

parser = argparse.ArgumentParser(description='Deep Leakage from Gradients with SVD.')
parser.add_argument('--dataset', type=str, default="MNIST",
                    help='dataset to do the experiment')
parser.add_argument('--model', type=str, default="MLP-3000",
                    help='MLP-{hidden_size}')
parser.add_argument('--shared_model', type=str, default="LeNet",
                    help='LeNet')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--epochs', type=int, default=200,
                    help='epochs for training')
parser.add_argument('--batch_size', type=int, default=256,
                    help='batch_size for training')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed for training set-up')
parser.add_argument('--leak_mode', type=str, default="sign",
                    help='sign/prune-{prune_rate}/batch-{batch_size}')
parser.add_argument('--get_validation_rec_first_thou', type=str, default=None,
                    help='checkpoint path')
parser.add_argument('--resume', type=str, default=None,
                    help='checkpoint path')
parser.add_argument('--type', type=str, default="sample",
                    help='unlearning type')
args = parser.parse_args()

if args.resume is not None:
    save_file_name = f"{args.dataset}_{args.shared_model}_{args.model}_{args.leak_mode}_{args.lr}_{args.epochs}_{args.batch_size}_{args.seed}_{args.resume}"
else:
    save_file_name = f"{args.dataset}_{args.shared_model}_{args.model}_{args.leak_mode}_{args.lr}_{args.epochs}_{args.batch_size}_{args.seed}"

logger = set_logger("", filepath=f"print_logs/{save_file_name}.txt")
logger.info(args)
logger.info(f"logs are saved at print_logs/{save_file_name}.txt")

save_dir = "./"

def train(grad_to_img_net, net, data_loader, sign=False, mask=None, prune_rate=None, leak_batch=1):
    grad_to_img_net.train()
    total_loss = 0
    total_num = 0
    cross_entropy = torch.nn.CrossEntropyLoss()
    for i, data in enumerate(tqdm(data_loader)):
        (images, labels) = data
        xs, ys = new_leakage_dataset(images,labels,full_net,unlearned_net)
        optimizer.zero_grad()
        batch_num = len(ys)
        batch_size = int(batch_num / leak_batch)
        batch_num = batch_size * leak_batch
        total_num += batch_num
        xs, ys = xs[:batch_num].cuda(), ys[:batch_num].cuda()
        
        if sign:
            xs = torch.sign(xs)
        if prune_rate is not None:
            rank = torch.topk(xs.abs(), int(xs.size()[1] * (1 - prune_rate)), dim=1).indices
            mask = torch.zeros(xs.size())
            mask[torch.arange(len(ys)).view(-1, 1).expand(rank.size()), rank] = 1  
        if mask is not None:
            xs.mul_(mask)
        
        # 使用 SVD 降维矩阵投影梯度
        xs = torch.mm(xs, V_k)  # xs: (batch_num, model_size) -> (batch_num, k)
        if gauss_noise > 0:
            xs = xs + torch.randn(*xs.shape).cuda() * gauss_noise
        

        xs = xs.view(batch_size, leak_batch, -1).mean(1).cuda()  # (batch_size, k)
        ys = ys.view(batch_size, leak_batch, -1).cuda()
        preds = grad_to_img_net(xs).view(batch_size, leak_batch, -1)


        # batch_wise_mse = (torch.cdist(ys, preds) ** 2) / image_size
        # loss = 0
        # for mse_mat in batch_wise_mse:
        #     row_ind, col_ind = linear_sum_assignment(mse_mat.detach().cpu().numpy())
        #     loss += mse_mat[row_ind, col_ind].mean()
        # loss /= batch_size
        # if i % 10 == 0:
        #     logger.info(f"train iter: {i}; loss: {loss}")
        # loss.backward()
        # optimizer.step()
        # total_loss += loss.item() * batch_num

        # Compute MSE loss with matching per sample
        mse_loss = 0
        matched_reconstructed = []
        matched_real = []
        for sample_id in range(batch_size):
            ys_sample = ys[sample_id]
            preds_sample = preds[sample_id]
            distance_matrix = torch.cdist(ys_sample, preds_sample)
            mse_mat = (distance_matrix ** 2) / image_size
            row_ind, col_ind = linear_sum_assignment(mse_mat.detach().cpu().numpy())
            mse_loss_sample = mse_mat[row_ind, col_ind].mean()
            mse_loss += mse_loss_sample
            # Collect matched reconstructed and real images
            matched_reconstructed.append(preds_sample[col_ind])
            matched_real.append(ys_sample[row_ind])
        
        mse_loss /= batch_size
        # Prepare matched images for perceptual loss
        matched_reconstructed_all = torch.stack(matched_reconstructed, dim=0).view(batch_size * leak_batch, image_size)
        matched_real_all = torch.stack(matched_real, dim=0).view(batch_size * leak_batch, image_size)
        reconstructed_images_matched = matched_reconstructed_all.view(batch_size * leak_batch, 3, 32, 32).to('cuda')
        real_images_matched = matched_real_all.view(batch_size * leak_batch, 3, 32, 32).to('cuda')
        loss_fn_vgg = lpips.LPIPS(net='vgg').to('cuda')

        real_images_matched = real_images_matched * 2 - 1   # [0,1] -> [-1,1]
        reconstructed_images_matched = reconstructed_images_matched * 2 - 1
        perceptual_loss = loss_fn_vgg(real_images_matched, reconstructed_images_matched).mean()

        total_loss = mse_loss + 0.1 * perceptual_loss
        print("perceptual_loss:", perceptual_loss)
        total_loss.backward()
        optimizer.step()
        total_loss_value = total_loss.item() * batch_num
        total_loss += total_loss_value
            
    total_loss = total_loss / total_num
    return (total_loss, None)

def test(grad_to_img_net, net, data_loader, sign=False, mask=None, prune_rate=None, leak_batch=1, num_test=None):
    grad_to_img_net.eval()
    total_loss = 0
    total_num = 0
    reconstructed_data = []
    gt_data = []
    cross_entropy = torch.nn.CrossEntropyLoss()
    if num_test is None:
        num_test = len(data_loader.dataset)
    for i, data in enumerate(tqdm(data_loader)):
        if i * args.batch_size >= num_test:
            break
        (images, labels) = data
        xs, ys = new_leakage_dataset(images,labels,full_net,unlearned_net)
        with torch.no_grad():
            batch_num = len(ys)
            batch_size = int(batch_num / leak_batch)
            batch_num = batch_size * leak_batch
            total_num += batch_num
            xs, ys = xs[:batch_num], ys[:batch_num]
            if sign:
                xs = torch.sign(xs)
            if prune_rate is not None:
                mask = torch.zeros(xs.size())
                rank = torch.topk(xs.abs(), int(xs.size()[1] * (1 - prune_rate)), dim=1).indices
                mask[torch.arange(len(ys)).view(-1, 1).expand(rank.size()), rank] = 1   
            if mask is not None:
                xs = xs * mask
                
            # 使用 SVD 降维矩阵投影梯度
            xs = torch.mm(xs, V_k)  # xs: (batch_num, model_size) -> (batch_num, k)
            if gauss_noise > 0:
                xs = xs + torch.randn(*xs.shape).cuda() * gauss_noise
                
            xs = xs.view(batch_size, leak_batch, -1).mean(1).cuda()  # (batch_size, k)
            ys = ys.view(batch_size, leak_batch, -1).cuda()
            grad_to_img_net.cuda()
            preds = grad_to_img_net(xs).view(batch_size, leak_batch, -1)
            batch_wise_mse = (torch.cdist(ys, preds) ** 2) / image_size
            loss = 0
            for batch_id, mse_mat in enumerate(batch_wise_mse):
                row_ind, col_ind = linear_sum_assignment(mse_mat.detach().cpu().numpy())
                loss += mse_mat[row_ind, col_ind].mean()
                sorted_preds = preds[batch_id, col_ind]
                sorted_preds[row_ind] = preds[batch_id, col_ind]
                sorted_preds = sorted_preds.view(leak_batch, -1).detach().cpu()
                reconstructed_data.append(sorted_preds)
                gt_data.append(ys[batch_id])
            loss /= batch_size
            total_loss += loss.item() * batch_num
            
    reconstructed_data = torch.cat(reconstructed_data)
    if args.dataset in ["MNIST", "FashionMNIST"]:
        reconstructed_data = reconstructed_data.view(-1, 1, 28, 28)
    elif args.dataset.startswith("CIFAR10"):
        reconstructed_data = reconstructed_data.view(-1, 3, 32, 32)

    total_loss = total_loss / total_num
    return (total_loss, None), (reconstructed_data, gt_data)

# 输入模型
if args.dataset in ["FashionMNIST", "MNIST"]:
    image_size = 784
    num_classes = 10
elif args.dataset.startswith("CIFAR10"):
    image_size = 3 * 32 * 32
    num_classes = 10
elif args.dataset.startswith("CIFAR100"):
    image_size = 3 * 32 * 32
    num_classes = 100

prune_rate = None
leak_batch = 1
sign = False
gauss_noise = 0
leak_mode_list = args.leak_mode.split("-")
for i in range(len(leak_mode_list)):
    if leak_mode_list[i] == "sign":
        sign = True
    elif leak_mode_list[i] == "prune":
        prune_rate = float(leak_mode_list[i+1])
    elif leak_mode_list[i] == "batch":
        leak_batch = int(leak_mode_list[i+1])
    elif leak_mode_list[i] == "gauss":
        gauss_noise = float(leak_mode_list[i+1])

if args.shared_model == "ResNet20":
    net = resnet20(num_classes).to("cuda")
    compress_rate = 0.5
    torch.manual_seed(1234)
    net.apply(weights_init)

model_size = 0
for i, parameters in enumerate(net.parameters()):
    if parameters.requires_grad:
        model_size += np.prod(parameters.size())
logger.info(f"model size: {model_size}")

logger.info("loading dataset...")
if args.dataset == "MNIST":
    transform = transforms.Compose([transforms.ToTensor()])
    dst_train = datasets.MNIST("~/.torch", download=True, train=True, transform=transform)
    dst_test = datasets.MNIST("~/.torch", download=True, train=False, transform=transform)
elif args.dataset == "FashionMNIST":
    transform = transforms.Compose([transforms.ToTensor()])
    dst_train = datasets.FashionMNIST("~/.torch", download=True, train=True, transform=transform)
    dst_test = datasets.FashionMNIST("~/.torch", download=True, train=False, transform=transform)
elif args.dataset.startswith("CIFAR10"):
    transform = transforms.Compose([transforms.ToTensor()])
    dst_train = datasets.CIFAR10("~/.torch", download=True, train=True, transform=transform)
    dst_validation = datasets.CIFAR10("~/.torch", download=True, train=False, transform=transform)    

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
client_batch_size = 128
# 定义客户端数量和遗忘参数
CLIENT_NUM = 4
FORGOTTEN_CLIENT_IDX = 3  # 要遗忘的客户端索引
FORGET_SIZE = 1000        # 固定遗忘样本数
FORGOTTEN_CLASS =1

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
    client_datasets[FORGOTTEN_CLIENT_IDX] = torch.utils.data.Subset(dst_train, remaining_indices)

    # 创建遗忘数据集加载器
    forgotten_dataset = torch.utils.data.Subset(dst_train, fixed_forgotten_indices)
    forgotten_loader = torch.utils.data.DataLoader(
        forgotten_dataset, 
        batch_size=128, 
        shuffle=False
    )
    # 创建客户端加载器（包含更新后的数据集）
    client_loaders = [
        torch.utils.data.DataLoader(
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
    client_datasets[FORGOTTEN_CLIENT_IDX] = torch.utils.data.Subset(dst_train, [])
        

    # 创建遗忘数据集加载器
    forgotten_dataset = torch.utils.data.Subset(dst_train, fixed_forgotten_indices)
    forgotten_loader = torch.utils.data.DataLoader(
        forgotten_dataset, 
        batch_size=128, 
        shuffle=False
    )

    # 创建客户端加载器
    client_loaders = [
    torch.utils.data.DataLoader(
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
    client_datasets[FORGOTTEN_CLIENT_IDX] = torch.utils.data.Subset(dst_train, remaining_indices)

    # 创建遗忘数据集加载器
    forgotten_dataset = torch.utils.data.Subset(dst_train, fixed_forgotten_indices)
    forgotten_loader = torch.utils.data.DataLoader(
        forgotten_dataset, 
        batch_size=128, 
        shuffle=False
    )

    # 客户端加载器
    client_loaders = [
        torch.utils.data.DataLoader(
            ds, 
            batch_size=128, 
            shuffle=True,
            generator=torch.Generator().manual_seed(SEED)
        ) for ds in client_datasets
    ]

aux_dataset= dst_validation


print("aux_dataset 长度:",len(aux_dataset))
print("forgotten_dataset 长度:",len(forgotten_dataset))
batch_size = args.batch_size
train_loader = torch.utils.data.DataLoader(dataset=aux_dataset, batch_size=(batch_size * leak_batch), shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=forgotten_dataset, batch_size=(batch_size * leak_batch), shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("开始实例化全局模型")
# 实例化全样本模型
full_net = resnet20(num_classes).to(device)
criterion = nn.CrossEntropyLoss()  # 交叉熵损失
optimizer_full = torch.optim.Adam(full_net.parameters(), lr=0.001)  # Adam优化器，学习率0.001

unlearned_net = resnet20(num_classes).to(device)
optimizer_unlearned = torch.optim.Adam(unlearned_net.parameters(), lr=0.001)

full_model_path = "/home/ecs-user/fgi/resnet_CIFAR10_sample_federated_full_sample_round_20_partial.pth"
print(f"Found existing full model at '{full_model_path}', loading weights...")
full_net.load_state_dict(torch.load(full_model_path))


unlearned_model_path = "/home/ecs-user/fgi/resnet_CIFAR10_sample_federated_unlearned_round_20_partial.pth"
print(f"Found existing unlearned model at '{unlearned_model_path}', loading weights...")
unlearned_net.load_state_dict(torch.load(unlearned_model_path))

def new_leakage_dataset(images, labels, full_net, unlearn_net):
    full_net.eval()
    unlearn_net.eval()
    criterion = cross_entropy_for_onehot
    batch_size = len(images)
    targets = torch.zeros([batch_size, images.view(batch_size, -1).size()[-1]])
    features = None
    for i, (image, label) in enumerate(zip(images, labels)):
        image = image.unsqueeze(0)
        label = label.unsqueeze(0)
        onehot_label = label_to_onehot(label, num_classes=num_classes)
        image, onehot_label = image.cuda(), onehot_label.cuda()

        pred_full = full_net(image)
        loss_full = criterion(pred_full, onehot_label)
        dy_dx_full = torch.autograd.grad(loss_full, [para for para in full_net.parameters() if para.requires_grad])
        original_dy_dx_full = torch.cat(list((_.detach().clone().view(-1) for _ in dy_dx_full)))

        pred_unlearn = unlearn_net(image)
        loss_unlearn= criterion(pred_unlearn, onehot_label)
        dy_dx_unlearn = torch.autograd.grad(loss_unlearn, [para for para in unlearn_net.parameters() if para.requires_grad])
        original_dy_dx_unlearn = torch.cat(list((_.detach().clone().view(-1) for _ in dy_dx_unlearn)))

        diff_grad = original_dy_dx_full - original_dy_dx_unlearn
        
        targets[i] = image.view(-1)
        if features is None:
                features = torch.zeros([batch_size, len(diff_grad)],device='cuda')
        features[i] = diff_grad
    return features, targets

def leakage_dataset(images, labels, net):
    net.eval()
    criterion = cross_entropy_for_onehot
    batch_size = len(images)
    targets = torch.zeros([batch_size, images.view(batch_size, -1).size()[-1]])
    features = None
    for i, (image, label) in enumerate(zip(images, labels)):
        image = image.unsqueeze(0).cuda()
        label = label.unsqueeze(0).cuda()
        onehot_label = label_to_onehot(label, num_classes=num_classes)
        image, onehot_label = image.cuda(), onehot_label.cuda()
        pred = net(image)
        loss = criterion(pred, onehot_label)
        dy_dx = torch.autograd.grad(loss, [para for para in net.parameters() if para.requires_grad])
        original_dy_dx = torch.cat(list((_.detach().clone().view(-1) for _ in dy_dx)))
        targets[i] = image.view(-1)
        if features is None:
            features = torch.zeros([batch_size, len(original_dy_dx)], device='cuda')
        features[i] = original_dy_dx
    return features, targets

# 计算 SVD 降维矩阵
logger.info("Computing SVD for gradient compression...")
num_samples_for_svd = 5000  # 用于 SVD 的样本数量
svd_loader = torch.utils.data.DataLoader(dataset=aux_dataset, batch_size=num_samples_for_svd, shuffle=True)
data = next(iter(svd_loader))
images, labels = data
# xs, _ = leakage_dataset(images, labels, net)
xs, _ = new_leakage_dataset(images,labels,full_net,unlearned_net)
U, S, V = torch.svd(xs.cpu())  # 对梯度矩阵进行 SVD 分解
k = 5000 # 目标降维维度
V_k = V[:, :k].cuda()  # 降维矩阵，形状为 (model_size, k)
logger.info(f"SVD completed, reduced dimension from {model_size} to {k}")

single_infer = False

if args.model.startswith("MLP"):
    hidden_size = int(args.model.split("-")[-1])
    output_size = image_size if single_infer else image_size * leak_batch
    grad_to_img_net = nn.Sequential(
        nn.Linear(int(k), hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size)
    )
    grad_to_img_net = grad_to_img_net.cuda()


# 定义梯度到图像的模型
# class ConvDecoder(nn.Module):
#     def __init__(self, input_size, output_channels=3, leak_batch=1):
#         super().__init__()
#         self.input_size = input_size
#         self.fc = nn.Linear(input_size, 512 * 4 * 4)  # input_size 应为 k
#         self.decoder = nn.Sequential(
#             nn.Conv2d(512, 512, 3, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#             nn.PixelShuffle(2),  # 8x8
#             nn.Conv2d(128, 256, 3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.PixelShuffle(2),  # 16x16
#             nn.Conv2d(64, 128, 3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.PixelShuffle(2),  # 32x32
#             nn.Conv2d(32, 3 * leak_batch, 3, padding=1),
#             nn.Sigmoid()
#         )
    
#     def forward(self, x):
#         x = self.fc(x)
#         x = x.view(-1, 512, 4, 4)
#         x = self.decoder(x)
#         return x.view(x.size(0), -1)

# grad_to_img_net = ConvDecoder(k, leak_batch=leak_batch).cuda()  # 输入维度为 k

size = 0
for parameters in grad_to_img_net.parameters():
    size += np.prod(parameters.size())
logger.info(f"net size: {size}")

# 训练设置
lr = args.lr
epochs = args.epochs
optimizer = torch.optim.Adam(grad_to_img_net.parameters(), lr=lr)
    
if args.get_validation_rec_first_thou is not None:
    validation_loader = torch.utils.data.DataLoader(dataset=dst_validation, batch_size=(batch_size * leak_batch), shuffle=False)
    checkpoint = torch.load(args.get_validation_rec_first_thou)
    grad_to_img_net.load_state_dict(checkpoint["state_dict"])
    (val_loss, val_acc), reconstructed_imgs = test(grad_to_img_net, net, validation_loader, sign, prune_rate=prune_rate, leak_batch=leak_batch, num_test=4000)
    checkpoint["val_loss"] = val_loss
    checkpoint["val_acc"] = val_acc
    checkpoint["val_reconstructed_imgs"] = reconstructed_imgs
    checkpoint["val_gt_data"] = dst_validation["labels"]
    torch.save(checkpoint, f"{args.get_validation_rec_first_thou}_val")
    exit()
    
if args.resume is not None:
    checkpoint = torch.load(f"checkpoint/{args.resume}")
    grad_to_img_net.load_state_dict(checkpoint["state_dict"])
    
best_test_loss = 1000000
best_state_dict = None
for epoch in tqdm(range(args.epochs)):
    train_loss, train_acc = train(grad_to_img_net, net, train_loader, sign, prune_rate=prune_rate, leak_batch=leak_batch)
    (test_loss, test_acc), reconstructed_imgs = test(grad_to_img_net, net, test_loader, sign, prune_rate=prune_rate, leak_batch=leak_batch)
    grad_to_img_net = grad_to_img_net.cpu()
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        best_state_dict = deepcopy(grad_to_img_net).cpu().state_dict()
    logger.info(f"epoch: {epoch}, train_loss: {train_loss}, train_acc: {train_acc}, test_loss: {test_loss}, test_acc: {test_acc}, best_test_loss: {best_test_loss}")
    if (epoch+1) == int(0.5 * args.epochs):
        for g in optimizer.param_groups:
            g['lr'] *= 0.1


    checkpoint = {}
    checkpoint["train_loss"] = train_loss
    checkpoint["val_loss"] = test_loss
    checkpoint["train_acc"] = train_acc
    checkpoint["val_acc"] = test_acc
    checkpoint["state_dict"] = grad_to_img_net.state_dict()
    checkpoint["best_test_loss"] = best_test_loss
    checkpoint["best_state_dict"] = best_state_dict
    checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if args.dataset.startswith("wikitext") or args.dataset.startswith("cola"):
        checkpoint["val_reconstructed_imgs"] = reconstructed_imgs
        checkpoint["gt_data"] = dst_validation["labels"]
    elif args.dataset.startswith("CIFAR10") or args.dataset.startswith("CIFAR100"):
        checkpoint["reconstructed_imgs"] = reconstructed_imgs[0]
        checkpoint["gt_data"] = reconstructed_imgs[1]
    checkpoint["epoch"] = epoch
    torch.save(checkpoint, f"{save_dir}/checkpoint/{save_file_name}_version1.pt")
    grad_to_img_net = grad_to_img_net.cuda()