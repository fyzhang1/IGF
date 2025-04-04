import argparse
import numpy as np
from tqdm import tqdm
import math
from copy import deepcopy
import os
os.environ['KMP_WARNINGS'] = '0'
# import cPickle as pickle
import pickle
import joblib
from pytorch_msssim import ssim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import models, datasets, transforms
from scipy.optimize import linear_sum_assignment
from scipy.fftpack import dct, idct
from scipy import stats
from utils_com.utils import label_to_onehot, cross_entropy_for_onehot
from models.vision import LeNetMnist, weights_init, LeNet
from models.resnet import resnet20
from utils_com.logger import set_logger
import random
import lpips
import matplotlib.pyplot as plt
"""
参数定义
"""

parser = argparse.ArgumentParser(description='Deep Leakage from Gradients.')
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
parser.add_argument('--leak_mode', type=str, default="none",
                    help='sign/prune-{prune_rate}/batch-{batch_size}')
parser.add_argument('--trainset', type=str, default="full")
args = parser.parse_args()
logger = set_logger("", f"print_logs/{args.dataset}_{args.shared_model}_{args.model}_{args.leak_mode}_{args.lr}_{args.epochs}_{args.batch_size}.txt")
logger.info(args)




# 按类别分组
def get_class_samples(dataset, num_samples_per_class=10):
    class_samples = {}
    for i in range(len(dataset)):
        _, label = dataset[i]  # 每个样本返回 (image, label)
        if label not in class_samples:
            class_samples[label] = []
        class_samples[label].append(i)
    
    selected_indices = []
    for label, indices in class_samples.items():
        if len(indices) >= num_samples_per_class:
            # 随机选择指定数量的样本
            selected_indices += random.sample(indices, num_samples_per_class)
    
    return selected_indices



"""
反演模型训练时的输入是全局模型在辅助数据集上的梯度然后与辅助数据集样本优化; 产生一种梯度与数据的映射关系
之后test的过程中再将全局模型的梯度作为输入去反演出数据
"""

def train(grad_to_img_net, data_loader, sign=False, mask=None, prune_rate=None, leak_batch=1):
    grad_to_img_net.train()
    total_loss = 0
    total_num = 0
    for i, (xs, ys) in enumerate(tqdm(data_loader)):
        optimizer.zero_grad()
        batch_num = len(ys)
        batch_size = int(batch_num / leak_batch)
        batch_num = batch_size * leak_batch
        total_num += batch_num
        xs, ys = xs[:batch_num, selected_para].cuda(), ys[:batch_num].cuda()
        vgg = models.vgg16(pretrained=True).features.eval().to('cuda')
        
        if sign:
            xs = torch.sign(xs)
        if prune_rate is not None:
            mask = torch.zeros(xs.size()).cuda()
            rank = torch.argsort(xs.abs(), dim=1)[:,  -int(xs.size()[1] * (1 - prune_rate)):]
            mask[torch.arange(len(ys)).view(-1, 1).expand(rank.size()), rank] = 1   
        if mask is not None:
            xs = xs * mask
        if gauss_noise > 0:
            xs = xs + torch.randn(*xs.shape).cuda() * gauss_noise

        xs = xs.view(batch_size, leak_batch, -1).mean(1)
        ys = ys.view(batch_size, leak_batch, image_size)
        preds = grad_to_img_net(xs).view(batch_size, leak_batch, image_size)
        
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
        # # Compute perceptual loss
        # perceptual_loss = loss_fn_vgg(real_images_matched, reconstructed_images_matched).mean()

        real_images_matched = real_images_matched * 2 - 1   # [0,1] -> [-1,1]
        reconstructed_images_matched = reconstructed_images_matched * 2 - 1
        perceptual_loss = loss_fn_vgg(real_images_matched, reconstructed_images_matched).mean()

        
        # Total loss
        total_loss = mse_loss + 0.1 * perceptual_loss

        print("perceptual_loss:", perceptual_loss)
        
        total_loss.backward()
        optimizer.step()
        total_loss_value = total_loss.item() * batch_num
        total_loss += total_loss_value
            
    total_loss = total_loss / len(data_loader.dataset)
    return total_loss


#感知损失+MSE损失+SSIM损失
# def train(grad_to_img_net, data_loader, sign=False, mask=None, prune_rate=None, leak_batch=1):
#     grad_to_img_net.train()
#     total_loss = 0
#     total_num = 0
#     for i, (xs, ys) in enumerate(tqdm(data_loader)):
#         optimizer.zero_grad()
#         batch_num = len(ys)
#         batch_size = int(batch_num / leak_batch)
#         batch_num = batch_size * leak_batch
#         total_num += batch_num
#         xs, ys = xs[:batch_num, selected_para].cuda(), ys[:batch_num].cuda()
#         vgg = models.vgg16(pretrained=True).features.eval().to('cuda')
        
#         if sign:
#             xs = torch.sign(xs)
#         if prune_rate is not None:
#             mask = torch.zeros(xs.size()).cuda()
#             rank = torch.argsort(xs.abs(), dim=1)[:, -int(xs.size()[1] * (1 - prune_rate)):]
#             mask[torch.arange(len(ys)).view(-1, 1).expand(rank.size()), rank] = 1   
#         if mask is not None:
#             xs = xs * mask
#         if gauss_noise > 0:
#             xs = xs + torch.randn(*xs.shape).cuda() * gauss_noise

#         xs = xs.view(batch_size, leak_batch, -1).mean(1)
#         ys = ys.view(batch_size, leak_batch, image_size)
#         preds = grad_to_img_net(xs).view(batch_size, leak_batch, image_size)
        
#         # Compute MSE loss with matching per sample
#         mse_loss = 0
#         matched_reconstructed = []
#         matched_real = []
#         for sample_id in range(batch_size):
#             ys_sample = ys[sample_id]
#             preds_sample = preds[sample_id]
#             distance_matrix = torch.cdist(ys_sample, preds_sample)
#             mse_mat = (distance_matrix ** 2) / image_size
#             row_ind, col_ind = linear_sum_assignment(mse_mat.detach().cpu().numpy())
#             mse_loss_sample = mse_mat[row_ind, col_ind].mean()
#             mse_loss += mse_loss_sample
#             # Collect matched reconstructed and real images
#             matched_reconstructed.append(preds_sample[col_ind])
#             matched_real.append(ys_sample[row_ind])
        
#         mse_loss /= batch_size
        
#         # Prepare matched images for perceptual loss and SSIM
#         matched_reconstructed_all = torch.stack(matched_reconstructed, dim=0).view(batch_size * leak_batch, image_size)
#         matched_real_all = torch.stack(matched_real, dim=0).view(batch_size * leak_batch, image_size)
#         reconstructed_images_matched = matched_reconstructed_all.view(batch_size * leak_batch, 3, 32, 32).to('cuda')
#         real_images_matched = matched_real_all.view(batch_size * leak_batch, 3, 32, 32).to('cuda')
        
#         # Compute perceptual loss
#         loss_fn_vgg = lpips.LPIPS(net='vgg').to('cuda')
#         real_images_matched_perc = real_images_matched * 2 - 1  # [0,1] -> [-1,1] for perceptual loss
#         reconstructed_images_matched_perc = reconstructed_images_matched * 2 - 1
#         perceptual_loss = loss_fn_vgg(real_images_matched_perc, reconstructed_images_matched_perc).mean()
        
#         # Compute SSIM loss
#         # SSIM expects images in [0,1] range, which matches reconstructed_images_matched and real_images_matched
#         ssim_value = ssim(real_images_matched, reconstructed_images_matched, data_range=1.0, size_average=True)
#         ssim_loss = 1 - ssim_value  # Convert SSIM to a loss term
        
#         # Total loss: Combine MSE, perceptual, and SSIM losses
#         total_loss = mse_loss + 0.1 * perceptual_loss + 0.1 * ssim_loss  # Weights can be adjusted
        
#         print("perceptual_loss:", perceptual_loss.item(), "ssim_loss:", ssim_loss.item())
        
#         total_loss.backward()
#         optimizer.step()
#         total_loss_value = total_loss.item() * batch_num
#         total_loss += total_loss_value
            
#     total_loss = total_loss / len(data_loader.dataset)
#     return total_loss

def test(grad_to_img_net, data_loader, sign=False, mask=None, prune_rate=None, leak_batch=1):
    grad_to_img_net.eval()
    total_loss = 0
    total_num = 0
    reconstructed_data = []
    with torch.no_grad():
        for i, (xs, ys) in enumerate(tqdm(data_loader)):
            batch_num = len(ys)
            batch_size = int(batch_num / leak_batch)
            batch_num = batch_size * leak_batch
            total_num += batch_num
            xs, ys = xs[:batch_num, selected_para].cuda(), ys[:batch_num].cuda()
            # if sign:
            #     xs = torch.sign(xs)
            # if prune_rate is not None:
            #     mask = torch.zeros(xs.size()).cuda()
            #     rank = torch.argsort(xs.abs(), dim=1)[:,  -int(xs.size()[1] * (1 - prune_rate)):]
            #     mask[torch.arange(len(ys)).view(-1, 1).expand(rank.size()), rank] = 1   
            # if mask is not None:
            #     xs = xs * mask
            # if gauss_noise > 0:
            #     xs = xs + torch.randn(*xs.shape).cuda() * gauss_noise
            xs = xs.view(batch_size, leak_batch, -1).mean(1)
            ys = ys.view(batch_size, leak_batch, -1)
            preds = grad_to_img_net(xs).view(batch_size, leak_batch, -1)
            batch_wise_mse = (torch.cdist(ys, preds) ** 2) / image_size
            loss = 0
            for batch_id, mse_mat in enumerate(batch_wise_mse):
                row_ind, col_ind = linear_sum_assignment(mse_mat.detach().cpu().numpy())
                loss += mse_mat[row_ind, col_ind].sum()

                #save the reconstructed data in order
                sorted_preds = preds[batch_id, col_ind]
                sorted_preds[row_ind] = preds[batch_id, col_ind]
                sorted_preds = sorted_preds.view(leak_batch, -1).detach().cpu()
                reconstructed_data.append(sorted_preds)
            total_loss += loss.item()
            
    reconstructed_data = torch.cat(reconstructed_data)
    reconstructed_data = reconstructed_data.view(-1, 3, 32, 32)
    total_loss = total_loss / total_num
    return total_loss, reconstructed_data

#input the model shared among parties
image_size = 3 * 32 * 32
num_classes = 10


"""
全局模型定义LeNet (net)
"""

if args.shared_model == "LeNet":
    net = LeNet(num_classes).to("cuda")
    compress_rate = 1.0
    torch.manual_seed(1234)
    net.apply(weights_init)
    criterion = cross_entropy_for_onehot
model_size = 0
for i, parameters in enumerate(net.parameters()):
    model_size += np.prod(parameters.size())
logger.info(f"model size: {model_size}")

# net.load_state_dict(torch.load("cifar_lenet.pth"))


"""
训练和测试集定义
"""
if args.trainset == "full":
    checkpoint_name = f"data/{args.dataset}_{args.shared_model}_grad_to_img.pl"
else:
    checkpoint_name = f"data/{args.dataset}_{args.shared_model}_{args.trainset}_grad_to_img.pl"
if not os.path.exists(checkpoint_name):
    print("generating dataset...")
    if args.dataset == "MNIST":
        transform = transforms.Compose([
                    transforms.ToTensor(),
                    ])
        dst_train = datasets.MNIST("~/.torch", download=True, train=True, transform=transform)
        dst_test = datasets.MNIST("~/.torch", download=True, train=False, transform=transform)
    elif args.dataset == "FashionMNIST":
        transform = transforms.Compose([
                    transforms.ToTensor(),
                    ])
        dst_train = datasets.FashionMNIST("~/.torch", download=True, train=True, transform=transform)
        dst_test = datasets.FashionMNIST("~/.torch", download=True, train=False, transform=transform)
    elif args.dataset == "CIFAR10":
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
        ])
        dst_train = datasets.CIFAR10(root="~/.torch", train=True, download=True, transform=transform)
        dst_test = datasets.CIFAR10("~/.torch", download=True, train=False, transform=transform)

        SEED = 42
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        # 定义客户端数量和遗忘参数
        CLIENT_NUM = 4
        FORGOTTEN_CLIENT_IDX = 3  # 要遗忘的客户端索引
        FORGET_SIZE = 1000       # 固定遗忘样本数
    
    print("加载fedrated learning和fedrated unlearning数据集")

    # 固定划分客户端数据（使用确定性的随机划分）
    client_datasets = torch.utils.data.random_split(
        dst_train,
        [len(dst_train)//CLIENT_NUM]*CLIENT_NUM,
        generator=torch.Generator().manual_seed(SEED)  # 固定划分随机种子
    )
    
    # 获取目标客户端原始数据索引
    target_dataset = client_datasets[FORGOTTEN_CLIENT_IDX]
    original_indices = target_dataset.indices.copy()  # 原始索引列表

    # 确定性地选择前N个样本作为遗忘集（方法1：绝对位置）
    fixed_forgotten_indices = sorted(original_indices)[:FORGET_SIZE]  # 按原始顺序取前1000

    # 更新客户端数据集划分
    remaining_indices = list(set(original_indices) - set(fixed_forgotten_indices))
    client_datasets[FORGOTTEN_CLIENT_IDX] = torch.utils.data.Subset(dst_train, remaining_indices)

    # 创建遗忘数据集加载器
    forgotten_dataset = torch.utils.data.Subset(dst_train, fixed_forgotten_indices)
    forgotten_loader = torch.utils.data.DataLoader(
        forgotten_dataset, 
        batch_size=32, 
        shuffle=False
    )

    # 创建客户端加载器（包含更新后的数据集）
    client_loaders = [
        torch.utils.data.DataLoader(
            ds, 
            batch_size=32, 
            shuffle=True,  # 训练时保持shuffle但随机种子固定
            generator=torch.Generator().manual_seed(SEED))
        for ds in client_datasets ]

    test_forgotten_loader = torch.utils.data.DataLoader(
        forgotten_dataset, 
        batch_size=1, 
        shuffle=False
    )

    aux_loader = torch.utils.data.DataLoader(dst_test, batch_size=1, shuffle=False)  # 测试集

    # selected_indices = get_class_samples(dst_test, num_samples_per_class=300)

    # # 创建新的数据集
    # custom_subset = torch.utils.data.Subset(dst_test, selected_indices)

    # # 创建新的 DataLoader
    # aux_loader = torch.utils.data.DataLoader(custom_subset, batch_size=1, shuffle=False)
    
    # 训练反演模型用 aux_loader
    # 测试用 test_forgotten_loader

    # print("print dataset")
    # print(len(fed_loader_full.dataset))         # 10000
    # print(len(fed_loader_unlearned.dataset))    # 9900
    # print(len(aux_loader.dataset))              # 50000
    # print(len(forgotten_loader.dataset))        # 100

    def leakage_dataset(data_loader, full_net, unlearned_net, criterion, is_forgotten=False):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        full_net.eval()
        unlearned_net.eval()
        
        # Initialize tensors for features and targets
        features = torch.zeros([len(data_loader.dataset), model_size], device=device)
        targets = torch.zeros([len(data_loader.dataset), image_size], device=device)

        for i, (images, labels) in enumerate(tqdm(data_loader)):
            onehot_labels = label_to_onehot(labels, num_classes=10)
            images, onehot_labels = images.to(device), onehot_labels.to(device)
            
            # Calculate gradients for the full model
            pred_full = full_net(images)
            loss_full = criterion(pred_full, onehot_labels)
            dy_dx_full = torch.autograd.grad(loss_full, full_net.parameters(), create_graph=False)
            grad_full = torch.cat([g.detach().view(-1) for g in dy_dx_full])

            # Calculate gradients for the unlearned model
            pred_unlearned = unlearned_net(images)
            loss_unlearned = criterion(pred_unlearned, onehot_labels)
            dy_dx_unlearned = torch.autograd.grad(loss_unlearned, unlearned_net.parameters(), create_graph=False)
            grad_unlearned = torch.cat([g.detach().view(-1) for g in dy_dx_unlearned])

            # Compute the difference between gradients
            diff_grad = grad_full - grad_unlearned

            # Store the difference gradient (features) and the original image (targets)
            features[i] = diff_grad
            targets[i] = images.view(-1)

        return features, targets


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("开始实例化全局模型")
    # 实例化全样本模型
    full_net = LeNet(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失
    optimizer_full = torch.optim.Adam(full_net.parameters(), lr=0.001)  # Adam优化器，学习率0.001

    unlearned_net = LeNet(num_classes=10).to(device)
    optimizer_unlearned = torch.optim.Adam(unlearned_net.parameters(), lr=0.001)

    full_model_path = "/home/ecs-user/fgi/federated_weight/federated_full_sample_1000_round_20_partial.pth"
    print(f"Found existing full model at '{full_model_path}', loading weights...")
    full_net.load_state_dict(torch.load(full_model_path))


    unlearned_model_path = "/home/ecs-user/fgi/federated_weight/efficient/efficient_federated_unlearned_gradient_sample_1000_round_20.pth"
    print(f"Found existing unlearned model at '{unlearned_model_path}', loading weights...")
    unlearned_net.load_state_dict(torch.load(unlearned_model_path))


    checkpoint = {}
    # 生成训练数据（被遗忘样本的差异梯度）
    print("Generating training leakage dataset...")
    # features, targets = leakage_dataset(forgotten_loader)

    features, targets = leakage_dataset(aux_loader, full_net, unlearned_net, criterion, is_forgotten=False)
    checkpoint["train_features"] = features
    checkpoint["train_targets"] = targets


    # 生成测试数据（测试集的差异梯度）
    print("Generating testing leakage dataset...")

    features, targets = leakage_dataset(test_forgotten_loader, full_net, unlearned_net, criterion, is_forgotten=False)
    checkpoint["test_features"] = features
    checkpoint["test_targets"] = targets
    torch.save(checkpoint, checkpoint_name)
else:
    checkpoint = torch.load(checkpoint_name)
del net
    
    
print("loading dataset...")
trainset = torch.utils.data.TensorDataset(checkpoint["train_features"], checkpoint["train_targets"])

testset = torch.utils.data.TensorDataset(checkpoint["test_features"], checkpoint["test_targets"])





"""
模式定义; 这里我们不需要, 我们设置为none
"""
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
print(prune_rate, leak_batch, sign, gauss_noise)


"""
反演模型定义
"""
torch.manual_seed(0)
selected_para = torch.randperm(model_size)[:int(model_size * compress_rate)]
# if args.model.startswith("MLP"):
#     print(image_size)
#     hidden_size = int(args.model.split("-")[-1])
#     grad_to_img_net = nn.Sequential(
#         nn.Linear(len(selected_para), hidden_size),
#         nn.ReLU(),
#         nn.Linear(hidden_size, hidden_size),
#         nn.ReLU(),
#         nn.Linear(hidden_size, image_size * leak_batch),
#         torch.nn.Sigmoid()
#     )
#     grad_to_img_net = grad_to_img_net.cuda()

class ConvDecoder(nn.Module):
    def __init__(self, input_size, output_channels=3, leak_batch=1):
        super().__init__()
        self.input_size = input_size
        self.fc = nn.Linear(input_size, 512 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.PixelShuffle(2),  # 8x8
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.PixelShuffle(2),  # 16x16
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.PixelShuffle(2),  # 32x32
            
            nn.Conv2d(32, 3 * leak_batch, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 512, 4, 4)
        x = self.decoder(x)
        return x.view(x.size(0), -1)

grad_to_img_net = ConvDecoder(len(selected_para), leak_batch=leak_batch).cuda()

size = 0
for parameters in grad_to_img_net.parameters():
    size += np.prod(parameters.size())
print(f"net size: {size}")



"""
训练中参数的一些定义
"""
lr = args.lr
epochs = args.epochs
optimizer = torch.optim.Adam(grad_to_img_net.parameters(), lr=lr)
# optimizer = torch.optim.Adam(grad_to_img_net.parameters(), lr=lr, weight_decay=1e-5)

    
"""
加载训练和测试集
"""

batch_size = args.batch_size
# train_loader = torch.utils.data.DataLoader(dataset=trainset,
#                                               batch_size=(batch_size * leak_batch), 
#                                               shuffle=True)
# test_loader = torch.utils.data.DataLoader(dataset=testset,
#                                               batch_size=(batch_size * leak_batch), 
#                                               shuffle=False)

train_loader_inversion = torch.utils.data.DataLoader(trainset, batch_size=(batch_size * leak_batch), shuffle=True)

test_loader_inversion = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
#reformate the gt_data
gt_data = checkpoint["test_targets"]
gt_data = gt_data.view(-1, 3, 32, 32)
del checkpoint

#learning
if args.trainset == "full":
    checkpoint_name = f"checkpoint/{args.dataset}_{args.shared_model}_{args.model}_{args.leak_mode}_{args.lr}_{args.epochs}_{args.batch_size}.pt"
else:
    checkpoint_name = f"checkpoint/{args.dataset}_{args.trainset}_{args.shared_model}_{args.model}_{args.leak_mode}_{args.lr}_{args.epochs}_{args.batch_size}.pt"
    
if os.path.exists(checkpoint_name):
    checkpoint = torch.load(checkpoint_name)
    print("checkpoint exists!")
    print(checkpoint["test_loss"], checkpoint["best_test_loss"])
    exit()




best_test_loss = 10000
best_state_dict = None

"""
训练反演模型主函数
"""

for epoch in tqdm(range(args.epochs)):
    train_loss = train(grad_to_img_net, train_loader_inversion, sign, prune_rate=prune_rate, leak_batch=leak_batch)
    test_loss, reconstructed_imgs = test(grad_to_img_net, test_loader_inversion, sign, prune_rate=prune_rate, leak_batch=leak_batch)



    
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        best_state_dict = deepcopy(grad_to_img_net.to("cpu").state_dict())
    grad_to_img_net.to("cuda")
    logger.info(f"epoch: {epoch}, train_loss: {train_loss}, test_loss: {test_loss}, best_test_loss: {best_test_loss}")
    if (epoch+1)%100 == 0:
        checkpoint = {}
        checkpoint["train_loss"] = train_loss
        checkpoint["test_loss"] = test_loss
        checkpoint["reconstructed_imgs"] = reconstructed_imgs
        checkpoint["gt_data"] = gt_data
        if args.trainset == "full":
            torch.save(checkpoint, f"checkpoint/{args.dataset}_{args.shared_model}_{args.model}_{args.leak_mode}_{args.lr}_{args.epochs}_{args.batch_size}_best.pt")
        else:
            torch.save(checkpoint, f"checkpoint/{args.dataset}_{args.trainset}_{args.shared_model}_{args.model}_{args.leak_mode}_{args.lr}_{args.epochs}_{args.batch_size}.pt")
    if (epoch+1) == int(0.75 * args.epochs):
        for g in optimizer.param_groups:
            g['lr'] *= 0.1
checkpoint = {}
checkpoint["train_loss"] = train_loss
checkpoint["test_loss"] = test_loss
checkpoint["state_dict"] = grad_to_img_net.state_dict()
checkpoint["best_test_loss"] = best_test_loss
checkpoint["best_state_dict"] = best_state_dict
checkpoint["reconstructed_imgs"] = reconstructed_imgs
checkpoint["gt_data"] = gt_data

import os

# Create the 'checkpoint' directory if it doesn't exist
os.makedirs("checkpoint", exist_ok=True)

if args.trainset == "full":
    torch.save(checkpoint, f"checkpoint/{args.dataset}_{args.shared_model}_{args.model}_{args.leak_mode}_{args.lr}_{args.epochs}_{args.batch_size}.pt")
else:
    torch.save(checkpoint, f"checkpoint/{args.dataset}_{args.trainset}_{args.shared_model}_{args.model}_{args.leak_mode}_{args.lr}_{args.epochs}_{args.batch_size}.pt")