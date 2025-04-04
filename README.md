# A Crisis of Curiosity: Gradient Inversion Attacks in Federated Unlearning

The repositorie is the code of "A Crisis of Curiosity: Gradient Inversion Attacks in Federated Unlearning".

##  Create and activate environment
```python
conda create -n attackfu python=3.10
conda activate attackfu
```

##  Install the required repository

```python
pip install -r requirements.txt
cd GIAFU
```

##  Datasets
- CIFAR10
- CIFAR100
- MNIST
- FashionMNIST



## Generate Federate Model

```python
python training.py --model lenet --dataset CIFAR10 --type sample --unlearning retrain --aggregation fedavg
```

```python
option:
--dataset CIFAR10, CIFAR100, MNIST, FashionMNIST
--model lenet, lenetmnist, resnet20, mnist_resnet20
--type sample, class, client
--unlearning retrain, efficient
--aggregation fedavg, fedprox, fedopt
```

## Quick Start

```python
python main.py --lr 1e-4 --epochs 30 --leak_mode none --dataset CIFAR10 --batch_size 256 --shared_model LeNet --type sample --unlearning retrain --state attack
```