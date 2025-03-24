# Gradient-Inversion-Attacks-in-Federated-Unlearning
The repositorie is the code of Gradient Inversion Attacks in Federated Unlearning

####  Create and activate environment.
```python
conda create -n attackfu python=3.10
conda activate attackfu
```

####  Install the required repository.

```python
pip install -r requirements.txt
```

```python
cd Gradient-Inversion-Attacks-in-Federated-Unlearning
```


## 1.Generate the federared global model

```python
python training.py --model mnist_resnet20 --dataset MNIST --type sample --unlearning retrain --aggregation fedavg
```

```python
option:
--dataset CIFAR10, CIFAR100, MNIST, FashionMNIST
--model lenet, lenetmnist, resnet20, mnist_resnet20
--type sample, class, client
--unlearning retrain, efficient
--aggregation fedavg, fedprox, fedopt
```

## 2.Importing model to attack
find the definition of ```full_net``` and ```unlearn_net```; import the path of federated model weight

```python
python main.py --lr 1e-4 --epochs 30 --leak_mode none --dataset CIFAR10 --batch_size 256 --shared_model LeNet --type sample --unlearning retrain --state attack
```

```python
option:
--dataset CIFAR10, CIFAR100, MNIST, FashionMNIST
--leak_mode sign/prune-{prune_rate}/batch-{batch_size}/perturb-0.01/smooth-0.1
--shared_model LeNet
--type sample, class, client
--unlearning retrain, efficient
--state attack, defense
```

```python
python main.py --lr 1e-4 --epochs 30 --leak_mode perturb-0.01 --dataset CIFAR10 --batch_size 256 --shared_model Resnet20 --type sample --unlearning retrain
```