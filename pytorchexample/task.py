"""pytorchexample: A Flower / PyTorch app."""

import random
import numpy as np
import albumentations

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import NaturalIdPartitioner
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score


class Net(nn.Module):
    def __init__(self): 
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(84, 8)  #  classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.adaptive_pool(x)

        x = x.view(-1, 16 * 4 * 4)

        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))  #Doing this so we get fc2 output after ReLU
        return self.fc3(x)

#model = Net()

fds = None  # Cache FederatedDataset

def apply_train_transforms(batch):
    """Apply FLamby-style train transforms."""
    size = 200
    train_transforms = albumentations.Compose(
        [
            albumentations.RandomScale(scale_limit=0.07, p=1.0),
            albumentations.Rotate(limit=50, p=1.0),
            albumentations.RandomBrightnessContrast(
                brightness_limit=0.15, contrast_limit=0.10, p=1.0
            ),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.Affine(shear=0.1, p=1.0),
            albumentations.RandomCrop(height=size, width=size, p=1.0),
            albumentations.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(16, 16),
                hole_width_range=(16, 16),
                p=1.0,
            ),
            albumentations.Normalize(p=1.0),
        ]
    )

    images = []
    for image in batch["image"]:
        augmented = train_transforms(image=np.array(image))["image"]
        transposed = np.transpose(augmented, (2, 0, 1)).astype(np.float32)
        images.append(torch.tensor(transposed, dtype=torch.float32))

    batch["image"] = images
    return batch


def apply_test_transforms(batch):
    """Apply FLamby-style test transforms."""
    size = 200
    test_transforms = albumentations.Compose(
        [
            albumentations.CenterCrop(height=size, width=size, p=1.0),
            albumentations.Normalize(p=1.0),
        ]
    )

    images = []
    for image in batch["image"]:
        augmented = test_transforms(image=np.array(image))["image"]
        transposed = np.transpose(augmented, (2, 0, 1)).astype(np.float32)
        images.append(torch.tensor(transposed, dtype=torch.float32))

    batch["image"] = images
    return batch


def load_data(partition_id: int, num_partitions: int, batch_size: int):
    """Load one client's train and test data."""

    global fds
    if fds is None:
        fds = FederatedDataset(
            dataset="flwrlabs/fed-isic2019",
            partitioners={
                "train": NaturalIdPartitioner(partition_by="center"),
                "test": NaturalIdPartitioner(partition_by="center"),
            },
        )
        

    train_partition = fds.load_partition(partition_id=partition_id, split="train")
    test_partition = fds.load_partition(partition_id=partition_id, split="test")

    train_partition = train_partition.with_transform(apply_train_transforms)
    test_partition = test_partition.with_transform(apply_test_transforms)

    trainloader = DataLoader(train_partition, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_partition, batch_size=batch_size, shuffle=False)

    return trainloader, testloader


def load_centralized_dataset(batch_size: int = 128):
    """Load the full centralized test set for server-side evaluation."""
    test_dataset = load_dataset("flwrlabs/fed-isic2019", split="test")
    test_dataset = test_dataset.with_transform(apply_test_transforms)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set (standard FedAvg / FedProx)."""
    net.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
 
    net.train()
    running_loss = 0.0
    correct = 0
 
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
 
            optimizer.zero_grad()
            outputs = net(images.float())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
 
            running_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
 
    avg_train_loss = running_loss / (epochs * len(trainloader))
    accuracy = correct / len(trainloader.dataset)
    return avg_train_loss, accuracy

def scaffold_train(net, trainloader, epochs, lr, device, global_cv, local_cv):
    net.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.0)
 
    # Save initial global parameters
    init_global_params: dict[str, torch.Tensor] = {
        key: value.detach().clone() for key, value in net.state_dict().items()
        }

    net.train()
    running_loss = 0.
    correct = 0
    num_steps = 0
 
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
 
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()

            #scaffold gradient correction
            with torch.no_grad():
                for name, param in net.named_parameters():
                    if param.grad is None:
                        continue
                    if name not in local_cv or name not in global_cv:
                        continue
                    param.grad.data.add_(                                       #add correction term to original gradient
                        global_cv[name].to(device) - local_cv[name].to(device)  #subtract client bias
                    )

            optimizer.step()
 
            running_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            num_steps += 1
 
    avg_train_loss = running_loss / (epochs * len(trainloader))
    accuracy = correct / len(trainloader.dataset)

    #update local model
    updated_model: dict[str, torch.Tensor] = {
        key: value.detach().clone() for key, value in net.state_dict().items()
        }
    
    total_steps = max(num_steps, 1)
    scaling_factor = 1.0 / (total_steps * lr)

    #compute new local control variate
    new_local_cv: dict[str, torch.Tensor] = {}
    cv_diff: dict[str, torch.Tensor] = {}

    
    with torch.no_grad():
        for key in init_global_params:
            # 1. Calculate drift on the active training device (GPU/CPU)
            client_drift = init_global_params[key] - updated_model[key]                         
            
            # 2. Explicitly move control variates to the same device for the math
            new_client_cv = local_cv[key].to(device) - global_cv[key].to(device) + scaling_factor * client_drift      
            
            # 3. Bring them back to the CPU so Flower can package them safely
            new_local_cv[key] = new_client_cv.cpu()                                                   
            cv_diff[key] = (new_client_cv - local_cv[key].to(device)).cpu()                                        

    return avg_train_loss, accuracy, net, new_local_cv, cv_diff

def test(net, testloader, device, global_eval=False):
    """Evaluate the model on the test set."""
    net.to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    net.eval()
    correct = 0
    loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in testloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = net(images.float())
            loss += criterion(outputs, labels).item()
            
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = loss / len(testloader)
    accuracy = correct / len(testloader.dataset)

    if global_eval:
        precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
        recall    = recall_score(   all_labels, all_preds, average="weighted", zero_division=0)
        f1        = f1_score(       all_labels, all_preds, average="weighted", zero_division=0)
        support   = np.bincount(all_labels).tolist()
        return avg_loss, accuracy, precision, recall, f1, support
    else:
        return avg_loss, accuracy