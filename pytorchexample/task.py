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
from .models.xception import Xception


class Net(Xception):
    def __init__(self) -> None:
        super(Net, self).__init__(num_classes=8) #Should run the 2048 dimension Xception architecture, for 8 classes

    def forward(self, x):
        #forward logic is found in xception.py
        return super().forward(x)

fds = None  # Cache FederatedDataset

def apply_train_transforms(batch):
    """Apply FLamby-style train transforms."""
    size = 299 #I changed 200 to 299
    train_transforms = albumentations.Compose(
        [
            albumentations.Resize(333, 333), # Ensure the image is large enough
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
    size = 299 #Changed 200 to 299
    test_transforms = albumentations.Compose(
        [
            albumentations.Resize(333, 333),
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


def load_centralized_dataset(batch_size: int = 16): #I changed batch_size from 128 to 16
    """Load the full centralized test set for server-side evaluation."""
    test_dataset = load_dataset("flwrlabs/fed-isic2019", split="test")
    test_dataset = test_dataset.with_transform(apply_test_transforms)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    net.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    net.train()
    running_loss = 0.0

    for _ in range(epochs):
        print("WE ARE AT LEAST TRAINING")
        #print(f"Starting Epoch{epochs}")
        for batch in trainloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

    avg_train_loss = running_loss / (epochs * len(trainloader))
    return avg_train_loss


def test(net, testloader, device):
    """Evaluate the model on the test set."""
    net.to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    net.eval()
    correct = 0
    loss = 0.0

    with torch.no_grad():
        for batch in testloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()

    avg_loss = loss / len(testloader)
    accuracy = correct / len(testloader.dataset)
    return avg_loss, accuracy