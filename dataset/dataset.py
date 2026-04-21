import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import albumentations
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import NaturalIdPartitioner
from flwr_datasets.visualization import plot_label_distributions

#Constants
SIZE_IMG = 299


class FedISIC2019_Dataset():
    fds = None
    labels = None
    labels_for_the_labels = ["mel", "mel-nev", "bcc", "ak", "bk", "df", "vl", "scc"]
    seed = None

    def __init__(self, seed: int):
        self.seed = seed
        #Loading Fed-ISIC2019 via flwr_datasets
        self.fds = FederatedDataset(
            dataset="flwrlabs/fed-isic2019", 
            partitioners={
                "train" : NaturalIdPartitioner(partition_by="center"),
                "test" : NaturalIdPartitioner(partition_by="center")
            }
        )

    def centralized_dataset(self):
        full_train = self.fds.load_split("train")
        full_test = self.fds.load_split("test")

        self.labels = full_train.features["label"].names
        return full_train, full_test
    
    def plot_centralized_train_class_distribution(self):
        full_train, _ = self.centralized_dataset()
        label_counter = [0, 0, 0, 0, 0, 0, 0, 0]
        
        for sample in full_train:
            label_on_sample = sample["label"]
            label_counter[label_on_sample] += 1
        
        colors = ['red', 'gold', 'limegreen', 'dodgerblue', 'orange', 'violet', 'tomato', 'teal']
        label_bars = plt.bar(self.labels_for_the_labels, label_counter, color=colors)
        plt.title("Label distribution for the centralized Fed-ISIC2019")
        plt.xlabel("labels_for_the_labels")
        plt.ylabel("Number of Images")
        plt.bar_label(label_bars, labels=label_counter, padding=5)
        plt.show()

        return
    
    def plot_in_partitions_train_class_distribution(self):
        partitioner = self.fds.partitioners["train"]

        #labels_for_labels to plot compatible dict
        lfl_dict = {
                    0: self.labels_for_the_labels[0], 
                    1: self.labels_for_the_labels[1],
                    2: self.labels_for_the_labels[2],
                    3: self.labels_for_the_labels[3],
                    4: self.labels_for_the_labels[4],
                    5: self.labels_for_the_labels[5],
                    6: self.labels_for_the_labels[6],
                    7: self.labels_for_the_labels[7]
                }

        fig, ax, df = plot_label_distributions(
            partitioner,
            label_name="label",
            plot_type="bar",
            size_unit="absolute",
            partition_id_axis="x",
            legend=True,
            verbose_labels=True,
            title="Per Partition Labels Distribution"
        )

        #Adding sample counts per partitions to the figure
        partitions_sample_count_list = []
        for i_partition in range(6):
            partition_number_of_samples = 0

            for j_label in range(8):
                partition_number_of_samples += int(df.loc[i_partition, str(j_label)])
            
            partitions_sample_count_list.append(partition_number_of_samples)
            partition_number_of_samples = 0
        
        for i_bar, partition_total in enumerate(partitions_sample_count_list):
            
            ax.text(
                i_bar,
                partition_total,
                str(partition_total),
                fontsize=10,
                ha="center",
                va="bottom"
            )

        plt.show()

        return
    
    def apply_oversampling_train_transform(self, pil_img):
        if self.seed != None:
            np.random.seed(self.seed)
        
        transform = albumentations.Compose([
            albumentations.PadIfNeeded(min_height=SIZE_IMG, min_width=SIZE_IMG, border_mode=0, value=0),
            albumentations.CenterCrop(height=SIZE_IMG, width=SIZE_IMG),
            albumentations.RandomScale(0.07),
            albumentations.RandomRotate90(),
            albumentations.ShiftScaleRotate(),
            albumentations.Normalize(normalization="min_max_per_channel")
        ])

        #Taking the Pillow formated image from the dataset and make it into a Numpy Array
        img_np = np.array(pil_img)
        
        #Applying the transform
        augmented = transform(image=img_np)["image"]

        #Trasposing the image into a Pytorch-friendly tensor
        tensor = torch.tensor(
            np.transpose(augmented, (2, 0, 1)),
            dtype=torch.float32
        )

        return tensor
    



    

    

    
    



dataset = FedISIC2019_Dataset(None)

#print(dataset.fds.partitioners["test"].dataset)

#fta, ftt = dataset.centralized_dataset()

#print(fta)

#full_train, full_test = dataset.centralized_dataset()

#print(full_train[0]["label"])

#dataset.plot_centralized_train_class_distribution()

dataset.plot_in_partitions_train_class_distribution()