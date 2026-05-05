import random
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import albumentations
import datasets
from datasets import Dataset
from PIL import Image
import multiprocessing

from torch.utils.data import DataLoader
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import NaturalIdPartitioner
from flwr_datasets.visualization import plot_label_distributions

#Constants
SIZE_IMG = 299


class FedISIC2019_Dataset():
    fds = None
    labels = None
    labels_for_the_labels = ["mel", "mel-nev", "bcc", "ak", "bk", "df", "vl", "scc"]
    amt_labels = 8
    seed = None

    def __init__(self, seed: int):
        self.seed = seed

        #Seeding RNG
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        else:
            print("Dataset RNG not seeded")
            exit(1)

        #Loading Fed-ISIC2019 via flwr_datasets
        self.fds = FederatedDataset(
            dataset="flwrlabs/fed-isic2019", 
            partitioners={
                "train" : NaturalIdPartitioner(partition_by="center"),
                "test" : NaturalIdPartitioner(partition_by="center")
            }
        )

    def get_partition_label_count(self, partition: Dataset, partition_id: int, quiet_output = True):
        if(not quiet_output):
            print(f"Counting Labels for Partition {partition_id}")

        label_counters = [0 for n in range(self.amt_labels)]
        for row in partition:
            label_counters[row["label"]] += 1
        return label_counters
    
    def __to_numpy(self, img):
        if isinstance(img, np.ndarray):
            return img
        return np.array(img)

    def apply_train_val_test_standard_transform(self, pil_img):
        transform = albumentations.Compose([
            albumentations.PadIfNeeded(min_height=SIZE_IMG, min_width=SIZE_IMG, border_mode=0),
            albumentations.CenterCrop(height=SIZE_IMG, width=SIZE_IMG)
        ], seed=self.seed)

        #Taking the Pillow formated image from the dataset and make it into a Numpy Array
        img_np = self.__to_numpy(pil_img)
        
        #Applying the transform
        augmented = transform(image=img_np)["image"]

        #Transforming back into PIL Image for memory conservation
        augmented_pil_img = Image.fromarray(augmented)
        
        return augmented_pil_img
    
    def apply_oversampling_train_transform(self, pil_img):
        transform = albumentations.Compose([
            albumentations.RandomScale(0.07),
            albumentations.RandomRotate90(),
            albumentations.ShiftScaleRotate(),
            albumentations.PadIfNeeded(min_height=SIZE_IMG, min_width=SIZE_IMG, border_mode=0),
            albumentations.CenterCrop(height=SIZE_IMG, width=SIZE_IMG),
        ], seed=self.seed)

        #Taking the Pillow formated image from the dataset and make it into a Numpy Array
        img_np = self.__to_numpy(pil_img)
        
        #Applying the transform
        augmented = transform(image=img_np)["image"]

        #Transforming back into PIL Image for memory conservation
        augmented_pil_img = Image.fromarray(augmented)
        
        return augmented_pil_img
    
    def __map_image_to_standard_transformed_image(self, row):
        row["image"] = self.apply_train_val_test_standard_transform(row["image"])
        return row
    
    def __calc_distr(self, num_labels, total_examples):
        return [x/total_examples for x in num_labels]
    
    def __calc_partition_change_list(self, distributions: list, partition_total_samples: list, partition_label_counts: list, num_of_partitions: int, representative_partition: int):
        pc_list = [[0 for label in range(self.amt_labels)] for i in range(num_of_partitions)]
        
        #For all partitions, we...
        for i in range(num_of_partitions):
            if i == representative_partition:
                continue

            missing_label_percentage = 0
            #...add the missing_label_percentage of the labels that dont exist in the partition, then...
            for j in [x for x in range(self.amt_labels) if distributions[i][x] == 0]:
                missing_label_percentage += distributions[representative_partition][j]
            
            partition_change_list = [0 for i in range(self.amt_labels)]

            #...for the labels that do exist...
            for j in [x for x in range(self.amt_labels) if not distributions[i][x] == 0]:
                #...calculate the number of pictures to add/remove from the set...
                #Ratio (r) = desired ration of label/The_total_ratio_of_existing_labels_in_partition (1 - missing_label_percentage)
                #Number of images to add or remove (n) = (r * all_samples_in_partition)/The_total_ratio_of_existing_labels_in_partition - number_of_n_label_img_in_partition
                n = math.ceil((distributions[representative_partition][j]/(1 - missing_label_percentage))*partition_total_samples[i] - partition_label_counts[i][j])
                partition_change_list[j] = n
            
            pc_list[i] = partition_change_list

        return pc_list
    
    def augment_partition(self, partition_id: int, partition_change_list: list, quiet_output = True):
        if(not quiet_output):
            print(f"Augmenting Partition {partition_id}")
        
        partition_data = self.fds.load_partition(partition_id, "train")

        #Adding oversampled images
        new_train = []
        for label_index, partition_label_change in enumerate(partition_change_list):
            if partition_label_change > 0:
                temp_filtered_ds = partition_data.filter(lambda row: row["label"] == label_index, num_proc=4)
                temp_to_transform = temp_filtered_ds.select([np.random.randint(0,temp_filtered_ds.num_rows) for _ in range(partition_label_change)])
                new_train.extend({"center":partition_id,"label":label_index,"image":self.apply_oversampling_train_transform(row["image"])} for row in temp_to_transform)

                #Freeing memory
                temp_filtered_ds = []
                temp_to_transform = []

        #Guarding against an empty new_train from the representative partition
        new_partition_ds = None
        if new_train:    
            new_train_ds = datasets.Dataset.from_list(new_train)
            new_train = []

            #Casting the features to match the original dataset
            new_train_ds = new_train_ds.cast(partition_data.features)

            new_partition_ds =  datasets.concatenate_datasets([partition_data, new_train_ds])
        else:
            new_partition_ds = partition_data    
        
        new_partition_ds.save_to_disk(f"dataset_proccesed_data/partition{partition_id}")
        new_partition_ds = []


        #Removing unnecessary images
        new_partition_data = datasets.Dataset.load_from_disk(f"dataset_proccesed_data/partition{partition_id}")

        temp_to_rmv = []
        for label_index, partition_label_change in enumerate(partition_change_list):
            if partition_label_change < 0:
                temp_filtered_ds = new_partition_data.filter(lambda row: row["label"] == label_index, num_proc=4)

                rows_indicies_to_remove = np.random.choice([x for x in range(0, temp_filtered_ds.num_rows)],abs(partition_label_change), replace=False)
                temp_rows_dataset = temp_filtered_ds.select(rows_indicies_to_remove)
                for row in temp_rows_dataset:
                    temp_to_rmv.append(row)
                
                temp_filtered_ds = [] #Free memory
            
            # Build a set of fingerprints from rows to remove
            fingerprints_to_remove = set()
            for row in temp_to_rmv:
                img_bytes = np.array(row["image"]).tobytes()
                fingerprints_to_remove.add((row["center"], row["label"], img_bytes))

        def should_keep(row):
            img_bytes = np.array(row["image"]).tobytes()
            return (row["center"], row["label"], img_bytes) not in fingerprints_to_remove

        new_partition_ds = new_partition_data.filter(should_keep, num_proc=1)  # num_proc=1 since fingerprints_to_remove can't be pickled
        new_partition_ds = new_partition_ds.cast(new_partition_data.features)
        temp_to_rmv = [] # Free memory

        return new_partition_ds





    def augment_dataset(self, representative_partition: int, quiet_output = False):
        #Stage 1 - Loading a Partiton, Standardizing and counting labels.
        num_of_partitions = self.fds.partitioners["train"].num_partitions
        partition_label_counts = [[0 for n in range(self.amt_labels)] for n in range(num_of_partitions)]

        for partition_index in range(num_of_partitions):
            partition_data = self.fds.load_partition(partition_id=partition_index, split="train")
            partition_label_counts[partition_index] = self.get_partition_label_count(partition=partition_data, partition_id=partition_index, quiet_output=quiet_output)
            standardized_dataset = partition_data.map(self.__map_image_to_standard_transformed_image, num_proc=4)
            standardized_dataset.save_to_disk(f"dataset_proccesed_data/partition{partition_index}")
        

        #Stage 2 - Calculate partition distributions and amount of images to add/remove.
        partition_total_samples = [0 for partitions in range(num_of_partitions)]
        for i_part in range(num_of_partitions):
            for j_label in range(self.amt_labels):
                partition_total_samples[i_part] += partition_label_counts[i_part][j_label]
        
        distributions = [self.__calc_distr(partition_label_counts[i],partition_total_samples[i]) for i in range(num_of_partitions)]
        partition_change_lists = self.__calc_partition_change_list(distributions, partition_total_samples, partition_label_counts, num_of_partitions, representative_partition)

        
        #Stage 3 - Adding/Removing images
        for partition_index in range(num_of_partitions):
            augmented_partition = self.augment_partition(partition_id=partition_index, partition_change_list=partition_change_lists[partition_index], quiet_output=quiet_output)
            augmented_partition.save_to_disk(f"dataset_proccesed_data/partition{partition_index}_augmented")
            if(not quiet_output):
                print(f"Finished Augmenting Partition{partition_index}")
            



        return


if __name__ == "__main__":
    dataset = FedISIC2019_Dataset(67)
    dataset.augment_dataset(0)