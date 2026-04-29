import random
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import albumentations
import csv
import datasets
import multiprocessing
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
    
    def augment_dataset(self, representative):
        return self.__apply_augmentations(representative=representative, quiet=False)
         

    def __apply_augmentations(self, representative, quiet = True):
        amt_labels = 8

        partitions = self.fds.partitioners["train"]
        
        data = [partitions.load_partition(i) for i in range(0,partitions.num_partitions)] 

        num_labels = [[0 for i in range(0,amt_labels)] for i in range(0,partitions.num_partitions)]
        if(not quiet):
            print("Couting labels")
        #Count labels for each of the partitions
        for d in range(0, partitions.num_partitions):
            for row in data[d]:
                num_labels[d][row["label"]] += 1

        #List of lists with the distributions per partition
        distributions = [self.__calc_distr(num_labels[i],len(data[i])) for i in range(0,partitions.num_partitions)]

        new_train = [[] for x in range(0,partitions.num_partitions)]
        #For all partitions, we...
        for i in range(0,partitions.num_partitions):
            if(not quiet):
                print(f"augmenting partition {i}")
            if(i == representative):
                for row in data[representative]:
                    temp_img = self.apply_train_val_test_standard_transform(row["image"])
                    tensorified_temp_img = self.__to_torch_tensor(temp_img)

                    new_train[i].append({"center":representative,"image":tensorified_temp_img ,"label":row["label"]})
                continue    

            missing_label_percentage = 0
            #...add the missing_label_percentage of the labels that dont exist in the partition, then...
            for j in [x for x in range(0,amt_labels) if np.isclose(distributions[i][x],0)]:
                missing_label_percentage += distributions[representative][j]
            #n = [0 for i in range(0, amt_labels)]
            #...for the labels that do exist...
            for j in [x for x in range(0,amt_labels) if not np.isclose(distributions[i][x],0)]:
                #...calculate the number of pictures to add/remove from the set...
                #Ratio (r) = desired ration of label/The_total_ratio_of_existing_labels_in_partition (1 - missing_label_percentage)
                #Number of images to add or remove (n) = (r * all_samples_in_partition)/The_total_ratio_of_existing_labels_in_partition - number_of_n_label_img_in_partition
                n = math.ceil((distributions[representative][j]/np.round((1 - missing_label_percentage)))*data[i].num_rows - num_labels[i][j])
                
                temp = data[i].filter(lambda e: e['label'] == j)
                if(n > 0):
                    new_train[i] = [{"center":i,"label":j,"image":self.__to_torch_tensor(row["image"])} for row in temp]
                    elem = temp.select([np.random.randint(0,temp.num_rows) for _ in range(0, n)])
                    for m in range(0, n):
                        transformed_image = self.apply_oversampling_train_transform(elem[m]["image"])
                        tensorified_image = self.__to_torch_tensor(transformed_image)
                        new_train[i].append({"center": i,"label":j,"image":tensorified_image})
                elif(n < 0):
                    rmv = np.random.choice([x for x in range(0, temp.num_rows)],abs(n), replace=False)
                    for x in range(0,temp.num_rows):
                        if(x not in rmv):
                            new_train[i].append({"center":i,"label":j,"image":self.__to_torch_tensor(temp[x]["image"])})
            data[i] = datasets.Dataset.from_list(new_train[i])
        if(not quiet):
            print("augmenting complete")
        

        return data
       
    def __calc_distr(self, num_labels, total_examples):
        return [x/total_examples for x in num_labels]

    def __to_torch_tensor(self, pil):

        return self.normalize_and_tensorify(pil)

    def __to_numpy(self, img):
        if isinstance(img, np.ndarray):
            return img
        return np.array(img)

    def plot_in_partitions_train_class_distribution(self):
        partitioner = self.fds.partitioners["train"]

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

        #Fixing the legend
        old_legend = fig.legends[0]
        handles = old_legend.legend_handles

        old_texts = [t.get_text() for t in old_legend.get_texts()]
        print("Original legend order:", old_texts)
        reverse_lfl_list = self.labels_for_the_labels
        reverse_lfl_list.reverse()
        print("Your labels order:", reverse_lfl_list)

        ax.legend(
            handles,
            reverse_lfl_list,
            loc="center right",
            bbox_to_anchor=(1.01,0.5),
            borderaxespad=-10
        )

        fig.legends.clear()
        fig.set_constrained_layout(True)

        plt.show()

        return
    
    def apply_train_val_test_standard_transform(self, pil_img):
        if self.seed != None:
            np.random.seed(self.seed)
        
        transform = albumentations.Compose([
            albumentations.PadIfNeeded(min_height=SIZE_IMG, min_width=SIZE_IMG, border_mode=0),
            albumentations.CenterCrop(height=SIZE_IMG, width=SIZE_IMG)
        ])

        #Taking the Pillow formated image from the dataset and make it into a Numpy Array
        img_np = self.__to_numpy(pil_img)
        
        #Applying the transform
        augmented = transform(image=img_np)["image"]
        
        return augmented

    def apply_oversampling_train_transform(self, pil_img):
        if self.seed != None:
            np.random.seed(self.seed)
        
        transform = albumentations.Compose([
            albumentations.RandomScale(0.07),
            albumentations.RandomRotate90(),
            albumentations.ShiftScaleRotate(),
            albumentations.PadIfNeeded(min_height=SIZE_IMG, min_width=SIZE_IMG, border_mode=0),
            albumentations.CenterCrop(height=SIZE_IMG, width=SIZE_IMG)
        ])

        #Taking the Pillow formated image from the dataset and make it into a Numpy Array
        img_np = self.__to_numpy(pil_img)
        
        #Applying the transform
        augmented = transform(image=img_np)["image"]

        return augmented
    
    def normalize_and_tensorify(self, transformed_img):
        transformed_img = self.__to_numpy(transformed_img)
        normalize = albumentations.Compose([albumentations.Normalize(normalization="min_max_per_channel"), albumentations.ToTensorV2()])
        final_img = normalize(image=transformed_img)["image"]
        return final_img

    def generate_dataloader_for_dataset(self, partition_dataset):
        if self.seed == None:
            print("No seed given to the dataset object")
            exit(1)

        partition_train_test = partition_dataset.train_test_split(test_size=0.2, seed=self.seed)
        partition_train = partition_train_test["train"]
        partition_test = partition_train_test["test"]

        #Setting up the dataloader
        generator = torch.Generator()
        generator.manual_seed(self.seed)

        #Setting up shared list across processes
        #mp_manager = multiprocessing.Manager()
        train_worker_seeds = []
        test_worker_seeds = []


        def make_seed_worker(split_name, seeds_list):
            def seed_worker(worker_id):
                worker_seed = torch.initial_seed()
                np.random.seed(worker_seed)
                random.seed(worker_seed)
                seeds_list.append({"split": split_name, "worker_id": worker_id, "seed": worker_seed})
            return seed_worker

        dataloader_train = DataLoader(
            partition_train,
            batch_size=32,
            shuffle=True,
            generator=generator,
            worker_init_fn=make_seed_worker("train", train_worker_seeds),
            num_workers=0
        )

        dataloader_test = DataLoader(
            partition_test,
            batch_size=32,
            shuffle=False,
            worker_init_fn=make_seed_worker("test", test_worker_seeds),
            num_workers=0
        )

        return dataloader_train, dataloader_test, {"train": list(train_worker_seeds), "test": list(test_worker_seeds)}
    



def plot_dataloader_batch(dataloader, num_images=8):
        # Grab one batch
        images, labels = next(iter(dataloader))
    
        # Clamp to [0,1] in case of any floating point overshoot
        images = images.clamp(0, 1)
    
        num_images = min(num_images, len(images))
        fig, axes = plt.subplots(1, num_images, figsize=(num_images * 2, 3))
    
        for i in range(num_images):
            # Tensor is (C, H, W) → matplotlib needs (H, W, C)
            img = images[i].permute(1, 2, 0).numpy()
        
            axes[i].imshow(img)
            axes[i].set_title(f"Label: {labels[i].item()}")
            axes[i].axis("off")
    
        plt.tight_layout()
        plt.show()
  
dataset = FedISIC2019_Dataset(67)

#print(dataset.fds.load_partition(0, "train")[0])

#fta, ftt = dataset.centralized_dataset()

#print(fta)

#full_train, full_test = dataset.centralized_dataset()

#print(full_train[0])

#dataset.plot_centralized_train_class_distribution()
#dataset.plot_centralized_train_class_distribution()

##dataset.plot_in_partitions_train_class_distribution()
#dataset.augment_dataset(0)


augmented_partitions = dataset.augment_dataset(0)
dataloader_train_part1, dataloader_test_part1, seed_logs = dataset.generate_dataloader_for_dataset(augmented_partitions[1])

batch = next(iter(dataloader_train_part1))
print(batch["image"])
print(batch["label"])
print(seed_logs)

#plot_dataloader_batch(dataloader_train_part1)