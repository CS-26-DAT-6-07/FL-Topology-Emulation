import random
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import albumentations
import csv
import datasets
import multiprocessing
import os

from torch.utils.data import Dataset, DataLoader
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import NaturalIdPartitioner
from flwr_datasets.visualization import plot_label_distributions

#Constants
SIZE_IMG = 299

#Pickable unlike a nested function / For Pytorch dataloader
class SeedWorker:
    def __init__(self, split_name, seeds_list):
        self.split_name = split_name
        self.seeds_list = seeds_list
    
    def __call__(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        self.seeds_list.append({
            "split": self.split_name,
            "worker_id": worker_id,
            "seed": worker_seed
        })

class FedISIC2019_Dataset():
    fds = None
    labels = None
    labels_for_the_labels = ["mel", "mel-nev", "bcc", "ak", "bk", "df", "vl", "scc"]
    seed = None
    augmented_dataset_partitions = None

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
        self.augmented_dataset_partitions = self.__apply_augmentations(representative=representative, quiet=False)
        return self.augmented_dataset_partitions        

    def __apply_augmentations(self, representative, quiet = True):
        dataset_path = f"dataset/augmented_train/{self.seed}"
        if(os.path.isdir(dataset_path)):
            data = []
            for dir in os.listdir(f"{dataset_path}/{self.seed}"):
                data = data.append(datasets.load_from_disk(dir))
            return data

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
                #for row in data[representative]:
                #    temp_img = self.apply_train_val_test_standard_transform(row["image"])
                #    tensorified_temp_img = self.__to_torch_tensor(temp_img)
                #    new_train[i].append({"center":representative ,"label":row["label"],"image":tensorified_temp_img})
                #data[i] = datasets.Dataset.from_list(new_train[i]).with_format("torch")   
                data[i] = data[i].map(self.__transform_image).with_format('torch')
                continue    

            missing_label_percentage = 0
            #...add the missing_label_percentage of the labels that dont exist in the partition, then...
            for j in [x for x in range(0,amt_labels) if distributions[i][x] == 0]:
                missing_label_percentage += distributions[representative][j]
            #...for the labels that do exist...
            rmv = []
            add = []
            for j in [x for x in range(0,amt_labels) if not distributions[i][x] == 0]:
                #...calculate the number of pictures to add/remove from the set...
                #Ratio (r) = desired ration of label/The_total_ratio_of_existing_labels_in_partition (1 - missing_label_percentage)
                #Number of images to add or remove (n) = (r * all_samples_in_partition)/The_total_ratio_of_existing_labels_in_partition - number_of_n_label_img_in_partition
                n = math.ceil((distributions[representative][j]/(1 - missing_label_percentage))*data[i].num_rows - num_labels[i][j])

                #Find indexes with labels matching the one currently being looked at            
                candidates = []
                for k in range(0, data[i].num_rows):
                    if(data[i][k]['label'] == j):
                        candidates.append(k)
                
                #if images need to be added
                if(n > 0):
                    #select n candidates (can be duplicates) and use them to make images
                    for k in np.random.choice([x for x in candidates],n, replace=True):
                        temp = data[i][k]
                        temp['image'] =  self.__to_torch_tensor(self.apply_oversampling_train_transform(temp["image"]))
                        add.append(temp) 
                elif(n < 0):
                    #save the indices to removed later
                    rmv = np.append(rmv,np.random.choice([x for x in candidates],abs(n), replace=False))
            
            data[i] = data[i].map(self.__transform_image) #normalize
            data[i] = data[i].select([k for k in range(0, data[i].num_rows) if k not in rmv]) #remove images

            
            if(len(add) > 0):
                target_features = data[i].features
                def gen():
                    for x in add: yield x
                add_dataset = datasets.Dataset.from_generator(gen,features=target_features) #Create a new dataset with the new images
                data[i] = datasets.concatenate_datasets([data[i], add_dataset])
            data[i] = data[i].with_format("torch")
        if(not quiet):
            print("augmenting complete")
        
        for datasets in data:
            os.mkdir(dataset_path)
            for i in range(0, partitions.num_partitions):
                data[i].save_to_disk(dataset_path=f"{dataset_path}")


        return data

    def __transform_image(self, example):
       example['image'] = self.__to_torch_tensor(self.apply_train_val_test_standard_transform(example['image']))
       return example

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
        transform = albumentations.Compose([
            albumentations.PadIfNeeded(min_height=SIZE_IMG, min_width=SIZE_IMG, border_mode=0),
            albumentations.CenterCrop(height=SIZE_IMG, width=SIZE_IMG)
        ], seed=self.seed)

        #Taking the Pillow formated image from the dataset and make it into a Numpy Array
        img_np = self.__to_numpy(pil_img)
        
        #Applying the transform
        augmented = transform(image=img_np)["image"]
        
        return augmented

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

        return augmented
    
    def normalize_and_tensorify(self, transformed_img):
        transformed_img = self.__to_numpy(transformed_img)
        normalize = albumentations.Compose([albumentations.Normalize(normalization="min_max_per_channel")])
        final_img = normalize(image=transformed_img)["image"]
        return final_img

    def generate_dataloader_for_dataset(self, partition_dataset):
        partition_train_test = partition_dataset.train_test_split(test_size=0.2, seed=self.seed)
        partition_train = partition_train_test["train"]
        partition_test = partition_train_test["test"]

        #Setting up the dataloader
        generator = torch.Generator()
        generator.manual_seed(self.seed)

        #Setting up shared list across processes
        mp_manager = multiprocessing.Manager()
        train_worker_seeds = mp_manager.list()
        test_worker_seeds = mp_manager.list()

        dataloader_train = DataLoader(
            partition_train,
            batch_size=32,
            shuffle=True,
            generator=generator,
            worker_init_fn=SeedWorker("train", train_worker_seeds),
            num_workers=4
        )

        dataloader_test = DataLoader(
            partition_test,
            batch_size=32,
            shuffle=False,
            worker_init_fn=SeedWorker("test", test_worker_seeds),
            num_workers=4
        )

        return dataloader_train, dataloader_test, train_worker_seeds, test_worker_seeds
    
    def plot_in_partitions_augmented_train_class_distribution(self, representative: int):
        if self.augmented_dataset_partitions is None:
            if representative is not None:
                self.augment_dataset(representative=representative)
            else:
                print("Need representative partition id")
                exit(1)
        
        bar_width = 0.5
        indicies_partitions = [n for n in range(len(self.augmented_dataset_partitions))]

        mel_counters     = np.zeros(len(self.augmented_dataset_partitions), dtype=int)
        mel_nev_counters = np.zeros(len(self.augmented_dataset_partitions), dtype=int)
        bcc_counters     = np.zeros(len(self.augmented_dataset_partitions), dtype=int)
        ak_counters      = np.zeros(len(self.augmented_dataset_partitions), dtype=int)
        bk_counters      = np.zeros(len(self.augmented_dataset_partitions), dtype=int)
        df_counters      = np.zeros(len(self.augmented_dataset_partitions), dtype=int)
        vl_counters      = np.zeros(len(self.augmented_dataset_partitions), dtype=int)
        scc_counters     = np.zeros(len(self.augmented_dataset_partitions), dtype=int)

        for i, partition in enumerate(self.augmented_dataset_partitions):
            partition_label_count = [0 for label in self.labels_for_the_labels]
            for sample in partition:
                partition_label_count[sample["label"]] += 1

            mel_counters[i] += partition_label_count[0]
            mel_nev_counters[i] += partition_label_count[1]
            bcc_counters[i] += partition_label_count[2]
            ak_counters[i] += partition_label_count[3]
            bk_counters[i] += partition_label_count[4]
            df_counters[i] += partition_label_count[5]
            vl_counters[i] += partition_label_count[6]
            scc_counters[i] += partition_label_count[7]

        plt.bar(indicies_partitions, mel_counters, bar_width, label="mel")
        plt.bar(indicies_partitions, mel_nev_counters, bar_width, bottom=mel_counters, label="mel-nev")
        plt.bar(indicies_partitions, bcc_counters, bar_width, bottom=mel_counters + mel_nev_counters, label="bcc")
        plt.bar(indicies_partitions, ak_counters, bar_width, bottom=mel_counters + mel_nev_counters + bcc_counters, label="ak")
        plt.bar(indicies_partitions, bk_counters, bar_width, bottom=mel_counters + mel_nev_counters + bcc_counters + ak_counters, label="bk")
        plt.bar(indicies_partitions, df_counters, bar_width, bottom=mel_counters + mel_nev_counters + bcc_counters + ak_counters + bk_counters, label="df")
        plt.bar(indicies_partitions, vl_counters, bar_width, bottom=mel_counters + mel_nev_counters + bcc_counters + ak_counters + bk_counters + df_counters, label="vl")
        plt.bar(indicies_partitions, scc_counters, bar_width, bottom=mel_counters + mel_nev_counters + bcc_counters + ak_counters + bk_counters + df_counters + vl_counters, label="scc")
        
        totals = (mel_counters + mel_nev_counters + bcc_counters + ak_counters + bk_counters + df_counters + vl_counters + scc_counters)

        for i, total in enumerate(totals):
            plt.text(i, total, str(total), ha="center", va="bottom", fontsize=10)

        plt.xlabel("Partition ID")
        plt.ylabel("Count")
        plt.xticks(indicies_partitions, [f"{n}" for n in indicies_partitions])
        plt.title("Per partition Labels Distribution")
        plt.legend()
        
        plt.show()

        return




    



def plot_dataloader_batch(dataloader, num_images=8):
        # Grab one batch
        batch = next(iter(dataloader))
    
        # Clamp to [0,1] in case of any floating point overshoot
        batch['image'] = [img.clamp(0,1) for img in batch['image']]
    
        num_images = min(num_images, len(batch['image']))
        fig, axes = plt.subplots(1, num_images, figsize=(num_images * 2, 3))
    
        for i in range(num_images):
            # Tensor is (C, H, W) → matplotlib needs (H, W, C)
            img = batch['image'][i].permute(1, 2, 0).numpy()
        
            axes[i].imshow(img)
            axes[i].set_title(f"Label: {batch['label'][i].item()}")
            axes[i].axis("off")
    
        plt.tight_layout()
        plt.show()
  



if __name__ == "__main__":
    dataset = FedISIC2019_Dataset(67)

    #augmented_partitions = dataset.augment_dataset(0)
    #dataloader_train_part1, dataloader_test_part1, train_worker_seeds, test_worker_seeds = dataset.generate_dataloader_for_dataset(augmented_partitions[1])


    #plot_dataloader_batch(dataloader_train_part1)
    #print({"DatasetObjSeed": dataset.seed, "train": list(train_worker_seeds), "test": list(test_worker_seeds)})
    aug_partitions = dataset.augment_dataset(0)
    for partition in aug_partitions:
        print(partition)