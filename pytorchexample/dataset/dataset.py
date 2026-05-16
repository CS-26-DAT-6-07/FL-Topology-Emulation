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
import os
import json

from torch.utils.data import DataLoader
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import NaturalIdPartitioner
from flwr_datasets.visualization import plot_label_distributions

#Constants
SIZE_IMG = 299

dataset = None

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
    amt_labels = 8
    seed = None
    normalize_transform = None
    _dataset_is_augmented = False
    dataloaders = None
    global_dataloader = None
    __num_proc = 0

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

        self.normalize_transform = albumentations.Compose([
            albumentations.Normalize(normalization="min_max_per_channel"),
            albumentations.pytorch.ToTensorV2()
        ])

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
    
    def wrapper_mitosti(self, row):
        return self.__map_image_to_standard_transformed_image(row)

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

        standardized_partition_data = datasets.Dataset.load_from_disk(f"dataset_proccesed_data/partition{partition_id}")

        #Adding oversampled images
        new_train = []
        for label_index, partition_label_change in enumerate(partition_change_list):
            if partition_label_change > 0:
                temp_filtered_ds = partition_data.filter(lambda row: row["label"] == label_index, num_proc=self.__num_proc)
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

            new_partition_ds =  datasets.concatenate_datasets([standardized_partition_data, new_train_ds])
        else:
            new_partition_ds = standardized_partition_data    
        
        new_partition_ds.save_to_disk(f"dataset_proccesed_data/partition{partition_id}_temp")
        new_partition_ds = []


        #Removing unnecessary images
        new_partition_data = datasets.Dataset.load_from_disk(f"dataset_proccesed_data/partition{partition_id}_temp")

        temp_to_rmv = []
        for label_index, partition_label_change in enumerate(partition_change_list):
            if partition_label_change < 0:
                temp_filtered_ds = new_partition_data.filter(lambda row: row["label"] == label_index, num_proc=self.__num_proc)

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

        new_partition_ds = new_partition_data.filter(should_keep, num_proc=0)  # num_proc=0 since fingerprints_to_remove can't be pickled
        new_partition_ds = new_partition_ds.cast(new_partition_data.features)
        temp_to_rmv = [] # Free memory

        return new_partition_ds

    def __save_seed_totem(self):
        seed_totem = {"seed": self.seed}
        seed_totem = json.dumps(seed_totem)
        with open("seed.json", "w") as f:
            f.write(seed_totem)
    
    def __read_seed_totem(self):
        if os.path.exists("seed.json"):   
            seed_totem = None
            with open("seed.json") as f:
                seed_totem =  f.read()
            return seed_totem
        else:
            print("PWD PATH::" + os.getcwd())
            return None

    def augment_dataset(self, representative_partition: int, quiet_output = False):
        #Stage 1 - Loading a Partiton, Standardizing and counting labels.
        num_of_partitions = self.fds.partitioners["train"].num_partitions
        partition_label_counts = [[0 for n in range(self.amt_labels)] for n in range(num_of_partitions)]

        for partition_index in range(num_of_partitions):
            partition_data = self.fds.load_partition(partition_id=partition_index, split="train")
            partition_label_counts[partition_index] = self.get_partition_label_count(partition=partition_data, partition_id=partition_index, quiet_output=quiet_output)
            standardized_dataset = partition_data.map(self.__map_image_to_standard_transformed_image, num_proc=self.__num_proc)
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
            
        if(not quiet_output):
            print("Finished augmenting the dataset")
        self._dataset_is_augmented = True
        self.__save_seed_totem()
        return

    def normalize_and_tensorify_batch(self, batch):
        batch["image"] = [
            self.normalize_transform(image=self.__to_numpy(img))["image"]
            for img in batch["image"]
        ]
        return batch
    
    def test_dataset_transform(self, batch):
        batch = [self.__map_image_to_standard_transformed_image(e) for e in batch]
        self.normalize_and_tensorify_batch(batch)
        return batch

    def generate_dataloader_for_dataset(self, partition_dataset: Dataset):
        partition_dataset = partition_dataset.with_transform(self.normalize_and_tensorify_batch)

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
            batch_size=16,
            shuffle=True,
            generator=generator,
            worker_init_fn=SeedWorker("train", train_worker_seeds),
            num_workers=4
        )

        dataloader_test = DataLoader(
            partition_test,
            batch_size=16,
            shuffle=False,
            worker_init_fn=SeedWorker("test", test_worker_seeds),
            num_workers=4
        )

        return dataloader_train, dataloader_test, train_worker_seeds, test_worker_seeds
    
    def load_partition(self,partition, rep = 0):
        if self.__read_seed_totem() is not None:
            augmented_path = f"dataset_proccesed_data/partition{partition}_augmented"
            if os.path.exists(augmented_path):
                dataset_partition = datasets.Dataset.load_from_disk(augmented_path)
                dataloader_train, dataloader_test, _, _ = self.generate_dataloader_for_dataset(dataset_partition)
                return dataloader_train, dataloader_test
        if self.dataloaders == None:
            self.dataloaders, self.worker_seeds = self.generate_all_dataloaders(rep)
        return self.dataloaders[partition]

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

    def plot_in_partitions_augmented_train_class_distribution(self, representative_partition: int):
        if self._dataset_is_augmented is False:
            if representative_partition is not None:
                self.augment_dataset(representative_partition=representative_partition)
            else:
                print("Need representative partition id")
                exit(1)
        
        num_of_partitions = self.fds.partitioners["train"].num_partitions
        
        bar_width = 0.5
        indicies_partitions = [n for n in range(num_of_partitions)]

        mel_counters     = np.zeros(num_of_partitions, dtype=int)
        mel_nev_counters = np.zeros(num_of_partitions, dtype=int)
        bcc_counters     = np.zeros(num_of_partitions, dtype=int)
        ak_counters      = np.zeros(num_of_partitions, dtype=int)
        bk_counters      = np.zeros(num_of_partitions, dtype=int)
        df_counters      = np.zeros(num_of_partitions, dtype=int)
        vl_counters      = np.zeros(num_of_partitions, dtype=int)
        scc_counters     = np.zeros(num_of_partitions, dtype=int)

        for partition_index in range(num_of_partitions):
            partition_data = datasets.Dataset.load_from_disk(f"dataset_proccesed_data/partition{partition_index}_augmented")

            partition_label_count = self.get_partition_label_count(partition=partition_data, partition_id=partition_index)

            mel_counters[partition_index] += partition_label_count[0]
            mel_nev_counters[partition_index] += partition_label_count[1]
            bcc_counters[partition_index] += partition_label_count[2]
            ak_counters[partition_index] += partition_label_count[3]
            bk_counters[partition_index] += partition_label_count[4]
            df_counters[partition_index] += partition_label_count[5]
            vl_counters[partition_index] += partition_label_count[6]
            scc_counters[partition_index] += partition_label_count[7]

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
    
    def generate_all_dataloaders(self,rep):
        self.augment_dataset(rep)
        num_of_partitions = self.fds.partitioners["train"].num_partitions
        worker_seeds = []
        dataloaders = []

        for i in range(num_of_partitions):
            dataset = datasets.Dataset.load_from_disk(f"dataset_proccesed_data/partition{i}_augmented").with_format("torch")
            dataloader_train, dataloader_test, train_worker_seeds, test_worker_seeds = self.generate_dataloader_for_dataset(dataset)
            dataloaders.append((dataloader_train,dataloader_test))
            worker_seeds.append((train_worker_seeds,test_worker_seeds))
        
        return dataloaders, worker_seeds

def init_dataset(seed, rep):
    global dataset
    dataset = FedISIC2019_Dataset(seed)  
    dataset.load_partition(0, rep=rep)


def load_partition(partition):
    global dataset

    if dataset is None:
        seed = 0
        try:
            if os.path.exists("seed.json"):
                with open("seed.json") as f:
                    seed_data = json.load(f)
                    seed = seed_data.get("seed", 0)
        except Exception:
            pass
        dataset = FedISIC2019_Dataset(seed)

    augmented_path = f"dataset_proccesed_data/partition{partition}_augmented"
    if os.path.exists(augmented_path):
        partition_dataset = datasets.Dataset.load_from_disk(augmented_path)
        dataloader_train, dataloader_test, _, _ = dataset.generate_dataloader_for_dataset(partition_dataset)
        return dataloader_train, dataloader_test

    return dataset.load_partition(partition)

def load_centralized_dataset():
    global dataset
    if dataset.global_dataloader == None:
        temp = dataset.fds.load_split(split = "test").map(dataset.wrapper_mitosti)
        dataset.global_dataloader = DataLoader(
                dataset=temp.with_transform(dataset.normalize_and_tensorify_batch),#dataset.fds.load_split(split = "test").with_transform(dataset.normalize_and_tensorify_batch),
                batch_size=32,
                shuffle=False
            )

    return dataset.global_dataloader



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
    dataset.augment_dataset(0)

    augmented_partition = datasets.Dataset.load_from_disk("dataset_proccesed_data/partition0_augmented")
    dataloader_train_part1, dataloader_test_part1, train_worker_seeds, test_worker_seeds = dataset.generate_dataloader_for_dataset(augmented_partition)


    plot_dataloader_batch(dataloader_train_part1)
    print({"DatasetObjSeed": dataset.seed, "train": list(train_worker_seeds), "test": list(test_worker_seeds)})
    dataset.plot_in_partitions_augmented_train_class_distribution(0)