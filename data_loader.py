from pathlib import Path
import os
from unittest import loader
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import Constant
colab = Constant.colab
import torch.nn.functional as F
from sklearn import preprocessing

torch.manual_seed(0)
# TODO:debug data loader
class FeatherData(Dataset):
    def __init__(self, paths, labels,transform):
      if colab:
        content_path = '/content/gdrive/Othercomputers/My MacBook Pro/'
        self.data =  paths
        self.labels = labels
      else:
        self.data =  paths
        self.labels = labels

      self.length = len(self.data)

      self.transform = transform
  
    def __getitem__(self, index):
      sample = Image.open(self.data[index])
      if sample.size[0] > sample.size[1]:
          sample = sample.rotate(90)

      if self.transform:
          sample = self.transform(sample)
      return sample, self.labels[index]

    def __len__(self):
      return self.length

def csv_to_paths(dataset_dir, csv_file):
    with open(csv_file, "r") as readfile:
        readfile.readline()  # skip header

        csv_data = readfile.readlines()

    image_paths = []

    for line in csv_data:
        image_path_parts = [
            x.lower().replace(" ", "_") for x in line.strip().split(",")
        ]

        image_path = (
            dataset_dir
            / "images"
            / image_path_parts[1]
            / image_path_parts[2]
            / image_path_parts[0]
        )

        image_paths.append(image_path.as_posix())

    return image_paths

def read_labels(file_path, label_type):
    # Select position of label in CSV file by type.
    if label_type == "order":
        pos = 1
    elif label_type == "species":
        pos = 2
    else:
        raise Exception("undefined label type")

    # Return classes from CSV file.
    with open(file_path, "r") as readfile:
        readfile.readline()  # skip header
        return [line.strip().split(",")[pos] for line in readfile.readlines()]


class Data_Loader():
    def __init__(self, batch_size, shuf=False):
        self.batch = batch_size
        self.shuf = shuf

    def loader(self):
        # FeatherData
        if colab is False:
            DATASET_DIR = Path(".") / "feathersv1-dataset"
        else:
            DATASET_DIR = Path("/content/gdrive/MyDrive/Feather/feathersv1-dataset")

        CLASSES_COUNT = 100
        TRAIN_CSV = DATASET_DIR / "data" / f"train_top_50_species.csv"
        TEST_CSV = DATASET_DIR / "data" / f"test_top_50_species.csv"
        # TRAIN_CSV = DATASET_DIR / "data" / f"train_all_species.csv"
        # TEST_CSV = DATASET_DIR / "data" / f"test_all_species.csv"

        IMG_WIDTH, IMG_HEIGHT = 240, 40
        # tr_data = self.fungi()

        transform = transforms.Compose([    
            # transforms.CenterCrop(299),
            transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),
            transforms.ToTensor(),
        ])

        # dataset = FungiData(tr_data,transform)
        paths = csv_to_paths(DATASET_DIR, TRAIN_CSV)
        paths_test = csv_to_paths(DATASET_DIR, TEST_CSV)

        labels = read_labels(TRAIN_CSV, label_type="species")
        labels_test = read_labels(TEST_CSV, label_type="species")

        le = preprocessing.LabelEncoder()



        _ = le.fit(list({*labels, *labels_test}))

        # train + val
        labels = le.transform(labels)
        # test
        labels_test = le.transform(labels_test)



        total_count = len(paths)

        # train + val
        targets = torch.as_tensor(labels)
        # test
        targets_test = torch.as_tensor(labels_test)
        
        
        dataset = FeatherData(paths, targets, transform)
        dataset_test = FeatherData(paths_test, targets_test, transform)
        train_data, val_data = torch.utils.data.random_split(dataset, [int(total_count * 0.7), total_count - int(total_count * 0.7)]) 
        if Constant.GPU:
            loader_train = torch.utils.data.DataLoader(dataset=train_data,
                                                batch_size=self.batch,
                                                shuffle=self.shuf,
                                                num_workers=2,
                                                drop_last=True)


            loader_val = torch.utils.data.DataLoader(dataset=val_data,
                                                batch_size=self.batch,
                                                shuffle=self.shuf,
                                                num_workers=2,
                                                drop_last=True)

            loader_test = torch.utils.data.DataLoader(dataset=dataset_test,
                                                batch_size=self.batch,
                                                shuffle=self.shuf,
                                                num_workers=2,
                                                drop_last=True)
        else:
            loader_train = torch.utils.data.DataLoader(dataset=train_data,
                                                batch_size=self.batch,
                                                shuffle=self.shuf,
                                                # num_workers=2,
                                                drop_last=True)


            loader_val = torch.utils.data.DataLoader(dataset=val_data,
                                                batch_size=self.batch,
                                                shuffle=self.shuf,
                                                # num_workers=2,
                                                drop_last=True)

            loader_test = torch.utils.data.DataLoader(dataset=dataset_test,
                                                batch_size=self.batch,
                                                shuffle=self.shuf,
                                                # num_workers=2,
                                                drop_last=True)
            

        return loader_train, loader_val, loader_test

if __name__ == "__main__":
    x = Data_Loader(64)
    _,_,_ = x.loader()