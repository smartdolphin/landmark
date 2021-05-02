import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from glob import glob
from PIL import Image


class TrainDataset(Dataset):
    def __init__(self, args, transform=None):
        self.train_dir = args.train_dir
        self.train_csv_dir = args.train_csv_dir
        self.train_csv_exist_dir = args.train_csv_exist_dir
        self.args = args
        self.transform = transform
        self.train_image = list()
        self.train_label = list()
        if not os.path.isfile(self.train_csv_exist_dir) :
            self.train_csv = pd.read_csv(self.train_csv_dir, encoding='utf-8')
            self.train_csv_exist = self.train_csv.copy()
            self.load_full_data()
            self.train_csv_exist.to_csv(self.train_csv_exist_dir, index=False)
        else:
            self.load_exist_data()

    def load_full_data(self):
        for i in tqdm(range(len(self.train_csv))):
            filename = self.train_csv['id'][i]
            fullpath = glob(self.train_dir + "*/*/" + filename.replace('[', '[[]') + ".JPG")[0]
            label = self.train_csv['landmark_id'][i]
            self.train_csv_exist.loc[i,'id'] = fullpath
            self.train_image.append(fullpath)
            self.train_label.append(label)

    def load_exist_data(self):
        self.train_csv_exist = pd.read_csv(self.train_csv_exist_dir, encoding='utf-8')
        for i in tqdm(range(len(self.train_csv_exist))):
            fullpath = self.train_csv_exist['id'][i]
            label = self.train_csv_exist['landmark_id'][i]
            self.train_image.append(fullpath)
            self.train_label.append(label)

    def __len__(self):
        return len(self.train_image)

    def __getitem__(self, idx):
        image = Image.open(self.train_image[idx])
        if self.transform is not None:
            image = np.array(image)
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            image = image.resize((self.args.image_size, self.args.image_size))
            image = np.array(image) / 255.
            image = np.transpose(image, axes=(2, 0, 1))
        label = self.train_label[idx]
        return image, label


class TestDataset(Dataset):
    def __init__(self, args, transform=None):
        self.test_dir = args.test_dir
        self.test_csv_dir = args.test_csv_dir
        self.test_csv_exist_dir = args.test_csv_exist_dir
        self.args = args
        self.transform = transform
        self.test_image = list()
        self.test_label = list()
        if not os.path.isfile(self.test_csv_exist_dir):
            self.test_csv = pd.read_csv(self.test_csv_dir, encoding='utf-8')
            self.test_csv_exist = self.test_csv.copy()
            self.load_full_data()
            self.test_csv_exist.to_csv(self.test_csv_exist_dir, index=False)
        else :
            self.load_exist_data()

    def load_full_data(self):
        for i in tqdm(range(len(self.test_csv))):
            filename = self.test_csv['id'][i]
            fullpath = glob(self.test_dir + "*/" + filename.replace('[', '[[]') + ".JPG")[0]
            label = self.test_csv['id'][i]

            self.test_csv_exist.loc[i,'id'] = fullpath
            self.test_image.append(fullpath)
            self.test_label.append(label)

    def load_exist_data(self):
        self.test_csv_exist = pd.read_csv(self.test_csv_exist_dir)
        for i in tqdm(range(len(self.test_csv_exist))):
            fullpath = self.test_csv_exist['id'][i]
            label = self.test_csv_exist['id'][i]

            self.test_image.append(fullpath)
            self.test_label.append(label)

    def __len__(self):
        return len(self.test_image)

    def __getitem__(self, idx):
        image = Image.open(self.test_image[idx])
        if self.transform is not None:
            image = np.array(image)
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            image = image.resize((self.args.image_size, self.args.image_size))
            image = np.array(image) / 255.
            image = np.transpose(image, axes=(2, 0, 1))
        label = self.test_label[idx]
        return image, label

