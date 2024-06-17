import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pytorch_lightning as pl
import torch




class ImageDataset_with_labels_correct_anomaly(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Root directory containing "good" and "Anomalies" subfolders.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = []
        self.labels = []

        # Populate image_files and labels
        for subdir in os.listdir(root_dir):
            subdir_path = os.path.join(root_dir, subdir)
            # print(f"Subdirectory_path: {subdir_path}")

            for file in os.listdir(subdir_path):
                # print(file)
                if os.path.isfile(os.path.join(subdir_path, file)):
                    self.image_files.append(os.path.join(subdir_path, file))
                    if subdir.lower() == "good":
                        self.labels.append(0)
                    elif subdir.lower() == "anomalies":
                        self.labels.append(1)
                    else:
                        raise ValueError(f"Unexpected subdirectory: {subdir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image = Image.open(img_name).convert("RGB")  # Convert image to RGB

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label


class CustomImageDatasetFromCSV(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with image names and labels.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.data_frame.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        label = int(self.data_frame.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)

        return image, label

class DataClass_4classes(Dataset):
    """
    A custom dataset class to load images and corresponding labels from a CSV file.

    Args:
        data (pd.DataFrame): DataFrame containing the image paths and labels.
        root_dir (str): Directory with all the images.
        transform (callable):Transform to be applied on an image sample.

    """
    def __init__(self, data, root_dir, transform=None):
        self.data = data
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        label = int(self.data.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)

        return image, label


class DataModule_4classes(pl.LightningDataModule):
    """
    A PyTorch Lightning DataModule for loading and splitting image datasets.

    Args:
        csv_file (str): Path to the CSV file containing image paths and labels.
        root_dir (str): Directory with all the images.
        batch_size (int, optional): Batch size for DataLoader. Default is 32.
        val_split (float, optional): Fraction of the data to use for validation. Default is 0.2.
        test_split (float, optional): Fraction of the data to use for testing. Default is 0.1.
        num_workers (int, optional): Number of workers for DataLoader. Default is 4.
        transform (callable): Transform to be applied on an image sample.

    """
    def __init__(self, csv, root_dir, batch_size=32, val_split=0.15,test_split=0.15, num_workers=0,transform_train=None, transform_val_test=None):
        super().__init__()
        self.csv_file = csv
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.num_workers = num_workers

        self.transform_train = transform_train
        self.transform_val_test = transform_val_test

    def prepare_data(self):
        # Read the CSV files
        self.data = pd.read_csv(self.csv_file)
        
    def setup(self, stage=None):
        # Split the data into train, val, and test

        # get the shapes
        unique_shapes = self.data['shape'].unique()
        # divide the shapes into train, val, and test --> shapes that are in train should not be in val or test
        shapes_train, shapes_temp = train_test_split(unique_shapes, test_size=(self.val_split + self.test_split), random_state=42)
        

        # split the datasate according to the shapes
        train_df = self.data[self.data['shape'].isin(shapes_train)]
        val_test_df = self.data[~self.data['shape'].isin(shapes_train)]

        val_df, test_df = train_test_split(val_test_df, test_size=self.test_split/(self.test_split + self.val_split), stratify=val_test_df['class'], random_state=42)



        self.train_dataset = DataClass_4classes(train_df, self.root_dir, transform=self.transform_train)
        self.val_dataset = DataClass_4classes(val_df, self.root_dir, transform=self.transform_val_test)
        self.test_dataset = DataClass_4classes(test_df, self.root_dir, transform=self.transform_val_test)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
class DataModule_4classes_kfold_cross_validation(pl.LightningDataModule):
    """
    A PyTorch Lightning DataModule for loading and splitting image datasets.

    Args:
        csv_file (str): Path to the CSV file containing image paths and labels.
        root_dir (str): Directory with all the images.
        batch_size (int, optional): Batch size for DataLoader. Default is 32.
        val_split (float, optional): Fraction of the data to use for validation. Default is 0.2.
        test_split (float, optional): Fraction of the data to use for testing. Default is 0.1.
        num_workers (int, optional): Number of workers for DataLoader. Default is 4.
        transform (callable): Transform to be applied on an image sample.

    """
    def __init__(self, csv, root_dir, batch_size=32, val_split=0.15,test_split=0.15, num_workers=0,transform_train=None, transform_val_test=None):
        super().__init__()
        self.csv_file = csv
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.num_workers = num_workers

        self.transform_train = transform_train
        self.transform_val_test = transform_val_test

    def prepare_data(self):
        # Read the CSV files
        self.data = pd.read_csv(self.csv_file)
        
    def setup(self, stage=None):
        # Split the data into train, val, and test

        # get the shapes
        unique_shapes = self.data['shape'].unique()
        # divide the shapes into train, val, and test --> shapes that are in train should not be in val or test
        shapes_train, shapes_temp = train_test_split(unique_shapes, test_size=(self.val_split + self.test_split), random_state=42)
        shapes_val, shapes_test = train_test_split(shapes_temp, test_size=self.test_split/(self.test_split + self.val_split), random_state=42)

        # split the datasate according to the shapes
        train_df = self.data[self.data['shape'].isin(shapes_train)]
        val_test_df = self.data[~self.data['shape'].isin(shapes_train)]

        val_df, test_df = train_test_split(val_test_df, test_size=self.test_split/(self.test_split + self.val_split), stratify=val_test_df['class'], random_state=42)



        self.train_dataset = DataClass_4classes(train_df, self.root_dir, transform=self.transform_train)
        self.val_dataset = DataClass_4classes(val_df, self.root_dir, transform=self.transform_val_test)
        self.test_dataset = DataClass_4classes(test_df, self.root_dir, transform=self.transform_val_test)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
class DataClass_4classes_crossvalidation(Dataset):
    """
    A custom dataset class to load images and corresponding labels from a CSV file.

    Args:
        data (pd.DataFrame): DataFrame containing the image paths and labels.
        root_dir (str): Directory with all the images.
        transform (callable):Transform to be applied on an image sample.

    """
    def __init__(self, data, root_dir, transform=None):
        self.data = data
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        label = int(self.data.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)

        return image, label