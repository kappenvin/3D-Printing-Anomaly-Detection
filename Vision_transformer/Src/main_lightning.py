import torch
import timm
import os

# import tifffile as tifi
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import h5py
import pandas as pd
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F
import sys
import pdb
from torch.utils.data import random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ProgressBar, TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger,WandbLogger,CSVLogger
from sklearn.metrics import confusion_matrix

sys.path.append(r"C:\Users\Vincent\Documents\Masterarbeit\Src\Deep_Learning")
from utils.datamodule import DataModule_4classes,DataClass_4classes
from model.Vit_base16 import ViTLightningModule
import csv 


# define the path where the images are an the csv file
image_path_all_images = r"C:\Users\Vincent\Documents\Masterarbeit\Data\external_dataset\KIT_dataset_from_kaggle\Printing_Errors\images\all_images256"
csv_path_all_images = r"C:\Users\Vincent\Documents\Masterarbeit\Data\external_dataset\KIT_dataset_from_kaggle\Printing_Errors\general_data\all_images_no_filter_modified.csv"
csv_path_black_bed=r"C:\Users\Vincent\Documents\Masterarbeit\Data\external_dataset\KIT_dataset_from_kaggle\Printing_Errors\general_data\black_bed_all_modified.csv"

# define csv files where you want to save the predictions
losses_val_train_path = 'losses_val_train_final.csv'
preds_target_train_val_path = 'preds_target_val_final.csv'
preds_targets_test_path = 'preds_targets_test_final.csv'

#define the transforms

transform_augmentation = T.Compose(
    [
        T.RandomHorizontalFlip(),
        T.Resize((224, 224)),
        T.RandomRotation(30),
        T.ToTensor(),
        T.RandomGrayscale(p=0.2),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.ConvertImageDtype(torch.float32),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

transform_normal = T.Compose(
    [
        T.ToTensor(),
        T.Resize((224, 224)),
        T.ConvertImageDtype(torch.float32),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# initinialize the datamodule


data_module = DataModule_4classes(csv=csv_path_black_bed,root_dir=image_path_all_images,transform_train=transform_augmentation,val_split=0.1,test_split=0.1,transform_val_test=transform_normal ,batch_size=32)

## write the header of the csv files
with open(losses_val_train_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['epoch','train_loss', 'train_accuracy', 'val_loss', 'val_accuracy', 'train_f1_score', 'val_f1_score'])
with open(preds_target_train_val_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['epoch', 'val_preds', 'val_targets'])
with open(preds_targets_test_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['epoch','test_preds', 'test_targets'])

# initialize the model

model= ViTLightningModule(pretrained=True,num_labels=3,patch=32,learning_rate=2e-04,path_val_train_losses=losses_val_train_path,path_preds_target_train_val=preds_target_train_val_path,path_preds_targets=preds_targets_test_path)

#define the logger 

# define the logger
tb_logger = TensorBoardLogger("tb_logs/", name="vitbase_32_pretrained")
#csv_logger = CSVLogger(save_dir='cv_logs/', name='vitbase_32_pretrained')




#define the callbacks
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5,verbose=True, mode='max')
progress_bar = TQDMProgressBar(refresh_rate=10)
checkpoint_callback = ModelCheckpoint(
    monitor='val_accuracy',
    mode='max',
    filename='best_model_{epoch:02d}-{val_accuracy:.2f}',
    save_top_k=3,
    dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
)



# define the trainer
trainer = pl.Trainer(
    accelerator="gpu",
    min_epochs=1, 
    max_epochs=2,
    fast_dev_run=True,
    callbacks=[early_stopping, progress_bar,checkpoint_callback],
    logger=tb_logger,
)

trainer.fit(model, data_module)
#trainer.validate(model, data_module)
best_model_path = checkpoint_callback.best_model_path
best_model = ViTLightningModule.load_from_checkpoint(best_model_path)
trainer.test(best_model, data_module)