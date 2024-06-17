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

sys.path.append(r"C:\Users\Vincent\Documents\Masterarbeit\Src\Deep_Learning\utils")
from datamodule import ImageDataset_with_labels_correct_anomaly,CustomImageDatasetFromCSV


torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


"""
# was specified as transform_train before 
transform_augmentation = T.Compose(
    [
        T.RandomHorizontalFlip(),
        T.Resize((224, 224)),
        T.RandomVerticalFlip(),
        T.RandomRotation(30),
        T.ToTensor(),
        T.ConvertImageDtype(torch.float32),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
"""
# Todo: implement different transforms for training and validation with the csv file (maybe with sklearn or with subset)


transform_normal = T.Compose(
    [
        T.ToTensor(),
        T.Resize((224, 224)),
        T.ConvertImageDtype(torch.float32),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(0)


# for 2 class classification
# path_to_folders = r"C:\Users\Vincent\Documents\Masterarbeit\Data\external_dataset\KIT_dataset_from_kaggle\Anomalieerkennung_2_classes"
# dataset = ImageDataset_with_labels(root_dir=path_to_folders, transform=transform_train)

# for multiclass classification
image_path = r"C:\Users\Vincent\Documents\Masterarbeit\Data\external_dataset\KIT_dataset_from_kaggle\Printing_Errors\images\all_images256"
csv_path = r"C:\Users\Vincent\Documents\Masterarbeit\Data\external_dataset\KIT_dataset_from_kaggle\Printing_Errors\general_data\all_images_no_filter_modified.csv"

dataset_normal = CustomImageDatasetFromCSV(
    csv_path, image_path, transform=transform_normal
)

# Define the size of the train and validation sets
train_size = int(0.8 * len(dataset_normal))
val_size = len(dataset_normal) - train_size

# Split the dataset
train_dataset, val_dataset = random_split(dataset_normal, [train_size, val_size])


validation_dataloader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=True,
    worker_init_fn=seed_worker,
    generator=g,
)
train_dataloader = DataLoader(
    train_dataset, batch_size=16, shuffle=True, worker_init_fn=seed_worker, generator=g
)


class ViTBase16(nn.Module):
    def __init__(self, n_classes, pretrained=False):

        super(ViTBase16, self).__init__()

        if pretrained:
            self.model = timm.create_model("vit_base_patch16_224", pretrained=True)
        else:
            self.model = timm.create_model("vit_base_patch16_224", pretrained=False)

        self.model.head = nn.Linear(self.model.head.in_features, n_classes)

    def forward(self, x):
        x = self.model(x)
        return x


epoch = 20
vit_model = ViTBase16(4, pretrained=True)
loss_history = [[], []]  # train, val
accuracy_history = [[], []]  # train, val
acc_epoch_history = [[], []]
loss_epoch_history = [[], []]

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Selected device: {device}")

vit_model.to(device)

optimizer = torch.optim.Adam(vit_model.parameters(), lr=2e-04)
criterion = nn.CrossEntropyLoss()

for e in range(epoch):
    vit_model.train()
    print(f"====================== EPOCH {e+1} ======================")
    print("Training.....")
    for i, (data, target) in enumerate(train_dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = vit_model(data)
        # pdb.set_trace()
        loss = criterion(output, target)
        loss.backward()

        nn.utils.clip_grad_norm_(vit_model.parameters(), 3)
        accuracy = (output.argmax(dim=1) == target).float().mean()

        loss_history[0].append(loss.item())
        accuracy_history[0].append(accuracy)

        optimizer.step()

        if i % 10 == 0:
            print(
                f"MINIBATCH {i+1}/{train_dataloader.__len__()} TRAIN ACC : {accuracy_history[0][-1]}  TRAIN LOSS : {loss_history[0][-1]}"
            )


print("Validation.....")
vit_model.eval()

with torch.no_grad():
    for i, (data, target) in enumerate(validation_dataloader):
        data, target = data.to(device), target.to(device)
        output = vit_model(data)
        loss = criterion(
            output,
            target.view(
                -1,
            ),
        )
        accuracy = (output.argmax(dim=1) == target).float().mean()
        loss_history[1].append(loss.item())
        accuracy_history[1].append(accuracy)

acc_epoch_history[0].append(
    sum(accuracy_history[0][-1 : -train_dataloader.__len__() : -1])
    / train_dataloader.__len__()
)
acc_epoch_history[1].append(
    sum(accuracy_history[1][-1 : -validation_dataloader.__len__() : -1])
    / validation_dataloader.__len__()
)

loss_epoch_history[0].append(
    sum(loss_history[0][-1 : -train_dataloader.__len__() : -1])
    / train_dataloader.__len__()
)
loss_epoch_history[1].append(
    sum(loss_history[1][-1 : -validation_dataloader.__len__() : -1])
    / validation_dataloader.__len__()
)

print("====================================================")
print(
    f"TRAIN ACC : {acc_epoch_history[0][-1]}  TRAIN LOSS : {loss_epoch_history[0][-1]}"
)
print(f"VALL ACC : {acc_epoch_history[1][-1]}  VAL LOSS : {loss_epoch_history[1][-1]}")
print("====================================================")

torch.save(
    {
        "epoch": e,
        "model_state_dict": vit_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss_epoch_history[0][-1],
        "acc": acc_epoch_history[0][-1],
    },
    "./model_checkpoint_4_clasess.pt",
)
