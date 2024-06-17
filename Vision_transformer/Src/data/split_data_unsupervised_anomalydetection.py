import os
import shutil
import random

# Define the input directories
data_dir = r"C:\Users\Vincent\Documents\Masterarbeit\Data\external_dataset\KIT_dataset_from_kaggle"
anomalies_dir = os.path.join(data_dir, "anomalies")
good_dir = os.path.join(data_dir, "good")


# Define the output directories
output_dirs = {
    "train/good": os.path.join(data_dir, "train/good"),
    "val/good": os.path.join(data_dir, "val/good"),
    "val/anomalies": os.path.join(data_dir, "val/anomalies"),
    "test/good": os.path.join(data_dir, "test/good"),
    "test/anomalies": os.path.join(data_dir, "test/anomalies"),
}

# Create output directories if they do not exist
for path in output_dirs.values():
    os.makedirs(path, exist_ok=True)


# Function to split data into train, val, and test
def split_data(src_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    files = os.listdir(src_dir)
    random.shuffle(files)
    train_idx = int(len(files) * train_ratio)
    val_idx = int(len(files) * (train_ratio + val_ratio))

    train_files = files[:train_idx]
    val_files = files[train_idx:val_idx]
    test_files = files[val_idx:]

    return train_files, val_files, test_files


# Split the good images
good_train_files, good_val_files, good_test_files = split_data(good_dir)


# Move the good images to the appropriate folders
for file in good_train_files:
    shutil.copy(os.path.join(good_dir, file), output_dirs["train/good"])
for file in good_val_files:
    shutil.copy(os.path.join(good_dir, file), output_dirs["val/good"])
for file in good_test_files:
    shutil.copy(os.path.join(good_dir, file), output_dirs["test/good"])

# Split the anomaly images
_, anomaly_val_files, anomaly_test_files = split_data(
    anomalies_dir, train_ratio=0.0, val_ratio=0.5, test_ratio=0.5
)

# Move the anomaly images to the appropriate folders
for file in anomaly_val_files:
    shutil.copy(os.path.join(anomalies_dir, file), output_dirs["val/anomalies"])
for file in anomaly_test_files:
    shutil.copy(os.path.join(anomalies_dir, file), output_dirs["test/anomalies"])

print("Data split and copied successfully!")
