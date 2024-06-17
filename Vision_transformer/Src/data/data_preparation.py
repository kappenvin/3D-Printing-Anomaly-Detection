import pandas as pd
from sklearn.model_selection import train_test_split
import os 
from pytorch_lightning.loggers import TensorBoardLogger
def split_dataset_shapes(csv_path, ratio=0.2):
    """
    Split the dataset based on unique shapes.

    Parameters:
    - csv_path (str): The path to the CSV file containing the dataset.
    - ratio (float): The ratio of the test dataset size to the total dataset size. Default is 0.2.

    Returns:
    - train_df (pandas.DataFrame): The training dataset containing the shapes in the training set.
    - test_df (pandas.DataFrame): The test dataset containing the shapes in the test set.
    - shapes_train (numpy.ndarray): The array of shapes in the training set.
    """
    data = pd.read_csv(csv_path)

    unique_shapes = data['shape'].unique()
    shapes_train, shapes_test = train_test_split(unique_shapes, test_size=ratio, random_state=42)
    train_df = data[data['shape'].isin(shapes_train)]
    test_df = data[data['shape'].isin(shapes_test)]

    return train_df, test_df, shapes_train


def create_tensorboard_logger(fold):
    """
    Creates a TensorBoardLogger for a specific fold.

    Parameters:
    - fold (int): The fold number.

    Returns:
    - TensorBoardLogger: The TensorBoardLogger object.

    """
    log_dir = os.path.join("tb_logs", f"vitbase_32_pretrained_fold{fold}","checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    return TensorBoardLogger(log_dir)








