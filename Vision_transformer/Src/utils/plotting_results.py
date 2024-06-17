import pandas as pd
import torch
import torchmetrics
from sklearn.metrics import accuracy_score, f1_score,confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os 

import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_and_save_confusion_matrix(targets, preds, path_to_save_output, normalize=False, dataset_type='test'):
    """
    Plots and saves either absolute or normalized confusion matrix using seaborn.
     
    Parameters:
    targets (array-like): True labels.
    preds (array-like): Predicted labels.
    path_to_save_output (str): Directory path where the confusion matrix image will be saved.
    normalize (bool): If True, the confusion matrix will be normalized. Default is False.
    dataset_type (str): The type of dataset, either 'validation' or 'test'. Default is 'test'.
    """
    
    # Compute confusion matrix
    cm = confusion_matrix(targets, preds, normalize='true' if normalize else None)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues', 
                xticklabels=['Good', 'Underextrusion', 'Stringing'], 
                yticklabels=['Good', 'Underextrusion', 'Stringing'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('Actual Labels')
    
    # Determine the title and file name based on normalization and dataset type
    if normalize:
        title = f'Confusion Matrix relative for {dataset_type.capitalize()}'
        file_name = f'confusion_matrix_normalized_{dataset_type}.png'
    else:
        title = f'Confusion Matrix Absolute for {dataset_type.capitalize()}'
        file_name = f'confusion_matrix_absolute_{dataset_type}.png'
    
    plt.title(title)
    
    # Save the figure instead of showing
    output_directory = os.path.join(path_to_save_output, file_name)
    plt.savefig(output_directory)
    plt.close()

import numpy as np
import matplotlib.pyplot as plt
import os

def plot_and_save_train_val_losses(losses_df, save_path):
    """
    Calculates and plots train and validation losses over epochs, and saves the plot as an image.

    Parameters:
    losses_df (pd.DataFrame): DataFrame containing columns 'epoch', 'train_loss', and 'val_loss'.
    save_path (str): Path to save the plot image.
    """
    # Get unique epochs and sort them
    epochs = sorted(losses_df['epoch'].unique())
    train_losses = []
    val_losses = []

    # Calculate mean train and validation loss for each epoch
    for epoch in epochs:
        epoch_data = losses_df[losses_df['epoch'] == epoch]
        train_loss = np.mean(epoch_data['train_loss'].values)
        val_loss = np.mean(epoch_data['val_loss'].values)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

    # Plotting
    plt.figure(figsize=(12, 6))

    # Plot train losses
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')

    # Plot validation losses
    plt.plot(epochs, val_losses, label='Validation Loss', marker='o')

    # Set x-axis and y-axis limits
    plt.xlim(0, max(epochs) + 1)
    plt.ylim(0, max(max(train_losses), max(val_losses)) + 0.1)

    # Add title and labels
    plt.title('Training and Validation Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Show plot with grid
    plt.grid(True)

    output_directory = os.path.join(save_path, "train_val_losses.png")
    # Save the plot
    plt.savefig(output_directory)
    plt.close()
    
    print(f"Plot saved at: {save_path}")

def plot_and_save_accuracy_f1score(df_val, save_path,dataset_type='Validation'):
    """
    Calculates and plots accuracy and F1 score per epoch, and saves the plot as an image.

    Parameters:
    df_val (pd.DataFrame): DataFrame containing columns 'epoch', 'val_targets', and 'val_preds'.
    save_path (str): Path to save the plot image.
    """
    # Get unique epochs and sort them
    epochs = sorted(df_val['epoch'].unique())
    accuracies = []
    f1_scores = []

    # Calculate accuracy and F1 score for each epoch
    for epoch in epochs:
        epoch_data = df_val[df_val['epoch'] == epoch]
        targets = epoch_data['val_targets']
        preds = epoch_data['val_preds']
        
        acc = accuracy_score(targets, preds)
        f1 = f1_score(targets, preds, average='weighted')  # You can change the average method as needed
        
        accuracies.append(acc)
        f1_scores.append(f1)

    # Plotting
    plt.figure(figsize=(12, 6))

    # Plot accuracy
    plt.plot(epochs, accuracies, label='Accuracy', marker='o')

    # Plot F1 score
    plt.plot(epochs, f1_scores, label='F1 Score', marker='o')

    # Add title and labels
    plt.title('Accuracy and F1 Score '+ dataset_type)
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()

    # Show plot with grid
    plt.grid(True)

    output_directory = os.path.join(save_path, "val_accuracy_f1score.png")
    # Save the plot
    plt.savefig(output_directory)
    plt.close()
    
    print(f"Plot saved at: {output_directory}")

# Example usage:
# plot_and_save_train_val_losses(losses_df, 'path/to/save/plot.png')


def main():
    # define path to csv file with targets and predictions for test and  val
    path_to_csv_val = r'C:\Users\Vincent\Documents\Masterarbeit\Src\Deep_Learning\Vision_transformer\Src\results\own_csv_files\preds_target_val_final.csv'
    df_val=pd.read_csv(path_to_csv_val)
    path_to_csv_test = r'C:\Users\Vincent\Documents\Masterarbeit\Src\Deep_Learning\Vision_transformer\Src\results\own_csv_files\preds_targets_test_final.csv'
    df_test=pd.read_csv(path_to_csv_test)

    # define path to losses
    path_to_losses = r'C:\Users\Vincent\Documents\Masterarbeit\Src\Deep_Learning\Vision_transformer\Src\results\own_csv_files\losses_val_train_final.csv'
    losses_df=pd.read_csv(path_to_losses)



    # define the path to save the plots of the confusion matrices
    path_to_save_output=r'C:\Users\Vincent\Documents\Masterarbeit\Src\Deep_Learning\Vision_transformer\Src\results\plots'
    # get best mdodel predictions,based on model with highest accuracy on validation set
    df_val_best=df_val[df_val['epoch'] == 9]



    # plot and save confusion matrix 
    '''
    plot_and_save_confusion_matrix(df_val_best['val_targets'], df_val_best['val_preds'], path_to_save_output, normalize=False, dataset_type='validation')
    plot_and_save_confusion_matrix(df_val_best['val_targets'], df_val_best['val_preds'], path_to_save_output, normalize=True, dataset_type='validation')
    plot_and_save_confusion_matrix(df_test['test_targets'], df_test['test_preds'], path_to_save_output, normalize=False, dataset_type='test')
    plot_and_save_confusion_matrix(df_test['test_targets'], df_test['test_preds'], path_to_save_output, normalize=True, dataset_type='test')
    '''
    #plot_and_save_train_val_losses(losses_df, path_to_save_output)
    plot_and_save_accuracy_f1score(df_val, path_to_save_output,dataset_type='Validation')

if __name__ == '__main__':
    main()