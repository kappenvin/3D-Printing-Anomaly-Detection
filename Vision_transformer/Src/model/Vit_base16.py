import torch
import timm
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os 
import pandas as pd
import csv
import pdb





class ViTBase16(nn.Module):
    def __init__(self, n_classes, pretrained=False,patch=16):

        super(ViTBase16, self).__init__()
        if patch==16:
            if pretrained:
                self.model = timm.create_model("vit_base_patch16_224", pretrained=True)
            else:
                self.model = timm.create_model("vit_base_patch16_224", pretrained=False)
        else:
            if pretrained:
                self.model = timm.create_model("vit_base_patch32_224", pretrained=True)
            else:
                self.model = timm.create_model("vit_base_patch32_224", pretrained=False)

        self.model.head = nn.Linear(self.model.head.in_features, n_classes)

    def forward(self, x):
        x = self.model(x)
        return x
    
class ViTLightningModule(pl.LightningModule):
    def __init__(self,pretrained, num_labels=4,learning_rate=2e-04,patch=16,path_val_train_losses=None,path_preds_target_train_val=None,path_preds_targets=None):
        super(ViTLightningModule, self).__init__()

    
        #self.save_hyperparameters()

        if patch==16:
            if pretrained:
                self.model = timm.create_model("vit_base_patch16_224", pretrained=True)
            else:
                self.model = timm.create_model("vit_base_patch16_224", pretrained=False)
        else:
            if pretrained:
                self.model = timm.create_model("vit_base_patch32_224", pretrained=True)
            else:
                self.model = timm.create_model("vit_base_patch32_224", pretrained=False)
        self.model.head = nn.Linear(self.model.head.in_features, num_labels)
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_labels)
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=num_labels)

        # define the paths to the csv files
        self.path_val_train_losses = path_val_train_losses
        self.path_preds_target_train_val = path_preds_target_train_val
        self.path_preds_targets = path_preds_targets

        # store the predictions in arrays
        self.val_preds = []
        self.val_targets = []
        self.test_preds = []
        self.test_targets = []
        self.train_losses = []
        self.val_losses = []
        self.f1_score_list_train=[]
        self.accuracy_list_train=[]
        self.f1_score_list_val=[]
        self.accuracy_list_val=[]
        self.f1_score_list_test=[]
        self.accuracy_list_test=[]


    def forward(self, x):
        x= self.model(x)
        return x
    
    
    def common_step(self, batch):
        data, target = batch
        logits = self(data)# forwrad pass for the model --> it is the output
        loss = self.criterion(logits, target)
        predictions = logits.argmax(dim=1)

        return loss, predictions,target,logits
    
    def training_step(self, batch):
        loss_train, predictions_train,target_train,logits_train = self.common_step(batch)
        accuracy_train = self.accuracy(predictions_train, target_train) 
        f1_score_train = self.f1_score(predictions_train, target_train)
        
        self.log_dict({'loss/train': loss_train, 'train_accuracy': accuracy_train, 'train_f1_score': f1_score_train, 'step': self.current_epoch},
                      on_step=False, on_epoch=True, prog_bar=True)
        

        self.train_losses.append(loss_train.item())
        self.f1_score_list_train.append(f1_score_train.item())
        self.accuracy_list_train.append(accuracy_train.item())
        
        return loss_train
    
    def validation_step(self, batch):
        loss_val, predictions_val,target_val,logits_val = self.common_step(batch)
        accuracy = self.accuracy(predictions_val, target_val) 
        f1_score = self.f1_score(predictions_val, target_val)
        
        self.log_dict({'loss/val': loss_val, 'val_accuracy': accuracy, 'val_f1_score': f1_score,'step': self.current_epoch},
                      on_step=False, on_epoch=True, prog_bar=True)
        
        self.val_losses.append(loss_val.item())
        
        # append the predictions and targets to the buffers for the confusion matrix
        self.val_preds.append(predictions_val.cpu().numpy())
        self.val_targets.append(target_val.cpu().numpy())
        self.f1_score_list_val.append(f1_score.item())
        self.accuracy_list_val.append(accuracy.item())



        return loss_val
    
    def test_step(self, batch):
        loss_test, predictions_test,target_test,logits = self.common_step(batch)
        accuracy = self.accuracy(predictions_test, target_test) 
        f1_score = self.f1_score(predictions_test, target_test)
        

        self.log_dict({'test_loss': loss_test, 'test_accuracy': accuracy, 'test_f1_score': f1_score},
                      on_step=False, on_epoch=True, prog_bar=True)
        
        
        self.test_preds.append(predictions_test.cpu().numpy())
        self.test_targets.append(target_test.cpu().numpy())
        self.f1_score_list_test.append(f1_score.item())
        self.accuracy_list_test.append(accuracy.item())


        return loss_test
    
    
    def on_validation_epoch_end(self):
        # Compute confusion matrix
        val_preds = np.concatenate(self.val_preds)
        val_targets = np.concatenate(self.val_targets)
        
        

        # Save losses and mtrices to  a CSV file
        with open(self.path_val_train_losses,'a' ,newline='') as csvfile:
            writer = csv.writer(csvfile)
            for train_loss, accuracy_train,val_losses, accuracy_val,f1_train,f1_val  in zip(self.train_losses, self.accuracy_list_train,self.val_losses,self.accuracy_list_val,self.f1_score_list_train,self.f1_score_list_val):
                writer.writerow([self.current_epoch,train_loss, accuracy_train, val_losses, accuracy_val, f1_train, f1_val])

        with open(self.path_preds_target_train_val, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for pred, target in zip(val_preds, val_targets):
                writer.writerow([self.current_epoch,pred, target])
            

        # clean the buffers
        self.val_preds = []
        self.val_targets = []
        self.train_losses = []
        self.val_losses = []
        self.f1_score_list_train=[]
        self.accuracy_list_train=[]
    
    def on_test_epoch_end(self):
        # Compute confusion matrix
        test_preds = np.concatenate(self.test_preds)
        test_targets = np.concatenate(self.test_targets)
        

        
        # Clear stored predictions and targets
        self.test_preds = []
        self.test_targets = []

        with open(self.path_preds_targets, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for pred, target in zip(test_preds, test_targets):
                writer.writerow([self.current_epoch,pred, target])
            

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer