import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import matplotlib.pyplot as plt

import hydra
from omegaconf import DictConfig, OmegaConf

from models import ResNet50, ResNet101, ResNet152

data_dir = '/content/gsn-hydra/dataset'

transforms = {
    'Training': transforms.Compose
    ([
        transforms.RandomResizedCrop(224),
        #transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'Validation': transforms.Compose
    ([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

val_losses = []
train_losses = []
val_accuracy = []
train_accuracy = []

def train(model, criterion, optimizer, scheduler, epochs, loader, dataset_sizes):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    val_acc = 0.0
    

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))

        for x in range(2):
            if x == 0:
                model.train()  
                stage = 'Training'
            elif x == 1:
                model.eval()  
                stage = 'Validation' 
            else: 
               break

            partial_loss = 0.0
            partial_corrects = 0
            val_loss = 0.0

            for data, targets in loader[stage]:
                device_data = data.to(device)
                device_targets = targets.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(x == 0): # Training
                    outputs = model(device_data)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, device_targets)

                    if x == 0: # Training
                        loss.backward()
                        optimizer.step()

              
                partial_loss += loss.item() * device_data.size(0)
                partial_corrects += torch.sum(preds == device_targets.data)
            if x == 0: # Training
                scheduler.step()

            loss_ = partial_loss / dataset_sizes[stage]
            acc = partial_corrects.double() / dataset_sizes[stage]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(stage, loss_, acc))
            if x == 1:
              val_losses.append(loss_)
              val_accuracy.append(acc)
            else:
              train_losses.append(loss_)
              train_accuracy.append(acc)
            
            if x == 1 and acc > val_acc: # Validation
                val_acc = acc
                best_model_wts = copy.deepcopy(model.state_dict())
           
            if x == 1: # Validation, early stopping
              if val_loss < loss_ and val_loss != 0:
                  model.load_state_dict(best_model_wts)
                  return model
              else:
                val_loss = loss_

        print()

    model.load_state_dict(best_model_wts)
    return model


@hydra.main(config_path="./conf", config_name="config")
def parameters(cfg: DictConfig) -> None:
    #print(OmegaConf.to_yaml(cfg))
    workers = 2
    optimizer = cfg['optimizer']['name']
    learning_r = cfg['learning_rate']['lr']
    momentum_ = cfg['momentum']['value']
    dropout = cfg['dropout']['value']
    batch = cfg['batch_size']['size']
    epochs = cfg['epochs']['number']
    model =cfg['model']['name']
    

    image_split = {i: datasets.ImageFolder(os.path.join(data_dir, i), transforms[i])for i in ['Training', 'Validation']}

    loader = {i: torch.utils.data.DataLoader(image_split[i], batch_size=batch, shuffle=True, num_workers=workers) for i in ['Training', 'Validation']}

    dataset_sizes = {i: len(image_split[i]) for i in ['Training', 'Validation']}

    class_names = image_split['Training'].classes

    nr_epochs = epochs

    if model == "resnet50":
        res = ResNet50()
    elif model == "resnet101":
        res = ResNet101
    elif model == "resnet152":
        res = ResNet152
    
    res.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.SGD(res.parameters(), lr=learning_r, momentum=momentum_)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train(res, criterion, optimizer_ft, exp_lr_scheduler, nr_epochs, loader, dataset_sizes)

    PATH = "model.pt"
    torch.save({
                'model_state_dict': model_ft.state_dict(),
                'optimizer_state_dict': optimizer_ft.state_dict(),
                }, PATH)

    
if __name__ == "__main__":

    parameters()
