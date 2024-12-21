import os
import torch
import time
import numpy as np

from tqdm import tqdm
from org.symplesys.ocr.datasets import KMNISTImageDataset

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

def train(model, device, source_folder, batch_size, num_epochs=25):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    return train_model(model, criterion, optimizer, scheduler, device, source_folder, batch_size, num_epochs)

def train_model(model, criterion, optimizer, scheduler, device, source_folder, batch_size, num_epochs=25):
    num_workers = 4
    dataloaders = {}
    data_loader_train = torch.utils.data.DataLoader(KMNISTImageDataset(is_train=True, source_folder=source_folder),batch_size=batch_size,shuffle=True,num_workers=num_workers)
    data_loader_val = torch.utils.data.DataLoader(KMNISTImageDataset(is_train=False, source_folder=source_folder),batch_size=batch_size,shuffle=True,num_workers=num_workers)
    dataloaders["train"] = data_loader_train
    dataloaders["val"] = data_loader_val
    dataset_sizes = {"train": len(data_loader_train), "val": len(data_loader_val)}

    since = time.time()

    best_model_params_path = os.path.join("checkpoints", 'best_model_params.pt')

    torch.save(model.state_dict(), best_model_params_path)
    best_acc = 0.0

    model.to(device)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), best_model_params_path)

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(torch.load(best_model_params_path, weights_only=True))
    return model









