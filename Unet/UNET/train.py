import os
import time
from glob import glob
import datetime
import sys
import getopt

import torch
from torch.utils.data import DataLoader
import torch.nn as nn


from data import DriveDataset
from model import build_unet
from loss import DiceLoss, DiceBCELoss
from utils import seeding, create_dir, epoch_time

def train(model, loader, optimizer, loss_fn, device):
    epoch_loss = 0.0

    model.train()
    for x, y in loader:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss = epoch_loss/len(loader)
    return epoch_loss

def evaluate(model, loader, loss_fn, device):
    epoch_loss = 0.0

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()

        epoch_loss = epoch_loss/len(loader)
    return epoch_loss

if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    # Remove 1st argument from the
    # list of command line arguments
    argumentList = sys.argv[1:]
     
    # Options
    options = "b:e:d:"
     
    # Long options
    long_options = ["batch=", "epochs=", "data=","img="]
     
    try:
        # Parsing argument
        arguments, values = getopt.getopt(argumentList, options, long_options)
         
        # checking each argument
        for currentArgument, currentValue in arguments:
     
            if currentArgument in ("-b", "--batch"):
                batch_size = int(currentValue)
                 
            elif currentArgument in ("-e", "--epochs"):
                num_epochs = int(currentValue)
                 
            elif currentArgument in ("-d", "--data"):
                path = currentValue
                
             
    except getopt.error as err:
        # output error, and return with an error code
        print (str(err))

    # Get the current date and time
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M")

    """ Directories """
    create_dir("files")
    

    """ Load dataset """
    train_x = sorted(glob(os.path.join("..",path,"train","image","*")))
    train_y = sorted(glob(os.path.join("..",path,"train","mask","*")))

    valid_x = sorted(glob(os.path.join("..",path,"val","image","*")))
    valid_y = sorted(glob(os.path.join("..",path,"val","mask","*")))

    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n"
    print(data_str)

    """ Hyperparameters """
    H = 512
    W = 512
    size = (H, W)
    #batch_size = 2
    #num_epochs = 5
    lr = 1e-4
    #checkpoint_path = os.path.join(".","files","checkpoint.pth")
    checkpoint_path = os.path.join(".","files",f"{formatted_datetime}_checkpoint.pth")

    """ Dataset and loader """
    train_dataset = DriveDataset(train_x, train_y)
    valid_dataset = DriveDataset(valid_x, valid_y)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    #device = torch.device('cuda')   ## GTX 1060 6GB
    device = torch.device('cpu')   ## GTX 1060 6GB
    model = build_unet()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    loss_fn = DiceBCELoss()

    """ Training the model """
    best_valid_loss = float("inf")

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss = train(model, train_loader, optimizer, loss_fn, device)
        valid_loss = evaluate(model, valid_loader, loss_fn, device)

        
        """ Saving the model """
        if valid_loss < best_valid_loss:
            data_str = f"Valid loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}. Saving checkpoint: {checkpoint_path}"
            print(data_str)

            best_valid_loss = valid_loss
            torch.save(model.state_dict(), checkpoint_path)
        
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n'
        data_str += f'\tTrain Loss: {train_loss:.3f}\n'
        data_str += f'\t Val. Loss: {valid_loss:.3f}\n'
        print(data_str)
