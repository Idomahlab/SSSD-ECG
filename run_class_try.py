import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from resnetEcg import ResNet1d  # Make sure resnetEcg.py is in the same directory or in the Python path
torch.cuda.manual_seed(10)
from tqdm import tqdm
import pandas as pd
import pathlib
from classifier_utils import train
from classifier_utils import evaluate
# 1. Load and prepare the data
def load_data(data_path, labels_path):
    data = np.load(data_path)
    labels = np.load(labels_path)
    return data, labels

# 2. Create a PyTorch dataset and dataloader
def create_dataloader(data, labels, batch_size=32, shuffle=True):
    dataset = TensorDataset(torch.FloatTensor(data), torch.FloatTensor(labels))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

# 3. Define the training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"Val Loss: {val_loss/len(val_loader):.4f}")
        print()
    torch.save(model.state_dict(), 'res_model_weights.pth')


# Main script
if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    foldername = pathlib.PurePath("./check_points/resnet_model")
    print('model folder:', foldername)
    os.makedirs(foldername, exist_ok=True)
    # Load data
    x_train, y_train = load_data('/home/ido.mahlab/SSSD-ECG/data/ptbxl_train_data.npy',
                                         "/home/ido.mahlab/SSSD-ECG/labels/ptbxl_train_labels.npy")
                                       
    x_val, y_val = load_data('/home/ido.mahlab/SSSD-ECG/data/ptbxl_validation_data.npy',
                                     "/home/ido.mahlab/SSSD-ECG/labels/ptbxl_validation_labels.npy")
    # label_to_drop = 61  # Replace with the index of the label you want to drop
    # # Drop label column from training and validation data
    # y_train = np.delete(y_train, label_to_drop, axis=1)
    # y_val = np.delete(y_val, label_to_drop, axis=1)

    # Create dataloaders
    train_loader = create_dataloader(x_train, y_train, batch_size=32, shuffle=True)
    val_loader = create_dataloader(x_val, y_val, batch_size=32, shuffle=False)
    
    # Initialize model, criterion, and optimizer
    # filter_size = [64, 128, 196, 256]
    # net_seq_lengh = [1000, 500, 250, 125]
    filter_size = [64, 128, 196, 256, 512, 512]
    net_seq_lengh = [1000, 500, 250, 125, 125, 125]
    model = ResNet1d(input_dim=(12, 1000), blocks_dim=list(zip(filter_size, net_seq_lengh)), n_classes=71,kernel_size = 17, dropout_rate = 0.8)
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    tqdm.write("Define loss...")
    criterion = nn.BCELoss()
    training_generator = torch.utils.data.DataLoader([*zip(x_train, y_train)], batch_size=64)
    validation_generator = torch.utils.data.DataLoader([*zip(x_val, y_val)], batch_size=64)
    tqdm.write("Define scheduler...")
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=12,
                                                         min_lr=1e-7,
                                                         factor=0.1)

    tqdm.write("Training...")
    start_epoch = 0
    best_loss = np.Inf
    history = pd.DataFrame(columns=['epoch', 'train_loss', 'valid_loss', 'lr'])
    for ep in range(start_epoch, 80):
        train_loss = train(model, ep, train_loader, device, criterion, optimizer)
        valid_loss = evaluate(model, ep, val_loader, device, criterion)
        # Save best model
        if valid_loss < best_loss:
            print(f"{valid_loss} better than best loss saving model")
            # Save model
            torch.save({'epoch': ep,
                        'model': model.state_dict(),
                        'valid_loss': valid_loss,
                        'optimizer': optimizer.state_dict()},
                        str(foldername / 'model_tmp.pth'))
            # Update best validation loss
            best_loss = valid_loss
        # Get learning rate
        for param_group in optimizer.param_groups:
            learning_rate = param_group["lr"]
        # Interrupt for minimum learning rate
        if learning_rate < 1e-7:
            break
        # Print message
        tqdm.write('Epoch {:2d}: \tTrain Loss {:.6f} ' \
                    '\tValid Loss {:.6f} \tLearning Rate {:.7f}\t'
                    .format(ep, train_loss, valid_loss, learning_rate))
        # Save history
        history = pd.concat([history, pd.DataFrame({"epoch": ep, "train_loss": train_loss,
                                                    "valid_loss": valid_loss, "lr": learning_rate}, index=[0])],
                            ignore_index=True)
        history.to_csv(foldername / 'history.csv', index=False)
        # Update learning rate
        scheduler.step(valid_loss)
    tqdm.write("Done!")