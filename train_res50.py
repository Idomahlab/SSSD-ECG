import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from resnetEcg import ResNet1d 
torch.cuda.manual_seed(10)
from tqdm import tqdm
import pandas as pd
import pathlib
from classifier_utils import train
from classifier_utils import evaluate
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold

class BottleneckBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BottleneckBlock1D, self).__init__()
        mid_channels = out_channels // 4
        self.conv1 = nn.Conv1d(in_channels, mid_channels, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm1d(mid_channels)
        self.conv2 = nn.Conv1d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(mid_channels)
        self.conv3 = nn.Conv1d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet1D(nn.Module):
    def __init__(self, block, layers, num_classes=71):
        super(ResNet1D, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv1d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 256, layers[0])
        self.layer2 = self._make_layer(block, 512, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 1024, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 2048, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(2048, num_classes)
    
    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# Initialize a ResNet-50
model = ResNet1D(BottleneckBlock1D, [3, 4, 6, 3], num_classes=71)

# 1. Load and prepare the data
def load_data(data_path, labels_path):
    data = np.load(data_path)
    labels = np.load(labels_path)
    return data, labels

# 2. Standardize the data
def standardize_data(x_train, x_val):
    scaler = StandardScaler()
    n_samples, n_channels, n_features = x_train.shape
    
    x_train_reshaped = x_train.reshape(n_samples * n_channels, n_features)
    x_train_scaled = scaler.fit_transform(x_train_reshaped).reshape(n_samples, n_channels, n_features)
    
    x_val_reshaped = x_val.reshape(x_val.shape[0] * x_val.shape[1], x_val.shape[2])
    x_val_scaled = scaler.transform(x_val_reshaped).reshape(x_val.shape[0], x_val.shape[1], x_val.shape[2])
    
    return x_train_scaled, x_val_scaled

# 3. Create a PyTorch dataset and dataloader
def create_dataloader(data, labels, batch_size=128, shuffle=True):
    dataset = TensorDataset(torch.FloatTensor(data), torch.FloatTensor(labels))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

# 4. Define the training function
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
    x, y = load_data('/home/ido.mahlab/SSSD-ECG/Res50_Data/combined_samples.npy',
                     "/home/ido.mahlab/SSSD-ECG/Res50_Data/combined_labels.npy")
    
    # Standardize the data
    scaler = StandardScaler()
    x = x.reshape(-1, x.shape[2])
    x = scaler.fit_transform(x)
    x = x.reshape(-1, 12, x.shape[1])

    # Define the cross-validation strategy
    kf = KFold(n_splits=5, shuffle=True, random_state=10)
    
    start_epoch = 0
    best_loss = np.Inf
    history = pd.DataFrame(columns=['epoch', 'train_loss', 'valid_loss', 'lr'])

    for fold, (train_idx, val_idx) in enumerate(kf.split(x)):
        print(f"Starting fold {fold + 1}...")
        
        x_train, x_val = x[train_idx], x[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Create dataloaders
        train_loader = create_dataloader(x_train, y_train, batch_size=128, shuffle=True)
        val_loader = create_dataloader(x_val, y_val, batch_size=128, shuffle=False)
        
        # Initialize model, criterion, and optimizer
        model = ResNet1D(BottleneckBlock1D, [3, 4, 6, 3], num_classes=71)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.00001)
        
        tqdm.write("Define loss...")
        criterion = nn.BCEWithLogitsLoss()
        
        tqdm.write("Define scheduler...")
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=12, min_lr=1e-7, factor=0.1)
        
        tqdm.write("Training...")
        for ep in range(start_epoch, 30):
            train_loss = train(model, ep, train_loader, device, criterion, optimizer)
            valid_loss = evaluate(model, ep, val_loader, device, criterion)
            
            # Save best model
            if valid_loss < best_loss:
                print(f"{valid_loss} better than best loss saving model")
                torch.save({'epoch': ep,
                            'model': model.state_dict(),
                            'valid_loss': valid_loss,
                            'optimizer': optimizer.state_dict()},
                            str(foldername / f'model_fold_{fold+1}_tmp.pth'))
                best_loss = valid_loss
            
            # Get learning rate
            for param_group in optimizer.param_groups:
                learning_rate = param_group["lr"]
            
            # Interrupt for minimum learning rate
            if learning_rate < 1e-7:
                break
            
            # Print message
            tqdm.write(f'Epoch {ep:2d}: \tTrain Loss {train_loss:.6f} ' \
                        f'\tValid Loss {valid_loss:.6f} \tLearning Rate {learning_rate:.7f}\t')
            
            # Save history
            history = pd.concat([history, pd.DataFrame({"epoch": ep, "train_loss": train_loss,
                                                        "valid_loss": valid_loss, "lr": learning_rate}, index=[0])],
                                ignore_index=True)
            history.to_csv(foldername / f'history_fold_{fold+1}.csv', index=False)
            
            # Update learning rate
            scheduler.step(valid_loss)
        
        tqdm.write(f"Finished fold {fold + 1}")
        
    tqdm.write("Cross-validation Done!")

