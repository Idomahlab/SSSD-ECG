import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from resnetEcg import ResNet1d  # Make sure resnetEcg.py is in the same directory or in the Python path
torch.cuda.manual_seed(10)
from sklearn.metrics import classification_report
from classifier_utils import train
from classifier_utils import calculate_accuracy
from torchmetrics.classification import Recall, Specificity, F1Score
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
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
def get_optimal_precision_recall(y_true, y_score):
    """Find precision and recall values that maximize F1 score."""
    n = np.shape(y_true)[1]
    opt_precision = []
    opt_recall = []
    opt_threshold = []
    
    for k in range(n):
        # Get precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_true[:, k], y_score[:, k])
        # Compute F1 score for each point
        f1_score = np.nan_to_num(2 * precision * recall / (precision + recall + 1e-10))  # Add epsilon to avoid division by zero
        # Select threshold that maximizes F1 score
        index = np.argmax(f1_score)
        opt_precision.append(precision[index])
        opt_recall.append(recall[index])
        t = thresholds[index - 1] if index != 0 else thresholds[0] - 1e-10
        opt_threshold.append(t)
    
    return np.array(opt_precision), np.array(opt_recall), np.array(opt_threshold)
def report_performance(y_true, y_pred, output_file='classification_report.csv', plot_file='roc_curve.png'):
    """Print and save the classification report."""
    # Capture the classification report as a string
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Convert the classification report to a pandas DataFrame
    df = pd.DataFrame(report).transpose()
    output_file = 'classification_report/' + output_file
    # Save the DataFrame to a CSV file
    df.to_csv(output_file)
    print(f"Classification report saved to {output_file}")
    
    # Initialize dictionaries for storing ROC AUC and ROC curves
    roc_auc = {}
    fpr = {}
    tpr = {}
    
    # Calculate ROC AUC for each class
    for i in range(y_true.shape[1]):
        roc_auc[i] = roc_auc_score(y_true[:, i], y_pred[:, i])
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
    
    # Calculate and store macro-average ROC AUC
    roc_auc["macro"] = roc_auc_score(y_true, y_pred, average="macro")
    
    # Plot ROC curves
    plt.figure(figsize=(12, 8))
    
    for i in range(y_true.shape[1]):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc='lower right')
    plt.savefig(plot_file)
    print(f"ROC curve plot saved to {plot_file}")

    # Optionally, add the macro-average AUC to the classification report
    with open(output_file, 'a') as f:
        f.write(f'\nMacro-average ROC AUC: {roc_auc["macro"]:.2f}')
# Define BottleneckBlock1D and ResNet1D as previously described

# Load test data and labels
x_test = np.load('/home/ido.mahlab/SSSD-ECG/data/ptbxl_test_data.npy')  # Adjust the path to your x_test file
y_test = np.load('/home/ido.mahlab/SSSD-ECG/labels/ptbxl_test_labels.npy').astype(int)  # Adjust the path to your y_test file

# Load training data to fit the scaler
x_train = np.load('/home/ido.mahlab/SSSD-ECG/data/ptbxl_train_data.npy')  # Adjust the path to your x_train file

# Standardize the data based on training data statistics
scaler = StandardScaler()
n_samples, n_channels, n_length = x_train.shape
x_train_reshaped = x_train.reshape(n_samples * n_channels, n_length)
scaler.fit(x_train_reshaped)

# Reshape and standardize test data
x_test_reshaped = x_test.reshape(-1, x_test.shape[-1])
x_test_standardized = scaler.transform(x_test_reshaped)
x_test = x_test_standardized.reshape(x_test.shape)

# Convert numpy arrays to PyTorch tensors
x_test_tensor = torch.FloatTensor(x_test)
y_test_tensor = torch.IntTensor(y_test)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create a DataLoader for the test data
batch_size = 64  # Adjust this as needed
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load model checkpoint
ckpt = torch.load(str('/home/ido.mahlab/SSSD-ECG/check_points/resnet_model/model_fold_2_tmp.pth'), map_location=lambda storage, loc: storage)

# Define the model
model = ResNet1D(BottleneckBlock1D, [3, 4, 6, 3], num_classes=71)

# Load model weights
model.load_state_dict(ckpt["model"])

model = model.to(device)

# Evaluate the model
model.eval()

predicted_y = np.zeros((len(x_test), y_test.shape[1]))

print('Evaluating...')
with torch.no_grad():
    for batch_no in range(len(x_test) // batch_size + 1):
        batch_x = x_test[batch_no * batch_size: (batch_no + 1) * batch_size]
        batch_x = torch.FloatTensor(batch_x)
        batch_x = batch_x.to(device, dtype=torch.float32)
        y_pred = model(batch_x)
        predicted_y[batch_no * batch_size:(batch_no + 1) * batch_size, :] = y_pred.cpu().numpy()

# Get optimal thresholds
opt_precision, opt_recall, opt_threshold = get_optimal_precision_recall(y_test, predicted_y)

# Apply optimal thresholds to make final predictions
final_predictions = np.zeros_like(predicted_y)
for k in range(y_test.shape[1]):
    final_predictions[:, k] = (predicted_y[:, k] > opt_threshold[k]).astype(int)

# Report performance
report_performance(y_test, final_predictions)
