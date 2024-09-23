import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from resnetEcg import ResNet1d  # Make sure resnetEcg.py is in the same directory or in the Python path
torch.cuda.manual_seed(10)
from sklearn.metrics import classification_report
#from classifier_utils import report_performance
from classifier_utils import train
from classifier_utils import calculate_accuracy
from torchmetrics.classification import Recall, Specificity, F1Score
from sklearn.metrics import precision_recall_curve,  roc_auc_score, roc_curve
import pandas as pd
import matplotlib.pyplot as plt

# score_fun = {
#     'Recall': Recall(num_classes=71, average='macro', compute_on_step=False),
#     'Specificity': Specificity(num_classes=71, average='macro', compute_on_step=False),
#     'F1 score': F1Score(num_classes=71, average='macro', compute_on_step=False)
# }
def report_performance(y_true, y_pred, output_file='classification_report.csv',plot_file='roc_curve.png'):
    """Print and save the classification report."""
    # Capture the classification report as a string
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Convert the classification report to a pandas DataFrame
    df = pd.DataFrame(report).transpose()
    
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
def get_optimal_precision_recall(y_true, y_score):
    """Find precision and recall values that maximize f1 score."""
    n = np.shape(y_true)[1]
    opt_precision = []
    opt_recall = []
    opt_threshold = []
    
    for k in range(n):
        # Get precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_true[:, k], y_score[:, k])
        # Compute f1 score for each point
        f1_score = np.nan_to_num(2 * precision * recall / (precision + recall + 1e-10))  # Add epsilon to avoid division by zero
        # Select threshold that maximizes f1 score
        index = np.argmax(f1_score)
        opt_precision.append(precision[index])
        opt_recall.append(recall[index])
        t = thresholds[index - 1] if index != 0 else thresholds[0] - 1e-10
        opt_threshold.append(t)
    
    return np.array(opt_precision), np.array(opt_recall), np.array(opt_threshold)

def check_wheights(model):
    # Assuming `model` is your neural network model
    total_sum = 0
    total_params = 0

    for param in model.parameters():
        total_sum += param.data.sum()
        total_params += param.numel()

    average = total_sum / total_params
    print(f"Average of model parameters: {average}")
    
@torch.no_grad
def test(model, test_loader, device):
    accuracies = []
    for batch in test_loader:
        data = batch[0].to(device)
        labels = batch[1]
        pred = model(data)
        #accuracy = calculate_accuracy(pred, labels)
        #accuracies.append(accuracy)
    #accuracies = np.array(accuracies)
    #print(f"Overall accuracy over all batch in test data {(accuracies.mean()):.4f}%")

# Load test data and labels
x_test = np.load('/home/ido.mahlab/SSSD-ECG/data/ptbxl_test_data.npy')  # Adjust the path to your x_test file
y_test = np.load('/home/ido.mahlab/SSSD-ECG/labels/ptbxl_test_labels.npy').astype(int)  # Adjust the path to your y_test file
# label_to_drop = 61  # Replace with the index of the label you want to drop
# # Drop label column from training and validation data
# y_test = np.delete(y_test, label_to_drop, axis=1)
# Convert numpy arrays to PyTorch tensors
x_test_tensor = torch.FloatTensor(x_test)
y_test_tensor = torch.IntTensor(y_test)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Create a DataLoader for the test data
batch_size = 64  # Adjust this as needed
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# Load model checkpoint
# Get checkpoint
ckpt = torch.load(str('/home/ido.mahlab/SSSD-ECG/check_points/resnet_model/model_tmp.pth'), map_location=lambda storage, loc: storage)
# Get model  # Adjust the path to your checkpoint file

# Define the model
N_LEADS = 12  # Adjust according to your model's input dimensions
N_CLASSES = y_test.shape[1]  # Number of classes
# net_filter_size = [64, 128, 196, 256]
# net_seq_lengh = [1000, 500, 250, 125]
net_filter_size = [64, 128, 196, 256, 512, 512]
net_seq_lengh = [1000, 500, 250, 125, 125, 125]
# Load model weights
model = ResNet1d(
    input_dim=(N_LEADS,1000),
    blocks_dim=list(zip(net_filter_size, net_seq_lengh)),  # Example; adjust according to your configuration
    n_classes=N_CLASSES,

    
)
print(ckpt.keys)
check_wheights(model)
# load model checkpoint
model.load_state_dict(ckpt["model"])
check_wheights(model)

model = model.to(device)

# Evaluate the model
model.eval()
#test(model, test_loader, device)

predicted_y = np.zeros((len(x_test), N_CLASSES))

print('Evaluating...')
with torch.no_grad():
    # for batch_no, (batch_x, _) in enumerate(test_loader):
    #     batch_x = batch_x.to(device, dtype=torch.float32)
    #     y_pred = model(batch_x)
    #     start = batch_no * batch_size
    #     end = start + len(batch_x)
    #     predicted_y[start:end, : ] = y_pred.cpu()
            for batch_no in range(len(x_test) // 64 + 1):
                batch_x = x_test[batch_no * 64: (batch_no + 1) * 64]
                #batch_x = np.transpose(batch_x, [0, 2, 1])
                batch_x = torch.FloatTensor(batch_x)
                with torch.no_grad():
                    batch_x = batch_x.to(device, dtype=torch.float32)
                    y_pred = model(batch_x)
                    # threshold = 0.5  # This is an example threshold; adjust based on your task
                    # y = (y_pred > threshold).int()
                predicted_y[batch_no * 64:(batch_no + 1) * 64, :] = y_pred.cpu().numpy()
# Get optimal thresholds
opt_precision, opt_recall, opt_threshold = get_optimal_precision_recall(y_test, predicted_y)

# Apply optimal thresholds to make final predictions
final_predictions = np.zeros_like(predicted_y)
for k in range(N_CLASSES):
    final_predictions[:, k] = (predicted_y[:, k] > opt_threshold[k]).astype(int)

report_performance(y_test, final_predictions)
#report_performance(y_test, predicted_y)
