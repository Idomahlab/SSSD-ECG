# ---------------------------------------------------------------
# This file has been modified from automatic-ecg-diagnosis.
#
# Source:
# https://github.com/antonior92/automatic-ecg-diagnosis/blob/tensorflow-v1/generate_figures_and_tables.py
#
# ---------------------------------------------------------------

import pandas as pd
import numpy as np
import torch
from torchmetrics import AveragePrecision, Recall, Specificity, F1Score, PrecisionRecallCurve

from tqdm import tqdm

score_fun = {
    'Recall': Recall(task="binary"),
    'Specificity': Specificity(task="binary"),
    'F1 score': F1Score(task="binary")}
def calculate_accuracy(outputs, labels, threshold=0.5):
    # outputs: should be in device
    # labels: should be in device
    predicted = (outputs >= threshold).float()
    correct = (predicted == labels).float().sum()
    accuracy = correct / (labels.size(0) * labels.size(1))
    return accuracy.item()


def get_data(y, traces_ids):
    # ----- Data settings ----- #
    # change to 71 options by label itos
    diagnosis = ["1dAVb", "RBBB", "LBBB", "SB", "AF", "ST"]
    # ------------------------- #
    y.set_index('id_exam', drop=True, inplace=True)
    y = y.reindex(traces_ids, copy=False)
    df_diagnosis = y.reindex(columns=[d for d in diagnosis])
    y = df_diagnosis.values
    return y


# %% Auxiliar functions
def get_scores(y_true, y_pred, score_fun):
    nclasses = np.shape(y_true)[1]
    scores = []
    for name, fun in score_fun.items():
        scores += [[fun(y_pred[:, k], y_true[:, k]) for k in range(nclasses)]]
    return np.array(scores).T


def get_optimal_precision_recall(y_true, y_score):
    """Find precision and recall values that maximize f1 score."""
    n = np.shape(y_true)[1]
    opt_precision = []
    opt_recall = []
    opt_threshold = []
    pr_curve = PrecisionRecallCurve(task="multilabel", num_labels=n)
    for k in range(n):
        # Get precision-recall curve
        precision, recall, threshold = pr_curve(y_score[:, k], y_true[:, k])
        # Compute f1 score for each point (use nan_to_num to avoid nans messing up the results)
        f1_score = np.nan_to_num(2 * precision * recall / (precision + recall))
        # Select threshold that maximize f1 score
        index = np.argmax(f1_score)
        opt_precision.append(precision[index])
        opt_recall.append(recall[index])
        t = threshold[index - 1] if index != 0 else threshold[0] - 1e-10
        opt_threshold.append(t)
    return np.array(opt_precision), np.array(opt_recall), np.array(opt_threshold)


def load_annotators(path_to_files):
    # Get true values
    y_true = pd.read_csv(path_to_files / 'gold_standard.csv').values
    # Get two annotators
    y_cardiologist1 = pd.read_csv(path_to_files / 'cardiologist1.csv').values
    y_cardiologist2 = pd.read_csv(path_to_files / 'cardiologist2.csv').values
    # Get residents and students performance
    y_cardio = pd.read_csv(path_to_files / 'cardiology_residents.csv').values
    y_emerg = pd.read_csv(path_to_files / 'emergency_residents.csv').values
    y_student = pd.read_csv(path_to_files / 'medical_students.csv').values
    return y_true, y_cardiologist1, y_cardiologist2, y_cardio, y_emerg, y_student


def report_performance(output_path, model_path, y_true, y_pred):
    _, _, our_threshold = get_optimal_precision_recall(torch.from_numpy(y_true), torch.from_numpy(y_pred))
    mask = y_pred > our_threshold
    y_ours = np.zeros_like(y_pred, dtype=int)
    y_ours[mask] = 1

    # evaluation metrics
    # diagnosis = ['1dAVb', 'RBBB', 'LBBB', 'SB', 'AF', 'ST']
    # nclasses = len(diagnosis)
    # predictor_names = ['DNN', 'cardio.', 'emerg.', 'stud.']

    # %% Generate table with scores for the average model (Table 2)
    scores_list = [] # load y_true vs y_ours
    scores = get_scores(torch.from_numpy(y_true), torch.from_numpy(y), score_fun)
    scores_df = pd.DataFrame(scores, range(len(y_ours))#index=diagnosis
                             , columns=score_fun.keys())
    # for y in [y_ours, y_cardio, y_emerg, y_student]:
    #     # Compute scores
    #     
    #     # Put them into a data frame
    #     
    #     # Append
    #     scores_list.append(scores_df)
    # # Concatenate dataframes
    # scores_all_df = pd.concat(scores_list, axis=1, keys=predictor_names)

    # Change multiindex levels
    scores_all_df = scores_df.swaplevel(0, 1, axis=1)
    scores_all_df = scores_df.reindex(level=0, columns=score_fun.keys())

    # Save results
    scores_all_df.to_excel(output_path / "compare_scores.xlsx", float_format='%.3f')
    scores_all_df.to_csv(output_path / "compare_scores.csv", float_format='%.3f')
    np.save(model_path / "thresholds.npy", our_threshold, allow_pickle=True)
    print(f'Saved threshold path: {model_path}/thresholds.npy')
    return


def train(model, ep, train_loader, device, criterion, optimizer):
    model.train()
    total_loss = 0
    n_entries = 0
    train_desc = "Epoch {:2d}: train - Loss: {:.6f}"
    train_bar = tqdm(initial=0, leave=True, total=len(train_loader),
                     desc=train_desc.format(ep, 0, 0), position=0)
    #accuracies = []
    for batch_x, batch_y in train_loader:
        #batch_x = batch_x.transpose(2,1)
        #print(batch_x.shape)
        batch_x, batch_y = batch_x.to(device, dtype=torch.float), batch_y.to(device)
        # Reinitialize grad
        model.zero_grad()
        # Send to device
        # Forward pass
        pred_y = model(batch_x)
        
        loss = criterion(pred_y, batch_y.float())
        # Backward pass
        loss.backward()
        # Optimize
        optimizer.step()
        # Update
        bs = len(batch_x)
        total_loss += loss.detach().cpu().numpy()
        n_entries += bs
        # Update train bar
        train_bar.desc = train_desc.format(ep, total_loss / n_entries)
        train_bar.update(1)
        #accuracies.append(calculate_accuracy(pred_y,batch_y))
    #accuracies = np.array(accuracies)
    #print(f"Overall accuracy over all batches in train loader data {(accuracies.mean() * 100):.4f}%")
    train_bar.close()
    return total_loss / n_entries


def evaluate(model, ep, test_loader, device, criterion):
    model.eval()
    total_loss = 0
    n_entries = 0
    eval_desc = "Epoch {:2d}: valid - Loss: {:.6f}"
    eval_bar = tqdm(initial=0, leave=True, total=len(test_loader),
                    desc=eval_desc.format(ep, 0, 0), position=0)
    #accuracies = []
    for batch_x, batch_y in test_loader:
        #batch_x = batch_x.transpose(1, 2)
        batch_x, batch_y = batch_x.to(device, dtype=torch.float), batch_y.to(device)
        with torch.no_grad():
            # Forward pass
            pred_y = model(batch_x)
            loss = criterion(pred_y, batch_y.float())
            # Update outputs
            bs = len(batch_x)
            # Update ids
            total_loss += loss.detach().cpu().numpy()
            n_entries += bs
            # Print result
            eval_bar.desc = eval_desc.format(ep, total_loss / n_entries)
            eval_bar.update(1)
            #accuracies.append(calculate_accuracy(pred_y,batch_y))
    #accuracies = np.array(accuracies)
    #print(f"Overall accuracy over all batches in valid loader data {(accuracies.mean() * 100):.4f}%")
    eval_bar.close()
    return total_loss / n_entries
