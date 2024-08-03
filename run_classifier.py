# ---------------------------------------------------------------
# This file has been modified from automatic-ecg-diagnosis.
#
# Source:
# https://github.com/antonior92/automatic-ecg-diagnosis/blob/tensorflow-v1/train.py
#
# ---------------------------------------------------------------

import json
import pathlib
import torch
import os
from tqdm import tqdm
import torch
torch.backends.cudnn.enabled = False
from resnetEcg import ResNet1d
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
from classifier_utils import train, evaluate
import ast
import wfdb
from torchmetrics import AveragePrecision, Recall, Specificity, F1Score, PrecisionRecallCurve

score_fun = {
    'Recall': Recall(task="binary"),
    'Specificity': Specificity(task="binary"),
    'F1 score': F1Score(task="binary")}

# %% Auxiliar functions
def get_scores(y_true, y_pred, score_fun):
    scores = []
    for name, fun in score_fun.items():
        scores += [[fun(y_pred, y_true)]]
    return np.array(scores).T

def report_performance(y_true, y_pred):
    scores = get_scores(torch.from_numpy(y_true), torch.from_numpy(y_pred), score_fun)[0]
    for sc in zip(score_fun.keys(), scores):
        print(sc)

def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data


def load_data(path_to_database, sampling_rate):
    '''
    This function load the ECG data and meta data.
    :param path_to_database: location of raw signals and csv files with annotations
    :param sampling_rate: choose between 100 or 500.
    :return: X: of shape [number of recordings, recording length, n leads]
    .        Y: dataframe.
    '''
    Y = pd.read_csv(path_to_database + 'ptbxl_database.csv', index_col='ecg_id')[:100]
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
    X = load_raw_data(Y[:100], sampling_rate, path_to_database)
    agg_df = pd.read_csv(path_to_database + 'scp_statements.csv', index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    def aggregate_diagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))

    Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)
    # X = np.array([])
    # Y = np.array([])
    return X, Y


def split_data(X, y_classes, Y):
    '''
    This function splits the data into train and test.
    :param X: raw ECG signals
    :param y_classes: one hot encoder on classes
    :param Y: annotations dataframe with strat_fold column
    :return: x_train, y_train, x_test, y_test.
    '''
    test_fold = 10
    # Train
    x_train = X[np.where(Y.strat_fold != test_fold)]
    y_train = y_classes[(Y.strat_fold != test_fold)].to_numpy()
    # Test
    x_test = X[np.where(Y.strat_fold == test_fold)]
    y_test = y_classes[Y.strat_fold == test_fold].to_numpy()

    x_val = x_train[:10]
    y_val = y_train[:10]
    # x_train = np.array([])
    # y_train = np.array([])
    # x_val = np.array([])
    # y_val = np.array([])
    # x_test = np.array([])
    # y_test = np.array([])
    return x_train, y_train, x_val, y_val, x_test, y_test


if __name__ == "__main__":
    import pandas as pd
    import argparse
    from warnings import warn

    # Arguments that will be saved in config file
    parser = argparse.ArgumentParser(add_help=True,
                                     description='Train model to predict arrhythmias from the raw ecg tracing.')
    parser.add_argument("--config", type=str, default="resnet_base.json",
                        help="model hyperparameters")
    parser.add_argument('--device', default='cuda:2', help='Device')
    parser.add_argument('--path_to_database', type=str, default='data/',
                        help='path to folder containing tnmg database')
    parser.add_argument('--train', default=True, action='store_true',
                        help='train the classifier from scratch (default: False)')
    parser.add_argument('--n_leads', type=int, default=12,
                        help='how many leads to train on, choose between [1,12] (default: lead 6)')
    parser.add_argument('--epochs', type=int, default=70,
                        help='maximum number of epochs (default: 70)')
    parser.add_argument('--seed', type=int, default=2,
                        help='random seed for number generator (default: 2)')
    parser.add_argument('--sample_freq', type=int, default=500,
                        help='sample frequency (in Hz) in which all traces will be resampled at (default: 400)')
    parser.add_argument('--seq_length', type=int, default=5000,
                        help='size (in # of samples) for all traces. If needed traces will be zeropadded'
                             'to fit into the given size. (default: 4096)')
    parser.add_argument('--scale_multiplier', type=int, default=10,
                        help='multiplicative factor used to rescale inputs.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size (default: 64).')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument("--patience", type=int, default=12,
                        help='maximum number of epochs without reducing the learning rate (default: 12)')
    parser.add_argument("--min_lr", type=float, default=1e-7,
                        help='minimum learning rate (default: 1e-7)')
    parser.add_argument("--lr_factor", type=float, default=0.1,
                        help='reducing factor for the lr in a plateu (default: 0.1)')
    parser.add_argument('--net_filter_size', type=int, nargs='+', default=[64, 128, 196, 256, 320],
                        help='filter size in resnet layers (default: [64, 128, 196, 256, 320]).')
    parser.add_argument('--net_seq_lengh', type=int, nargs='+', default=[5000, 1000, 200, 40, 8],
                        help='number of samples per resnet layer (default: [4096, 1024, 256, 64, 16]).')
    parser.add_argument('--dropout_rate', type=float, default=0.8,
                        help='dropout rate (default: 0.8).')
    parser.add_argument('--kernel_size', type=int, default=17,
                        help='kernel size in convolutional layers (default: 17).')
    args, unk = parser.parse_known_args()
    # Check for unknown options
    if unk:
        warn("Unknown arguments:" + str(unk) + ".")

    torch.manual_seed(args.seed)
    print(args)
    # Set device
    device = torch.device(args.device)

    # Save config file
    config_dir = "./config/"
    config_path = pathlib.Path(config_dir)
    config_path.mkdir(parents=True, exist_ok=True)
    config_path = config_dir + args.config

    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent='\t')

    foldername = pathlib.PurePath("./check_points/resnet_model")
    print('model folder:', foldername)
    os.makedirs(foldername, exist_ok=True)

    # Database paths
    """
    Provide database paths
    """
    X, Y = load_data(args.path_to_database, 100)
    y_classes = pd.get_dummies(Y.diagnostic_superclass.astype(str),
                               columns=Y.diagnostic_superclass.astype(str).unique()).astype(int)

    tqdm.write("Building data loaders...")
    x_train, y_train, x_val, y_val, x_test, y_test = split_data(X, y_classes, Y)

    if args.train:
        params = {'batch_size': args.batch_size,
                  'shuffle': True}
        training_generator = torch.utils.data.DataLoader([*zip(x_train, y_train)], batch_size=64)

        validation_generator = torch.utils.data.DataLoader([*zip(x_val, y_val)], batch_size=64)

        tqdm.write("Define model...")
        N_LEADS = args.n_leads
        N_CLASSES = len(y_classes.columns)
        model = ResNet1d(input_dim=(N_LEADS, args.seq_length),
                         blocks_dim=list(zip(args.net_filter_size, args.net_seq_lengh)),
                         n_classes=N_CLASSES,
                         kernel_size=args.kernel_size,
                         dropout_rate=args.dropout_rate)
        model.to(device=device)

        tqdm.write("Define loss...")
        criterion = nn.BCELoss()

        tqdm.write("Define optimizer...")
        optimizer = optim.Adam(model.parameters(), args.lr)

        tqdm.write("Define scheduler...")
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.patience,
                                                         min_lr=args.lr_factor * args.min_lr,
                                                         factor=args.lr_factor)

        tqdm.write("Training...")
        start_epoch = 0
        best_loss = np.Inf
        history = pd.DataFrame(columns=['epoch', 'train_loss', 'valid_loss', 'lr'])
        for ep in range(start_epoch, args.epochs):
            train_loss = train(model, ep, training_generator, device, criterion, optimizer)
            valid_loss = evaluate(model, ep, validation_generator, device, criterion)
            # Save best model
            if valid_loss < best_loss:
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
            if learning_rate < args.min_lr:
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
    else:
        # Get checkpoint
        ckpt = torch.load(str(foldername / "model_tmp.pth"), map_location=lambda storage, loc: storage)
        # Get model
        N_LEADS = args.n_leads
        N_CLASSES = len(y_classes.columns)
        model = ResNet1d(input_dim=(N_LEADS, args.seq_length),
                         blocks_dim=list(zip(args.net_filter_size, args.net_seq_lengh)),
                         n_classes=N_CLASSES,
                         kernel_size=args.kernel_size,
                         dropout_rate=args.dropout_rate)
        # load model checkpoint
        model.load_state_dict(ckpt["model"])
        model = model.to(device)

        # get data
        # Import data

        # Evaluate on test data
        model.eval()

        # Compute gradients
        predicted_y = np.zeros((len(x_test),N_CLASSES))
        end = 0

        print('Evaluating...')
        for batch_no in range(len(x_test) // args.batch_size + 1):
            batch_x = x_test[batch_no * args.batch_size: (batch_no + 1) * args.batch_size]
            batch_x = np.transpose(batch_x, [0, 2, 1])
            batch_x = torch.FloatTensor(batch_x)
            with torch.no_grad():
                batch_x = batch_x.to(device, dtype=torch.float32)
                y_pred = model(batch_x)
                y = torch.argmax(y_pred, dim=1)
            predicted_y[batch_no * args.batch_size:(batch_no + 1) * args.batch_size, y.detach().cpu().numpy()] = 1
        report_performance(y_test, predicted_y)