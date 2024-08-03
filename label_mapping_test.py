import pandas as pd
import numpy as np
import wfdb
import ast
from pathlib import Path

def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

# path = 'path/to/ptbxl/'
# sampling_rate=100

# load and convert annotation data
Y = pd.read_csv('/home/ido.mahlab/SSSD-ECG/ptbxl_database.csv', index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

# # Load raw signal data
# X = load_raw_data(Y, sampling_rate, path)

# Load scp_statements.csv for diagnostic aggregation
agg_df = pd.read_csv('/home/ido.mahlab/SSSD-ECG/scp_statements.csv', index_col=0)
#print(agg_df.columns)
agg_df = agg_df[agg_df.diagnostic == 1]

def aggregate_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

# Apply diagnostic superclass
Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

# # Split data into train and test
# test_fold = 10
# # Train
# #X_train = X[np.where(Y.strat_fold != test_fold)]
# y_train = Y[(Y.strat_fold != test_fold)].diagnostic_superclass
# # Test
# #X_test = X[np.where(Y.strat_fold == test_fold)]
# y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass
# #print(agg_df.shape)
# def reformat_as_memmap(df, target_filename, data_folder=None, annotation=False, max_len=0, delete_npys=True,col_data="data",col_label="label", batch_length=0):
#     target_filename = Path(target_filename)
#     data_folder = Path(data_folder)
    
#     npys_data = []
#     npys_label = []

#     for _,row in df.iterrows():
#         #npys_data.append(data_folder/row[col_data] if data_folder is not None else row[col_data])
#         #if(annotation):
#             #npys_label.append(data_folder/row[col_label] if data_folder is not None else row[col_label])
#     #if(batch_length==0):
#         #npys_to_memmap(npys_data, target_filename, max_len=max_len, delete_npys=delete_npys)
#     #else:
#         #npys_to_memmap_batched(npys_data, target_filename, max_len=max_len, delete_npys=delete_npys,batch_length=batch_length)
#         if(annotation):
#             if(batch_length==0):
#                 npys_to_memmap(npys_label, target_filename.parent/(target_filename.stem+"_label.npy"), max_len=max_len, delete_npys=delete_npys)
#         else:
#             #npys_to_memmap_batched(npys_label, target_filename.parent/(target_filename.stem+"_label.npy"), max_len=max_len, delete_npys=delete_npys, batch_length=batch_length)

#         #replace data(filename) by integer
#             df_mapped = df.copy()
#             #df_mapped[col_data+"_original"]=df_mapped.data
#             df_mapped[col_data]=np.arange(len(df_mapped))

#     #df_mapped.to_pickle(target_filename/("df_"+target_filename.stem+".pkl"))
#     print(df_mapped)
#     return df_mapped
# def npys_to_memmap(npys, target_filename, max_len=0, delete_npys=True):
#     memmap = None
#     start = []#start_idx in current memmap file
#     length = []#length of segment
#     filenames= []#memmap files
#     file_idx=[]#corresponding memmap file for sample
#     shape=[]
# df_ptb_xl = pd.read_csv('/home/ido.mahlab/SSSD-ECG/ptbxl_database.csv', index_col='ecg_id')
# target_folder_ptb_xl = Path("home/ido.mahlab/SSSD-ECG")

# reformat_as_memmap(df_ptb_xl, target_folder_ptb_xl/("memmap.npy"),data_folder=target_folder_ptb_xl,delete_npys=True)
labels = np.load('/home/ido.mahlab/SSSD-ECG/src/sssd/sssd_label_cond/ch256_T200_betaT0.02/0_labels.npy')

print(labels)