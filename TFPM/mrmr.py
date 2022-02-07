# import numpy as np
# import pandas as pd
# from sklearn.datasets import make_classification
# import pymrmr
#
# X, y = make_classification(n_samples=10000,
#                            n_features=7,
#                            n_informative=3,
#                            n_classes=2,
#                            random_state=0,
#                            shuffle=False)
#
# # Creating a dataFrame
# df = pd.DataFrame({'Feature 1': X[:, 0],
#                    'Feature 2': X[:, 1],
#                    'Feature 3': X[:, 2],
#                    'Feature 4': X[:, 3],
#                    'Feature 5': X[:, 4],
#                    'Feature 6': X[:, 5],
#                    'Feature 7': X[:, 6],
#                    'Class': y})
#
# y_train = df['Class']
# X_train = df.drop('Class', axis=1)
#
# print(pymrmr.mRMR(X_train, 'MIQ', 7))


from utils.helpers import *
import pymrmr
from sklearn.preprocessing import StandardScaler


if __name__ == '__main__':
    # TFPM_used, _ = read_fasta('../data/TFPM vs TFPNM/TFPM_training_dataset_used.txt')
    # TFPM_unused, _ = read_fasta('../data/TFPM vs TFPNM/TFPM_training_dataset_unused.txt')
    # TFPNM, _ = read_fasta('../data/TFPM vs TFPNM/TFPNM_training_dataset.txt')
    #
    # TFPM_val, _ = read_fasta('../data/TFPM vs TFPNM/TFPM_independent_dataset.txt')
    # TFPNM_val, _ = read_fasta('../data/TFPM vs TFPNM/TFPNM_independent_dataset.txt')

    TFPM_used, _ = read_fasta('../data/TF vs non-TF/TF_training_dataset.txt')
    TFPNM, _ = read_fasta('../data/TF vs non-TF/non-TF_training_dataset.txt')

    TFPM_val, _ = read_fasta('../data/TF vs non-TF/TF_independent_dataset.txt')
    TFPNM_val, _ = read_fasta('../data/TF vs non-TF/non-TF_independent_dataset.txt')

    gap = 2
    # op = 'op13'
    # ops = ['op5', 'op8', 'op9', 'op11', 'op13']
    ops = ['op13']

    TFPM_used_all = [[] for i in range(len(TFPM_used))]
    # TFPM_unused_all = [[] for i in range(len(TFPM_unused))]
    TFPNM_all = [[] for i in range(len(TFPNM))]

    TFPM_val_all = [[] for i in range(len(TFPM_val))]
    TFPNM_val_all = [[] for i in range(len(TFPNM_val))]

    for op in ops:
        reduced_TFPM_used = RAACs(TFPM_used, op=op)
        RGDP_TFPM_used = RGDPs(reduced_TFPM_used, gap=gap, op=op)
        # reduced_TFPM_unused = RAACs(TFPM_unused, op=op)
        # RGDP_TFPM_unused = RGDPs(reduced_TFPM_unused, gap=gap, op=op)
        reduced_TFPNM = RAACs(TFPNM, op=op)
        RGDP_TFPNM = RGDPs(reduced_TFPNM, gap=gap, op=op)

        reduced_TFPM_val = RAACs(TFPM_val, op=op)
        RGDP_TFPM_val = RGDPs(reduced_TFPM_val, gap=gap, op=op)
        reduced_TFPNM_val = RAACs(TFPNM_val, op=op)
        RGDP_TFPNM_val = RGDPs(reduced_TFPNM_val, gap=gap, op=op)

        TFPM_used_all = np.append(TFPM_used_all, RGDP_TFPM_used, axis=-1)
        # TFPM_unused_all = np.append(TFPM_unused_all, RGDP_TFPM_unused, axis=-1)
        TFPNM_all = np.append(TFPNM_all, RGDP_TFPNM, axis=-1)

        TFPM_val_all = np.append(TFPM_val_all, RGDP_TFPM_val, axis=-1)
        TFPNM_val_all = np.append(TFPNM_val_all, RGDP_TFPNM_val, axis=-1)

    # G-gap dipeptide composition
    # RGDP_TFPM_used = GDPs(TFPM_used, gap=gap)
    # RGDP_TFPM_unused = GDPs(TFPM_unused, gap=gap)
    # RGDP_TFPNM = GDPs(TFPNM, gap=gap)
    #
    # RGDP_TFPM_val = GDPs(TFPM_val, gap=gap)
    # RGDP_TFPNM_val = GDPs(TFPNM_val, gap=gap)

    # training data
    # data = np.append(TFPM_used_all, TFPM_unused_all, axis=0)
    data = TFPM_used_all
    # data = resample(data, replace=False, n_samples=int(len(TFPNM_all)), random_state=7)
    # RGDP_TFPNM = resample(RGDP_TFPNM, replace=True, n_samples=int(len(data)), random_state=7)
    labels = np.ones(len(data))
    data = np.append(data, TFPNM_all, axis=0)
    labels = np.append(labels, np.zeros(len(TFPNM_all)), axis=0)
    # validating data
    data_val = np.append(TFPM_val_all, TFPNM_val_all, axis=0)
    labels_val = np.append(np.ones(len(TFPM_val_all)), np.zeros(len(TFPNM_val_all)), axis=0)

    scaler = StandardScaler()
    scaler.fit(data)
    data = scaler.transform(data)

    # Creating a dataFrame
    df = pd.DataFrame({str(i): data[:, i] for i in range(len(data[0]))})

    importance = pymrmr.mRMR(df, 'MIQ', len(data[0]))
    importance_int = []
    for i in importance:
        importance_int.append(int(i))

    print(importance_int)

    f = h5py.File('mrmr' + '.h5', 'w')
    f.create_dataset('mrmr', data=importance_int)
    f.close()

    # file = h5py.File('mrmr' + '.h5', 'r')
    # importance_int = np.asarray(file['mrmr'])
    # print(importance_int)

