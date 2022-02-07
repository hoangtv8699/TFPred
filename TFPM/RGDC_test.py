# -*- coding: utf-8 -*-
import json

import numpy as np
import math
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tensorflow import keras
from joblib import dump, load
from utils.helpers import *
# from RAAC import read_fasta


if __name__ == '__main__':
    # TFPM_val, TFPM_val_labels = read_fasta('../data/TFPM vs TFPNM/TFPM_independent_dataset.txt')
    # TFPNM_val, TFPNM_val_labels = read_fasta('../data/TFPM vs TFPNM/TFPNM_independent_dataset.txt')
    #
    # path = 'F:/BioInformatic/Dataset/TFPM vs TFPNM/PSSM/'
    # TFPM_val_pssm, TFPM_val_pssm_labels = read_data(path + 'TFPM_independent_dataset/', padding='none')
    # TFPNM_val_pssm, TFPNM_val_pssm_labels = read_data(path + 'TFPNM_independent_dataset/', padding='none')
    #
    # reduced_TFPM, _ = reduce_pssms(TFPM_val, TFPM_val_labels, TFPM_val_pssm, TFPM_val_pssm_labels)
    # reduced_TFPNM, _ = reduce_pssms(TFPNM_val, TFPNM_val_labels, TFPNM_val_pssm, TFPNM_val_pssm_labels)
    #
    # data = np.append(reduced_TFPM, reduced_TFPNM, axis=0)
    # labels = np.append(np.ones(len(reduced_TFPM)), np.zeros(len(reduced_TFPNM)))
    #
    # data = data.reshape((data.shape[0], -1))

    TFPM_val, _ = read_fasta('../data/TFPM vs TFPNM/TFPM_independent_dataset.txt')
    TFPNM_val, _ = read_fasta('../data/TFPM vs TFPNM/TFPNM_independent_dataset.txt')

    # TFPM_val, _ = read_fasta('../data/TF vs non-TF/TF_independent_dataset.txt')
    # TFPNM_val, _ = read_fasta('../data/TF vs non-TF/non-TF_independent_dataset.txt')

    gap = 4
    # op = 'op13'
    # ops = ['op5', 'op8', 'op9', 'op11', 'op13']
    ops = ['op13']

    TFPM_val_all = [[] for i in range(len(TFPM_val))]
    TFPNM_val_all = [[] for i in range(len(TFPNM_val))]

    for op in ops:
        reduced_TFPM_val = RAACs(TFPM_val, op=op)
        RGDP_TFPM_val = RGDPs(reduced_TFPM_val, gap=gap, op=op)
        reduced_TFPNM_val = RAACs(TFPNM_val, op=op)
        RGDP_TFPNM_val = RGDPs(reduced_TFPNM_val, gap=gap, op=op)

        TFPM_val_all = np.append(TFPM_val_all, RGDP_TFPM_val, axis=-1)
        TFPNM_val_all = np.append(TFPNM_val_all, RGDP_TFPNM_val, axis=-1)

    # reduced_TFPM_val = RAACs(TFPM_val, op=op)
    # RGDP_TFPM_val = RGDPs(reduced_TFPM_val, gap=gap, op=op)
    # reduced_TFPNM_val = RAACs(TFPNM_val, op=op)
    # RGDP_TFPNM_val = RGDPs(reduced_TFPNM_val, gap=gap, op=op)

    # RGDP_TFPM_val = GDPs(TFPM_val, gap=gap)
    # RGDP_TFPNM_val = GDPs(TFPNM_val, gap=gap)

    # validating data
    data = np.append(TFPM_val_all, TFPNM_val_all, axis=0)
    labels = np.append(np.ones(len(TFPM_val_all)), np.zeros(len(TFPNM_val_all)), axis=0)

    # # BERT
    # path = 'F:/BioInformatic/Dataset/TFPM vs TFPNM/BERT/'
    # data, labels = read_h5py(path, 'TFPM_testing_top100')
    # print(data.shape)
    # data = data[:, 0]

    # DCT_TFPM_GGAP = load('../saved_models/DCT_TFPM_GGAP.joblib')
    # # importance = DCT_TFPM_GGAP.feature_importances_
    # importance = DCT_TFPM_GGAP.coef_[0]
    # importance = np.argsort(importance)[::-1]

    # file = h5py.File('mrmr' + '.h5', 'r')
    # importance = np.asarray(file['mrmr'])

    # pick = importance[:37]
    # data = data[:, pick]

    path = "../saved_models/TFPM/"
    model_paths = os.listdir(path)
    model = []
    i = 0
    for model_path in model_paths:
        model.append(load('../saved_models/TFPM op5/' + str(i) + '_SVM.joblib'))
        i += 1

    i = 0
    a = []
    b = []
    for i in range(len(model)):
        pre = model[i].predict_proba(data)
        print("model: " + str(i))
        sen = sensitivity(labels, pre)
        spe = specificity(labels, pre)
        accc = acc(labels, pre)
        mccc = mcc(labels, pre)
        aucc = auc(labels, pre)
        b.append(math.floor(sen * 1000) / 1000)
        b.append(math.floor(spe * 1000) / 1000)
        b.append(math.floor(accc * 1000) / 1000)
        b.append(math.floor(mccc * 1000) / 1000)
        b.append(math.floor(aucc * 1000) / 1000)
        a.append(b)
        b = []
        i += 1

    vote = voting(model, data)
    ave = average(model, data)
    med = median(model, data)

    print("voting:")
    sen = sensitivity(labels, vote)
    spe = specificity(labels, vote)
    accc = acc(labels, vote)
    mccc = mcc(labels, vote)
    aucc = auc(labels, vote)
    b.append(math.floor(sen * 1000) / 1000)
    b.append(math.floor(spe * 1000) / 1000)
    b.append(math.floor(accc * 1000) / 1000)
    b.append(math.floor(mccc * 1000) / 1000)
    b.append(math.floor(aucc * 1000) / 1000)
    a.append(b)
    b = []

    print("ave:")
    sen = sensitivity(labels, ave)
    spe = specificity(labels, ave)
    accc = acc(labels, ave)
    mccc = mcc(labels, ave)
    aucc = auc(labels, ave)
    b.append(math.floor(sen * 1000) / 1000)
    b.append(math.floor(spe * 1000) / 1000)
    b.append(math.floor(accc * 1000) / 1000)
    b.append(math.floor(mccc * 1000) / 1000)
    b.append(math.floor(aucc * 1000) / 1000)
    a.append(b)
    b = []

    print("med:")
    sen = sensitivity(labels, med)
    spe = specificity(labels, med)
    accc = acc(labels, med)
    mccc = mcc(labels, med)
    aucc = auc(labels, med)
    b.append(math.floor(sen * 1000) / 1000)
    b.append(math.floor(spe * 1000) / 1000)
    b.append(math.floor(accc * 1000) / 1000)
    b.append(math.floor(mccc * 1000) / 1000)
    b.append(math.floor(aucc * 1000) / 1000)
    a.append(b)
    b = []

    pd.DataFrame(a).to_csv('../test.csv')
