# -*- coding: utf-8 -*-
import json

import numpy as np
import math
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tensorflow import keras
from utils.helpers import *


if __name__ == '__main__':
    non_TF_pssm = 'non_TF_independent'
    TF_pssm = 'TF_independent'
    path = 'F:/BioInformatic/Dataset/TF vs non-TF/PSSM/'
    # read data
    # data_non_TF_pssm = read_data(path + non_TF_pssm + '/', padding="pad_sequence", maxlen=1000)
    # labels_non_TF_pssm = np.zeros(len(data_non_TF_pssm))
    # data_TF_pssm = read_data(path + TF_pssm + '/', padding="pad_sequence", maxlen=1000)
    # labels_TF_pssm = np.ones(len(data_TF_pssm))
    #
    # save_h5py(np.array(data_non_TF_pssm).astype('float32'), labels_non_TF_pssm, path, non_TF_pssm + ' 1000')
    # save_h5py(np.array(data_TF_pssm).astype('float32'), labels_TF_pssm, path, TF_pssm + ' 1000')
    data_non_TF_pssm, labels_non_TF_pssm = read_h5py(path, non_TF_pssm + ' 1000')
    data_TF_pssm, labels_TF_pssm = read_h5py(path, TF_pssm + ' 1000')

    data = np.append(data_non_TF_pssm, data_TF_pssm, axis=0)
    labels = np.append(labels_non_TF_pssm, labels_TF_pssm, axis=0)

    print(data.shape)
    data = np.expand_dims(data, axis=-1).astype(np.float32)
    path = "../saved_models/18 deeper/"
    model_paths = os.listdir(path)
    model = []
    for model_path in model_paths:
        model.append(keras.models.load_model(path + model_path,
                                             custom_objects={"sensitivity": sensitivity,
                                                             "specificity": specificity,
                                                             "mcc": mcc,
                                                             # 'AdasOptimizer': AdasOptimizer
                                                             }, compile=False))

    i = 0
    a = []
    b = []
    for i in range(len(model)):
        pre = model[i].predict(data)
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
