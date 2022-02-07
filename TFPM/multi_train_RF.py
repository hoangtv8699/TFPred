import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn
from sklearn import tree
from sklearn import ensemble
from sklearn.metrics import make_scorer, recall_score
from sklearn.svm import SVC
from joblib import dump, load
from utils.helpers import *
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


path = 'F:/BioInformatic/Dataset/TFPM vs TFPNM/BERT/'
data_bert, labels_bert = read_h5py(path, 'TFPM_training_top100_balance')
data_bert_val, labels_bert_val = read_h5py(path, 'TFPM_testing_top100')

# TFPM, labels_TFPM = read_h5py(path, 'TFPM_training_dataset_used')
# TFPM_unused, labels_TFPM_unused = read_h5py(path, 'TFPM_training_dataset_unused')
# TFPNM, labels_TFPNM = read_h5py(path, 'TFPNM_training_dataset')
# TFPM_val, labels_TFPM_val = read_h5py(path, 'TFPM_independent_dataset')
# TFPNM_val, labels_TFPNM_val = read_h5py(path, 'TFPNM_independent_dataset')
#
# data_bert = np.append(TFPM, TFPNM, axis=0)
# # data_bert = np.append(data_bert, TFPM_unused, axis=0)
# labels_bert = np.append(labels_TFPM, labels_TFPNM, axis=0)
# # labels_bert = np.append(labels_bert, labels_TFPM_unused, axis=0)
#
# data_bert_val = np.append(TFPM_val, TFPNM_val, axis=0)
# labels_bert_val = np.append(labels_TFPM_val, labels_TFPNM_val, axis=0)

a = []
for i in range(1024):
    print(i)
    b = []
    data = np.copy(data_bert[:, i, :1000])
    labels = np.copy(labels_bert)
    data_val = np.copy(data_bert_val[:, i, :1000])
    labels_val = np.copy(labels_bert_val)

    # # resampling
    # posi = data_bert[labels_bert == 1]
    # nega = data_bert[labels_bert == 0]
    # # up resample
    # # posi = resample(posi, replace=True, n_samples=int(len(nega)), random_state=101)
    # # down resample
    # nega = resample(nega, replace=True, n_samples=int(len(posi)), random_state=1010)
    # # concat data
    # print(len(posi))
    # print(len(nega))
    # labels_posi = [1 for i in range(len(posi))]
    # labels_nega = [0 for i in range(len(nega))]
    # data_bert = np.append(posi, nega, axis=0)
    # labels_bert = np.append(labels_posi, labels_nega)
    # data_bert, labels_bert = balance_data(data_bert, labels_bert, 31)

    # SMOTE
    # sm = SMOTE(random_state=1)
    # data, labels = sm.fit_resample(data, labels)

    clf = tree.DecisionTreeClassifier()
    clf.fit(data, labels)
    dump(clf, 'saved_models/TFPM all 1024/' + str(i) + ' RF_TFPM_SMOTE.joblib')

    # clf = load('saved_models/TFPM all 1024/' + str(i) + ' RF_TFPM_used.joblib')

    pre = clf.predict_proba(data_val)
    pre = np.asarray(pre)

    sen = sensitivity(labels_val, pre)
    spe = specificity(labels_val, pre)
    accc = acc(labels_val, pre)
    mccc = mcc(labels_val, pre)
    aucc = auc(labels_val, pre)
    b.append(math.floor(sen * 1000) / 1000)
    b.append(math.floor(spe * 1000) / 1000)
    b.append(math.floor(accc * 1000) / 1000)
    b.append(math.floor(mccc * 1000) / 1000)
    b.append(math.floor(aucc * 1000) / 1000)
    a.append(b)

pd.DataFrame(a).to_csv('saved_models/TFPM all 1024/SMOTE.csv')



