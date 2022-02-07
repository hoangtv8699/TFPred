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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold


path = 'F:/BioInformatic/Dataset/TFPM vs TFNPM/BERT/'
data_bert, labels_bert = read_h5py(path, 'TFPM_training_top100')
data_bert_val, labels_bert_val = read_h5py(path, 'TFPM_testing_top100')
data_bert = data_bert[:, 0, :1000]
data_bert_val = data_bert_val[:, 0, :1000]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=18)

for train_index, val_index in skf.split(data_bert, labels_bert):
    # split data
    train_data = data_bert[train_index]
    train_labels = labels_bert[train_index]
    val_data = data_bert[val_index]
    val_labels = labels_bert[val_index]

    # # resampling
    # posi = train_data[train_labels == 1]
    # nega = train_data[train_labels == 0]
    # # up resample
    # # posi = resample(posi, replace=True, n_samples=int(len(nega)))
    # # down resample
    # nega = resample(nega, replace=True, n_samples=int(len(posi)))
    # # concat data
    # print(len(posi))
    # print(len(nega))
    # labels_posi = [1 for i in range(len(posi))]
    # labels_nega = [0 for i in range(len(nega))]
    # train_data = np.append(posi, nega, axis=0)
    # train_labels = np.append(labels_posi, labels_nega)
    train_data, train_labels = balance_data(train_data, train_labels, 31)
    val_data, val_labels = balance_data(val_data, val_labels, 31)

    # # SMOTE
    # sm = SMOTE(random_state=1)
    # train_data, train_labels = sm.fit_resample( train_data, train_labels)

    # data_bert, sm, sd = normalize_common(data_bert)
    # data_bert_val = normalize(data_bert_val, sm, sd)

    # data_bert = normalize(data_bert)
    # data_bert_val = normalize(data_bert_val)
    sample_weight = np.ones(len(train_data))
    for i in range(len(sample_weight)):
        if train_labels[i] == 0:
            sample_weight[i] = 10

    # clf = ensemble.GradientBoostingClassifier()
    clf = SVC(gamma='auto', probability=True)
    clf.fit(train_data, train_labels, sample_weight=sample_weight)
    # dump(clf, '../saved_models/RF_TFPM_SMOTE_used.joblib')

    # clf = load('../saved_models/RF_TFPM_SMOTE_used.joblib')

    pre = clf.predict_proba(train_data)
    pre = np.asarray(pre)
    print('train test:')
    sen = sensitivity(train_labels, pre)
    spe = specificity(train_labels, pre)
    accc = acc(train_labels, pre)
    mccc = mcc(train_labels, pre)
    aucc = auc(train_labels, pre)
    print('SEN:' + str(sen))
    print('SPE:' + str(spe))
    print('ACC:' + str(accc))
    print('MCC:' + str(mccc))
    print('AUC:' + str(aucc))

    pre = clf.predict_proba(val_data)
    pre = np.asarray(pre)
    print('validation test:')
    sen = sensitivity(val_labels, pre)
    spe = specificity(val_labels, pre)
    accc = acc(val_labels, pre)
    mccc = mcc(val_labels, pre)
    aucc = auc(val_labels, pre)
    print('SEN:' + str(sen))
    print('SPE:' + str(spe))
    print('ACC:' + str(accc))
    print('MCC:' + str(mccc))
    print('AUC:' + str(aucc))

    pre = clf.predict_proba(data_bert_val)
    pre = np.asarray(pre)
    print('independent test:')
    sen = sensitivity(labels_bert_val, pre)
    spe = specificity(labels_bert_val, pre)
    accc = acc(labels_bert_val, pre)
    mccc = mcc(labels_bert_val, pre)
    aucc = auc(labels_bert_val, pre)
    print('SEN:' + str(sen))
    print('SPE:' + str(spe))
    print('ACC:' + str(accc))
    print('MCC:' + str(mccc))
    print('AUC:' + str(aucc))
    break



