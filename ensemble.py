import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import tree
from sklearn import ensemble
from sklearn.svm import SVC
from joblib import dump, load
from utils.helpers import *


data_bert_val, labels_bert_val = read_h5py('F:/BioInformatic/Dataset/TF vs non-TF/BERT/', 'independent_top100')
data_bert_val = data_bert_val[:, 0]

clf = load('saved_models/RF_up_resample.joblib')
pre = clf.predict_proba(data_bert_val)
pre = np.asarray(pre)

data_non_TF_pssm, labels_non_TF_pssm = read_h5py('F:/BioInformatic/Dataset/TF vs non-TF/PSSM/', 'non_TF_independent')
data_TF_pssm, labels_TF_pssm = read_h5py('F:/BioInformatic/Dataset/TF vs non-TF/PSSM/', 'TF_independent')

data = np.append(data_non_TF_pssm, data_TF_pssm, axis=0)
labels = np.append(labels_non_TF_pssm, labels_TF_pssm, axis=0)
data = np.expand_dims(data, axis=-1).astype(np.float32)

count = 0
for i in range(len(labels)):
    if labels[i] != labels_bert_val[i]:
        count += 1
print(count)

path = "saved_models/18 deeper/"
model_paths = os.listdir(path)
model = []
for model_path in model_paths:
    model.append(tf.keras.models.load_model(path + model_path, compile=False))

vote = voting(model, data)
# ave = average(model, data)
# med = median(model, data)

ave = []
for i in range(len(pre)):
    tmp = [(pre[i][0] + vote[i][0]) / 2, (pre[i][1] + vote[i][1]) / 2]
    ave.append(tmp)

ave = np.asarray(ave)

sen = sensitivity(labels, ave)
spe = specificity(labels, ave)
accc = acc(labels, ave)
mccc = mcc(labels, ave)
aucc = auc(labels, ave)
print('SEN:' + str(sen))
print('SPE:' + str(spe))
print('ACC:' + str(accc))
print('MCC:' + str(mccc))
print('AUC:' + str(aucc))
