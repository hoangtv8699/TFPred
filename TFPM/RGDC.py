# from utils.helpers import *
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from Bio import SeqIO
from sklearn import metrics
from sklearn import ensemble
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from utils.helpers import *
from sklearn.utils import resample
from sklearn.model_selection import StratifiedKFold
from joblib import dump, load
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn import tree
from sklearn.linear_model import LinearRegression

if __name__ == '__main__':
    TFPM_used, _ = read_fasta('../data/TFPM vs TFPNM/TFPM_training_dataset_used.txt')
    TFPM_unused, _ = read_fasta('../data/TFPM vs TFPNM/TFPM_training_dataset_unused.txt')
    TFPNM, _ = read_fasta('../data/TFPM vs TFPNM/TFPNM_training_dataset.txt')

    TFPM_val, _ = read_fasta('../data/TFPM vs TFPNM/TFPM_independent_dataset.txt')
    TFPNM_val, _ = read_fasta('../data/TFPM vs TFPNM/TFPNM_independent_dataset.txt')

    # TFPM_used, _ = read_fasta('../data/TF vs non-TF/TF_training_dataset.txt')
    # TFPNM, _ = read_fasta('../data/TF vs non-TF/non-TF_training_dataset.txt')
    #
    # TFPM_val, _ = read_fasta('../data/TF vs non-TF/TF_independent_dataset.txt')
    # TFPNM_val, _ = read_fasta('../data/TF vs non-TF/non-TF_independent_dataset.txt')

    gap = 4
    # op = 'op13'
    # ops = ['op5', 'op8', 'op9', 'op11', 'op13']
    ops = ['op13']

    TFPM_used_all = [[] for i in range(len(TFPM_used))]
    TFPM_unused_all = [[] for i in range(len(TFPM_unused))]
    TFPNM_all = [[] for i in range(len(TFPNM))]

    TFPM_val_all = [[] for i in range(len(TFPM_val))]
    TFPNM_val_all = [[] for i in range(len(TFPNM_val))]

    for op in ops:
        reduced_TFPM_used = RAACs(TFPM_used, op=op)
        RGDP_TFPM_used = RGDPs(reduced_TFPM_used, gap=gap, op=op)
        reduced_TFPM_unused = RAACs(TFPM_unused, op=op)
        RGDP_TFPM_unused = RGDPs(reduced_TFPM_unused, gap=gap, op=op)
        reduced_TFPNM = RAACs(TFPNM, op=op)
        RGDP_TFPNM = RGDPs(reduced_TFPNM, gap=gap, op=op)

        reduced_TFPM_val = RAACs(TFPM_val, op=op)
        RGDP_TFPM_val = RGDPs(reduced_TFPM_val, gap=gap, op=op)
        reduced_TFPNM_val = RAACs(TFPNM_val, op=op)
        RGDP_TFPNM_val = RGDPs(reduced_TFPNM_val, gap=gap, op=op)

        TFPM_used_all = np.append(TFPM_used_all, RGDP_TFPM_used, axis=-1)
        TFPM_unused_all = np.append(TFPM_unused_all, RGDP_TFPM_unused, axis=-1)
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
    data = np.append(TFPM_used_all, TFPM_unused_all, axis=0)
    # data = TFPM_used_all
    # data = resample(data, replace=False, n_samples=int(len(TFPNM_all)), random_state=7)
    # RGDP_TFPNM = resample(RGDP_TFPNM, replace=True, n_samples=int(len(data)), random_state=7)
    labels = np.ones(len(data))
    data = np.append(data, TFPNM_all, axis=0)
    labels = np.append(labels, np.zeros(len(TFPNM_all)), axis=0)
    # validating data
    data_val = np.append(TFPM_val_all, TFPNM_val_all, axis=0)
    labels_val = np.append(np.ones(len(TFPM_val_all)), np.zeros(len(TFPNM_val_all)), axis=0)

    # # BERT
    # path = 'F:/BioInformatic/Dataset/TFPM vs TFPNM/BERT/'
    # data, labels = read_h5py(path, 'TFPM_training_top100')
    # print(data.shape)
    # data = data[:, 0]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=18)

    i = 0
    for train_index, val_index in skf.split(data, labels):
        # split data
        train_data = data[train_index]
        train_labels = labels[train_index]
        val_data = data[val_index]
        val_labels = labels[val_index]

        # # resampling
        # posi = train_data[train_labels == 1]
        # nega = train_data[train_labels == 0]
        # # up resample
        # # posi = resample(posi, replace=False, n_samples=int(len(nega)))
        # # down resample
        # nega = resample(nega, replace=True, n_samples=int(len(posi)))
        # # concat data
        # print(len(posi))
        # print(len(nega))
        # labels_posi = [1 for i in range(len(posi))]
        # labels_nega = [0 for i in range(len(nega))]
        # train_data = np.append(posi, nega, axis=0)
        # train_labels = np.append(labels_posi, labels_nega)

        # # SMOTE
        # sm = SMOTE(random_state=1)
        # train_data, train_labels = sm.fit_resample(train_data, train_labels)

        # get importance
        # DCT_TFPM_GGAP = load('../saved_models/DCT_TFPM_GGAP.joblib')
        # # importance = DCT_TFPM_GGAP.feature_importances_
        # importance = DCT_TFPM_GGAP.coef_[0]
        # importance = np.argsort(importance)[::-1]

        # file = h5py.File('mrmr' + '.h5', 'r')
        # importance = np.asarray(file['mrmr'])

        # sen_val = []
        # spe_val = []
        # acc_val = []
        # auc_val = []
        # mcc_val = []
        # for k in range(1, len(train_data[0]) + 1):
        #     # pick = importance[:k]
        #     #
        #     # train_data_pick = train_data[:, pick]
        #     # val_data_pick = val_data[:, pick]
        #     # feature ranking ANOVA
        #     anova_filter = SelectKBest(f_classif, k=k)
        #     # clf = ensemble.RandomForestClassifier(random_state=101)
        #     clf = SVC(gamma='auto', probability=True, random_state=7)
        #     clf = make_pipeline(StandardScaler(), anova_filter, clf)
        #     clf.fit(train_data, train_labels)
        #
        #     # dump(clf, '../saved_models/TFPM ANOVA/' + str(i) + '_SVM.joblib')
        #
        #     pre = clf.predict_proba(val_data)
        #     pre = np.asarray(pre)
        #
        #     # print('train test:')
        #     sen = sensitivity(val_labels, pre)
        #     spe = specificity(val_labels, pre)
        #     accc = acc(val_labels, pre)
        #     mccc = mcc(val_labels, pre)
        #     aucc = auc(val_labels, pre)
        #     sen_val.append(sen)
        #     spe_val.append(spe)
        #     acc_val.append(accc)
        #     auc_val.append(aucc)
        #     mcc_val.append(mccc)
        #     print(k)

        # pick = importance[:37]
        # train_data = train_data[:, pick]
        # val_data = val_data[:, pick]

        # anova_filter = SelectKBest(f_classif, k=1358)
        # clf = ensemble.RandomForestClassifier(n_estimators=20, random_state=7)
        # clf = tree.DecisionTreeClassifier(random_state=7)
        # clf = LinearRegression()
        clf = SVC(gamma='auto', probability=True, random_state=7)
        clf = make_pipeline(StandardScaler(), clf)
        clf.fit(train_data, train_labels)

        dump(clf, '../saved_models/TFPM op5/' + str(i) + '_SVM.joblib')

        pre = clf.predict_proba(val_data)
        pre = np.asarray(pre)

        print('independent test:')
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

        # pre = clf.predict_proba(data_val)
        # pre = np.asarray(pre)

        # print('independent test:')
        # sen = sensitivity(labels_val, pre)
        # spe = specificity(labels_val, pre)
        # accc = acc(labels_val, pre)
        # mccc = mcc(labels_val, pre)
        # aucc = auc(labels_val, pre)
        # print('SEN:' + str(sen))
        # print('SPE:' + str(spe))
        # print('ACC:' + str(accc))
        # print('MCC:' + str(mccc))
        # print('AUC:' + str(aucc))

        # f = h5py.File('IFS.h5', 'w')
        # f.create_dataset('sen', data=sen_val)
        # f.create_dataset('spe', data=spe_val)
        # f.create_dataset('acc', data=acc_val)
        # f.create_dataset('auc', data=auc_val)
        # f.create_dataset('mcc', data=mcc_val)
        # f.close()
        # break
        i += 1

