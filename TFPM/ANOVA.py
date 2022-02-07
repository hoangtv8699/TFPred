# from utils.helpers import *
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
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

if __name__ == '__main__':
    # TFPM_used, _ = read_fasta('../data/TFPM vs TFPNM/TFPM_training_dataset_used.txt')
    # TFPM_unused, _ = read_fasta('../data/TFPM vs TFPNM/TFPM_training_dataset_unused.txt')
    # TFPNM, _ = read_fasta('../data/TFPM vs TFPNM/TFPNM_training_dataset.txt')
    #
    # TFPM_val, _ = read_fasta('../data/TFPM vs TFPNM/TFPM_independent_dataset.txt')
    # TFPNM_val, _ = read_fasta('../data/TFPM vs TFPNM/TFPNM_independent_dataset.txt')
    #
    # gap = 0
    #
    # # G-gap dipeptide composition
    # RGDP_TFPM_used = GDPs(TFPM_used, gap=gap)
    # RGDP_TFPM_unused = GDPs(TFPM_unused, gap=gap)
    # RGDP_TFPNM = GDPs(TFPNM, gap=gap)
    #
    # RGDP_TFPM_val = GDPs(TFPM_val, gap=gap)
    # RGDP_TFPNM_val = GDPs(TFPNM_val, gap=gap)
    #
    # # training data
    # data = np.append(RGDP_TFPM_used, RGDP_TFPM_unused, axis=0)
    # # data = resample(data, replace=True, n_samples=int(len(RGDP_TFPNM)))
    # data = np.append(data, RGDP_TFPNM, axis=0)
    # labels = np.append(np.ones(len(RGDP_TFPM_used)), np.ones(len(RGDP_TFPM_unused)), axis=0)
    # # labels = np.ones(len(RGDP_TFPNM))
    # labels = np.append(labels, np.zeros(len(RGDP_TFPNM)), axis=0)
    # # validating data
    # data_val = np.append(RGDP_TFPM_val, RGDP_TFPNM_val, axis=0)
    # labels_val = np.append(np.ones(len(RGDP_TFPM_val)), np.zeros(len(RGDP_TFPNM_val)), axis=0)

    # fasta
    TFPM_used, TFPM_used_labels = read_fasta('../data/TFPM vs TFPNM/TFPM_training_dataset_used.txt')
    TFPM_unused, TFPM_unused_labels = read_fasta('../data/TFPM vs TFPNM/TFPM_training_dataset_unused.txt')
    TFPNM, TFPNM_labels = read_fasta('../data/TFPM vs TFPNM/TFPNM_training_dataset.txt')
    #
    TFPM_val, TFPM_val_labels = read_fasta('../data/TFPM vs TFPNM/TFPM_independent_dataset.txt')
    TFPNM_val, TFPNM_val_labels = read_fasta('../data/TFPM vs TFPNM/TFPNM_independent_dataset.txt')

    # pssm
    path = 'F:/BioInformatic/Dataset/TFPM vs TFPNM/PSSM/'
    TFPM_pssm_used, TFPM_pssm_used_labels = read_data(path + 'TFPM_training_dataset_used/', padding='none')
    TFPM_pssm_unused, TFPM_pssm_unused_labels = read_data(path + 'TFPM_training_dataset_unused/', padding='none')
    TFPNM_pssm, TFPNM_pssm_labels = read_data(path + 'TFPNM_training_dataset/', padding='none')
    #
    TFPM_val_pssm, TFPM_val_pssm_labels = read_data(path + 'TFPM_independent_dataset/', padding='none')
    TFPNM_val_pssm, TFPNM_val_pssm_labels = read_data(path + 'TFPNM_independent_dataset/', padding='none')

    reduced_TFPM_used, _ = reduce_pssms(TFPM_used, TFPM_used_labels, TFPM_pssm_used, TFPM_pssm_used_labels)
    reduced_TFPM_unused, _ = reduce_pssms(TFPM_unused, TFPM_unused_labels, TFPM_pssm_unused, TFPM_pssm_unused_labels)
    reduced_TFPNM, _ = reduce_pssms(TFPNM, TFPNM_labels, TFPNM_pssm, TFPNM_pssm_labels)

    reduced_TFPM_val, _ = reduce_pssms(TFPM_val, TFPM_val_labels, TFPM_val_pssm, TFPM_val_pssm_labels)
    reduced_TFPNM_val, _ = reduce_pssms(TFPNM_val, TFPNM_val_labels, TFPNM_val_pssm, TFPNM_val_pssm_labels)

    data = np.append(reduced_TFPM_used, reduced_TFPM_unused, axis=0)
    # data = resample(data, replace=False, n_samples=int(len(reduced_TFPNM)))
    labels = np.ones((len(data)))

    # reduced_TFPNM = resample(reduced_TFPNM, replace=True, n_samples=int(len(data)))

    data = np.append(data, reduced_TFPNM, axis=0)
    labels = np.append(labels, np.zeros(len(reduced_TFPNM)))

    # data = np.append(reduced_TFPM_used, reduced_TFPNM, axis=0)
    # labels = np.append(np.ones(len(reduced_TFPM_used)), np.zeros(len(reduced_TFPNM)))

    data_val = np.append(reduced_TFPM_val, reduced_TFPNM_val, axis=0)
    labels_val = np.append(np.ones(len(reduced_TFPM_val)), np.zeros(len(reduced_TFPNM_val)))

    data = data.reshape((data.shape[0], -1))
    data_val = data_val.reshape((data_val.shape[0], -1))

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

        # clf = ensemble.RandomForestClassifier(random_state=101)
        # clf = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True))

        sen_val = []
        spe_val = []
        acc_val = []
        auc_val = []
        mcc_val = []
        for k in range(1, len(train_data[0]) + 1):
            # feature ranking ANOVA
            anova_filter = SelectKBest(f_classif, k=k)
            clf = ensemble.RandomForestClassifier(random_state=101)
            # clf = SVC(gamma='auto', probability=True)
            anova_svm = make_pipeline(anova_filter, StandardScaler(), clf)
            anova_svm.fit(train_data, train_labels)

            # dump(clf, '../saved_models/TFPM ANOVA/' + str(i) + '_SVM.joblib')

            pre = anova_svm.predict_proba(val_data)
            pre = np.asarray(pre)

            # print('train test:')
            sen = sensitivity(val_labels, pre)
            spe = specificity(val_labels, pre)
            accc = acc(val_labels, pre)
            mccc = mcc(val_labels, pre)
            aucc = auc(val_labels, pre)
            sen_val.append(sen)
            spe_val.append(spe)
            acc_val.append(accc)
            auc_val.append(aucc)
            mcc_val.append(mccc)
            print(k)
            # print('SEN:' + str(sen))
            # print('SPE:' + str(spe))
            # print('ACC:' + str(accc))
            # print('MCC:' + str(mccc))
            # print('AUC:' + str(aucc))

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

        f = h5py.File('ANOVA.h5', 'w')
        f.create_dataset('sen', data=sen_val)
        f.create_dataset('spe', data=spe_val)
        f.create_dataset('acc', data=acc_val)
        f.create_dataset('auc', data=auc_val)
        f.create_dataset('mcc', data=mcc_val)
        f.close()
        i += 1
        break
