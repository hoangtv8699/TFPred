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
import time
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from reduce_pssm import reduce_pssms
from RAAC import GDPs
from RAAC import read_fasta

import seaborn as sns

sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})
RS = 123

# path = 'F:/BioInformatic/Dataset/TFPM vs TFNPM/BERT/'
# data_bert, labels_bert = read_h5py(path, 'TFPM_training_top100')
# data_bert, labels_bert = read_h5py(path, 'TFPM_testing_top100')
# data_bert = data_bert[:, 0, :1000]
# data_bert = data_bert[:, 0, :1000]
# data_bert = normalize(data_bert)
# data_bert = data_bert.reshape((data_bert.shape[0], -1))

# # resampling
# posi = data_bert[labels_bert == 1]
# nega = data_bert[labels_bert == 0]
# # up resample
# posi = resample(posi, replace=True, n_samples=int(len(nega)))
# # down resample
# # nega = resample(nega, replace=True, n_samples=int(len(posi)), random_state=1010)
# # concat data
# print(len(posi))
# print(len(nega))
# labels_posi = [1 for i in range(len(posi))]
# labels_nega = [0 for i in range(len(nega))]
# data_bert = np.append(posi, nega, axis=0)
# labels_bert = np.append(labels_posi, labels_nega)
# data_bert, labels_bert = balance_data(data_bert, labels_bert, 31)

# # fasta
# TFPM_used, TFPM_used_labels = read_fasta('../data/TFPM vs TFPNM/TFPM_training_dataset_used.txt')
# TFPM_unused, TFPM_unused_labels = read_fasta('../data/TFPM vs TFPNM/TFPM_training_dataset_unused.txt')
# TFPNM, TFPNM_labels = read_fasta('../data/TFPM vs TFPNM/TFPNM_training_dataset.txt')
# #
# TFPM_val, TFPM_val_labels = read_fasta('../data/TFPM vs TFPNM/TFPM_independent_dataset.txt')
# TFPNM_val, TFPNM_val_labels = read_fasta('../data/TFPM vs TFPNM/TFPNM_independent_dataset.txt')
#
# # pssm
# path = 'F:/BioInformatic/Dataset/TFPM vs TFPNM/PSSM/'
# TFPM_pssm_used, TFPM_pssm_used_labels = read_data(path + 'TFPM_training_dataset_used/', padding='none')
# TFPM_pssm_unused, TFPM_pssm_unused_labels = read_data(path + 'TFPM_training_dataset_unused/', padding='none')
# TFPNM_pssm, TFPNM_pssm_labels = read_data(path + 'TFPNM_training_dataset/', padding='none')
# #
# TFPM_val_pssm, TFPM_val_pssm_labels = read_data(path + 'TFPM_independent_dataset/', padding='none')
# TFPNM_val_pssm, TFPNM_val_pssm_labels = read_data(path + 'TFPNM_independent_dataset/', padding='none')
#
#
# reduced_TFPM_used, _ = reduce_pssms(TFPM_used, TFPM_used_labels, TFPM_pssm_used, TFPM_pssm_used_labels)
# reduced_TFPM_unused, _ = reduce_pssms(TFPM_unused, TFPM_unused_labels, TFPM_pssm_unused, TFPM_pssm_unused_labels)
# reduced_TFPNM, _ = reduce_pssms(TFPNM, TFPNM_labels, TFPNM_pssm, TFPNM_pssm_labels)
#
# reduced_TFPM_val, _ = reduce_pssms(TFPM_val, TFPM_val_labels, TFPM_val_pssm, TFPM_val_pssm_labels)
# reduced_TFPNM_val, _ = reduce_pssms(TFPNM_val, TFPNM_val_labels, TFPNM_val_pssm, TFPNM_val_pssm_labels)
#
# data = np.append(reduced_TFPM_used, reduced_TFPM_unused, axis=0)
# # data = resample(data, replace=False, n_samples=int(len(reduced_TFPNM)))
# labels = np.ones((len(data)))
#
# # reduced_TFPNM = resample(reduced_TFPNM, replace=True, n_samples=int(len(data)))
#
# data = np.append(data, reduced_TFPNM, axis=0)
# labels = np.append(labels, np.zeros(len(reduced_TFPNM)))
#
# # data = np.append(reduced_TFPM_used, reduced_TFPNM, axis=0)
# # labels = np.append(np.ones(len(reduced_TFPM_used)), np.zeros(len(reduced_TFPNM)))
#
# data_val = np.append(reduced_TFPM_val, reduced_TFPNM_val, axis=0)
# labels_val = np.append(np.ones(len(reduced_TFPM_val)), np.zeros(len(reduced_TFPNM_val)))
#
# data = data.reshape((data.shape[0], -1))
# data_val = data_val.reshape((data_val.shape[0], -1))


TFPM_used = read_fasta('../data/TFPM vs TFPNM/TFPM_training_dataset_used.txt')
TFPM_unused = read_fasta('../data/TFPM vs TFPNM/TFPM_training_dataset_unused.txt')
TFPNM = read_fasta('../data/TFPM vs TFPNM/TFPNM_training_dataset.txt')

TFPM_val = read_fasta('../data/TFPM vs TFPNM/TFPM_independent_dataset.txt')
TFPNM_val = read_fasta('../data/TFPM vs TFPNM/TFPNM_independent_dataset.txt')

gap = 0

# reduced_TFPM_used = RAACs(TFPM_used)
# RGDP_TFPM_used = RGDPs(reduced_TFPM_used, gap=gap)
# reduced_TFPM_unused = RAACs(TFPM_unused)
# RGDP_TFPM_unused = RGDPs(reduced_TFPM_unused, gap=gap)
# reduced_TFPNM = RAACs(TFPNM)
# RGDP_TFPNM = RGDPs(reduced_TFPNM, gap=gap)
#
# reduced_TFPM_val = RAACs(TFPM_val)
# RGDP_TFPM_val = RGDPs(reduced_TFPM_val, gap=gap)
# reduced_TFPNM_val = RAACs(TFPNM_val)
# RGDP_TFPNM_val = RGDPs(reduced_TFPNM_val, gap=gap)

# G-gap dipeptide composition
RGDP_TFPM_used = GDPs(TFPM_used, gap=gap)
RGDP_TFPM_unused = GDPs(TFPM_unused, gap=gap)
RGDP_TFPNM = GDPs(TFPNM, gap=gap)

RGDP_TFPM_val = GDPs(TFPM_val, gap=gap)
RGDP_TFPNM_val = GDPs(TFPNM_val, gap=gap)

# training data
data = np.append(RGDP_TFPM_used, RGDP_TFPM_unused, axis=0)
# data = resample(data, replace=True, n_samples=int(len(RGDP_TFPNM)))
data = np.append(data, RGDP_TFPNM, axis=0)
labels = np.append(np.ones(len(RGDP_TFPM_used)), np.ones(len(RGDP_TFPM_unused)), axis=0)
# labels = np.ones(len(RGDP_TFPNM))
labels = np.append(labels, np.zeros(len(RGDP_TFPNM)), axis=0)
# validating data
data_val = np.append(RGDP_TFPM_val, RGDP_TFPNM_val, axis=0)
labels_val = np.append(np.ones(len(RGDP_TFPM_val)), np.zeros(len(RGDP_TFPNM_val)), axis=0)


def fashion_scatter(x, colors):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40, c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit corresponding to the label
    txts = []

    for i in range(num_classes):
        # Position of each label at median of data points.

        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
    plt.show()

    return f, ax, sc, txts


from sklearn.decomposition import PCA

time_start = time.time()

pca = PCA(n_components=4)
pca_result = pca.fit_transform(data)

print('PCA done! Time elapsed: {} seconds'.format(time.time() - time_start))

pca_df = pd.DataFrame(columns = ['pca1','pca2','pca3','pca4'])

pca_df['pca1'] = pca_result[:,0]
pca_df['pca2'] = pca_result[:,1]
pca_df['pca3'] = pca_result[:,2]
pca_df['pca4'] = pca_result[:,3]

print('Variance explained per principal component: {}'.format(pca.explained_variance_ratio_))

top_two_comp = pca_df[['pca1','pca2']] # taking first and second principal component

fashion_scatter(top_two_comp.values, labels) # Visualizing the PCA output

from sklearn.manifold import TSNE
import time
time_start = time.time()

fashion_tsne = TSNE(random_state=RS).fit_transform(data)

print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

fashion_scatter(fashion_tsne, labels)

time_start = time.time()

pca_50 = PCA(n_components=50)
pca_result_50 = pca_50.fit_transform(data)

print('PCA with 50 components done! Time elapsed: {} seconds'.format(time.time()-time_start))

print('Cumulative variance explained by 50 principal components: {}'.format(np.sum(pca_50.explained_variance_ratio_)))

import time
time_start = time.time()


fashion_pca_tsne = TSNE(random_state=RS).fit_transform(pca_result_50)

print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

fashion_scatter(fashion_pca_tsne, labels)