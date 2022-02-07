from joblib import dump, load
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from utils.helpers import *

# path = 'F:/BioInformatic/Dataset/TFPM vs TFPNM/BERT/'
# TFPM_used = 'TFPM_training_dataset_used'
# TFPM_unused = 'TFPM_training_dataset_unused'
# TFNPM = 'TFPNM_training_dataset'

# prepare h5py file
# data_TFPM = read_bert(path + TFPM_used + '/', padding="pad_sequence", maxlen=1500)
# labels_TFPM = np.ones(len(data_TFPM))
# data_TFPM_unused = read_bert(path + TFPM_unused + '/', padding="pad_sequence", maxlen=1500)
# labels_TFPM_unused = np.ones(len(data_TFPM_unused))
# data_TFNPM = read_bert(path + TFNPM + '/', padding="pad_sequence", maxlen=1500)
# labels_TFNPM = np.zeros(len(data_TFNPM))

# save_h5py(data_TFPM, labels_TFPM, path, TFPM_used)
# save_h5py(data_TFPM_unused, labels_TFPM_unused, path, TFPM_unused)
# save_h5py(data_TFNPM, labels_TFNPM, path, TFNPM)
# data_TFPM, labels_TFPM = read_h5py(path, TFPM_used)
# data_TFPM_unused, labels_TFPM_unused = read_h5py(path, TFPM_unused)
# data_TFNPM, labels_TFNPM = read_h5py(path, TFNPM)

# reduced G-Gap data
TFPM_used, _ = read_fasta('../data/TFPM vs TFPNM/TFPM_training_dataset_used.txt')
TFPM_unused, _ = read_fasta('../data/TFPM vs TFPNM/TFPM_training_dataset_unused.txt')
TFPNM, _ = read_fasta('../data/TFPM vs TFPNM/TFPNM_training_dataset.txt')

gap = 2
# op = 'op13'
# ops = ['op5', 'op8', 'op9', 'op11', 'op13']
ops = ['op13']

TFPM_used_all = [[] for i in range(len(TFPM_used))]
TFPM_unused_all = [[] for i in range(len(TFPM_unused))]
TFPNM_all = [[] for i in range(len(TFPNM))]

for op in ops:
    reduced_TFPM_used = RAACs(TFPM_used, op=op)
    RGDP_TFPM_used = RGDPs(reduced_TFPM_used, gap=gap, op=op)
    reduced_TFPM_unused = RAACs(TFPM_unused, op=op)
    RGDP_TFPM_unused = RGDPs(reduced_TFPM_unused, gap=gap, op=op)
    reduced_TFPNM = RAACs(TFPNM, op=op)
    RGDP_TFPNM = RGDPs(reduced_TFPNM, gap=gap, op=op)

    TFPM_used_all = np.append(TFPM_used_all, RGDP_TFPM_used, axis=-1)
    TFPM_unused_all = np.append(TFPM_unused_all, RGDP_TFPM_unused, axis=-1)
    TFPNM_all = np.append(TFPNM_all, RGDP_TFPNM, axis=-1)


# G-gap dipeptide composition
# RGDP_TFPM_used = GDPs(TFPM_used, gap=gap)
# RGDP_TFPM_unused = GDPs(TFPM_unused, gap=gap)
# RGDP_TFPNM = GDPs(TFPNM, gap=gap)
#
# RGDP_TFPM_val = GDPs(TFPM_val, gap=gap)
# RGDP_TFPNM_val = GDPs(TFPNM_val, gap=gap)

# training data
data = np.append(TFPM_used_all, TFPM_unused_all, axis=0)
# data = resample(data, replace=False, n_samples=int(len(TFPNM_all)), random_state=7)
# RGDP_TFPNM = resample(RGDP_TFPNM, replace=True, n_samples=int(len(data)), random_state=7)
labels = np.ones(len(data))
data = np.append(data, TFPNM_all, axis=0)
labels = np.append(labels, np.zeros(len(TFPNM_all)), axis=0)

scaler = StandardScaler()
scaler.fit(data)
data = scaler.transform(data)

# # Load dataset.
# X, y = read_h5py(path, 'training_DCT_TFPM')
# print(X.shape)
# print(y.shape)

# clf = tree.DecisionTreeClassifier()
clf = SVC(gamma='auto', probability=True, random_state=7, kernel='linear')
clf.fit(data, labels)
# print(clf.score(data, labels))
importance = clf.coef_
dump(clf, '../saved_models/DCT_TFPM_GGAP.joblib')

# top 100 feature
clf = load('../saved_models/DCT_TFPM_GGAP.joblib')
# importance = clf.feature_importances_
top100 = np.argsort(importance)[::-1]
print(importance)
print(importance[top100])
# summarize feature importance
for i, v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i, v))
# plot feature importance
color = []
for x in range(len(importance)):
    color.append('b')
color[top100[0]] = 'r'
f = plt.figure()
plt.rc('xtick', labelsize=24)
plt.rc('ytick', labelsize=24)
plt.bar([x for x in range(len(importance))], importance, color=color)
plt.xlabel('Feature number', fontsize=24)
plt.ylabel('Importance level (%)', fontsize=24)
plt.show()
# f.savefig('hinh 3.png')
# f.savefig('hinh 3.pdf', dpi=700)


# reduce 1024xN to 1024 by geting mean
# data_TFPM, labels_TFPM = read_h5py(path, TFPM_used)
# data_TFPM_unused, labels_TFPM_unused = read_h5py(path, TFPM_unused)
# data_TFNPM, labels_TFNPM = read_h5py(path, TFNPM)
#
# X = np.append(data_TFPM, data_TFPM_unused, axis=0)
# X = np.append(X, data_TFNPM, axis=0)
# y = np.append(labels_TFPM, labels_TFPM_unused, axis=0)
# y = np.append(y, labels_TFNPM, axis=0)
# x1 = []
# for i in range(len(X)):
#     x2 = []
#     for j in range(len(X[i])):
#         x2.append(np.mean(X[i][j]))
#     x1.append(x2)
#
# save_h5py(x1, y, path, 'training_DCT_TFPM')

# data_TFPM, labels_TFPM = read_h5py(path, TFPM_used)
# # data_TFPM_unused, labels_TFPM_unused = read_h5py(path, TFPM_unused)
# data_TFNPM, labels_TFNPM = read_h5py(path, TFNPM)
#
# X = np.append(data_TFPM, data_TFNPM, axis=0)
# # X = np.append(X, data_TFPM_unused, axis=0)
# y = np.append(labels_TFPM, labels_TFNPM, axis=0)
# # y = np.append(y, labels_TFPM_unused, axis=0)
#
# X = X[:, top100, :]
#
# save_h5py(X, y, path, 'TFPM_training_top100_balance')

# TFPM_independent = 'TFPM_independent_dataset'
# TFPNM_independent = 'TFPNM_independent_dataset'
#
# data_TFPM, labels_TFPM = read_h5py(path, TFPM_independent)
# data_TFNPM, labels_TFNPM = read_h5py(path, TFPNM_independent)
#
# X = np.append(data_TFPM, data_TFNPM, axis=0)
# y = np.append(labels_TFPM, labels_TFNPM, axis=0)
#
# X = X[:, top100, :]
#
# save_h5py(X, y, path, 'TFPM_testing_top100')
