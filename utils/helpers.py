# -*- coding: utf-8 -*-
import os

import numpy as np
from Bio import SeqIO
from Bio.SubsMat import MatrixInfo
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import one_hot

import pandas as pd
import tensorflow as tf
import random
import math
from matplotlib import pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import sequence
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import time
from sklearn import metrics
import torch

import h5py


# reduce pssm
def reduce_pssm(seq, pssm):
    header = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    header_dict = dict()
    for i in range(len(header)):
        header_dict[header[i]] = i
    reduced = np.zeros((20, 20))
    count = np.zeros(20)

    for i in range(len(seq)):
        pos = header_dict[seq[i]]
        reduced[pos] += pssm[i]
        count[pos] += 1

    for i in range(20):
        reduced[i] /= len(seq)

    return reduced


def reduce_pssms(fasta, fasta_labels, pssm, pssm_labels):
    TFPM_fasta_dict = dict()
    for data, label in zip(fasta, fasta_labels):
        TFPM_fasta_dict[label] = data

    reduced = []
    labels = []
    for data, label in zip(pssm, pssm_labels):
        labels.append(label)
        seq = TFPM_fasta_dict[label]
        reduced.append(reduce_pssm(seq, data))

    return np.asarray(reduced), labels


# GDCs, RAAC
def count_freq(path):
    header = ['A', 'R', 'N', 'C', 'D', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    count = {x: 0 for x in header}

    fasta_sequences = SeqIO.parse(open(path), 'fasta')
    sequences = []
    for fasta in fasta_sequences:
        # get name and value of each sequence
        name, seq = str(fasta.id), str(fasta.seq)
        sequences.append(seq)

    total = 0
    for seq in sequences:
        for char in seq:
            count[char] += 1
            total += 1

    return count, total


def find_group():
    TFPM_used, TFPM_used_total = count_freq('../data/TFPM vs TFPNM/TFPM_training_dataset_used.txt')
    TFPM_unused, TFPM_unused_total = count_freq('../data/TFPM vs TFPNM/TFPM_training_dataset_unused.txt')
    TFPNM, TFPNM_total = count_freq('../data/TFPM vs TFPNM/TFPNM_training_dataset.txt')
    TFPM = TFPM_used

    for key in TFPM_used.keys():
        TFPM[key] = TFPM_used[key] + TFPM_unused[key]

    TFPM_total = TFPM_used_total + TFPM_unused_total
    # TFPM_total = TFPM_used_total

    for key in TFPM.keys():
        TFPM[key] /= TFPM_total

    for key in TFPNM.keys():
        TFPNM[key] /= TFPNM_total

    X = np.arange(len(TFPM.values()))
    width = 0.15
    fig, ax = plt.subplots()
    rects1 = ax.bar(X - 0.1, list(TFPM.values()), color='blue', width=width, label='TFPM')
    rects2 = ax.bar(X + 0.1, list(TFPNM.values()), color='orange', width=width, label='TFPNM')

    ax.set_xticks(np.arange(len(TFPM.keys())))
    ax.set_xticklabels(list(TFPM.keys()), fontsize=10)
    plt.show()

    tfpm_char = []
    tfpnm_char = []

    for key in TFPM.keys():
        if TFPM[key] > TFPNM[key]:
            tfpm_char.append(key)
        else:
            tfpnm_char.append(key)

    print(tfpm_char)
    print(tfpnm_char)

    splits = [0.03, 0.06]
    a, b, c = [], [], []
    for key in tfpm_char:
        if TFPM[key] < splits[0]:
            a.append(key)
        elif splits[0] <= TFPM[key] <= splits[1]:
            b.append(key)
        else:
            c.append(key)

    x, y, z = [], [], []
    for key in tfpnm_char:
        if TFPM[key] < splits[0]:
            x.append(key)
        elif splits[0] <= TFPM[key] <= splits[1]:
            y.append(key)
        else:
            z.append(key)
    groups = [a, b, c, x, y, z]
    return groups


def RAAC(seq, op='op5'):
    scheme = {
        'op5': [['G'], ['I', 'V', 'F', 'Y', 'W'], ['A', 'L', 'M', 'E', 'Q', 'R', 'K'], ['P'], ['N', 'D', 'H', 'S', 'T', 'C']],
        'op8': [['G'], ['I', 'V'], ['F', 'Y', 'W'], ['A', 'L', 'M'], ['E', 'Q', 'R', 'K'], ['P'], ['N', 'D'], ['H', 'S', 'T', 'C']],
        'op9': [['G'], ['I', 'V'], ['F', 'Y', 'W'], ['A', 'L', 'M'], ['E', 'Q', 'R', 'K'], ['P'], ['N', 'D'], ['H', 'S'], ['T', 'C']],
        'op11': [['G'], ['I', 'V'], ['F', 'Y', 'W'], ['A'], ['L', 'M'], ['E', 'Q', 'R', 'K'], ['P'], ['N', 'D'],
                 ['H', 'S'], ['T'], ['C']],
        'op13': [['G'], ['I', 'V'], ['F', 'Y', 'W'], ['A'], ['L'], ['M'], ['E'], ['Q', 'R', 'K'], ['P'], ['N', 'D'],
                 ['H', 'S'], ['T'], ['C']],
        'op20': ['G', 'I', 'V', 'F', 'Y', 'W', 'A', 'L', 'M', 'E', 'Q', 'R', 'K', 'P', 'N', 'D', 'H', 'S', 'T', 'C']
    }
    # groups = [['C', 'H', 'F', 'Y'], ['N', 'D', 'Q', 'I', 'K', 'T', 'V'], ['E'], ['M', 'W'], ['R'],
    #           ['A', 'G', 'L', 'P', 'S']]
    groups = scheme[op]
    for group in groups:
        for key in group:
            seq = seq.replace(key, group[0])
    return seq


def RAACs(seqs, op='op5'):
    new_seqs = []
    for i in range(len(seqs)):
        new_seqs.append(RAAC(seqs[i], op=op))
    return new_seqs


def RGDP(seq, gap=1, op='op5'):
    scheme = {
        'op5': ['G', 'I', 'A', 'P', 'N'],
        'op8': ['G', 'I', 'F', 'A', 'E', 'P', 'N', 'H'],
        'op9': ['G', 'I', 'F', 'A', 'E', 'P', 'N', 'H', 'T'],
        'op11': ['G', 'I', 'F', 'A', 'L', 'E', 'P', 'N', 'H', 'T', 'C'],
        'op13': ['G', 'I', 'F', 'A', 'L', 'M', 'E', 'Q', 'P', 'N', 'H', 'T', 'C'],
        'op20': ['G', 'I', 'V', 'F', 'Y', 'W', 'A', 'L', 'M', 'E', 'Q', 'R', 'K', 'P', 'N', 'D', 'H', 'S', 'T', 'C']
    }
    header = scheme[op]
    reduced_dipeptide = [g1 + g2 for g1 in header for g2 in header]

    myDict = {}
    for t in reduced_dipeptide:
        myDict[t] = 0

    total = len(seq) - 1 - gap
    for j in range(total):
        myDict[seq[j] + seq[j + 1 + gap]] += 1

    RGPC = []
    for key in reduced_dipeptide:
        RGPC.append(myDict[key] / total)

    return RGPC


def RGDPs(seqs, gap=1, op='op5'):
    new_seqs = []
    for i in range(len(seqs)):
        new_seqs.append(RGDP(seqs[i], gap=gap, op=op))
    return new_seqs


def GDP(seq, gap=1):
    header = ['A', 'R', 'N', 'C', 'D', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    reduced_dipeptide = [g1 + g2 for g1 in header for g2 in header]

    myDict = {}
    for t in reduced_dipeptide:
        myDict[t] = 0

    total = len(seq) - 1 - gap
    for j in range(total):
        myDict[seq[j] + seq[j + 1 + gap]] += 1

    RGPC = []
    for key in reduced_dipeptide:
        RGPC.append(myDict[key] / total)

    return RGPC


def GDPs(seqs, gap=1):
    new_seqs = []
    for i in range(len(seqs)):
        new_seqs.append(GDP(seqs[i], gap=gap))
    return new_seqs


def normalize_data(datas):
    data_copy = np.copy(datas)
    for i in range(len(data_copy)):
        data = data_copy[i]
        for j in range(len(data)):
            row = data[j]
            sample_mean = sum(row) / (len(row) + K.epsilon())
            standard_deviation = math.sqrt(sum([pow(x - sample_mean, 2) for x in row]) / (len(row) + K.epsilon()))
            data[j] = [(x - sample_mean) / (standard_deviation + K.epsilon()) for x in row]
        data_copy[i] = data
    return data_copy


def normalize(data):
    for j in range(len(data)):
        row = data[j]
        sample_mean = sum(row) / (len(row) + K.epsilon())
        standard_deviation = math.sqrt(sum([pow(x - sample_mean, 2) for x in row]) / (len(row) + K.epsilon()))
        data[j] = [(x - sample_mean) / (standard_deviation + K.epsilon()) for x in row]
    return np.asarray(data)


def normalize_common(data):
    sample_mean = 0
    for sequence in data:
        sample_mean += sum(sequence)
    sample_mean /= (data.shape[0] * data.shape[1] + K.epsilon())

    standard_deviation = 0
    for sequence in data:
        standard_deviation += sum([pow(x - sample_mean, 2) for x in sequence])
    standard_deviation = math.sqrt(standard_deviation / (data.shape[0] * data.shape[1] + K.epsilon()))

    data = (data - sample_mean) / (standard_deviation + K.epsilon())
    return data, sample_mean, standard_deviation


# def normalize(data, sample_mean, standard_deviation):
#     data = (data - sample_mean) / (standard_deviation + K.epsilon())
#     return data


def balance_data(data, labels, random_state):
    posi = []
    nega = []
    balanced_data = []
    balanced_labels = []

    for i in range(len(data)):
        if labels[i] == 1:
            posi.append(data[i])
        else:
            nega.append(data[i])

    random.Random(random_state).shuffle(posi)
    # j = 0
    # for i in range(len(nega)):
    #     if j < len(posi):
    #         balanced_data.append(posi[j])
    #         balanced_labels.append(1)
    #         j += 1
    #     else:
    #         j = 0
    #         random.shuffle(posi)
    #     balanced_data.append(nega[i])
    #     balanced_labels.append(0)

    if len(posi) < len(nega):
        tmp = int(len(nega) / len(posi))
        j = 0
        for i in range(len(nega)):
            if i % tmp == 0:
                if j < len(posi):
                    balanced_data.append(posi[j])
                    balanced_labels.append(1)
                    j += 1
            balanced_data.append(nega[i])
            balanced_labels.append(0)
    else:
        tmp = int(len(posi) / len(nega))
        j = 0
        for i in range(len(posi)):
            if i % tmp == 0:
                if j < len(nega):
                    balanced_data.append(nega[j])
                    balanced_labels.append(0)
                    j += 1
            balanced_data.append(posi[i])
            balanced_labels.append(1)

    return np.asarray(balanced_data), np.asarray(balanced_labels)


def pad_same(sequence, maxlen=400):
    copy_sequence = np.copy(sequence)
    cur_len = len(sequence)
    if cur_len > maxlen:
        return sequence[:maxlen, :]
    step = int(maxlen / cur_len - 1)
    for i in range(step):
        sequence = np.append(sequence, copy_sequence, axis=0)
    append_len = maxlen - len(sequence)
    return np.append(sequence, copy_sequence[:append_len, :], axis=0)


def read_data(path, padding="pad_sequence", maxlen=400):
    pssm_files = os.listdir(path)
    data = []
    labels = []
    for pssm_file in pssm_files:
        df = pd.read_csv(path + pssm_file, sep=',', header=None)
        df = np.asarray(df, dtype=np.float32)
        # df = normalize(df.T)
        df = df.T
        if padding == "pad_sequence":
            df = sequence.pad_sequences(df, maxlen=maxlen, padding='post', truncating='post', dtype='float32',
                                        value=0.0)
        elif padding == "same":
            df = pad_same(df, maxlen=maxlen)
        data.append(df)
        labels.append(pssm_file.split('.')[0].split('_')[1])
    # data = np.asarray(data, dtype=np.float32)
    return data, labels


def read_bert(path, padding="pad_sequence", maxlen=1500, top=None):
    pssm_files = os.listdir(path)
    data = []
    count = 0
    for pssm_file in pssm_files:
        df = pd.read_csv(path + pssm_file, sep=',', header=None)
        df = np.asarray(df)
        df = df[1:1501, :]
        if padding == "pad_sequence":
            if df.shape[0] > maxlen:
                df = df[:1500, :]
            else:
                tmp = np.zeros((maxlen, df.shape[1]))
                tmp[:df.shape[0], :df.shape[1]] = df
                df = tmp
        elif padding == "same":
            df = pad_same(df, maxlen=maxlen)
        if top is not None:
            data.append(df.T[:100])
        else:
            data.append(df.T)
        count += 1
        if count % 10 == 0:
            print(count)
    data = np.asarray(data, dtype=float)
    return data


def read_fasta(path, maxlen=1500, encode='token'):
    # read the fasta sequences from input file
    fasta_sequences = SeqIO.parse(open(path), 'fasta')
    sequences = []
    labels = []
    for fasta in fasta_sequences:
        # get name and value of each sequence
        name, sequence = str(fasta.id), str(fasta.seq)
        sequences.append(sequence)
        labels.append(name)

    if encode == 'onehot':
        tk = Tokenizer(num_words=21, char_level=True)
        # Fitting
        tk.fit_on_texts(sequences)
        print(tk.word_index)
        sequences, labels = pad_sequences(tk.texts_to_sequences(sequences), maxlen=maxlen, padding='post',
                                          truncating='post'), np.asarray(labels)
        one_hot_sequences = []
        for sequence in sequences:
            b = np.zeros((maxlen, 21))
            b[np.arange(maxlen), sequence - 1] = 1
            one_hot_sequences.append(b.T)
        return np.asarray(one_hot_sequences), labels
    elif encode == 'token':
        return sequences, labels

    return None


def read_csv(path, maxlen=400, encode='token'):
    df = pd.read_csv(path, skipinitialspace=True)

    sequences = []
    labels = []
    for index, row in df.iterrows():
        label = 1
        if row['SEQCLASS'] == 'non-antioxidant':
            label = 0
        sequences.append(row['SEQUENCE'])
        labels.append(label)

    if encode == 'onehot':
        tk = Tokenizer(num_words=21, char_level=True)
        # Fitting
        tk.fit_on_texts(sequences)
        sequences, labels = pad_sequences(tk.texts_to_sequences(sequences),
                                          maxlen=maxlen, padding='post',
                                          truncating='post'), np.asarray(labels)
        one_hot_sequences = []
        for sequence in sequences:
            b = np.zeros((400, 21))
            b[np.arange(400), sequence - 1] = 1
            one_hot_sequences.append(b.T)
        return np.asarray(one_hot_sequences), labels
    elif encode == 'token':
        return sequences, labels

    return None


def score_match(pair, matrix):
    if pair not in matrix:
        pair = (tuple(reversed(pair)))
        if pair not in matrix:
            return 0
        else:
            return matrix[pair]
    else:
        return matrix[pair]


def create_matrix(sequence, matrix):
    header = ['A', 'R', 'N', 'C', 'D', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    blosum = []
    for c in sequence:
        row = []
        for v in header:
            pair = (c, v)
            row.append(score_match(pair, matrix))
        blosum.append(row)
    return blosum


def read_blosum(path, maxlen=400, type='fasta'):
    matrix = MatrixInfo.blosum62
    if type == 'fasta':
        data, labels = read_fasta(path, maxlen=maxlen, encode='token')
    else:
        data, labels = read_csv(path, maxlen=maxlen, encode='token')
    blosum_data = []
    for seq in data:
        blosum = create_matrix(seq, matrix)
        blosum = np.asarray(blosum)
        blosum = sequence.pad_sequences(blosum.T, maxlen=maxlen, padding='post', truncating='post')
        blosum_data.append(blosum)
    return np.asarray(blosum_data), np.asarray(labels)


def check_missing(path):
    missing = []
    label = np.zeros(1780)
    pssm_files = os.listdir(path)

    for pssm_file in pssm_files:
        id = pssm_file.split('_')[0]
        label[int(id)] += 1

    for i in range(1780):
        if label[i] < 1:
            missing.append(i)
    return missing


def encodes_amino_feature(sequences, maxlen=400):
    result = []
    for sequence in sequences:
        result.append(encode_amino_feature(sequence, maxlen=maxlen).T)
    return np.asarray(result)


def encode_amino_feature(sequence, maxlen=400):
    AvgMass = {
        'A': 89.09404, 'C': 121.15404, 'D': 133.10384, 'E': 147.13074, 'F': 165.19184,
        'G': 75.06714, 'H': 155.15634, 'I': 131.17464, 'K': 146.18934, 'L': 131.17464,
        'M': 149.20784, 'N': 132.11904, 'P': 115.13194, 'Q': 146.14594, 'R': 174.20274,
        'S': 105.09344, 'T': 119.12034, 'V': 117.14784, 'W': 204.22844, 'Y': 181.19124
    }

    volume = {
        'A': 67, 'C': 86, 'D': 91, 'E': 109, 'F': 135,
        'G': 48, 'H': 118, 'I': 124, 'K': 135, 'L': 124,
        'M': 124, 'N': 96, 'P': 90, 'Q': 114, 'R': 148,
        'S': 73, 'T': 93, 'V': 105, 'W': 163, 'Y': 141
    }

    scp = {
        'A': 0, 'C': 1, 'D': 1, 'E': 1, 'F': 1,
        'G': 1, 'H': 1, 'I': 0, 'K': 1, 'L': 0,
        'M': 0, 'N': 1, 'P': 0, 'Q': 0, 'R': 0,
        'S': 1, 'T': 1, 'V': 0, 'W': 1, 'Y': 0
    }

    sca = {
        'A': 0, 'C': -1, 'D': 0, 'E': 1, 'F': 0,
        'G': 1, 'H': 0, 'I': 0, 'K': -1, 'L': 0,
        'M': 0, 'N': -1, 'P': 0, 'Q': 0, 'R': 0,
        'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0
    }

    HIndex = {
        'A': 1.8, 'C': -4.5, 'D': -3.5, 'E': -3.5, 'F': 2.5,
        'G': -3.5, 'H': -3.5, 'I': -0.4, 'K': -3.2, 'L': 4.5,
        'M': 3.8, 'N': -3.9, 'P': 1.9, 'Q': 2.8, 'R': -1.6,
        'S': -0.8, 'T': -0.7, 'V': -0.9, 'W': -1.3, 'Y': 4.2
    }

    result = []
    for a in sequence:
        if a == 'U':
            result.append([0, 0, 0, 0, 0])
        else:
            result.append([AvgMass[a], volume[a], HIndex[a], scp[a], sca[a]])
    result = np.asarray(result).T
    result[:3] = normalize(result[:3])
    result = result.T

    if len(sequence) > maxlen:
        result = np.asarray(result[:maxlen])
    else:
        tmp = np.zeros((maxlen - len(sequence), 5))
        result = np.append(result, tmp, axis=0)
    return result


def get_model_name(k):
    return 'model_' + str(k) + '.h5'


def plot_loss(history, i):
    f = plt.figure()
    plt.rc('xtick', labelsize=24)
    plt.rc('ytick', labelsize=24)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.ylabel('loss', fontsize=24)
    plt.xlabel('epochs', fontsize=24)
    plt.legend(['train loss', 'val loss'], loc='upper right', prop={'size': 20})
    plt.show()
    # plt.savefig('saved_plots/loss_{}.png'.format(i))
    # f.savefig('hinh 4.png')
    # f.savefig('hinh 4.pdf', dpi=700)
    # plt.clf()


def plot_accuracy(history, i):
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # plt.savefig('saved_plots/accuracy_{}.png'.format(i))
    # plt.clf()


def plot_sensitivity(history, i):
    plt.plot(history['sensitivity'])
    plt.plot(history['val_sensitivity'])
    plt.title('model sensitivity')
    plt.ylabel('sensitivity')
    plt.xlabel('epochs')
    plt.legend(['train loss', 'val loss'], loc='upper left')
    plt.show()
    # plt.savefig('saved_plots/sensitivity_{}.png'.format(i))
    # plt.clf()


def plot_specificity(history, i):
    plt.plot(history['specificity'])
    plt.plot(history['val_specificity'])
    plt.title('model specificity')
    plt.ylabel('specificity')
    plt.xlabel('epochs')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # plt.savefig('saved_plots/specificity_{}.png'.format(i))
    # plt.clf()


def voting(models, data):
    pre = []
    pre_bin = []
    for model in models:
        # pre_tmp = model.predict(data)
        pre_tmp = model.predict_proba(data)
        pre_bin.append(np.argmax(pre_tmp, axis=-1))

    for i in range(len(pre_bin[0])):
        tmp = [0, 0]
        for x in pre_bin:
            tmp[x[i]] += 1
        tmp[0] = tmp[0] / 10
        tmp[1] = tmp[1] / 10
        pre.append(tmp)
    return np.asarray(pre)


def average(models, data):
    pre = []
    for model in models:
        # pre.append(model.predict(data))
        pre.append(model.predict_proba(data))

    ave = []
    for i in range(len(pre[0])):
        tmp = [0, 0]
        for j in range(len(pre)):
            tmp[0] += pre[j][i][0]
            tmp[1] += pre[j][i][1]
        tmp[0] /= 10
        tmp[1] /= 10
        ave.append(tmp)
    return np.asarray(ave)


def median(models, data):
    pre = []
    for model in models:
        # pre.append(model.predict(data))
        pre.append(model.predict_proba(data))

    med = []
    for i in range(len(pre[0])):
        b = []
        for j in range(len(pre)):
            b.append(pre[j][i])
        med.append(np.median(b, axis=0))
    return np.asarray(med)


# root = 'F:/BioInformatic/Dataset/TF vs non-TF/BERT/'


def save_h5py(data, labels, path, name):
    f = h5py.File(path + name + '.h5', 'w')
    f.create_dataset(name, data=data)
    f.create_dataset('labels', data=labels)
    f.close()


def read_h5py(path, name):
    file = h5py.File(path + name + '.h5', 'r')
    data = np.asarray(file[name], dtype=np.float32)
    labels = np.asarray(file['labels'], dtype=np.float32)
    return data, labels


def predict(model, data_loader, cuda=True):
    if cuda:
        model.cuda()

    y_true = []
    y_pred = []
    for idx, (e, target) in enumerate(data_loader):
        if cuda:
            e, target = e.cuda(), target.cuda()
        # pred = torch.argmax(model(e), dim=1)
        pred = model(e)
        y_true.extend(target.data.cpu().numpy())
        y_pred.extend(pred.data.cpu().numpy())
    return np.asarray(y_true), np.asarray(y_pred)


def predict_voting(models, data_loader, cuda=True):
    if cuda:
        for model in models:
            model.cuda()

    y_true = []
    y_pred = []
    for idx, (e, target) in enumerate(data_loader):
        if cuda:
            e = e.cuda()

        preds = []
        for model in models:
            preds.append(np.argmax(model(e).data.cpu().numpy(), axis=-1))

        for i in range(len(preds[0])):
            tmp = [0, 0]
            for x in preds:
                tmp[x[i]] += 1
            tmp[0] = tmp[0] / 10
            tmp[1] = tmp[1] / 10
            y_pred.append(tmp)

        y_true.extend(target.data.cpu().numpy())
    return np.asarray(y_true), np.asarray(y_pred)


def get_confusion_matrix(y_true, y_pred):
    return metrics.confusion_matrix(y_true, y_pred, labels=[0, 1])


# def sensitivity(y_true, y_pred):
#     y_pred_bin = np.argmax(y_pred, axis=-1)
#     epsilon = np.finfo(float).eps
#     cf_matrix = get_confusion_matrix(y_true, y_pred_bin)
#     tn, fp, fn, tp = cf_matrix.ravel()
#     # print('TN = {}, FP = {}, FN = {}, TP = {}'.format(tn, fp, fn, tp))
#
#     SEN = tp / (tp + fn + epsilon)
#     return SEN
#
#
# def specificity(y_true, y_pred):
#     y_pred_bin = np.argmax(y_pred, axis=-1)
#     epsilon = np.finfo(float).eps
#     cf_matrix = get_confusion_matrix(y_true, y_pred_bin)
#     tn, fp, fn, tp = cf_matrix.ravel()
#
#     SPE = tn / (tn + fp + epsilon)
#     return SPE
#
#
# def acc(y_true, y_pred):
#     y_pred_bin = np.argmax(y_pred, axis=-1)
#     return metrics.accuracy_score(y_true, y_pred_bin)
#
#
# def mcc(y_true, y_pred):
#     y_pred_bin = np.argmax(y_pred, axis=-1)
#     return metrics.matthews_corrcoef(y_true, y_pred_bin)
#
#
# def auc(y_true, y_pred):
#     fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred[:, 1], pos_label=1)
#     auc_score = metrics.auc(fpr, tpr)
#     return auc_score


def sensitivity(y_true, y_pred):
    y_pred_bin = K.argmax(y_pred, axis=-1)
    y_pred_bin = tf.cast(y_pred_bin, tf.float32)
    confusion_matrix = tf.math.confusion_matrix(y_true, y_pred_bin, num_classes=2)
    # as Keras Tensors
    TP = tf.cast(confusion_matrix[1, 1], dtype=tf.float32)
    FN = tf.cast(confusion_matrix[1, 0], dtype=tf.float32)

    sensitivity = TP / (TP + FN + K.epsilon())
    return sensitivity


def specificity(y_true, y_pred):
    y_pred_bin = K.argmax(y_pred, axis=-1)
    y_pred_bin = tf.cast(y_pred_bin, tf.float32)
    confusion_matrix = tf.math.confusion_matrix(y_true, y_pred_bin, num_classes=2)
    # as Keras Tensors
    TN = tf.cast(confusion_matrix[0, 0], dtype=tf.float32)
    FP = tf.cast(confusion_matrix[0, 1], dtype=tf.float32)

    specificity = TN / (TN + FP + K.epsilon())
    return specificity


def mcc(y_true, y_pred):
    y_pred_bin = K.argmax(y_pred, axis=-1)
    y_pred_bin = tf.cast(y_pred_bin, tf.float32)
    confusion_matrix = tf.math.confusion_matrix(y_true, y_pred_bin, num_classes=2)
    # as Keras Tensors
    TP = tf.cast(confusion_matrix[1, 1], dtype=tf.float32)
    FN = tf.cast(confusion_matrix[1, 0], dtype=tf.float32)
    TN = tf.cast(confusion_matrix[0, 0], dtype=tf.float32)
    FP = tf.cast(confusion_matrix[0, 1], dtype=tf.float32)

    MCC = (TP * TN) - (FP * FN)
    MCC /= (tf.math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) + K.epsilon())
    return MCC


def acc(y_true, y_pred):
    y_pred_bin = K.argmax(y_pred, axis=-1)
    confusion_matrix = tf.math.confusion_matrix(y_true, y_pred_bin, num_classes=2)
    # as Keras Tensors
    TP = tf.cast(confusion_matrix[1, 1], dtype=tf.float32)
    FN = tf.cast(confusion_matrix[1, 0], dtype=tf.float32)
    TN = tf.cast(confusion_matrix[0, 0], dtype=tf.float32)
    FP = tf.cast(confusion_matrix[0, 1], dtype=tf.float32)

    acc = (TP + TN) / (TP + TN + FP + FN + K.epsilon())
    return acc


def auc(y_true, y_pred):
    y_pred = np.asarray(y_pred)
    m = tf.keras.metrics.AUC()
    m.update_state(y_true, y_pred[:, 1])
    return m.result().numpy()
