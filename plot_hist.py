# -*- coding: utf-8 -*-
import os
import fnmatch
import shutil
import numpy as np
import pandas as pd
from Bio import SeqIO
from matplotlib import pyplot as plt
import re
import numpy as np
import math
import matplotlib.patches as mpatches
import torch


def main():
    n_bins = 100

    TF = 'data/TF vs non-TF/TF_training_dataset.txt'
    non_TF = 'data/TF vs non-TF/non-TF_training_dataset.txt'

    # read the fasta sequences from input file
    tfpm = SeqIO.parse(open(TF), 'fasta')
    non_tfpm = SeqIO.parse(open(non_TF), 'fasta')
    # loop through fasta sequences
    lengths = []

    for fasta in tfpm:
        # get name and value of each sequence
        name, sequence = fasta.id, str(fasta.seq)
        lengths.append(len(sequence))

    for fasta in non_tfpm:
        # get name and value of each sequence
        name, sequence = fasta.id, str(fasta.seq)
        lengths.append(len(sequence))

    print(lengths[np.argmax(lengths)])
    print(lengths[np.argmin(lengths)])

    less = 0
    more = 0
    for length in lengths:
        if length > 2000:
            more += 1
        else:
            less += 1

    print(less / (less + more))

    plt.rc('xtick', labelsize=24)
    plt.rc('ytick', labelsize=24)
    N, bins, patches = plt.hist(lengths, bins=np.arange(0, np.max(lengths), 10))
    for i in range(0, 150):
        patches[i].set_facecolor('blue')
    for i in range(150, int(np.max(lengths) / 10) - 1):
        patches[i].set_facecolor('red')
    plt.xlabel('Sample length', fontsize=24)
    plt.ylabel('Number of Sample', fontsize=24)
    plt.xticks([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000])
    blue_patch = mpatches.Patch(color='blue', label='97.23%')
    red_patch = mpatches.Patch(color='red', label='2.77%')
    plt.legend(handles=[blue_patch, red_patch], prop={'size': 20})
    plt.show()


if __name__ == '__main__':
    main()
