# -*- coding: utf-8 -*-
import json

import numpy as np
import math
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tensorflow import keras
from joblib import dump, load
from utils.helpers import *


if __name__ == '__main__':
    file = h5py.File('IFS.h5', 'r')
    sen_val = file['sen']
    spe_val = file['spe']
    acc_val = file['acc']
    auc_val = file['auc']
    mcc_val = file['mcc']

    print(np.argmax(acc_val))

    plt.plot(np.arange(1, len(acc_val) + 1), acc_val, label='acc')
    plt.plot(np.arange(1, len(sen_val) + 1), sen_val, label='sen')
    plt.plot(np.arange(1, len(spe_val) + 1), spe_val, label='spe')
    # plt.xlim(0, 400)
    # plt.ylim(0, 1)
    plt.legend()
    plt.show()

    # print(sen_val[np.argmax(sen_val)])





