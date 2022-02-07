# -*- coding: utf-8 -*-
import pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold
from tensorflow import keras
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.metrics import AUC

from utils.helpers import *
from sklearn.utils import resample



def models(maxlen=400):
    model = Sequential([
        Input(shape=(20, maxlen, 1)),

        ZeroPadding2D(1),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.2),

        ZeroPadding2D(1),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.2),

        ZeroPadding2D(1),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.2),

        ZeroPadding2D(1),
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.2),

        GlobalAveragePooling2D(),

        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(2, activation='softmax')
    ])

    optimizer = keras.optimizers.Adam(lr=0.0001)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', sensitivity, specificity, mcc]
    )
    return model


def train(n_splits, non_TF_pssm, TF_pssm, batch_size, epochs, random_state, maxlen=1500,
          save_path_model="saved_models/", save_path_his="saved_histories/"):
    path = 'F:/BioInformatic/Dataset/TF vs non-TF/PSSM/'
    # read data
    data_non_TF_pssm, _ = read_data(path + non_TF_pssm + '/', padding="pad_sequence", maxlen=maxlen)
    labels_non_TF_pssm = np.zeros(len(data_non_TF_pssm))
    data_TF_pssm, _ = read_data(path + TF_pssm + '/', padding="pad_sequence", maxlen=maxlen)
    labels_TF_pssm = np.ones(len(data_TF_pssm))
    #
    # save_h5py(np.array(data_non_TF_pssm).astype('float32'), labels_non_TF_pssm, path, non_TF_pssm + ' 1000')
    # save_h5py(np.array(data_TF_pssm).astype('float32'), labels_TF_pssm, path, TF_pssm + ' 1000')
    # data_non_TF_pssm, labels_non_TF_pssm = read_h5py(path, non_TF_pssm)
    # data_TF_pssm, labels_TF_pssm = read_h5py(path, TF_pssm)

    data = np.append(data_non_TF_pssm, data_TF_pssm, axis=0)
    labels = np.append(labels_non_TF_pssm, labels_TF_pssm, axis=0)

    # create 10-fold cross validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

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
        #
        # posi = resample(posi, replace=True, n_samples=len(nega), random_state=1)
        #
        # train_data = np.append(posi, nega, axis=0)
        # labels_posi = [1 for i in range(len(posi))]
        # labels_nega = [0 for i in range(len(nega))]
        # train_labels = np.append(labels_posi, labels_nega)

        # # SMOTE
        # sm = SMOTE(random_state=random_state)
        # train_data, train_labels = sm.fit_resample(train_data, train_labels)

        train_data, train_labels = balance_data(train_data, train_labels, random_state=random_state)
        val_data, val_labels = balance_data(val_data, val_labels, random_state=random_state)

        train_data = np.expand_dims(train_data, axis=-1).astype(np.float32)
        val_data = np.expand_dims(val_data, axis=-1).astype(np.float32)

        train_posi = sum(train_labels)
        train_nega = len(train_labels) - train_posi
        val_posi = sum(val_labels)
        val_nega = len(val_labels) - val_posi

        print("number of train positive: {}".format(train_posi))
        print("number of train negative: {}".format(train_nega))
        print("number of val positive: {}".format(val_posi))
        print("number of val negative: {}".format(val_nega))

        print(train_labels.shape)

        # create model
        model = models(maxlen)
        print(model.summary())

        # create weight
        weight = {0: 1, 1: 6}

        # callback
        es = EarlyStopping(
            monitor="val_loss",
            patience=10,
            mode='min',
            restore_best_weights=True
        )
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8,
                                      patience=5, verbose=1, mode='auto', min_delta=0.0001, cooldown=5,
                                      min_lr=0.00001)
        callbacks = [
            reduce_lr,
            es
        ]

        # train model
        history = model.fit(
            train_data,
            train_labels,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(val_data, val_labels),
            # class_weight=weight,
            callbacks=callbacks,
            # shuffle=True,
            verbose=2
        )
        pre = model.predict(val_data)
        aucc = auc(val_labels, pre)
        history.history['val_auc'] = aucc
        model.save(save_path_model + get_model_name(i))
        pickle.dump(history.history, open(save_path_his + "model_" + str(i), 'wb'), pickle.HIGHEST_PROTOCOL)
        # history = pickle.load(open(save_path_his + "model_" + str(i), 'wb'))
        i += 1


if __name__ == '__main__':
    non_TF_pssm = 'non_TF_training'
    TF_pssm = 'TF_training'
    n_splits = 5
    # random_state = random.randint(0, 19999)
    random_state = 18
    BATCH_SIZE = 16
    EPOCHS = 200
    print(random_state)
    save_path_model = "../saved_models/" + str(random_state) + " 2000/"
    save_path_his = "../saved_histories/" + str(random_state) + " 2000/"
    if not os.path.isdir(save_path_model):
        os.mkdir(save_path_model)
    if not os.path.isdir(save_path_his):
        os.mkdir(save_path_his)
    train(n_splits, non_TF_pssm, TF_pssm, BATCH_SIZE, EPOCHS, random_state, maxlen=2000,
          save_path_model=save_path_model, save_path_his=save_path_his)

