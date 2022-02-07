import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold
from utils.helpers import *


class TFPred(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(20, 32, 3, stride=1)
        self.conv2 = nn.Conv1d(32, 64, 3, stride=1)
        self.conv3 = nn.Conv1d(64, 128, 3, stride=1)
        # self.conv4 = nn.Conv1d(128, 256, 3, stride=1)

        self.batchnorm1 = nn.BatchNorm1d(32)
        self.batchnorm2 = nn.BatchNorm1d(64)
        self.batchnorm3 = nn.BatchNorm1d(128)
        # self.batchnorm4 = nn.BatchNorm1d(256)

        self.maxpool = nn.MaxPool1d(5, 5)
        self.dropout = nn.Dropout(p=0.2)
        self.flatten = nn.Flatten()
        self.zeropad = nn.ZeroPad2d(1)

        self.lstm = nn.LSTM(128, 512, batch_first=True, bidirectional=True)

        self.linear1 = nn.Linear(512, 512)
        # self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.batchnorm1(x))
        x = self.dropout(x)

        x = self.conv2(x)
        x = F.relu(self.batchnorm2(x))
        x = self.dropout(x)

        x = self.conv3(x)
        x = F.relu(self.batchnorm3(x))
        x = self.dropout(x)
        x = self.maxpool(x)

        # x = self.conv4(x)
        # x = F.relu(self.batchnorm4(x))
        # x = self.dropout(x)
        # x = self.maxpool(x)

        x = torch.transpose(x, 1, 2)
        # # print(x.size())
        out, (ht, ct) = self.lstm(x)
        # print(ht[-1].size())

        x = F.relu(self.linear1(ht[-1]))
        x = self.dropout(x)
        # x = F.relu(self.linear2(x))
        # x = self.dropout(x)
        x = F.softmax(self.linear3(x), dim=1)

        return x


class CombineDataset(Dataset):
    def __init__(self, pssm, labels):
        self.pssm = pssm
        self.labels = labels

    def __len__(self):
        return len(self.pssm)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        pssm = self.pssm[idx]
        label = self.labels[idx]

        return pssm, label


def train(n_splits, batch_size, epochs, random_state, maxlen=1500,
          save_path_model="saved_models/18 torch/", cuda=True):
    path_pssm = 'F:/BioInformatic/Dataset/TF vs non-TF/PSSM/'
    path_bert = 'F:/BioInformatic/Dataset/TF vs non-TF/BERT/'
    non_TF_pssm = 'non_TF_training'
    TF_pssm = 'TF_training'
    # read data
    # data_non_TF_pssm = read_data(path_pssm + non_TF_pssm + '/', padding="pad_sequence", maxlen=maxlen)
    # labels_non_TF_pssm = np.zeros(len(data_non_TF_pssm))
    # data_TF_pssm = read_data(path_pssm + TF_pssm + '/', padding="pad_sequence", maxlen=maxlen)
    # labels_TF_pssm = np.ones(len(data_TF_pssm))
    #
    # save_h5py(np.array(data_non_TF_pssm).astype('float32'), labels_non_TF_pssm, path_pssm, 'non_TF_training_no_norm')
    # save_h5py(np.array(data_TF_pssm).astype('float32'), labels_TF_pssm, path_pssm, 'TF_training_no_norm')
    data_non_TF_pssm, labels_non_TF_pssm = read_h5py(path_pssm, 'non_TF_training_no_norm')
    data_TF_pssm, labels_TF_pssm = read_h5py(path_pssm, 'TF_training_no_norm')

    data_pssm = np.append(data_non_TF_pssm, data_TF_pssm, axis=0)
    labels_pssm = np.append(labels_non_TF_pssm, labels_TF_pssm, axis=0)

    # create 10-fold cross validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold = 0
    for train_index, val_index in skf.split(data_pssm, labels_pssm):
        # split data
        train_data_pssm = data_pssm[train_index]
        train_labels_pssm = labels_pssm[train_index]
        val_data_pssm = data_pssm[val_index]
        val_labels_pssm = labels_pssm[val_index]

        # balance data
        train_data_pssm, train_labels_pssm = balance_data(train_data_pssm, train_labels_pssm, random_state=random_state)

        # train_data_pssm = np.expand_dims(train_data_pssm, axis=1).astype(np.float32)
        # val_data_pssm = np.expand_dims(val_data_pssm, axis=1).astype(np.float32)
        # train_labels_pssm = train_labels_pssm.astype(np.float32)
        # val_labels_pssm = val_labels_pssm.astype(np.float32)

        train_posi = sum(train_labels_pssm)
        train_nega = len(train_labels_pssm) - train_posi
        val_posi = sum(val_labels_pssm)
        val_nega = len(val_labels_pssm) - val_posi

        print("number of train positive: {}".format(train_posi))
        print("number of train negative: {}".format(train_nega))
        print("number of val positive: {}".format(val_posi))
        print("number of val negative: {}".format(val_nega))

        # create data loader
        trainset = CombineDataset(train_data_pssm, train_labels_pssm)
        valset = CombineDataset(val_data_pssm, val_labels_pssm)

        train_loader = DataLoader(
            trainset,
            batch_size=batch_size)

        val_loader = DataLoader(
            valset,
            batch_size=batch_size)

        # training model
        min_valid_loss = np.inf
        model = TFPred()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

        # print(model)

        if cuda:
            model.cuda()

        his = {}
        for i in ['loss', 'sen', 'spe', 'acc', 'mcc', 'auc']:
            his['train_' + i] = []
            his['val_' + i] = []

        earlyStop = 0
        for e in range(epochs):
            # training
            model.train()
            running_loss = 0.0
            for pssm, label in train_loader:
                if cuda:
                    pssm, label = pssm.cuda(), label.cuda()

                target = model(pssm)
                loss = criterion(target, label.long())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * label.size(0)

            # validating
            model.eval()
            y_true, y_pred = predict(model, train_loader)

            his['train_loss'].append(running_loss / len(train_loader))
            his['train_sen'].append(sensitivity(y_true, y_pred))
            his['train_spe'].append(specificity(y_true, y_pred))
            his['train_acc'].append(acc(y_true, y_pred))
            his['train_mcc'].append(mcc(y_true, y_pred))
            his['train_auc'].append(auc(y_true, y_pred))

            valid_loss = 0.0
            for pssm, label in val_loader:
                if cuda:
                    pssm, label = pssm.cuda(), label.cuda()

                target = model(pssm)
                loss = criterion(target, label.long())
                valid_loss += loss.item() * label.size(0)

            y_true, y_pred = predict(model, val_loader)

            his['val_loss'].append(valid_loss / len(val_loader))
            his['val_sen'].append(sensitivity(y_true, y_pred))
            his['val_spe'].append(specificity(y_true, y_pred))
            his['val_acc'].append(acc(y_true, y_pred))
            his['val_mcc'].append(mcc(y_true, y_pred))
            his['val_auc'].append(auc(y_true, y_pred))

            print('epoch: {}'.format(e))
            print('training... loss: {}, sen: {}, spe: {}, acc: {}, mcc: {}, auc: {}'
                  .format(his['train_loss'][-1], his['train_sen'][-1],
                          his['train_spe'][-1], his['train_acc'][-1], his['train_mcc'][-1], his['train_auc'][-1]))
            print('validating... loss: {}, sen: {}, spe: {}, acc: {}, mcc: {}, auc: {}\n'
                  .format(his['val_loss'][-1], his['val_sen'][-1],
                          his['val_spe'][-1], his['val_acc'][-1], his['val_mcc'][-1], his['val_auc'][-1]))

            if min_valid_loss > valid_loss:
                earlyStop = 0
                min_valid_loss = valid_loss
                # Saving State Dict
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'history': his,
                }, save_path_model + get_model_name(fold) + '.pt')
            else:
                earlyStop += 1
            if earlyStop == 10:
                break
        fold += 1


if __name__ == '__main__':
    train(10, 16, 200, 18)