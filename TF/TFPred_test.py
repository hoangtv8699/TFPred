from TF.TFPred import *

if __name__ == '__main__':
    path_pssm = 'F:/BioInformatic/Dataset/TF vs non-TF/PSSM/'
    path_bert = 'F:/BioInformatic/Dataset/TF vs non-TF/BERT/'
    data_non_TF_pssm, labels_non_TF_pssm = read_h5py(path_pssm, 'non_TF_independent')
    data_TF_pssm, labels_TF_pssm = read_h5py(path_pssm, 'TF_independent')

    data_pssm = np.append(data_non_TF_pssm, data_TF_pssm, axis=0)
    labels_pssm = np.append(labels_non_TF_pssm, labels_TF_pssm, axis=0)
    # data_pssm = np.expand_dims(data_pssm, axis=1).astype(np.float32)

    valset = CombineDataset(data_pssm, labels_pssm)
    val_loader = DataLoader(valset, batch_size=16)

    # checkpoint = torch.load('saved_models/18 torch/model_0.h5.pt')
    # model = TFPred()
    # model.load_state_dict(checkpoint['model_state_dict'])

    path = "../saved_models/18 torch/"
    model_paths = os.listdir(path)
    models = []
    for model_path in model_paths:
        checkpoint = torch.load(path + model_path)
        model = TFPred()
        model.load_state_dict(checkpoint['model_state_dict'])
        models.append(model)

    y_true, y_pred = predict_voting(models, val_loader)

    print(sensitivity(y_true, y_pred))
    print(specificity(y_true, y_pred))
    print(acc(y_true, y_pred))
    print(mcc(y_true, y_pred))
    print(auc(y_true, y_pred))

    # m = nn.BatchNorm1d(100)
    # # Without Learnable Parameters
    # m = nn.BatchNorm1d(100, affine=False)
    # input = torch.randn(20, 100)
    # output = m(input)

