from scipy.io import loadmat
from scipy.sparse import issparse
import numpy as np
from Data import VFLDataset
from torch.utils.data import DataLoader
import VFL
import torch
import os
from SFFS import get_f_stat_index


def add_noise_by_percentages(X, percentage: float):
    num_feat = X.shape[1]
    num_noise = int(num_feat * percentage)
    # 70 percent of normal noise
    # 20 percent of overwhelmed noise
    # 10 percent of shortcut noise
    num_normal = int(num_noise * 0.7)
    num_overwhelmed = int(num_noise * 0.2)
    num_shortcut = int(num_noise * 0.1)
    return num_normal, num_overwhelmed, num_shortcut



EPOCH_SETTING = {
     'arcene.mat': 30,
     'BASEHOCK.mat': 30,
     'COIL20.mat': 100,
     'gisette.mat': 30,
     'Isolet.mat': 100,
     'PCMAC.mat': 30,
    'RELATHE.mat': 30,
     }


if __name__ == "__main__":
    torch.manual_seed(0)
    DIR = "Data"
    file_names = os.listdir(DIR)
    # file_names = ['Isolet.mat']
    results = dict.fromkeys(file_names)
    percentages = np.arange(0.1, 0.6, 0.1)
    for file_name in file_names:
        
        if file_name == 'madelon.mat':
            break

        results = {

            'percentage': percentages,
            'FNN': [],
            'STG': [],
            'GINI+STG': [],
            'DualSTG': [],
            'DualSTG-double': [],
            'SFFS 0.5': [],
            'SFFS 0.1': [],
        }

        if file_name.endswith(".mat"):
            mat = loadmat(os.path.join(DIR, file_name))
            X = mat["X"]
            y = mat["Y"]
            if issparse(X):
                X = X.todense()
            y = y.flatten()
            print(file_name, X.shape, y.shape)
            if file_name in ['madelon.mat', 'arcene.mat', 'gisette.mat']:
                y[np.where(y == -1)] = 0
            if file_name in ['BASEHOCK.mat', 'RELATHE.mat', 'PCMAC.mat']:
                y[np.where(y == 1)] = 0
                y[np.where(y == 2)] = 1
            if file_name in ['COIL20.mat', 'Isolet.mat']:
                y = y - 1

            name = file_name.rstrip('.mat')
            print(name)
            RESULT_DIR = "Results"
            result_dir = os.path.join(RESULT_DIR, name)
            if not os.path.exists(result_dir):
                os.mkdir(result_dir)

            EPOCH = EPOCH_SETTING[file_name]
            NUM_TRAIL = 3
            EMBED_DIM = 8
            HIDDEN_DIM = [32, 16]
            if file_name == 'Isolet.mat':
                HIDDEN_DIM = [64, 32]
            
            for trail in range(NUM_TRAIL):
                for percentage in percentages:
                    num_normal, num_overwhelmed, num_shortcut = add_noise_by_percentages(X, percentage)
                    dataset = VFLDataset(data_source=(X, y),
                                         num_clients=3,
                                         gini_portion=None,
                                         insert_noise=True,
                                         num_random_samples=num_normal,
                                         num_overwhelmed=num_overwhelmed,
                                         num_shortcut=num_shortcut, 
                                         test_size=0.5)

                    train_loader = DataLoader(
                        dataset.train(), batch_size=128, shuffle=True)
                    val_loader = DataLoader(
                        dataset.valid(), batch_size=1000, shuffle=True)
                    test_loader = DataLoader(dataset.test(), batch_size=1000, shuffle=True)
                    input_dim_list = dataset.get_input_dim_list()
                    noisy_label = dataset.get_inserted_features_label()


                    ###########################
                    # FNN Model
                    ############################
                    print("FNN Model")
                    saving_name = f'FNN_{percentage}_{trail}'
                    output_dim = np.unique(y).size
                    criterion = torch.nn.CrossEntropyLoss()
                    models, top_model = VFL.make_binary_models(
                        input_dim_list=input_dim_list,
                        type='FNN',
                        emb_dim=EMBED_DIM,
                        output_dim=output_dim, hidden_dims=HIDDEN_DIM,
                        activation='relu')
                    history = VFL.train(
                        models, top_model, train_loader, val_loader, test_loader,
                        epochs=EPOCH,
                        criterion=criterion,
                        verbose=False,
                        save_mask_at=100000,
                        noise_label=noisy_label
                    )
                    fnn_acc = history.tail(3)['test_acc'].mean()
                    print(fnn_acc)

                    ###########################
                    # STG Model
                    ############################

                    print("STG")
                    saving_name = f'STG_{percentage}_{trail}'
                    models, top_model = VFL.make_binary_models(
                        input_dim_list=input_dim_list,
                        type='STG',
                        emb_dim=8,
                        output_dim=output_dim,
                        hidden_dims=[32, 16],
                        activation='relu')
                    VFL.train(
                        models, top_model, train_loader, val_loader, test_loader,
                        epochs=EPOCH,
                        criterion=criterion,
                        verbose=True,
                        log_dir=os.path.join(result_dir, saving_name) + ".csv",
                        save_mask_at=100000,
                        noise_label=noisy_label
                    )

                    ###########################
                    # STG with GINI Initialization
                    ############################
                    print("STG with GINI Initialization")
                    saving_name = f'STG_GINI_{percentage}_{trail}'
                    gini_labels = dataset.gini_filter(0.5)
                    feat_idx_list = dataset.get_feature_index_list()
                    mus = VFL.initialize_mu(gini_labels, feat_idx_list)

                    models, top_model = VFL.make_binary_models(
                        input_dim_list=input_dim_list,
                        type='STG',
                        emb_dim=8,
                        output_dim=output_dim,
                        hidden_dims=[32, 16],
                        activation='relu',
                        lam=0.1,
                        mus=mus)
                    VFL.train(models, top_model, train_loader, val_loader, test_loader,
                              epochs=EPOCH,
                              criterion=criterion,
                              verbose=True,
                              log_dir=os.path.join(result_dir, saving_name) + ".csv",
                              save_mask_at=100000,
                              noise_label=noisy_label)

                    ###########################
                    # Dual-STG  Model
                    ############################
                    print("Dual-STG")
                    saving_name = f'Dual-STG_{percentage}_{trail}'

                    mus = VFL.initialize_mu(gini_labels, feat_idx_list)

                    models, top_model = VFL.make_binary_models(
                        input_dim_list=input_dim_list,
                        type='DualSTG',
                        emb_dim=8,
                        output_dim=output_dim,
                        hidden_dims=[32, 16],
                        activation='relu',
                        lam=0.1, top_lam=0.1,
                        mus=mus)

                    VFL.train(
                        models, top_model, train_loader, val_loader, test_loader,
                        epochs=EPOCH,
                        criterion=criterion,
                        verbose=True,
                        log_dir=os.path.join(result_dir, saving_name) + ".csv",
                        save_mask_at=100000,
                        noise_label=noisy_label)

                    ###########################
                    # Double-Dual-STG  Model
                    ############################
                    print("Double-Dual-STG")
                    saving_name = f'Double-Dual-STG_{percentage}_{trail}'
                    mus = VFL.initialize_mu(gini_labels, feat_idx_list)
                    models, top_model = VFL.make_binary_models(
                        input_dim_list=input_dim_list,
                        type='DualSTG',
                        emb_dim=8,
                        output_dim=output_dim,
                        hidden_dims=[32, 16],
                        activation='relu',
                        lam=0.1, top_lam=0.1,
                        mus=mus)
                    VFL.train(
                        models, top_model, train_loader, val_loader, test_loader,
                        epochs=2 * EPOCH,
                        criterion=criterion,
                        verbose=True,
                        log_dir=os.path.join(result_dir, saving_name) + ".csv",
                        save_mask_at=100000,
                        noise_label=noisy_label)

                    ###########################
                    # SFFS Half Model
                    ############################
                    print("SFFS Half")
                    saving_name = f'SFFS_Half_{percentage}_{trail}'
                    # index = get_f_stat_index(X, y)
                    gini_labels = dataset.gini_filter(0.5)
                    gini_labels = gini_labels.flatten()
                    # print(gini_labels.shape)
                    X, y = dataset.get_data()
                    # print(X.shape, y.shape)
                    X_filtered = X[:, np.nonzero(gini_labels)].squeeze()
                    # print(X_filtered.shape)
                    dataset = VFLDataset(data_source=(X_filtered, y),
                                         num_clients=2,
                                         gini_portion=None,
                                         insert_noise=False,
                                         test_size=0.5)

                    train_loader = DataLoader(dataset.train(), batch_size=256, shuffle=True)
                    val_loader = DataLoader(dataset.valid(), batch_size=1000, shuffle=True)
                    test_loader = DataLoader(dataset.test(), batch_size=1000, shuffle=True)
                    input_dim_list = dataset.get_input_dim_list()
                    models, top_model = VFL.make_binary_models(
                        input_dim_list=input_dim_list,
                        type='FNN',
                        emb_dim=8,
                        output_dim=output_dim,
                        hidden_dims=[32, 16],
                        activation='relu')
                    VFL.train(models, top_model, train_loader, val_loader, test_loader,
                              epochs=EPOCH,
                              criterion=criterion,
                              verbose=True,
                              log_dir=os.path.join(result_dir, saving_name) + ".csv",
                              save_mask_at=100000,
                              noise_label=noisy_label)

                    ###########################
                    # SFFS Quarter Model
                    ############################
                    print("SFFS Quarter")
                    saving_name = f'SFFS_Quarter_{percentage}_{trail}'
                    gini_labels = dataset.gini_filter(0.25)
                    gini_labels = gini_labels.flatten()
                    # print(gini_labels.shape)
                    X, y = dataset.get_data()
                    # print(X.shape, y.shape)
                    X_filtered = X[:, np.nonzero(gini_labels)].squeeze()
                    # print(X_filtered.shape)
                    dataset = VFLDataset(data_source=(X_filtered, y),
                                         num_clients=2,
                                         gini_portion=None,
                                         insert_noise=False,
                                         test_size=0.5)

                    train_loader = DataLoader(dataset.train(), batch_size=256, shuffle=True)
                    val_loader = DataLoader(dataset.valid(), batch_size=1000, shuffle=True)
                    test_loader = DataLoader(dataset.test(), batch_size=1000, shuffle=True)
                    input_dim_list = dataset.get_input_dim_list()
                    models, top_model = VFL.make_binary_models(
                        input_dim_list=input_dim_list,
                        type='FNN',
                        emb_dim=8,
                        output_dim=output_dim,
                        hidden_dims=[32, 16],
                        activation='relu')
                    VFL.train(models, top_model, train_loader, val_loader, test_loader,
                              epochs=EPOCH,
                              criterion=criterion,
                              verbose=True,
                              log_dir=os.path.join(result_dir, saving_name) + ".csv",
                              save_mask_at=100000,
                              noise_label=noisy_label)
