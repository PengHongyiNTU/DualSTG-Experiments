
from scipy.io import loadmat
from scipy.sparse import issparse
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from Data import VFLDataset
from torch.utils.data import DataLoader
import VFL
import torch
import os


DIR = "Data"
# file_names = os.listdir(DIR)
file_names = ['RELATHE.mat']
for file_name in file_names:
    if file_name.endswith(".mat"):
        mat = loadmat(os.path.join(DIR, file_name))
        X = mat["X"]
        y = mat["Y"]
        if issparse(X):
                data_X = data_X.todense()
        y = y.flatten()
        print(file_name, X.shape, y.shape)
        if file_name in ['madelon.mat', 'arcene.mat', 'gisette.mat']:
            y[np.where(y == -1)] = 0
        if file_name in ['BASEHOCK.mat', 'RELATHE.mat', 'PCMAC.mat']:
            y[np.where(y == 1)] = 0
            y[np.where(y == 2)] = 1
        if file_name in ['COIL20.mat', 'isolet.mat']:
            y = y-1
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
        name = file_name.rstrip('.mat')
        print(name)
        RESULT_DIR = "Results"
        result_dir = os.path.join(RESULT_DIR, name)
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)

        EPOCH = 100
        NUM_TRAIL = 5
        NUM_NORMAL = np.arange(0, 200, 20)
        NUM_OVERWHELMED = [0]+[20]*(len(NUM_NORMAL)-1)
        NUM_SHORTCUT = [0, 0, 5, 5, 10, 10, 15, 15, 20, 20]
        # NUM_SHORTCUT = np.repeat(np.array(NUM_SHORTCUT), 5)
        assert len(NUM_NORMAL) == len(NUM_OVERWHELMED) == len(NUM_SHORTCUT)

        for trail in range(1):
            for i, num_normal in enumerate(NUM_NORMAL):
                num_over = NUM_OVERWHELMED[i]
                num_shortcut = NUM_SHORTCUT[i]

                dataset = VFLDataset(data_source=(X, y), 
                    num_clients=3,
                    gini_portion=None,
                    insert_noise=True, 
                    num_random_samples=num_normal,
                    num_shortcut=num_shortcut,
                    num_overwhelemd=num_over)

                
                train_loader = DataLoader(
                    dataset.train(), batch_size=128, shuffle=True)
                val_loader = DataLoader(
                    dataset.valid(), batch_size=1000, shuffle=True)
                test_loader = DataLoader(dataset.test(), batch_size=1000, shuffle=True)
                input_dim_list = dataset.get_input_dim_list()
                noisy_label = dataset.get_inserted_features_label()
                # print(noisy_label)
                # print(np.count_nonzero(noisy_label[0]))
                # print(np.count_nonzero(noisy_label[1]))
                # print(np.count_nonzero(noisy_label[2]))
        


                ###########################
                # FNN Model
                ############################
                print("FNN Model")
                saving_name = f'FNN_N_{name}_T_{trail}_N_{num_normal}_O_{num_over}_S_{num_shortcut}'
        
                output_dim = np.unique(y).size
                criterion = torch.nn.CrossEntropyLoss()
                
                type = "FNN" 
                models, top_model = VFL.make_binary_models(
                    input_dim_list=input_dim_list, 
                    type=type, 
                    emb_dim=128, 
                    output_dim=output_dim, hidden_dims=[256, 128],
                    activation='relu')
                VFL.train(
                    models, top_model, train_loader, val_loader, test_loader,
                    epochs=EPOCH, 
                    criterion=criterion,
                    verbose=True,
                    save_dir=os.path.join(result_dir, saving_name)+".pt",
                    log_dir=os.path.join(result_dir, saving_name)+".csv",
                    save_mask_at=100000, 
                    noise_label=noisy_label
                )
        
                ###########################
                # STG Model
                ############################
                print("STG")
                type = 'STG'
                saving_name = f'STG_N_{name}_T_{trail}_N_{num_normal}_O_{num_over}_S_{num_shortcut}'
                models, top_model = VFL.make_binary_models(
                    input_dim_list=input_dim_list,
                    type='STG',
                    emb_dim=128,
                    output_dim=output_dim, hidden_dims=[256, 128],
                    activation='relu')
               
                VFL.train(
                    models, top_model, train_loader, val_loader, test_loader,
                    epochs=EPOCH,
                    criterion=criterion,
                    verbose=True,
                    save_dir=os.path.join(result_dir, saving_name)+".pt",
                    log_dir=os.path.join(result_dir, saving_name)+".csv",
                    save_mask_at=100000, 
                    noise_label=noisy_label
                )


                ###########################
                # Dual-STG Model
                ###########################
                print("Dual-STG")
                type = 'Dual-STG'
                saving_name = f'DualSTG_N_{name}_T_{trail}_N_{num_normal}_O_{num_over}_S_{num_shortcut}'
            
                models, top_model = VFL.make_binary_models(
                    input_dim_list=input_dim_list,
                    type='DualSTG',
                    emb_dim=128,
                    output_dim=output_dim, hidden_dims=[256, 128],
                    activation='relu')
                
                VFL.train(
                    models, top_model, train_loader, val_loader, test_loader,
                    epochs=EPOCH,
                    criterion=criterion,
                    verbose=True,
                    save_dir=os.path.join(result_dir, saving_name)+".pt",
                    log_dir=os.path.join(result_dir, saving_name)+".csv",
                    save_mask_at=100000,
                    noise_label=noisy_label)


                ###########################
                # STG Witi GINI Initialization Model
                ###########################
                print("STG with GINI Initialization")
                type = 'STG-GINI'
                saving_name = f'STG-GINI_N_{name}_T_{trail}_N_{num_normal}_O_{num_over}_S_{num_shortcut}'
                gini_labels = dataset.gini_filter(0.5)
                feat_idx_list = dataset.get_feature_index_list()
                mus = VFL.initialize_mu(gini_labels, feat_idx_list)
                models, top_model = VFL.make_binary_models(
                    input_dim_list=input_dim_list,
                    type='STG',
                    emb_dim=128,
                    output_dim=output_dim, hidden_dims=[256, 128],
                    activation='relu', mus=mus)
                VFL.train(models, top_model, train_loader, val_loader, test_loader,
                    epochs=EPOCH,
                    criterion=criterion,
                    verbose=True,
                    save_dir=os.path.join(result_dir, saving_name)+".pt",
                    log_dir=os.path.join(result_dir, saving_name)+".csv",
                    save_mask_at=100000,
                    noise_label=noisy_label)        

                ########################
                # Dual-STG with GINI Initialization Model     
                #########################   
                print('Dual-STG with GINI Initialization Model')
                type = 'Dual-STG-GINI'
                saving_name = f'DualSTG-GINI_N_{name}_T_{trail}_N_{num_normal}_O_{num_over}_S_{num_shortcut}'
                gini_labels = dataset.gini_filter(0.5)
                feat_idx_list = dataset.get_feature_index_list()
                mus = VFL.initialize_mu(gini_labels, feat_idx_list)
                models, top_model = VFL.make_binary_models(
                    input_dim_list=input_dim_list,
                    type='DualSTG',
                    emb_dim=128,
                    output_dim=output_dim, hidden_dims=[256, 128],
                    activation='relu', mus=mus)
                VFL.train(models, top_model, train_loader, val_loader, test_loader,
                    epochs=EPOCH,
                    criterion=criterion,
                    verbose=True,
                    save_dir=os.path.join(result_dir, saving_name)+".pt",
                    log_dir=os.path.join(result_dir, saving_name)+".csv",
                    save_mask_at=100000,
                    noise_label=noisy_label)   
           
              
       



        