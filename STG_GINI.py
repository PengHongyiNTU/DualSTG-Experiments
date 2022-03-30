from Data import VFLDataset
import VFL
from torch.utils.data import DataLoader
dataset = VFLDataset('BASEHOCK', scale=True, num_clients=3,
    gini_portion=0.8, insert_noise=True, num_random_samples=10, 
    num_overwhelemd=5, num_shortcut=5)
gini_labels = dataset.gini_filter()
feat_idx_list = dataset.get_feature_index_list()
print(gini_labels)
mus = VFL.initialize_mu(gini_labels, feat_idx_list)
train_loader = DataLoader(dataset.train(), batch_size=128, shuffle=True)
val_loader = DataLoader(dataset.valid(), batch_size=1000, shuffle=True)
test_loader = DataLoader(dataset.test(), batch_size=1000, shuffle=True)
input_dim_list = dataset.get_input_dim_list()
models, top_model = VFL.make_binary_models(
    input_dim_list, type='STG', emb_dim=128, output_dim=1, hidden_dims=[512, 256],
    sigma=1.0, lam=0.1, top_sigma=1.0, top_lam=0.1, mus=mus)
noisy_label = dataset.get_inserted_features_label()
VFL.train(models, top_model, train_loader, val_loader, test_loader, 
    epochs=100, noise_label=noisy_label)