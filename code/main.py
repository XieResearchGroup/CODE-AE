import numpy as np
import pandas as pd
import torch
import data_config

if __name__=='__main__':
    gex_features_df = pd.read_csv(data_config.gex_feature_file, index_col=0)
    unlabeled_training_params = {
        'batch_size': 64,
        'lr': 1e-4,
        'pretrain_num_epochs': 50,
        'train_num_epochs': 50,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'input_dim': gex_features_df.shape[-1],
        'latent_dim': 128,
        'encoder_hidden_dims': [512, 256],
        'classifier_hidden_dims': [64, 32, 16],
        'model_save_folder': './model_save'
    }
    labeled_training_params = {
        'batch_size': 64,
        'lr': 1e-4,
        'pretrain_num_epochs': 50,
        'train_num_epochs': 50,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'input_dim': gex_features_df.shape[-1],
        'latent_dim': 128,
        'encoder_hidden_dims': [512, 256],
        'classifier_hidden_dims': [64, 32, 16],
        'decay_coefficient': 0.1,
        'model_save_folder': './model_save'
    }


















