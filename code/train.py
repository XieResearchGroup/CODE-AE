import torch
import torch.nn as nn
from ae import AE
from vae import VAE
from mlp import MLP
from encoder_decoder import EncoderDecoder
from collections import defaultdict
from train_dsn import *
from evaluation_utils import *
from dsn_ae import DSNAE
from itertools import chain

def train_DSNAE(s_dataloaders, t_dataloaders, **kwargs):
    s_train_dataloader = s_dataloaders[0]
    s_test_dataloader = s_dataloaders[1]

    t_train_dataloader = t_dataloaders[0]
    t_test_dataloader = t_dataloaders[1]

    shared_encoder = MLP(input_dim=kwargs['input_dim'],
                         output_dim=kwargs['latent_dim'],
                         hidden_dims=kwargs['encoder_hidden_dims'])

    s_dsnae = DSNAE(shared_encoder=shared_encoder,
                    input_dim=kwargs['input_dim'],
                    latent_dim=kwargs['latent_dim'],
                    hidden_dims=kwargs['encoder_hidden_dims'])

    t_dsnae = DSNAE(shared_encoder=shared_encoder,
                    input_dim=kwargs['input_dim'],
                    latent_dim=kwargs['latent_dim'],
                    hidden_dims=kwargs['encoder_hidden_dims'])

    # target_decoder = MLP(input_dim=kwargs['latent_dim'], output_dim=1, hidden_dims=kwargs['classifier_hidden_dims'])
    # target_classifier = EncoderDecoder(encoder=shared_encoder, decoder=target_decoder)

    ae_params = [s_dsnae.parameters(), t_dsnae.parameters()]
    ae_optimizer = torch.optim.AdamW(chain(*ae_params), lr=kwargs['lr'])

    device = kwargs['device']

    dsnae_train_history = defaultdict(list)
    dsnae_val_history = defaultdict(list)

    # dsnae training
    for epoch in range(kwargs['num_epochs']):
        print(f'DSNAE training epoch {epoch}')
        for step, t_batch in enumerate(t_train_dataloader):
            s_batch = next(iter(s_train_dataloader))
            dsnae_train_history = dsn_ae_train_step(s_dsnae=s_dsnae,
                                                    t_dsnae=t_dsnae,
                                                    s_batch=s_batch,
                                                    t_batch=t_batch,
                                                    device=device,
                                                    optimizer=ae_optimizer,
                                                    history=dsnae_train_history)







def train_ADAE(train_dataloader, test_dataloader, **kwargs):
    autoencoder = AE(input_dim=kwargs['input_dim'], latent_dim=kwargs['latent_dim'],
                     hidden_dims=kwargs['encoder_hidden_dims'])
    classifier = MLP(input_dim=kwargs['latent_dim'], output_dim=1, hidden_dims=kwargs['classifier_hidden_dims'])
    confounder_classifier = EncoderDecoder(encoder=autoencoder.encoder, decoder=classifier)
    confounded_loss = nn.BCEWithLogitsLoss()
    ae_optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=kwargs['lr'])
    classifier_optimizer = torch.optim.AdamW(confounder_classifier.decoder.parameters(), lr=kwargs['lr'])
    scheduler = None
    device = kwargs['device']

    ae_pretrain_history = defaultdict(list)
    classifier_pretrain_history = defaultdict(list)
    ae_eval_train_history = defaultdict(list)
    ae_eval_val_history = defaultdict(list)
    classifier_eval_train_history = defaultdict(list)
    classifier_eval_val_history = defaultdict(list)

    # autoencoder pre-training
    for epoch in range(kwargs['pretrain_num_epochs']):
        print(f'----Autoencoder  Pre-Training Epoch {epoch} ----')
        autoencoder = autoencoder.to(device)
        for batch in train_dataloader:
            ae_train_step(ae=autoencoder,
                          batch=batch,
                          device=device,
                          optimizer=ae_optimizer,
                          history=ae_pretrain_history)
        ae_eval_train_history = evaluate_ae_epoch(model=autoencoder,
                                                  data_loader=train_dataloader,
                                                  device=device,
                                                  history=ae_eval_train_history)
        ae_eval_val_history = evaluate_ae_epoch(model=autoencoder,
                                                data_loader=test_dataloader,
                                                device=device,
                                                history=ae_eval_val_history)
        # print some loss/metric messages
        save_flag, stop_flag = model_save_check(history=ae_eval_val_history, metric_name='mse_loss')
        if save_flag:
            torch.save(autoencoder.state_dict(), './model_save/ae.pt')
        if stop_flag:
            break
    autoencoder.load_state_dict(torch.load('./model_save/ae.pt'))

    # confounder classifier pre-training
    classifier_eval_train_history = evaluate_classification_epoch(model=confounder_classifier,
                                                                  data_loader=train_dataloader,
                                                                  device=device,
                                                                  history=classifier_eval_train_history)
    classifier_eval_val_history = evaluate_classification_epoch(model=confounder_classifier,
                                                                data_loader=test_dataloader,
                                                                device=device,
                                                                history=classifier_eval_val_history)
    for epoch in range(kwargs['pretrain_num_epochs']):
        print(f'---- Classifier Pre-Training Epoch {epoch} ----')
        confounder_classifier = confounder_classifier.to(device)
        for batch in train_dataloader:
            classification_train_step(model=confounder_classifier,
                                      batch=batch,
                                      loss_fn=confounded_loss,
                                      device=device,
                                      optimizer=classifier_optimizer,
                                      history=classifier_pretrain_history)
        classifier_eval_train_history = evaluate_classification_epoch(model=confounder_classifier,
                                                                      data_loader=train_dataloader,
                                                                      device=device,
                                                                      history=classifier_eval_train_history)
        classifier_eval_val_history = evaluate_classification_epoch(model=confounder_classifier,
                                                                    data_loader=test_dataloader,
                                                                    device=device,
                                                                    history=classifier_eval_val_history)

        save_flag, stop_flag = model_save_check(history=classifier_eval_val_history, metric_name='acc')
        if save_flag:
            torch.save(confounder_classifier.state_dict(), './model_save/enc_dec.pt')
        if stop_flag:
            break
    confounder_classifier.load_state_dict(torch.load('./model_save/enc_dec.pt'))

    # alternative training
    for epoch in range(kwargs['train_num_epochs']):
        print(f'---- Alternative Training Epoch {epoch} ----')
        autoencoder = autoencoder.to(device)
        confounder_classifier = confounder_classifier.to(device)
        for batch in train_dataloader:
            customized_ae_train_step(ae=autoencoder,
                                     enc_dec_model=confounder_classifier,
                                     batch=batch,
                                     enc_dec_loss_fn=confounded_loss,
                                     device=device,
                                     optimizer=ae_optimizer,
                                     alpha=1.0,
                                     ae_history=ae_pretrain_history,
                                     enc_dec_history=classifier_pretrain_history,
                                     scheduler=None)
        ae_eval_train_history = evaluate_ae_epoch(model=autoencoder,
                                                  data_loader=train_dataloader,
                                                  device=device,
                                                  history=ae_eval_train_history)
        ae_eval_val_history = evaluate_ae_epoch(model=autoencoder,
                                                data_loader=test_dataloader,
                                                device=device,
                                                history=ae_eval_val_history)
        classifier_eval_train_history = evaluate_classification_epoch(model=confounder_classifier,
                                                                      data_loader=train_dataloader,
                                                                      device=device,
                                                                      history=classifier_eval_train_history)
        classifier_eval_val_history = evaluate_classification_epoch(model=confounder_classifier,
                                                                    data_loader=test_dataloader,
                                                                    device=device,
                                                                    history=classifier_eval_val_history)
        for batch in train_dataloader:
            classification_train_step(model=confounder_classifier,
                                      batch=batch,
                                      loss_fn=confounded_loss,
                                      device=device,
                                      optimizer=classifier_optimizer,
                                      history=classifier_pretrain_history)
        ae_eval_train_history = evaluate_ae_epoch(model=autoencoder,
                                                  data_loader=train_dataloader,
                                                  device=device,
                                                  history=ae_eval_train_history)
        ae_eval_val_history = evaluate_ae_epoch(model=autoencoder,
                                                data_loader=test_dataloader,
                                                device=device,
                                                history=ae_eval_val_history)
        classifier_eval_train_history = evaluate_classification_epoch(model=confounder_classifier,
                                                                      data_loader=train_dataloader,
                                                                      device=device,
                                                                      history=classifier_eval_train_history)
        classifier_eval_val_history = evaluate_classification_epoch(model=confounder_classifier,
                                                                    data_loader=test_dataloader,
                                                                    device=device,
                                                                    history=classifier_eval_val_history)


        return
