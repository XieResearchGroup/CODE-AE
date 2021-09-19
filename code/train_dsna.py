import os
import torch.nn as nn
import torch.autograd as autograd
from itertools import chain
from dsn_ae import DSNAE
from evaluation_utils import *
from mlp import MLP
from encoder_decoder import EncoderDecoder
from train_code_base import eval_dsnae_epoch

def dsn_dann_train_step(classifier, s_dsnae, t_dsnae, s_batch, t_batch, loss_fn, device, optimizer, alpha, history,
                        scheduler=None):
    classifier.zero_grad()
    s_dsnae.zero_grad()
    t_dsnae.zero_grad()
    classifier.train()
    s_dsnae.train()
    t_dsnae.train()

    s_x = s_batch[0].to(device)
    t_x = t_batch[0].to(device)

    optimizer.zero_grad()
    s_loss_dict = s_dsnae.loss_function(*s_dsnae(s_x))
    t_loss_dict = t_dsnae.loss_function(*t_dsnae(t_x))
    recons_loss = s_loss_dict['loss'] + t_loss_dict['loss']

    outputs = torch.cat((classifier(s_x), classifier(t_x)), dim=0)
    truths = torch.cat((torch.zeros(s_x.shape[0], 1), torch.ones(t_x.shape[0], 1)), dim=0).to(device)
    dann_loss = loss_fn(outputs, truths)

    loss = recons_loss + alpha * dann_loss
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    loss_dict = {k: v.cpu().detach().item() + t_loss_dict[k].cpu().detach().item() for k, v in s_loss_dict.items()}

    for k, v in loss_dict.items():
        history[k].append(v)
    history['gen_loss'].append(dann_loss.cpu().detach().item())

    return history


def train_dsna(s_dataloaders, t_dataloaders, **kwargs):
    """

    :param s_dataloaders:
    :param t_dataloaders:
    :param kwargs:
    :return:
    """
    s_train_dataloader = s_dataloaders[0]
    s_test_dataloader = s_dataloaders[1]

    t_train_dataloader = t_dataloaders[0]
    t_test_dataloader = t_dataloaders[1]

    shared_encoder = MLP(input_dim=kwargs['input_dim'],
                         output_dim=kwargs['latent_dim'],
                         hidden_dims=kwargs['encoder_hidden_dims'],
                         dop=kwargs['dop']).to(kwargs['device'])

    shared_decoder = MLP(input_dim=2 * kwargs['latent_dim'],
                         output_dim=kwargs['input_dim'],
                         hidden_dims=kwargs['encoder_hidden_dims'][::-1],
                         dop=kwargs['dop']).to(kwargs['device'])

    s_dsnae = DSNAE(shared_encoder=shared_encoder,
                    decoder=shared_decoder,
                    alpha=kwargs['alpha'],
                    input_dim=kwargs['input_dim'],
                    latent_dim=kwargs['latent_dim'],
                    hidden_dims=kwargs['encoder_hidden_dims'],
                    dop=kwargs['dop']).to(kwargs['device'])

    t_dsnae = DSNAE(shared_encoder=shared_encoder,
                    decoder=shared_decoder,
                    alpha=kwargs['alpha'],
                    input_dim=kwargs['input_dim'],
                    latent_dim=kwargs['latent_dim'],
                    hidden_dims=kwargs['encoder_hidden_dims'],
                    dop=kwargs['dop']).to(kwargs['device'])



    classifier = MLP(input_dim=kwargs['latent_dim'],
                                 output_dim=1,
                                 hidden_dims=kwargs['classifier_hidden_dims'],
                                 dop=kwargs['dop'],
                                 gr_flag=True).to(kwargs['device'])
    confounder_classifier = EncoderDecoder(encoder=shared_encoder, decoder=classifier).to(kwargs['device'])


    dsnae_train_history = defaultdict(list)
    dsnae_val_history = defaultdict(list)
    device = kwargs['device']

    if kwargs['retrain_flag']:
        confounded_loss = nn.BCEWithLogitsLoss()
        params = [t_dsnae.private_encoder.parameters(),
                  s_dsnae.private_encoder.parameters(),
                  shared_decoder.parameters(),
                  shared_encoder.parameters(),
                  classifier.parameters()
                  ]
        optimizer = torch.optim.AdamW(chain(*params), lr=kwargs['lr'])
        for epoch in range(int(kwargs['train_num_epochs'])):
            if epoch % 50 == 0:
                print(f'AE training epoch {epoch}')
            for step, s_batch in enumerate(s_train_dataloader):
                t_batch = next(iter(t_train_dataloader))
                dsnae_train_history = dsn_dann_train_step(classifier=confounder_classifier,
                                                          s_dsnae=s_dsnae,
                                                          t_dsnae=t_dsnae,
                                                          s_batch=s_batch,
                                                          t_batch=t_batch,
                                                          loss_fn=confounded_loss,
                                                          device=device,
                                                          alpha=1.0,
                                                          optimizer=optimizer,
                                                          history=dsnae_train_history)
            dsnae_val_history = eval_dsnae_epoch(model=s_dsnae,
                                                 data_loader=s_test_dataloader,
                                                 device=device,
                                                 history=dsnae_val_history
                                                 )
            dsnae_val_history = eval_dsnae_epoch(model=t_dsnae,
                                                 data_loader=t_test_dataloader,
                                                 device=device,
                                                 history=dsnae_val_history
                                                 )
            for k in dsnae_val_history:
                if k != 'best_index':
                    dsnae_val_history[k][-2] += dsnae_val_history[k][-1]
                    dsnae_val_history[k].pop()

            save_flag, stop_flag = model_save_check(dsnae_val_history, metric_name='loss', tolerance_count=50)
            if kwargs['es_flag']:
                if save_flag:
                    torch.save(s_dsnae.state_dict(), os.path.join(kwargs['model_save_folder'], 'm_s_dsnae.pt'))
                    torch.save(t_dsnae.state_dict(), os.path.join(kwargs['model_save_folder'], 'm_t_dsnae.pt'))
                if stop_flag:
                    break

        if kwargs['es_flag']:
            s_dsnae.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 'dann_s_dsnae.pt')))
            t_dsnae.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 'dann_t_dsnae.pt')))

        torch.save(s_dsnae.state_dict(), os.path.join(kwargs['model_save_folder'], 'dann_s_dsnae.pt'))
        torch.save(t_dsnae.state_dict(), os.path.join(kwargs['model_save_folder'], 'dann_t_dsnae.pt'))

    else:
        try:
            t_dsnae.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 'dann_t_dsnae.pt')))
        except FileNotFoundError:
            raise Exception("No pre-trained encoder")

    return t_dsnae.shared_encoder, (dsnae_train_history, dsnae_val_history)
