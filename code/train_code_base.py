import os
from itertools import chain

from dsn_ae import DSNAE
from evaluation_utils import *
from mlp import MLP


def eval_dsnae_epoch(model, data_loader, device, history):
    """

    :param model:
    :param data_loader:
    :param device:
    :param history:
    :return:
    """
    model.eval()
    avg_loss_dict = defaultdict(float)
    for x_batch in data_loader:
        x_batch = x_batch[0].to(device)
        with torch.no_grad():
            loss_dict = model.loss_function(*(model(x_batch)))
            for k, v in loss_dict.items():
                avg_loss_dict[k] += v.cpu().detach().item() / len(data_loader)

    for k, v in avg_loss_dict.items():
        history[k].append(v)
    return history


def dsn_ae_train_step(s_dsnae, t_dsnae, s_batch, t_batch, device, optimizer, history, scheduler=None):
    s_dsnae.zero_grad()
    t_dsnae.zero_grad()
    s_dsnae.train()
    t_dsnae.train()

    s_x = s_batch[0].to(device)
    t_x = t_batch[0].to(device)

    s_loss_dict = s_dsnae.loss_function(*s_dsnae(s_x))
    t_loss_dict = t_dsnae.loss_function(*t_dsnae(t_x))

    optimizer.zero_grad()
    loss = s_loss_dict['loss'] + t_loss_dict['loss']
    loss.backward()

    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    loss_dict = {k: v.cpu().detach().item() + t_loss_dict[k].cpu().detach().item() for k, v in s_loss_dict.items()}

    for k, v in loss_dict.items():
        history[k].append(v)

    return history


def train_code_base(s_dataloaders, t_dataloaders, **kwargs):
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
                    dop=kwargs['dop'],
                    norm_flag=kwargs['norm_flag']).to(kwargs['device'])

    t_dsnae = DSNAE(shared_encoder=shared_encoder,
                    decoder=shared_decoder,
                    alpha=kwargs['alpha'],
                    input_dim=kwargs['input_dim'],
                    latent_dim=kwargs['latent_dim'],
                    hidden_dims=kwargs['encoder_hidden_dims'],
                    dop=kwargs['dop'],
                    norm_flag=kwargs['norm_flag']).to(kwargs['device'])

    device = kwargs['device']

    dsnae_train_history = defaultdict(list)
    dsnae_val_history = defaultdict(list)

    if kwargs['retrain_flag']:
        ae_params = [t_dsnae.private_encoder.parameters(),
                     s_dsnae.private_encoder.parameters(),
                     shared_decoder.parameters(),
                     shared_encoder.parameters()
                     ]

        ae_optimizer = torch.optim.AdamW(chain(*ae_params), lr=kwargs['lr'])
        for epoch in range(int(kwargs['train_num_epochs'])):
            if epoch % 50 == 0:
                print(f'AE training epoch {epoch}')
            for step, s_batch in enumerate(s_train_dataloader):
                t_batch = next(iter(t_train_dataloader))
                dsnae_train_history = dsn_ae_train_step(s_dsnae=s_dsnae,
                                                        t_dsnae=t_dsnae,
                                                        s_batch=s_batch,
                                                        t_batch=t_batch,
                                                        device=device,
                                                        optimizer=ae_optimizer,
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
                    torch.save(s_dsnae.state_dict(), os.path.join(kwargs['model_save_folder'], 's_dsnae.pt'))
                    torch.save(t_dsnae.state_dict(), os.path.join(kwargs['model_save_folder'], 't_dsnae.pt'))
                if stop_flag:
                    break

        if kwargs['es_flag']:
            s_dsnae.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 's_dsnae.pt')))
            t_dsnae.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 't_dsnae.pt')))

        torch.save(s_dsnae.state_dict(), os.path.join(kwargs['model_save_folder'], 's_dsnae.pt'))
        torch.save(t_dsnae.state_dict(), os.path.join(kwargs['model_save_folder'], 't_dsnae.pt'))

    else:
        try:
            loaded_model = torch.load(os.path.join(kwargs['model_save_folder'], 't_dsnae.pt'))
            print({key:val.shape for key,val in loaded_model.items()})
            print({key:val.shape for key,val in t_dsnae.state_dict().items()})
            t_dsnae.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 't_dsnae.pt')))
        except FileNotFoundError:
            raise Exception("No pre-trained encoder")


    return t_dsnae.shared_encoder, (dsnae_train_history, dsnae_val_history)
