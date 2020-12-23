import os
import torch
from loss_and_metrics import cov
from collections import defaultdict
from ae import AE
from evaluation_utils import model_save_check


def eval_ae_epoch(ae, s_dataloader, t_dataloader, device, history):
    ae.eval()
    avg_loss_dict = defaultdict(float)
    s_codes = None
    t_codes = None

    for x_batch in s_dataloader:
        x_batch = x_batch[0].to(device)
        with torch.no_grad():
            s_code = ae.encode(x_batch)
            # s_codes = s_code.cpu().detach().numpy() if s_codes is None else np.concatenate([s_codes, s_code.cpu().detach().numpy()], axis=0)
            s_codes = s_code if s_codes is None else torch.cat((s_codes, s_code))
            loss_dict = ae.loss_function(*(ae(x_batch)))
            for k, v in loss_dict.items():
                avg_loss_dict[k] += v.cpu().detach().item() / len(s_dataloader)

    for x_batch in t_dataloader:
        x_batch = x_batch[0].to(device)
        with torch.no_grad():
            t_code = ae.encode(x_batch)
            # t_codes = t_code.cpu().detach().numpy() if t_codes is None else np.concatenate([t_codes, t_code.cpu().detach().numpy()], axis=0)
            t_codes = t_code if t_codes is None else torch.cat((t_codes, t_code))

            loss_dict = ae.loss_function(*(ae(x_batch)))
            for k, v in loss_dict.items():
                avg_loss_dict[k] += v.cpu().detach().item() / len(t_dataloader)

    for k, v in avg_loss_dict.items():
        history[k].append(v)

    history['coral_loss'].append(((torch.square(torch.norm(cov(s_codes) - cov(t_codes), p='fro'))) / (
            4 * (s_codes.size()[-1] ** 2))).cpu().detach().numpy().ravel())
    # history['coral_loss'].append((torch.square(torch.norm(cov(s_codes)-cov(t_codes), p='fro'))).cpu().detach().numpy().ravel())
    history['loss'][-1] += history['coral_loss'][-1]

    return history


def coral_ae_train_step(ae, s_batch, t_batch, device, optimizer, alpha, history, scheduler=None):
    ae.zero_grad()
    ae.train()

    s_x = s_batch[0].to(device)
    t_x = t_batch[0].to(device)

    s_code = ae.encode(s_x)
    t_code = ae.encode(t_x)
    s_cov = cov(s_code)
    t_cov = cov(t_code)

    coral_loss = (torch.square(torch.norm(s_cov - t_cov, p='fro'))) / (4 * (s_code.size()[-1] ** 2))
    # coral_loss = torch.square(torch.norm(s_cov-t_cov, p='fro'))

    s_loss_dict = ae.loss_function(*ae(s_x))
    t_loss_dict = ae.loss_function(*ae(t_x))

    optimizer.zero_grad()
    loss = s_loss_dict['loss'] + t_loss_dict['loss'] + alpha * coral_loss
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    loss_dict = {k: v.cpu().detach().item() + t_loss_dict[k].cpu().detach().item() for k, v in s_loss_dict.items()}

    for k, v in loss_dict.items():
        history[k].append(v)

    history['coral_loss'].append(coral_loss.cpu().detach().item())

    return history


def train_coral(s_dataloaders, t_dataloaders, **kwargs):
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

    autoencoder = AE(input_dim=kwargs['input_dim'],
                     latent_dim=kwargs['latent_dim'],
                     hidden_dims=kwargs['encoder_hidden_dims'],
                     dop=kwargs['dop']).to(kwargs['device'])


    ae_train_history = defaultdict(list)
    # ae_eval_train_history = defaultdict(list)
    ae_eval_val_history = defaultdict(list)


    if kwargs['retrain_flag']:
        ae_optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=kwargs['lr'])
        for epoch in range(int(kwargs['train_num_epochs'])):
            if epoch % 50 == 0:
                print(f'AE training epoch {epoch}')
            for step, s_batch in enumerate(s_train_dataloader):
                t_batch = next(iter(t_train_dataloader))
                ae_train_history = coral_ae_train_step(ae=autoencoder,
                                                       s_batch=s_batch,
                                                       t_batch=t_batch,
                                                       device=kwargs['device'],
                                                       optimizer=ae_optimizer,
                                                       alpha=kwargs['alpha'],
                                                       history=ae_train_history)
            ae_eval_val_history = eval_ae_epoch(ae=autoencoder,
                                                s_dataloader=s_test_dataloader,
                                                t_dataloader=t_test_dataloader,
                                                device=kwargs['device'],
                                                history=ae_eval_val_history)
            save_flag, stop_flag = model_save_check(ae_eval_val_history, metric_name='loss', tolerance_count=50)
            if kwargs['es_flag']:
                if save_flag:
                    torch.save(autoencoder.state_dict(), os.path.join(kwargs['model_save_folder'], 'coral_ae.pt'))
                if stop_flag:
                    break

        if kwargs['es_flag']:
            autoencoder.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 'coral_ae.pt')))

        torch.save(autoencoder.state_dict(), os.path.join(kwargs['model_save_folder'], 'coral_ae.pt'))

    else:
        try:
            autoencoder.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 'coral_ae.pt')))
        except FileNotFoundError:
            raise Exception("No pre-trained encoder")

    return autoencoder.encoder, (ae_train_history, ae_eval_val_history)
