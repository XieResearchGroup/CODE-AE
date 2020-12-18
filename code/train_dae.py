import torch
import os
from evaluation_utils import eval_ae_epoch, model_save_check
from collections import defaultdict
from ae import AE



def ae_train_step(ae, s_batch, t_batch, device, optimizer, history, scheduler=None):
    ae.zero_grad()
    ae.train()

    s_x = s_batch[0].to(device)
    t_x = t_batch[0].to(device)

    s_loss_dict = ae.loss_function(*ae(s_x))
    t_loss_dict = ae.loss_function(*ae(t_x))

    optimizer.zero_grad()
    loss = s_loss_dict['loss'] + t_loss_dict['loss']
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    loss_dict = {k: v.cpu().detach().item() + t_loss_dict[k].cpu().detach().item() for k, v in s_loss_dict.items()}

    for k, v in loss_dict.items():
        history[k].append(v)

    return history


def train_dae(s_dataloaders, t_dataloaders, **kwargs):
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
                     noise_flag=True,
                     dop=kwargs['dop']).to(kwargs['device'])


    ae_eval_train_history = defaultdict(list)
    ae_eval_val_history = defaultdict(list)

    if kwargs['retrain_flag']:
        ae_optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=kwargs['lr'])
        # start autoencoder pretraining
        for epoch in range(int(kwargs['train_num_epochs'])):
            if epoch % 50 == 0:
                print(f'----Autoencoder Training Epoch {epoch} ----')
            for step, s_batch in enumerate(s_train_dataloader):
                t_batch = next(iter(t_train_dataloader))
                ae_eval_train_history = ae_train_step(ae=autoencoder,
                                                      s_batch=s_batch,
                                                      t_batch=t_batch,
                                                      device=kwargs['device'],
                                                      optimizer=ae_optimizer,
                                                      history=ae_eval_train_history)

            ae_eval_val_history = eval_ae_epoch(model=autoencoder,
                                                data_loader=s_test_dataloader,
                                                device=kwargs['device'],
                                                history=ae_eval_val_history
                                                )
            ae_eval_val_history = eval_ae_epoch(model=autoencoder,
                                                data_loader=t_test_dataloader,
                                                device=kwargs['device'],
                                                history=ae_eval_val_history
                                                )
            for k in ae_eval_val_history:
                if k != 'best_index':
                    ae_eval_val_history[k][-2] += ae_eval_val_history[k][-1]
                    ae_eval_val_history[k].pop()
            # print some loss/metric messages
            save_flag, stop_flag = model_save_check(history=ae_eval_val_history, metric_name='loss', tolerance_count=50)
            if kwargs['es_flag']:
                if save_flag:
                    torch.save(autoencoder.state_dict(), os.path.join(kwargs['model_save_folder'], 'dae.pt'))
                if stop_flag:
                    break

        if kwargs['es_flag']:
            autoencoder.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 'dae.pt')))

        torch.save(autoencoder.state_dict(), os.path.join(kwargs['model_save_folder'], 'dae.pt'))
    else:
        try:
            autoencoder.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 'dae.pt')))
        except FileNotFoundError:
            raise Exception("No pre-trained encoder")

    return autoencoder.encoder, (ae_eval_train_history,
                                 ae_eval_val_history)
