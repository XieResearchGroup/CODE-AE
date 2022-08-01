import os
import torch.autograd as autograd
from itertools import chain
from dsn_ae import DSNAE
from evaluation_utils import *
from mlp import MLP
from train_code_base import eval_dsnae_epoch, dsn_ae_train_step
from collections import OrderedDict

def compute_gradient_penalty(critic, real_samples, fake_samples, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand((real_samples.shape[0], 1)).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    critic_interpolates = critic(interpolates)
    fakes = torch.ones((real_samples.shape[0], 1)).to(device)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=fakes,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def critic_dsn_train_step(critic, s_dsnae, t_dsnae, s_batch, t_batch, device, optimizer, history, scheduler=None,
                          clip=None, gp=None):
    critic.zero_grad()
    s_dsnae.zero_grad()
    t_dsnae.zero_grad()
    s_dsnae.eval()
    t_dsnae.eval()
    critic.train()

    s_x = s_batch[0].to(device)
    t_x = t_batch[0].to(device)

    s_code = s_dsnae.encode(s_x)
    t_code = t_dsnae.encode(t_x)

    loss = torch.mean(critic(t_code)) - torch.mean(critic(s_code))

    if gp is not None:
        gradient_penalty = compute_gradient_penalty(critic,
                                                    real_samples=s_code,
                                                    fake_samples=t_code,
                                                    device=device)
        loss = loss + gp * gradient_penalty

    optimizer.zero_grad()
    loss.backward()
    #     if clip is not None:
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()

    if clip is not None:
        for p in critic.parameters():
            p.data.clamp_(-clip, clip)
    if scheduler is not None:
        scheduler.step()

    history['critic_loss'].append(loss.cpu().detach().item())

    return history


def gan_dsn_gen_train_step(critic, s_dsnae, t_dsnae, s_batch, t_batch, device, optimizer, alpha, history,
                           scheduler=None):
    critic.zero_grad()
    s_dsnae.zero_grad()
    t_dsnae.zero_grad()
    critic.eval()
    s_dsnae.train()
    t_dsnae.train()

    s_x = s_batch[0].to(device)
    t_x = t_batch[0].to(device)

    t_code = t_dsnae.encode(t_x)

    optimizer.zero_grad()
    gen_loss = -torch.mean(critic(t_code))
    s_loss_dict = s_dsnae.loss_function(*s_dsnae(s_x))
    t_loss_dict = t_dsnae.loss_function(*t_dsnae(t_x))
    recons_loss = s_loss_dict['loss'] + t_loss_dict['loss']
    loss = recons_loss + alpha * gen_loss
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    loss_dict = {k: v.cpu().detach().item() + t_loss_dict[k].cpu().detach().item() for k, v in s_loss_dict.items()}

    for k, v in loss_dict.items():
        history[k].append(v)
    history['gen_loss'].append(gen_loss.cpu().detach().item())

    return history


def train_code_adv(s_dataloaders, t_dataloaders, **kwargs):
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

    confounding_classifier = MLP(input_dim=kwargs['latent_dim'] * 2,
                                 output_dim=1,
                                 hidden_dims=kwargs['classifier_hidden_dims'],
                                 dop=kwargs['dop']).to(kwargs['device'])

    ae_params = [t_dsnae.private_encoder.parameters(),
                 s_dsnae.private_encoder.parameters(),
                 shared_decoder.parameters(),
                 shared_encoder.parameters()
                 ]
    t_ae_params = [t_dsnae.private_encoder.parameters(),
                   s_dsnae.private_encoder.parameters(),
                   shared_decoder.parameters(),
                   shared_encoder.parameters()
                   ]

    ae_optimizer = torch.optim.AdamW(chain(*ae_params), lr=kwargs['lr'])
    classifier_optimizer = torch.optim.RMSprop(confounding_classifier.parameters(), lr=kwargs['lr'])
    t_ae_optimizer = torch.optim.RMSprop(chain(*t_ae_params), lr=kwargs['lr'])

    dsnae_train_history = defaultdict(list)
    dsnae_val_history = defaultdict(list)
    critic_train_history = defaultdict(list)
    gen_train_history = defaultdict(list)
    # classification_eval_test_history = defaultdict(list)
    # classification_eval_train_history = defaultdict(list)

    if kwargs['retrain_flag']:

        # start dsnae pre-training
        for epoch in range(int(kwargs['pretrain_num_epochs'])):
            if epoch % 50 == 0:
                print(f'AE training epoch {epoch}')
            for step, s_batch in enumerate(s_train_dataloader):
                t_batch = next(iter(t_train_dataloader))
                dsnae_train_history = dsn_ae_train_step(s_dsnae=s_dsnae,
                                                        t_dsnae=t_dsnae,
                                                        s_batch=s_batch,
                                                        t_batch=t_batch,
                                                        device=kwargs['device'],
                                                        optimizer=ae_optimizer,
                                                        history=dsnae_train_history)
            dsnae_val_history = eval_dsnae_epoch(model=s_dsnae,
                                                 data_loader=s_test_dataloader,
                                                 device=kwargs['device'],
                                                 history=dsnae_val_history
                                                 )
            dsnae_val_history = eval_dsnae_epoch(model=t_dsnae,
                                                 data_loader=t_test_dataloader,
                                                 device=kwargs['device'],
                                                 history=dsnae_val_history
                                                 )
            for k in dsnae_val_history:
                if k != 'best_index':
                    dsnae_val_history[k][-2] += dsnae_val_history[k][-1]
                    dsnae_val_history[k].pop()
            if kwargs['es_flag']:
                save_flag, stop_flag = model_save_check(dsnae_val_history, metric_name='loss', tolerance_count=20)
                if save_flag:
                    torch.save(s_dsnae.state_dict(), os.path.join(kwargs['model_save_folder'], 'a_s_dsnae.pt'))
                    torch.save(t_dsnae.state_dict(), os.path.join(kwargs['model_save_folder'], 'a_t_dsnae.pt'))
                if stop_flag:
                    break
        if kwargs['es_flag']:
            s_dsnae.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 'a_s_dsnae.pt')))
            t_dsnae.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 'a_t_dsnae.pt')))

        # start critic pre-training
        # for epoch in range(100):
        #     if epoch % 10 == 0:
        #         print(f'confounder critic pre-training epoch {epoch}')
        #     for step, t_batch in enumerate(s_train_dataloader):
        #         s_batch = next(iter(t_train_dataloader))
        #         critic_train_history = critic_dsn_train_step(critic=confounding_classifier,
        #                                                      s_dsnae=s_dsnae,
        #                                                      t_dsnae=t_dsnae,
        #                                                      s_batch=s_batch,
        #                                                      t_batch=t_batch,
        #                                                      device=kwargs['device'],
        #                                                      optimizer=classifier_optimizer,
        #                                                      history=critic_train_history,
        #                                                      clip=None,
        #                                                      gp=None)
        # start GAN training
        for epoch in range(int(kwargs['train_num_epochs'])):
            if epoch % 50 == 0:
                print(f'confounder wgan training epoch {epoch}')
            for step, s_batch in enumerate(s_train_dataloader):
                t_batch = next(iter(t_train_dataloader))
                critic_train_history = critic_dsn_train_step(critic=confounding_classifier,
                                                             s_dsnae=s_dsnae,
                                                             t_dsnae=t_dsnae,
                                                             s_batch=s_batch,
                                                             t_batch=t_batch,
                                                             device=kwargs['device'],
                                                             optimizer=classifier_optimizer,
                                                             history=critic_train_history,
                                                             # clip=0.1,
                                                             gp=10.0)
                if (step + 1) % 5 == 0:
                    gen_train_history = gan_dsn_gen_train_step(critic=confounding_classifier,
                                                               s_dsnae=s_dsnae,
                                                               t_dsnae=t_dsnae,
                                                               s_batch=s_batch,
                                                               t_batch=t_batch,
                                                               device=kwargs['device'],
                                                               optimizer=t_ae_optimizer,
                                                               alpha=1.0,
                                                               history=gen_train_history)

        torch.save(s_dsnae.state_dict(), os.path.join(kwargs['model_save_folder'], 'a_s_dsnae.pt'))
        torch.save(t_dsnae.state_dict(), os.path.join(kwargs['model_save_folder'], 'a_t_dsnae.pt'))

    else:
        try:
            # if kwargs['norm_flag']:
            #     loaded_model = torch.load(os.path.join(kwargs['model_save_folder'], 'a_t_dsnae.pt'))
            #     new_loaded_model = {key: val for key, val in loaded_model.items() if key in t_dsnae.state_dict()}
            #     new_loaded_model['shared_encoder.output_layer.0.weight'] = loaded_model[
            #         'shared_encoder.output_layer.3.weight']
            #     new_loaded_model['shared_encoder.output_layer.0.bias'] = loaded_model[
            #         'shared_encoder.output_layer.3.bias']
            #     new_loaded_model['decoder.output_layer.0.weight'] = loaded_model['decoder.output_layer.3.weight']
            #     new_loaded_model['decoder.output_layer.0.bias'] = loaded_model['decoder.output_layer.3.bias']

            #     corrected_model = OrderedDict({key: new_loaded_model[key] for key in t_dsnae.state_dict()})
            #     t_dsnae.load_state_dict(corrected_model)
            # else:
                #t_dsnae.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 'a_t_dsnae.pt')))
                t_dsnae.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 'a_t_dsnae.pt')))

        except FileNotFoundError:
            raise Exception("No pre-trained encoder")

    return t_dsnae.shared_encoder, (dsnae_train_history, dsnae_val_history, critic_train_history, gen_train_history)
