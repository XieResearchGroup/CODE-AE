import torch
import torch.nn as nn
from torch.autograd import Variable
import os
from evaluation_utils import eval_ae_epoch, evaluate_adv_classification_epoch, model_save_check
from collections import defaultdict
from ae import AE
from mlp import MLP
from encoder_decoder import EncoderDecoder


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


def classification_train_step(classifier, s_batch, t_batch, loss_fn, device, optimizer, history, scheduler=None,
                              clip=None):
    classifier.zero_grad()
    classifier.train()

    s_x = s_batch[0].to(device)
    t_x = t_batch[0].to(device)
    outputs = torch.cat((classifier(s_x), classifier(t_x)), dim=0)
    truths = torch.cat((torch.zeros(s_x.shape[0], 1), torch.ones(t_x.shape[0], 1)), dim=0).to(device)
    loss = loss_fn(outputs, truths)

    # valid = torch.ones((s_x.shape[0], 1)).to(device)
    # fake = torch.zeros((t_x.shape[0], 1)).to(device)
    #
    # real_loss = loss_fn((classifier(s_x)), valid)
    # fake_loss = loss_fn(classifier(t_x), fake)
    # loss = 0.5 * (real_loss + fake_loss)

    optimizer.zero_grad()
    loss.backward()
    #     if clip is not None:
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()
    if clip is not None:
        for p in classifier.decoder.parameters():
            p.data.clamp_(-clip, clip)
    if scheduler is not None:
        scheduler.step()

    history['bce'].append(loss.cpu().detach().item())

    return history


def customized_ae_train_step(classifier, ae, s_batch, t_batch, loss_fn, alpha, device, optimizer, history,
                             scheduler=None):
    classifier.zero_grad()
    ae.zero_grad()
    classifier.eval()
    ae.train()

    s_x = s_batch[0].to(device)
    t_x = t_batch[0].to(device)

    outputs = torch.cat((classifier(s_x), classifier(t_x)), dim=0)
    truths = torch.cat((torch.zeros(s_x.shape[0], 1), torch.ones(t_x.shape[0], 1)), dim=0).to(device)
    adv_loss = loss_fn(outputs, truths)

    # valid = torch.ones((s_x.shape[0], 1)).to(device)
    # fake = torch.zeros((t_x.shape[0], 1)).to(device)
    #
    # real_loss = loss_fn((classifier(s_x)), valid)
    # fake_loss = loss_fn(classifier(t_x), fake)
    # adv_loss = 0.5 * (real_loss + fake_loss)

    s_loss_dict = ae.loss_function(*ae(s_x))
    t_loss_dict = ae.loss_function(*ae(t_x))
    loss = s_loss_dict['loss'] + t_loss_dict['loss'] - alpha * adv_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if scheduler is not None:
        scheduler.step()

    loss_dict = {k: v.cpu().detach().item() + t_loss_dict[k].cpu().detach().item() for k, v in s_loss_dict.items()}
    for k, v in loss_dict.items():
        history[k].append(v)
    # history['bce'].append(adv_loss.cpu().detach().item())

    return history


def train_adae(s_dataloaders, t_dataloaders, **kwargs):
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
    classifier = MLP(input_dim=kwargs['latent_dim'],
                     output_dim=1,
                     hidden_dims=kwargs['classifier_hidden_dims'],
                     dop=kwargs['dop']).to(kwargs['device'])
    confounder_classifier = EncoderDecoder(encoder=autoencoder.encoder, decoder=classifier).to(kwargs['device'])

    ae_eval_train_history = defaultdict(list)
    ae_eval_val_history = defaultdict(list)
    classifier_pretrain_history = defaultdict(list)
    classification_eval_test_history = defaultdict(list)
    classification_eval_train_history = defaultdict(list)

    if kwargs['retrain_flag']:
        confounded_loss = nn.BCEWithLogitsLoss()
        ae_optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=kwargs['lr'])
        classifier_optimizer = torch.optim.AdamW(confounder_classifier.decoder.parameters(), lr=kwargs['lr'])

        # start autoencoder pretraining
        for epoch in range(int(kwargs['pretrain_num_epochs'])):
            if epoch % 50 == 0:
                print(f'----Autoencoder  Pre-Training Epoch {epoch} ----')
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
            if kwargs['es_flag']:
                save_flag, stop_flag = model_save_check(history=ae_eval_val_history, metric_name='loss',
                                                        tolerance_count=10)
                if save_flag:
                    torch.save(autoencoder.state_dict(), os.path.join(kwargs['model_save_folder'], 'ae.pt'))
                if stop_flag:
                    break

        if kwargs['es_flag']:
            autoencoder.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 'ae.pt')))

        # start adversarial classifier pre-training
        for epoch in range(int(kwargs['pretrain_num_epochs'])):
            if epoch % 50 == 0:
                print(f'Adversarial classifier pre-training epoch {epoch}')
            for step, s_batch in enumerate(s_train_dataloader):
                t_batch = next(iter(t_train_dataloader))
                classifier_pretrain_history = classification_train_step(classifier=confounder_classifier,
                                                                        s_batch=s_batch,
                                                                        t_batch=t_batch,
                                                                        loss_fn=confounded_loss,
                                                                        device=kwargs['device'],
                                                                        optimizer=classifier_optimizer,
                                                                        history=classifier_pretrain_history)

            classification_eval_test_history = evaluate_adv_classification_epoch(classifier=confounder_classifier,
                                                                                 s_dataloader=s_test_dataloader,
                                                                                 t_dataloader=t_test_dataloader,
                                                                                 device=kwargs['device'],
                                                                                 history=classification_eval_test_history)
            classification_eval_train_history = evaluate_adv_classification_epoch(classifier=confounder_classifier,
                                                                                  s_dataloader=s_train_dataloader,
                                                                                  t_dataloader=t_train_dataloader,
                                                                                  device=kwargs['device'],
                                                                                  history=classification_eval_train_history)

            save_flag, stop_flag = model_save_check(history=classification_eval_test_history, metric_name='acc',
                                                    tolerance_count=50)
            if kwargs['es_flag']:
                if save_flag:
                    torch.save(confounder_classifier.state_dict(),
                               os.path.join(kwargs['model_save_folder'], 'adv_classifier.pt'))
                if stop_flag:
                    break

        if kwargs['es_flag']:
            confounder_classifier.load_state_dict(
                torch.load(os.path.join(kwargs['model_save_folder'], 'adv_classifier.pt')))

        # start alternative training
        for epoch in range(int(kwargs['train_num_epochs'])):
            if epoch % 50 == 0:
                print(f'Alternative training epoch {epoch}')
            # start autoencoder training epoch
            for step, s_batch in enumerate(s_train_dataloader):
                t_batch = next(iter(t_train_dataloader))
                ae_eval_train_history = customized_ae_train_step(classifier=confounder_classifier,
                                                                 ae=autoencoder,
                                                                 s_batch=s_batch,
                                                                 t_batch=t_batch,
                                                                 loss_fn=confounded_loss,
                                                                 alpha=kwargs['alpha'],
                                                                 device=kwargs['device'],
                                                                 optimizer=ae_optimizer,
                                                                 history=ae_eval_train_history,
                                                                 scheduler=None)

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

            classification_eval_test_history = evaluate_adv_classification_epoch(classifier=confounder_classifier,
                                                                                 s_dataloader=s_test_dataloader,
                                                                                 t_dataloader=t_test_dataloader,
                                                                                 device=kwargs['device'],
                                                                                 history=classification_eval_test_history)
            classification_eval_train_history = evaluate_adv_classification_epoch(classifier=confounder_classifier,
                                                                                  s_dataloader=s_train_dataloader,
                                                                                  t_dataloader=t_train_dataloader,
                                                                                  device=kwargs['device'],
                                                                                  history=classification_eval_train_history)

            for step, s_batch in enumerate(s_train_dataloader):
                t_batch = next(iter(t_train_dataloader))
                classifier_pretrain_history = classification_train_step(classifier=confounder_classifier,
                                                                        s_batch=s_batch,
                                                                        t_batch=t_batch,
                                                                        loss_fn=confounded_loss,
                                                                        device=kwargs['device'],
                                                                        optimizer=classifier_optimizer,
                                                                        history=classifier_pretrain_history)

            classification_eval_test_history = evaluate_adv_classification_epoch(classifier=confounder_classifier,
                                                                                 s_dataloader=s_test_dataloader,
                                                                                 t_dataloader=t_test_dataloader,
                                                                                 device=kwargs['device'],
                                                                                 history=classification_eval_test_history)
            classification_eval_train_history = evaluate_adv_classification_epoch(classifier=confounder_classifier,
                                                                                  s_dataloader=s_train_dataloader,
                                                                                  t_dataloader=t_test_dataloader,
                                                                                  device=kwargs['device'],
                                                                                  history=classification_eval_train_history)

        torch.save(autoencoder.state_dict(), os.path.join(kwargs['model_save_folder'], 'ae.pt'))

    else:
        try:
            autoencoder.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 'ae.pt')))
        except FileNotFoundError:
            raise Exception("No pre-trained encoder")

    return autoencoder.encoder, (ae_eval_train_history,
                                 ae_eval_val_history,
                                 classification_eval_train_history,
                                 classification_eval_test_history,
                                 classifier_pretrain_history)
