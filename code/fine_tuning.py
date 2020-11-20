from evaluation_utils import evaluate_target_classification_epoch, model_save_check
from collections import defaultdict
from itertools import chain
from mlp import MLP
from encoder_decoder import EncoderDecoder
import os
import torch
import torch.nn as nn


def classification_train_step(model, batch, loss_fn, device, optimizer, history, scheduler=None, clip=None):
    model.zero_grad()
    model.train()

    x = batch[0].to(device)
    y = batch[1].to(device)
    loss = loss_fn(model(x), y.double().unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    #     if clip is not None:
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

    optimizer.step()
    if clip is not None:
        for p in model.decoder.parameters():
            p.data.clamp_(-clip, clip)
    if scheduler is not None:
        scheduler.step()

    history['bce'].append(loss.cpu().detach().item())

    return history


def fine_tune_encoder(encoder, train_dataloader, val_dataloader, test_dataloader=None, **kwargs):
    target_decoder = MLP(input_dim=kwargs['latent_dim'],
                         output_dim=1,
                         hidden_dims=kwargs['classifier_hidden_dims'])
    target_classifier = EncoderDecoder(encoder=encoder, decoder=target_decoder)
    classification_loss = nn.BCEWithLogitsLoss()

    target_classification_train_history = defaultdict(list)
    target_classification_eval_train_history = defaultdict(list)
    target_classification_eval_val_history = defaultdict(list)
    target_classification_eval_test_history = defaultdict(list)

    encoder_module_indices = [i for i in range(len(list(encoder.modules())))
                              if str(list(encoder.modules())[i]).startswith('Linear')]

    reset_count = 1
    lr = kwargs['lr']

    target_classification_params = [target_classifier.decoder.parameters()]
    target_classification_optimizer = torch.optim.AdamW(chain(*target_classification_params),
                                                        lr=lr)

    for epoch in range(kwargs['train_num_epochs']):
        if epoch % 50 == 0:
            print(f'Fine tuning epoch {epoch}')
        for step, batch in enumerate(train_dataloader):
            target_classification_train_history = classification_train_step(model=target_classifier,
                                                                            batch=batch,
                                                                            loss_fn=classification_loss,
                                                                            device=kwargs['device'],
                                                                            optimizer=target_classification_optimizer,
                                                                            history=target_classification_train_history)
        target_classification_eval_train_history = evaluate_target_classification_epoch(classifier=target_classifier,
                                                                                        dataloader=train_dataloader,
                                                                                        device=kwargs['device'],
                                                                                        history=target_classification_eval_train_history)
        target_classification_eval_val_history = evaluate_target_classification_epoch(classifier=target_classifier,
                                                                                      dataloader=val_dataloader,
                                                                                      device=kwargs['device'],
                                                                                      history=target_classification_eval_val_history)

        if test_dataloader is not None:
            target_classification_eval_test_history = evaluate_target_classification_epoch(classifier=target_classifier,
                                                                                           dataloader=test_dataloader,
                                                                                           device=kwargs['device'],
                                                                                           history=target_classification_eval_test_history)
        save_flag, stop_flag = model_save_check(history=target_classification_eval_val_history,
                                                metric_name='auroc',
                                                tolerance_count=10,
                                                reset_count=reset_count)
        if save_flag:
            torch.save(target_classifier.state_dict(),
                       os.path.join(kwargs['model_save_folder'], 'target_classifier.pt'))
        if stop_flag:
            try:
                ind = encoder_module_indices.pop()
                print(f'Unfreezing {epoch}')
                target_classification_params.append(list(target_classifier.encoder.modules())[ind].parameters())
                lr = lr * kwargs['decay_coefficient']
                target_classification_optimizer = torch.optim.AdamW(chain(*target_classification_params), lr=lr)
                reset_count += 1
            except IndexError:
                break

    target_classifier.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 'target_classifier.pt')))

    return target_classifier, (target_classification_train_history, target_classification_eval_train_history,
                               target_classification_eval_val_history, target_classification_eval_test_history)
