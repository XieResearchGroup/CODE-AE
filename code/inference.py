from evaluation_utils import evaluate_target_classification_epoch, predict_target_classification
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
    if clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    history['bce'].append(loss.cpu().detach().item())

    return history


def make_inference(encoder, train_dataloader, test_df, task_save_folder, normalize_flag=False,retrain_flag=True,
                   **kwargs):
    target_decoder = MLP(input_dim=kwargs['latent_dim'],
                         output_dim=1,
                         hidden_dims=kwargs['classifier_hidden_dims']).to(kwargs['device'])
    target_classifier = EncoderDecoder(encoder=encoder, decoder=target_decoder, normalize_flag=normalize_flag).to(
        kwargs['device'])

    if not retrain_flag:
        target_classifier.load_state_dict(torch.load(os.path.join(task_save_folder, f'target_classifier.pt')))
    else:
        classification_loss = nn.BCEWithLogitsLoss()
        target_classification_train_history = defaultdict(list)
        target_classification_eval_train_history = defaultdict(list)

        encoder_module_indices = [i for i in range(len(list(encoder.modules())))
                                  if str(list(encoder.modules())[i]).startswith('Linear')]
        lr = kwargs['lr']
        target_classification_params = [target_classifier.decoder.parameters()]

        target_classification_optimizer = torch.optim.AdamW(chain(*target_classification_params),
                                                            lr=lr)

        for epoch in range(kwargs['train_num_epochs']):
            if epoch % 10 == 0:
                print(f'Fine tuning epoch {epoch}')
            for step, batch in enumerate(train_dataloader):
                target_classification_train_history = classification_train_step(model=target_classifier,
                                                                                batch=batch,
                                                                                loss_fn=classification_loss,
                                                                                device=kwargs['device'],
                                                                                optimizer=target_classification_optimizer,
                                                                                history=target_classification_train_history)
            target_classification_eval_train_history = evaluate_target_classification_epoch(
                classifier=target_classifier,
                dataloader=train_dataloader,
                device=kwargs['device'],
                history=target_classification_eval_train_history)

            if epoch > 20 and epoch % 10 == 0:
                try:
                    ind = encoder_module_indices.pop()
                    print(f'Unfreezing {epoch}')
                    target_classification_params.append(list(target_classifier.encoder.modules())[ind].parameters())
                    lr = lr * kwargs['decay_coefficient']
                    target_classification_optimizer = torch.optim.AdamW(chain(*target_classification_params), lr=lr)
                except IndexError:
                    break

        torch.save(target_classifier.state_dict(),
                   os.path.join(task_save_folder, f'target_classifier.pt'))

    prediction_df = predict_target_classification(classifier=target_classifier, test_df=test_df,
                                                  device=kwargs['device'])

    return prediction_df, (
    target_classification_train_history, target_classification_eval_train_history) if retrain_flag else prediction_df, None
