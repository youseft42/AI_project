import os
import abc
import sys
import tqdm
import torch
from typing import Any, Callable
from pathlib import Path
from torch.utils.data import DataLoader


def train_two_phase_rnn(model, dl_train, optimizer, sub_loss_fn, stance_loss_fn, max_batches=None):
    losses = []

    with tqdm.tqdm(total=(max_batches if max_batches else len(dl_train)), file=sys.stdout) as pbar:
        for idx_batch, batch in enumerate(dl_train, start=1):
            x, y_stance, y_sub = batch.TWEET, batch.STANCE, batch.SENTIMENT

            # Forward pass
            # Output
            y_stance_hat, y_sub_hat = model(x)
            S, B, V = y_stance_hat.shape

            y_stance_hat = y_stance_hat[y_stance_hat.shape[0] - 1, :, :]
            torch.squeeze(y_stance_hat, 0)
            y_sub_hat = y_sub_hat[y_sub_hat.shape[0] - 1, :, :]
            torch.squeeze(y_sub_hat, 0)

            # Calculate loss
            loss_stance = stance_loss_fn(y_stance_hat, y_stance)
            loss_sub = sub_loss_fn(y_sub_hat, y_sub)
            loss = loss_stance + loss_sub

            # Update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss_stance.item())

            pbar.update()
            pbar.set_description(f'train loss={losses[-1]},')
            if max_batches and idx_batch >= max_batches:
                break
    return losses


def eval_two_phase_rnn(model, dl_test):
    sub_accuracies = []
    stance_accuracies = []

    with tqdm.tqdm(total=(len(dl_test)), file=sys.stdout) as pbar:
        for idx_batch, batch in enumerate(dl_test):
            x, y_stance, y_sub = batch.TWEET, batch.STANCE, batch.SENTIMENT

            # Forward pass
            # Output
            with torch.no_grad():
                y_stance_hat, y_sub_hat = model(x)

            S, B, V = y_stance_hat.shape

            y_stance_hat = y_stance_hat[y_stance_hat.shape[0] - 1, :, :]
            torch.squeeze(y_stance_hat, 0)

            y_sub_hat = y_sub_hat[y_sub_hat.shape[0] - 1, :, :]
            torch.squeeze(y_sub_hat, 0)

            y_sub_pred = torch.argmax(y_sub_hat, dim=1)
            y_stance_pred = torch.argmax(y_stance_hat, dim=1)

            sub_accuracies.append(torch.sum(y_sub_pred == y_sub) / float(B))
            stance_accuracies.append(torch.sum(y_stance_pred == y_stance) / float(B))
            pbar.update()
            pbar.set_description(f' sentiment accuracy={sub_accuracies[-1]}, stance accuracy={stance_accuracies[-1]} ')

    return sub_accuracies, stance_accuracies
