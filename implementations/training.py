
import sys
import tqdm
import torch


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

def train_pe_two_phase_rnn(model, dl_train, optimizer, sub_loss_fn, stance_loss_fn, max_batches=None):
    losses = []

    with tqdm.tqdm(total=(max_batches if max_batches else len(dl_train)), file=sys.stdout) as pbar:
        for idx_batch, batch in enumerate(dl_train, start=1):
            x, y_stance, y_sub = batch.TWEET, batch.STANCE, batch.SENTIMENT

            # Forward pass
            # Output
            y_stance_hat, y_sub_hat = model(x)

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

def train_baseline_rnn(model, dl_train, optimizer, stance_loss_fn, max_batches=None):
    losses = []

    with tqdm.tqdm(total=(max_batches if max_batches else len(dl_train)), file=sys.stdout) as pbar:
        for idx_batch, batch in enumerate(dl_train, start=1):
            x, y_stance = batch.TWEET, batch.STANCE

            # Forward pass
            # Output
            y_stance_hat = model(x)

            # Calculate loss
            loss_stance = stance_loss_fn(y_stance_hat, y_stance)
            loss = loss_stance

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
            sub_to = torch.FloatTensor(sub_accuracies)
            stance_to = torch.FloatTensor(stance_accuracies)
            pbar.set_description(f' sentiment accuracy={torch.mean(sub_to)}, stance accuracy={torch.mean(stance_to)} ')

    return torch.Tensor(sub_accuracies).mean(), torch.Tensor(stance_accuracies).mean()

def eval_pe_two_phase_rnn(model, dl_test):
    sub_accuracies = []
    stance_accuracies = []

    with tqdm.tqdm(total=(len(dl_test)), file=sys.stdout) as pbar:
        for idx_batch, batch in enumerate(dl_test):
            x, y_stance, y_sub = batch.TWEET, batch.STANCE, batch.SENTIMENT

            # Forward pass
            with torch.no_grad():
                y_stance_hat, y_sub_hat = model(x)

            B, V = y_stance_hat.shape

            y_sub_pred = torch.argmax(y_sub_hat, dim=1)
            y_stance_pred = torch.argmax(y_stance_hat, dim=1)

            sub_accuracies.append(torch.sum(y_sub_pred == y_sub) / float(B))
            stance_accuracies.append(torch.sum(y_stance_pred == y_stance) / float(B))
            pbar.update()
            sub_to = torch.FloatTensor(sub_accuracies)
            stance_to = torch.FloatTensor(stance_accuracies)
            pbar.set_description(f' sentiment accuracy={torch.mean(sub_to)}, stance accuracy={torch.mean(stance_to)} ')

    return torch.Tensor(sub_accuracies).mean(), torch.Tensor(stance_accuracies).mean()

def eval_baseline_rnn(model, dl_test):
    stance_accuracies = []

    with tqdm.tqdm(total=(len(dl_test)), file=sys.stdout) as pbar:
        for idx_batch, batch in enumerate(dl_test):
            x, y_stance = batch.TWEET, batch.STANCE

            # Forward pass
            with torch.no_grad():
                y_stance_hat = model(x)

            B, V = y_stance_hat.shape

            y_stance_pred = torch.argmax(y_stance_hat, dim=1)

            stance_accuracies.append(torch.sum(y_stance_pred == y_stance) / float(B))
            pbar.update()
            stance_to = torch.FloatTensor(stance_accuracies)
            pbar.set_description(f' stance accuracy={torch.mean(stance_to)} ')

    return torch.Tensor(stance_accuracies).mean()
