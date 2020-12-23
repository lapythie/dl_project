import torch
import numpy as np 
from tqdm import tqdm


def train(model, loader, criterion, optimizer, device, 
          last_n_losses=500, verbose=True):
    """Train for an epoch"""
    losses = []

    progress_bar = tqdm(total=len(loader), disable=not verbose, desc="Train")

    model.train()

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        pred = model(x)

        loss = criterion(pred.view(-1, pred.size(-1)), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        progress_bar.set_postfix(loss=np.mean(losses[-last_n_losses:]),
                                perplexity=np.exp(np.mean(losses[-last_n_losses:])))
        progress_bar.update()

    progress_bar.close()

    return losses


@torch.no_grad()
def evaluate(model, loader, criterion, device, last_n_losses=500, verbose=True):

    losses = []

    progress_bar = tqdm(total=len(loader), disable=not verbose, desc="Evaluate")

    model.eval()

    for x, y in loader:

        x = x.to(device)
        y = y.to(device)

        pred = model(x)

        loss = criterion(pred.view(-1, pred.size(-1)), y.view(-1))

        losses.append(loss.item())

        progress_bar.set_postfix(loss=np.mean(losses[-last_n_losses:]),
                                 perplexity=np.exp(np.mean(losses[-last_n_losses:])))
        progress_bar.update()

    progress_bar.close()

    return losses


@torch.no_grad()
def generate(model, x, steps, sample=False, temperature=1):
    "Higher temperature ( > 1) means more diversity and more mistakes"
    "Probably need to add sampling from top k predictions"
    
    max_len = model.max_len
    model.eval()
    for k in range(steps):
        # crop beginning if sequence is too long
        x_cropped = x if x.size(1) <= max_len else x[:, -max_len:]
        # take only last predicted token, we already have previous ones
        logits = model(x_cropped)[:, -1, :] / temperature
        probas = torch.nn.functional.softmax(logits, dim=-1)
        if sample: # more probable words are more likely to be sampled
            indices = torch.multinomial(probas, num_samples=1)
        else: # greedy
            _, indices = torch.topk(probas, k=1, dim=-1)
        x = torch.cat(tensors=(x, indices), dim=1)
    return x
