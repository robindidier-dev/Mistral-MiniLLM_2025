
"""
Training utilities for the Transformer-based language model.

Contains:
- Learning rate scheduler (warmup + cosine decay) --> improvement from Karpathy's original tutorial
- Batch generation
- Loss estimation
- Training loop

Author: Robin
Date:   November 2025
        Improved in December 2025
"""

import math
import torch
import torch.xpu
from torch.nn import functional as F



# =========================
# Learning rate scheduler
# =========================

def get_lr(iter, warmup_iters, max_iters, lr_max):
    """Compute the learning rate at a given training iteration using a
    linear warmup followed by cosine decay.
    
    Args:
        iter (int): Current training iteration.
        warmup_iters (int): Number of iterations for linear warmup.
        max_iters (int): Total number of training iterations.
        lr_max (float): Maximum learning rate after warmup.

    Returns:
        lr (float): Learning rate for the current iteration.
    """

    # Linear warmup
    if iter < warmup_iters:
        return lr_max * iter / warmup_iters

    # Cosine decay
    if iter < max_iters:
        decay_ratio = (iter - warmup_iters) / (max_iters - warmup_iters)
        return lr_max * 0.5 * (1 + math.cos(math.pi * decay_ratio))

    return 0.0



# =========================
# Batch generation
# =========================

def get_batch(split, data, block_size, batch_size, device):
    """ Generate a batch of input sequences 'x' and target sequences 'y' for training or validation.

    Args:
        split (str): 'train' or 'val', selects the dataset split.
        data (dict): dictionary containing 'train' and 'val' tensors.
        block_size (int): length of each input sequence.
        batch_size (int): number of sequences per batch.
        device (str): device to move the tensors to.

    Returns:
        x (Tensor): shape (batch_size, block_size), input sequences.
        y (Tensor): shape (batch_size, block_size), target sequences (next characters).
    """
    
    dataset = data["train"] if split == "train" else data["val"]

    # Random starting points for each sequence in the batch
    index = torch.randint(0, len(dataset) - block_size, (batch_size,))

    x = torch.stack([dataset[i:i+block_size] for i in index])
    y = torch.stack([dataset[i+1:i+block_size+1] for i in index])

    x, y = x.to(device), y.to(device)
    return x, y



# =========================
# Loss estimation
# =========================

@torch.no_grad()  # Optimization: useless to store gradients
def estimate_loss(model, data, eval_iters, block_size, batch_size, device):
    """ Estimate average loss on train and validation sets. """

    out = {}
    model.eval()  # No dropout, etc.

    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, data, block_size, batch_size, device)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()

    model.train()
    return out



# =========================
# Training loop
# =========================

def train_model(model, data, optimizer,
                max_iters,
                eval_interval,
                warmup_iters,
                lr_max,
                eval_iters,
                block_size,
                batch_size,
                device,
                start_iter=0,
                checkpoint_path="checkpoint.pth",
                writer=None):
    """ Train the language model.

    Args:
        model (nn.Module): to train.
        data (dict): Dictionary containing 'train' and 'val' tensors.
        optimizer : For updating parameters.
        max_iters (int): Number of training iterations.
        eval_interval (int): Periodical loss evaluation.
        warmup_iters (int): Warmup duration.
        lr_max (float): Maximum learning rate.
        eval_iters (int): Number of batches for loss estimation.
        block_size (int): Sequence length.
        batch_size (int): Batch size.
        device (str): Device to use.
        start_iter (int): Iteration to resume from.
        checkpoint_path (str): Path to save checkpoints.
        writer (SummaryWriter, optional): For TensorBoard logging.
    """

    for iter in range(start_iter, max_iters):
        # Evaluate loss periodically
        if iter % eval_interval == 0:
            losses = estimate_loss(model, data, eval_iters, block_size, batch_size, device)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            # Log losses to TensorBoard
            if writer is not None:
                writer.add_scalar("Loss/train", losses["train"], iter)
                writer.add_scalar("Loss/val", losses["val"], iter)

        # Get a batch of training data
        xb, yb = get_batch("train", data, block_size, batch_size, device)
        # Forward + backward + update
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Update learning rate (warmup + cosine decay)
        lr = get_lr(iter, warmup_iters, max_iters, lr_max)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Log learning rate to TensorBoard
        if writer is not None:
            writer.add_scalar("LR", lr, iter)

        optimizer.step()
        torch.xpu.empty_cache()


        # Save checkpoint
        if iter % 50 == 0 and iter > 0:
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iter": iter + 1,
            }, checkpoint_path)
            print(f"Checkpoint saved at iteration {iter}")
        elif iter % 10 == 0:
            print(f"Iteration {iter} completed ({iter/max_iters*100:.2f}%)")