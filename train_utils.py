
"""
Training utilities for the Transformer-based language model.

Contains:
- Learning rate scheduler (warmup + cosine decay)
- Batch generation with random offsets
- Loss estimation with mixed precision
- Training loop with gradient accumulation

Author: Robin
Date:   December 2025
"""

import math
import random
import torch

from torch.nn import functional as F
from tokenizer import SPECIAL_TOKENS


def get_lr(iter, warmup_iters, max_iters, lr_max):
    """ Learning rate scheduler.
    
    Compute the learning rate using linear warmup and cosine decay.
    """
    
    # Linear warmup
    if iter < warmup_iters:
        return lr_max * iter / warmup_iters

    # Cosine decay
    if iter < max_iters:
        decay_ratio = (iter - warmup_iters) / (max_iters - warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return coeff * lr_max
    
    return 0.0



def get_batch(split, data, block_size, batch_size, device):
    """ Generate a batch of input/target sequences. """

    blocks = data["train"] if split == "train" else data["val"]
    x_list, y_list = [], []

    # Ensure we collect exactly batch_size samples
    while len(x_list) < batch_size:
        block = random.choice(blocks)
        if len(block) <= block_size:
            continue

        # Random offset to use different parts of the text
        max_offset = len(block) - block_size - 1
        offset = random.randint(0, max_offset) if max_offset > 0 else 0

        x = block[offset : offset + block_size]
        y = block[offset + 1 : offset + block_size + 1]

        x_list.append(x)
        y_list.append(y)

    x = torch.stack(x_list).to(device)
    y = torch.stack(y_list).to(device)
    return x, y



@torch.no_grad()
def estimate_loss(model, data, eval_iters, block_size, batch_size, device):
    """ Estimate average loss on train and validation sets. """

    out = {}
    model.eval()
    special_ids = torch.tensor(list(SPECIAL_TOKENS.values()), device=device)
    
    # Determine dtype for mixed precision
    pt_dtype = torch.bfloat16 if (device == "xpu" or device == "cuda") else torch.float32

    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            X, Y = get_batch(split, data, block_size, batch_size, device)
            
            # Mask special tokens in targets
            mask = torch.isin(Y, special_ids)
            Y = Y.masked_fill(mask, -100)

            # Mixed precision context (this is an optimization)
            with torch.autocast(device_type=device, dtype=pt_dtype):
                _, loss = model(X, Y)
            
            losses[k] = loss.item()
        out[split] = losses.mean()

    model.train()
    return out



def train_model(model, data, optimizer,
                max_iters,
                eval_interval,
                warmup_iters,
                lr_max,
                eval_iters,
                block_size,
                batch_size,
                grad_accum_steps,
                device,
                start_iter=0,
                checkpoint_path="checkpoint.pth",
                writer=None,
                config=None): 
    
    """ Train the language model with gradient accumulation. """

    # Set precision (Bfloat16 is optimal for my XPU)
    pt_dtype = torch.bfloat16 if (device == "xpu" or device == "cuda") else torch.float32

    # Initialize learning rate
    lr = get_lr(start_iter, warmup_iters, max_iters, lr_max)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    optimizer.zero_grad(set_to_none=True)

    for iter in range(start_iter, max_iters):
        
        # Periodic evaluation
        if iter % eval_interval == 0:
            losses = estimate_loss(model, data, eval_iters, block_size, batch_size, device)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            if writer is not None:
                writer.add_scalar("Loss/train", losses["train"], iter)
                writer.add_scalar("Loss/val", losses["val"], iter)

        # Gradient Accumulation Loop (micro-batches to fit in memory)
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            
            # Get micro-batch
            xb, yb = get_batch("train", data, block_size, batch_size, device)
            
            # Mask special tokens
            special_ids = torch.tensor(list(SPECIAL_TOKENS.values()), device=device)
            mask = torch.isin(yb, special_ids)
            yb = yb.masked_fill(mask, -100)

            # Forward pass with mixed precision
            with torch.autocast(device_type=device, dtype=pt_dtype):
                logits, loss = model(xb, yb)
                # Scale loss for accumulation
                loss = loss / grad_accum_steps 
            
            loss_accum += loss.item()
            
            # Backward pass accumulates gradients
            loss.backward()

        # Gradient clipping (limits gradient norm for stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Log gradient norm in Tensorboard
        if writer is not None and iter % 10 == 0:
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            grad_norm = total_norm ** 0.5
            writer.add_scalar("GradNorm", grad_norm, iter)

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # Update learning rate
        lr = get_lr(iter + 1, warmup_iters, max_iters, lr_max)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if writer is not None:
            writer.add_scalar("LR", lr, iter)

        # Clear cache occasionally 
        if device == "xpu" and iter % 20 == 0:
            torch.xpu.empty_cache()

        # Save checkpoint
        if iter % 50 == 0 and iter > 0:
            checkpoint_data = {
                "model_state_dict": model.state_dict(),     
                "optimizer_state_dict": optimizer.state_dict(), 
                "iter": iter + 1,
            }
            if config is not None:
                checkpoint_data['config'] = config
            
            torch.save(checkpoint_data, checkpoint_path)
            print(f"Checkpoint saved at iteration {iter}")
        
        elif iter % 10 == 0:
            print(f"Iteration {iter} completed ({iter/max_iters*100:.2f}%) Loss: {loss_accum:.4f}")