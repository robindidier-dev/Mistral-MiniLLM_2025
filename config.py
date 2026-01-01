
"""
Global hyperparameters and configuration values.

This file centralizes all constants used across:
- model.py
- train_utils.py
- main_train.py
- tokenizer pipeline

Author: Robin
Date:   December 2025
"""

import torch

# =========================
# Model hyperparameters
# =========================

VOCAB_SIZE = 8192        # Target vocabulary size (including byte tokens)
N_EMBD = 384             # Embedding dimension
N_HEAD = 6               # Number of attention heads
N_LAYERS = 8             # Number of transformer blocks
DROPOUT = 0.1            # Dropout rate
BLOCK_SIZE = 1024        # Context length (in tokens) augmented to 1024 to capture more context


# =========================
# Training hyperparameters
# =========================

# Gradient Accumulation Strategy for Intel Arc
# We want an effective batch size of 64, but 64 * 1024 context won't fit in VRAM.
# So we use a micro-batch of 8 and accumulate gradients 8 times.
BATCH_SIZE = 8           # Size of each micro-batch
GRAD_ACCUM_STEPS = 8     # Accumulate gradients to simulate Batch Size = 64

MAX_ITERS = 5000         # Number of training iterations
EVAL_INTERVAL = 100      # Interval for evaluation
EVAL_ITERS = 50          # Number of iterations for evaluation

WARMUP_ITERS = 200       # Number of warmup iterations
MAX_LEARNING_RATE = 6e-4 # Maximum learning rate of scaled-down by cosine decay


# =========================
# Device
# =========================

# I don't have a CUDA device to run my model, so I use a XPU.
# Maybe I'll try to train it on a CUDA device in the future.

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.xpu.is_available():
    DEVICE = "xpu"       
else:
    DEVICE = "cpu"