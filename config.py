
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

VOCAB_SIZE = 5000        # Target vocabulary size (including byte tokens)
N_EMBD = 384             # Embedding dimension
N_HEAD = 8               # Number of attention heads
N_LAYERS = 8            # Number of transformer blocks
DROPOUT = 0.2            # Dropout rate
BLOCK_SIZE = 256         # Maximum context length


# =========================
# Training hyperparameters
# =========================

BATCH_SIZE = 64                     # Number of sequences processed in parallel
MAX_ITERS = 10000                    # Total number of training iterations
EVAL_INTERVAL = 100                 # Interval for loss evaluation
EVAL_ITERS = MAX_ITERS//200          # Number of batches for loss estimation
WARMUP_ITERS = MAX_ITERS//50        # Warmup duration
MAX_LEARNING_RATE = 3e-4            # Peak learning rate


# =========================
# Device
# =========================

# I don't have a CUDA device to run my model, so I use a XPU.
# Maybe I'll try to train it on a CUDA device in the future.

if torch.cuda.is_available():
    DEVICE = "cuda"                                         
elif hasattr(torch, "xpu") and torch.xpu.is_available():
    DEVICE = "xpu"
else:
    DEVICE = "cpu"
