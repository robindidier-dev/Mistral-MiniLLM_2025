"""
Transformer - Decoder-only Language Model - Main Training Script

Based on the Transformer architecture introduced in the paper "Attention 
Is All You Need" (Vaswani et al., 2017).

This script trains a LLM adapted from the tutorial
by Andrej Karpathy: "Let's build GPT: from scratch, in code, spelled out"
(https://www.youtube.com/watch?v=kCc8FmEb1nY).

The model uses a BPE tokenizer also adapted from another Andrej Karpathy's tutorial.

Modifications of the Kapathy version include:
    - Weight tying between input embeddings and output logits to reduce the number of parameters.
    - Scheduler for learning rate (Cosine Annealing).
    - Monitoring with TensorBoard.
    - Checkpoint saving and loading.

Author: Robin
Date:   November 2025
        Improved in December 2025
"""

import os
from tokenizer import Tokenizer
import torch
import subprocess
import sys
import webbrowser
from torch.utils.tensorboard import SummaryWriter

from config import (
    VOCAB_SIZE, N_EMBD, N_HEAD, N_LAYERS, DROPOUT,
    BLOCK_SIZE, BATCH_SIZE, MAX_ITERS, EVAL_INTERVAL,
    WARMUP_ITERS, MAX_LEARNING_RATE, EVAL_ITERS, DEVICE
)


from data_utils import load_documents, prepare_dataset
from model import LargeLanguageModel
from train_utils import train_model


TOKENIZER_PATH = "tokenizer.json"

# =========================
# Model + optimizer creation
# =========================

def create_model_and_optimizer():
    """ Instantiate model and optimizer. """

    model = LargeLanguageModel(
        vocab_size=VOCAB_SIZE,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        n_layers=N_LAYERS,
        block_size=BLOCK_SIZE,
        dropout=DROPOUT,
        device=DEVICE
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LEARNING_RATE)

    # Print parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params/1e6:.2f} million parameters")

    return model, optimizer


# =========================
# Main
# =========================

if __name__ == "__main__":

    # Load corpus
    corpus = load_documents("corpus_docs")

    # Load or train tokenizer
    if os.path.exists(TOKENIZER_PATH):
        tokenizer = Tokenizer.load(TOKENIZER_PATH)
    else:
        print("Training tokenizer...")
        tokenizer = Tokenizer(vocab_size=VOCAB_SIZE)
        tokenizer.train(corpus)
        tokenizer.save(TOKENIZER_PATH)

    print("Tokenizer ready. Dataset preparation...")

    # Prepare dataset
    data = prepare_dataset(
        corpus_path="corpus.txt",
        tokenizer=tokenizer,
        block_size=BLOCK_SIZE,
        train_ratio=0.9
    )

    print("Dataset ready. Creating model...")

    # Create model + optimizer
    model, optimizer = create_model_and_optimizer()

    # Checkpoint logic
    checkpoint_path = "checkpoint.pth"
    start_iter = 0

    if os.path.exists(checkpoint_path):
        print("Checkpoint loaded.")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_iter = checkpoint.get("iter", 0)
        print(f"Resuming training from iteration {start_iter}")
        logdir = "runs/LLM"          
    else:
        print("Starting new training.")
        logdir = "runs/LLM_new"      
    # TensorBoard writer
    writer = SummaryWriter(log_dir=logdir)

    # Launch TensorBoard silently
    subprocess.Popen(
        [
            sys.executable, "-m", "tensorboard.main",
            "--logdir=runs",
            "--port=6006",
            "--host=localhost"
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    print("TensorBoard started at http://localhost:6006")
    webbrowser.open("http://localhost:6006")

    # Train
    train_model(
        model=model,
        data=data,
        optimizer=optimizer,
        max_iters=MAX_ITERS,
        eval_interval=EVAL_INTERVAL,
        warmup_iters=WARMUP_ITERS,
        lr_max=MAX_LEARNING_RATE,
        eval_iters=EVAL_ITERS,
        block_size=BLOCK_SIZE,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        start_iter=start_iter,
        checkpoint_path=checkpoint_path,
        writer=writer
    )

    writer.close()

    # Save final model
    torch.save(model.state_dict(), "LLM.pth")
    print("Model saved as LLM.pth")