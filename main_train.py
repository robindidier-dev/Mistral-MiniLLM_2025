
"""
Transformer - Decoder-only Language Model - Main Training Script

Based on the Transformer architecture introduced in the paper "Attention 
Is All You Need" (Vaswani et al., 2017).

This script trains a LLM adapted from the tutorial
by Andrej Karpathy.

Modifications:
    - Weight tying.
    - Cosine Annealing Scheduler.
    - TensorBoard support.
    - Gradient Accumulation for small VRAM.

Author: Robin
Date:   December 2025
"""

import os
import sys
import subprocess
import webbrowser
import torch
from torch.utils.tensorboard import SummaryWriter

from tokenizer import Tokenizer
from model import LargeLanguageModel
from data_utils import load_documents, prepare_dataset
from train_utils import train_model

from config import (
    VOCAB_SIZE, N_EMBD, N_HEAD, N_LAYERS, DROPOUT,
    BLOCK_SIZE, BATCH_SIZE, MAX_ITERS, EVAL_INTERVAL,
    WARMUP_ITERS, MAX_LEARNING_RATE, EVAL_ITERS, DEVICE,
    GRAD_ACCUM_STEPS  
)

TOKENIZER_PATH = "tokenizer.json"
CORPUS_PATH = "corpus_docs"

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

    # Print the number of parameters of the model
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params/1e6:.2f} million parameters")

    return model, optimizer


# =========================
# Main
# =========================

if __name__ == "__main__":

    # Load corpus
    print(f"Loading corpus from {CORPUS_PATH}...")
    corpus = load_documents(CORPUS_PATH)

    # Load or train tokenizer
    if os.path.exists(TOKENIZER_PATH):
        print(f"Loading tokenizer from {TOKENIZER_PATH}...")
        tokenizer = Tokenizer.load(TOKENIZER_PATH)
    else:
        print("Training tokenizer...")
        # We train the tokenizer on only 10% of the corpus for efficiency
        nb_docs_train = 500
        training_subset = corpus[:nb_docs_train] 
        print(f"Using the first {len(training_subset)} documents for tokenizer training.")

        tokenizer = Tokenizer(VOCAB_SIZE)
        tokenizer.train(training_subset)
        tokenizer.save(TOKENIZER_PATH)

    print("Tokenizer ready. Dataset preparation...")

    model_config = {
        'vocab_size': VOCAB_SIZE,
        'n_embd': N_EMBD,
        'n_head': N_HEAD,
        'n_layers': N_LAYERS,
        'block_size': BLOCK_SIZE,
        'dropout': DROPOUT,
        'device': str(DEVICE)
    }

    # Prepare dataset (creating tensors etc.)
    data = prepare_dataset(
        documents=corpus,
        tokenizer=tokenizer,
        block_size=BLOCK_SIZE,
        train_ratio=0.9
    )

    # Optimization : clear corpus from RAM to free memory for training
    del corpus
    import gc
    gc.collect()

    print("Dataset ready. Creating model...")

    model, optimizer = create_model_and_optimizer()

    resume_version = None
    i = 0

    # Searching the first existing checkpoint
    while True:
        candidate = f"checkpoint_LLM{i}.pth"
        if os.path.exists(candidate):
            resume_version = i
            break
        if i > 10000:
            break
        i += 1

    if resume_version is not None:
        version = resume_version
        print(f"Resuming from checkpoint_LLM{version}.pth")
    else:
        version = 0
        while os.path.exists(f"runs/LLM{version}"):
            version += 1
        print(f"No checkpoint found. Starting new training session: LLM{version}")

    # Paths for logs, checkpoints, and final model
    logdir = f"runs/LLM{version}"
    checkpoint_path = f"checkpoint_LLM{version}.pth"
    final_model_path = f"LLM{version}.pth"

    start_iter = 0

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_iter = checkpoint.get("iter", 0)
        print(f"Resuming training from iteration {start_iter}")

    else:
        print("No checkpoint found for this version. Training from scratch.")

    # Initialize TensorBoard
    writer = SummaryWriter(log_dir=logdir)

    try:
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
    except Exception as e:
        print(f"Could not launch TensorBoard automatically: {e}")

    # Train
    try:
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
            grad_accum_steps=GRAD_ACCUM_STEPS,
            device=DEVICE,
            start_iter=start_iter,
            checkpoint_path=checkpoint_path,
            writer=writer,
            config=model_config  
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current state...")

    writer.close()

    # Save final model with its configuration
    print(f"Saving final model to {final_model_path}...")
    final_state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': model_config, 
        'iter': MAX_ITERS
    }
    torch.save(final_state, final_model_path)
    print("Done.")