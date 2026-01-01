
"""
Provisory text generation script.

This script loads a trained LLM, restores the tokenizer,
and performs autoregressive text generation.

Author: Robin
Date:   December 2025
"""

import os
import sys
import time
import torch

from model import LargeLanguageModel
from tokenizer import Tokenizer
from config import (
    VOCAB_SIZE, N_EMBD, N_HEAD, N_LAYERS, DROPOUT,
    BLOCK_SIZE, DEVICE
)


# Tokenizer Loading
TOKENIZER_PATH = "tokenizer.json"
tokenizer = Tokenizer.load(TOKENIZER_PATH)


# =========================
# Model & Weight Loading
# =========================

final_path = input("Enter model file name (NAME.pth): ")

if not os.path.exists(final_path):
    raise FileNotFoundError("No model weights found.")

print(f"Loading model data from {final_path}...")
checkpoint = torch.load(final_path, map_location=DEVICE)

conf = checkpoint["config"]

# Instantiate model using the saved configuration
model = LargeLanguageModel(
    vocab_size=conf["vocab_size"],
    n_embd=conf["n_embd"],
    n_head=conf["n_head"],
    n_layers=conf["n_layers"],
    block_size=conf["block_size"],
    dropout=conf["dropout"],
    device=DEVICE
)

# Load weights
model.load_state_dict(checkpoint["model_state_dict"])

model.to(DEVICE)
model.eval()


# =========================
# Generation
# =========================

prompt = input("Enter a french start: ")
tokens = tokenizer.encode(prompt)

context = torch.tensor(
    [tokens],
    dtype=torch.long,
    device=DEVICE
)

# Truncate prompt if it exceeds model context size (BLOCK_SIZE)
if context.shape[1] > model.block_size:
    context = context[:, -model.block_size:]
    print(f"[WARN] Prompt truncated to last {model.block_size} tokens.")

# Temperature of generation
# 1.0 = standard, < 1.0 = focused, > 1.0 = creative/chaotic.
temperature = 0.8

with torch.no_grad():
    raw = model.generate(
        context,
        max_new_tokens=500,
        temperature=temperature
    )

generated = raw[0].tolist()
text = tokenizer.decode(generated)

# Cut at the last period 
last_period = text.rfind(".")
if last_period != -1:
    text = text[: last_period + 1]


# =========================
# Sequential Animation
# =========================

print("\nGenerated text:\n")

# Removing �
clean = text.replace("\uFFFD", "")

for ch in clean:
    sys.stdout.write(ch)
    sys.stdout.flush()
    time.sleep(0.01)  # Adjustable generation speed

print("\n")