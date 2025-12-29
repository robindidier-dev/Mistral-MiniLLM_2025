
"""
Provisory text generation script.

This script loads a trained LLM, restores the tokenizer, 
and performs autoregressive text generation.

Author: Robin 
Date:   December 2025
"""

import os
import torch

from model import LargeLanguageModel
from tokenizer import Tokenizer
from config import (
    VOCAB_SIZE, N_EMBD, N_HEAD, N_LAYERS, DROPOUT,
    BLOCK_SIZE, DEVICE
)


# =========================
# Tokenizer Loading
# =========================

TOKENIZER_PATH = "tokenizer.json"

tokenizer = Tokenizer.load(TOKENIZER_PATH)


# =========================
# Model Instantiation
# =========================

model = LargeLanguageModel(
    vocab_size=VOCAB_SIZE,
    n_embd=N_EMBD,
    n_head=N_HEAD,
    n_layers=N_LAYERS,
    block_size=BLOCK_SIZE,
    dropout=DROPOUT,
    device=DEVICE
).to(DEVICE)


# =========================
# Weight Loading
# =========================

final_path = "LLM.pth"

if os.path.exists(final_path):
    print("Loading model weights...")
    state = torch.load(final_path, map_location=DEVICE)
    model.load_state_dict(state)

else:
    raise FileNotFoundError("No model weights found.")

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

with torch.no_grad():
    generated = model.generate(
        context,
        max_new_tokens=500
    )[0].tolist()

text = tokenizer.decode(generated)

last_period = text.rfind(".")
if last_period != -1:
    text = text[: last_period + 1]


print("\nGenerated text:\n")
print(text)
print("\n")
