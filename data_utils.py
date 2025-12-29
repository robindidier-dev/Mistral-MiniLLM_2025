
"""
Utility functions for loading corpus data, preparing datasets,
splitting into blocks, and shuffling for training.

Author: Robin
Date:   December 2025
"""

import os
import torch
import random


# =========================
# Corpus loading
# =========================

def load_corpus(path="corpus.txt"):
    """ Load the full concatenated corpus as a raw string.

    Args:
        path (str): Path to the corpus file.

    Returns:
        str: The full text content of the corpus.
    """

    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_documents(folder="corpus_docs"):
    """ Load all .txt documents from a folder as raw strings. Used for tokenizer training.

    Args:
        folder (str): Folder containing .txt documents.

    Returns:
        list[str]: List of document contents.
    """

    docs = []
    for filename in sorted(os.listdir(folder)):
        if filename.endswith(".txt"):
            path = os.path.join(folder, filename)
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                docs.append(f.read())
    return docs



# =========================
# Encoding
# =========================

def encode_corpus(text, encode_fn, merges):
    """ Encode the full corpus using the tokenizer's encode() function.

    Args:
        text (str): Raw corpus text.
        encode_fn (callable): Tokenizer encode function.
        merges (dict): BPE merge rules.

    Returns:
        torch.Tensor: Tensor of token IDs.
    """

    tokens = encode_fn(text, merges)
    return torch.tensor(tokens, dtype=torch.long)



# =========================
# Block splitting and shuffling
# =========================

def split_into_blocks(data_tensor, block_size):
    """ Split a long tensor of tokens into fixed-size blocks.

    Args:
        data_tensor (torch.Tensor): Token sequence.
        block_size (int): Length of each block.

    Returns:
        list[torch.Tensor]: List of blocks of shape [block_size].
    """

    blocks = [
        data_tensor[i:i + block_size]
        for i in range(0, len(data_tensor) - block_size, block_size)
    ]
    return blocks


def shuffle_blocks(blocks):
    """ Shuffle a list of blocks to avoid style bias in train/val split.

    Args:
        blocks (list[torch.Tensor]): List of token blocks.

    Returns:
        torch.Tensor: Concatenated shuffled blocks.
    """

    random.shuffle(blocks)
    return torch.cat(blocks)



# =========================
# Dataset preparation
# =========================

def build_train_val(shuffled_tensor, train_ratio=0.8):
    """ Split the shuffled tensor into train and validation sets.

    Args:
        shuffled_tensor (torch.Tensor): Full shuffled token sequence.
        train_ratio (float): Fraction of data used for training.

    Returns:
        dict[str, torch.Tensor]: {"train": ..., "val": ...}
    """

    n = int(train_ratio * len(shuffled_tensor))
    return {
        "train": shuffled_tensor[:n],
        "val": shuffled_tensor[n:]
    }



# =========================
# Pipeline
# =========================

def prepare_dataset(corpus_path, tokenizer, block_size, train_ratio=0.8):
    """ Full dataset preparation pipeline:
        - load corpus
        - encode
        - split into blocks
        - shuffle
        - build train/val sets
    """

    text = load_corpus(corpus_path)
    encoded = tokenizer.encode(text)
    blocks = split_into_blocks(encoded, block_size)
    blocks = [torch.tensor(b, dtype=torch.long) for b in blocks]

    shuffled = shuffle_blocks(blocks)
    return build_train_val(shuffled, train_ratio)