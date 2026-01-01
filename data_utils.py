
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
    """ Load the full concatenated corpus as a raw string. """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Corpus file not found at: {path}")
        
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_documents(folder="corpus_docs"):
    """ Load all .txt documents from a folder as raw strings. """
    if not os.path.exists(folder):
        print(f"Folder '{folder}' did not exist.")
        return []

    docs = []
    for filename in sorted(os.listdir(folder)):
        if filename.endswith(".txt"):
            path = os.path.join(folder, filename)
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read().strip()
                if content: 
                    docs.append(content)
    return docs


# =========================
# Encoding
# =========================

def encode_corpus(text, tokenizer):
    """ Encode the full corpus using the tokenizer object.
    
    Args:
        text (str): Raw corpus text.
        tokenizer (Tokenizer): Instance of your Tokenizer class.
        
    Returns:
        torch.Tensor: Tensor of token IDs.
    """
    tokens = tokenizer.encode(text)
    return torch.tensor(tokens, dtype=torch.long)


# =========================
# Block splitting and shuffling
# =========================

def split_into_blocks(data_list, block_size):
    """ Split a list of tokens into blocks of size block_size + 1 for autoregressive training. 
    
    We drop the remainder (last chunk < block_size) to ensure consistent tensor shapes.
    """

    blocks = []
    step = block_size + 1 # '+1' because input is [0:-1] and target is [1:]

    # We iterate until len(data) - step to ensure we have a full block
    for i in range(0, len(data_list) - step + 1, step):
        block = data_list[i : i + step]
        blocks.append(block)

    return blocks


def shuffle_blocks(blocks):
    """ Shuffle a list of blocks to avoid style bias in train/val split. """
    random.shuffle(blocks)
    return blocks


# =========================
# Dataset preparation
# =========================

def build_train_val(blocks, train_ratio=0.9):
    """ Split a list of independent blocks into train and validation sets. """
    n = int(train_ratio * len(blocks))
    return {
        "train": blocks[:n],
        "val": blocks[n:]
    }


# =========================
# Pipeline
# =========================

def prepare_dataset(documents, tokenizer, block_size, train_ratio=0.9):
    """ Prepare dataset from a list of raw documents.
    
    Args:
        documents (list of str): List of raw text documents.
        tokenizer (Tokenizer): Instance of your Tokenizer class.
        block_size (int): Size of each training block.
        train_ratio (float): Ratio of data to use for training.
    """
    
    all_blocks = []
    dropped_docs = 0

    print(f"Encoding {len(documents)} documents...")

    for doc in documents:
        # 1. Encode via the class method
        encoded = tokenizer.encode(doc)

        # 2. Split into independent blocks
        # If document is shorter than (block_size + 1), it generates 0 block.
        doc_blocks = split_into_blocks(encoded, block_size)
        
        if not doc_blocks:
            dropped_docs += 1
            continue

        # 3. Convert to Tensors 
        tensor_blocks = [torch.tensor(b, dtype=torch.long) for b in doc_blocks]
        all_blocks.extend(tensor_blocks)

    if dropped_docs > 0:
        print(f" Dropped {dropped_docs} documents shorter than block_size ({block_size}).")

    print(f"Total blocks generated: {len(all_blocks)}")

    # 4. Global shuffle
    shuffled = shuffle_blocks(all_blocks)

    # 5. Split
    return build_train_val(shuffled, train_ratio)