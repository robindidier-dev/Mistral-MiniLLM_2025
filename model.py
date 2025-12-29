"""
 Transformer-based language model.
 
This script implements a transformer language model, adapted from the tutorial
by Andrej Karpathy: "Let's build GPT: from scratch, in code, spelled out"
(https://www.youtube.com/watch?v=kCc8FmEb1nY).

 Contains:
 - FeedForward submodule
 - Self-attention head
 - Multi-head attention
 - Transformer block
 - LargeLanguageModel (main model)
 
Author: Robin
Date:   November 2025
        Improved in December 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Submodel for feed forward mechanism
# =========================

class FeedForward(nn.Module):
    """ A simple feedforward neural network. """

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, emb):
        return self.net(emb)



# =========================
# Submodels for self-attention mechanism
# =========================

class Head(nn.Module):
    """ One head of self-attention. """

    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # Lower triangular matrix -> to prevent attending to future tokens
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, emb):
        """ We apply what is suggested in the paper """
        B, T, C = emb.shape

        k = self.key(emb)
        q = self.query(emb)
        v = self.value(emb)

        # Computing the attention scores = affinities
        affinity = q @ k.transpose(-2, -1) * C**-0.5
        affinity = affinity.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        affinity = F.softmax(affinity, dim=-1)
        affinity = self.dropout(affinity)

        # Weighted aggregation of values
        return affinity @ v



class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention in parallel. """

    def __init__(self, n_embd, num_heads, head_size, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([
            Head(n_embd, head_size, block_size, dropout) for _ in range(num_heads)
        ])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, emb):
        """ Concatenate the output of each head and project it back to N_EMBD dimensions. """
        out = torch.cat([h(emb) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))



# =========================
# Submodel for repeating the process {self-attention + feedforward} multiple times
# =========================

class Block(nn.Module):
    """ Transformer block: communication followed by computation. """

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head

        self.sa = MultiHeadAttention(n_embd, n_head, head_size, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)

        # Normalization for stabilizing training
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, emb):
        # Residual connection for gradient flow
        emb = emb + self.sa(self.ln1(emb))
        emb = emb + self.ffwd(self.ln2(emb))
        return emb



# =========================
# Main Model
# =========================

class LargeLanguageModel(nn.Module):
    """ LLM using self-attention mechanisms. """

    def __init__(self, vocab_size, n_embd, n_head, n_layers, block_size, dropout, device):
        super().__init__()

        self.block_size = block_size
        self.device = device

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layers)]
        )

        self.ln_f = nn.LayerNorm(n_embd)

        # Linear layer to project embeddings to vocab_size for logits
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.token_embedding_table.weight

        # Weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, y=None):
        """ Compute logits and (optionally) the training loss. """
        B, T = x.shape

        tok_emb = self.token_embedding_table(x)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))

        emb = tok_emb + pos_emb

        emb = self.blocks(emb)
        emb = self.ln_f(emb)
        logits = self.lm_head(emb)

        if y is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            y = y.view(B * T)
            loss = F.cross_entropy(logits, y)

        return logits, loss

    def generate(self, x, max_new_tokens):
        """ Utility method for sequential generation. """
        self.eval()

        for _ in range(max_new_tokens):
            x_cond = x[:, -self.block_size:]
            logits, _ = self.forward(x_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            x_next = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, x_next), dim=1)

        return x