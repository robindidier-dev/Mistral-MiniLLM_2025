
"""
Byte-Level BPE Tokenizer.

Minimal Byte-Pair Encoding (BPE) tokenizer, starting from raw UTF-8 bytes (0-255)
and learning merge rules to construct a compact subword vocabulary.

Inspired by Andrej Karpathy's educational tokenizer in:
"Let's build the GPT tokenizer" (https://www.youtube.com/watch?v=zduSFxRajkE)

Modifications of the Kapathy version include:
    - training over multiple docs
    - use of linked list for efficient merges
    - global pair statistics to avoid full rescans at each merge
    - support of special tokens (doc separation, role user/agent...)
    - max heap over frequency to avoid costly max() over pair stats
    - saving as json merge rules for later use

Author: Robin
Date:   December 2025
"""

import re
import heapq
import json
from data_utils import load_documents

# =========================
# Hyperparameters
# =========================

VOCAB_SIZE = 5000  # Target vocabulary size 

SPECIAL_TOKENS = {
    "<|pad|>": 256,
    "<|unk|>": 257,
    "<|bos|>": 258,
    "<|eos|>": 259,
    "<|doc|>": 260,
    "<|user|>": 261,
    "<|assistant|>": 262,
    "<|md_heading|>": 263,   # "###"
    "<|md_bold|>": 264,      # "**"
    "<|md_italic|>": 265,    # "*"
    "<|md_sep|>": 266,       # "---"
    "<|md_code|>": 267,      # "```"
}


# =========================
# Data structures
# =========================

NUM_SPECIAL = len(SPECIAL_TOKENS)
FIRST_MERGE_ID = 256 + NUM_SPECIAL


class Node:
    """ Class used in Document linked list representation. """

    __slots__ = ("token", "prev", "next")

    def __init__(self, token):
        self.token = token
        self.prev = None
        self.next = None


class Document:
    """ We represent documents as linked list of tokens for efficient merging. """

    def __init__(self, tokens):
        self.head = None
        self.tail = None
        self._build(tokens)

    def _build(self, tokens):
        prev = None
        for t in tokens:
            node = Node(t)
            if prev is None:
                self.head = node
            else:
                prev.next = node
                node.prev = prev
            prev = node
        self.tail = prev


# =========================
# Tokenizer class
# =========================

class Tokenizer:
    """ Byte-level BPE tokenizer with special tokens and persisted merges. 
    
    -> Reusablility (trains once, saves/loads its merges, encode/decode are methods).
    """

    def __init__(self, vocab_size=VOCAB_SIZE):
        self.vocab_size = vocab_size
        self.merges = {}

    def train(self, docs):
        """ Train BPE merges on a list of documents. """
        self.merges = train_bpe(docs, vocab_size=self.vocab_size)

    def encode(self, text):
        """ Encode text into token ids using learned merges. """
        return encode(text, self.merges)

    def decode(self, tokens):
        """ Decode token ids back into text using learned merges. """
        return decode(tokens, self.merges)

    def save(self, path):
        """ Save tokenizer configuration and merges to disk. """
        merges_list = [[a, b, new_token] for (a, b), new_token in self.merges.items()]
        data = {
            "vocab_size": self.vocab_size,
            "merges": merges_list,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        """
        Load a saved tokenizer from disk.
        Reconstructs internal structures without retraining.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tokenizer = cls.__new__(cls)
        tokenizer.vocab_size = data["vocab_size"]

        merges_list = data["merges"]
        tokenizer.merges = {(a, b): new_token for a, b, new_token in merges_list}

        return tokenizer


# =========================
# Utilities
# =========================

def preprocess_special_tokens(text):
    """ Replace frequent structural patterns (Markdown, roles, doc separators)
    with explicit special tokens before byte-level encoding.

    This ensures stable, non-ambiguous tokens for the BPE training.
    """

    # 1. Document separator 
    #    We normalize any variant of <|doc|> to the canonical form.
    text = text.replace("<|doc|>", " <|doc|> ")

    # 2. Conversation roles 
    text = text.replace("<|user|>", " <|user|> ")
    text = text.replace("<|assistant|>", " <|assistant|> ")

    # 3. Markdown patterns 
    # Code fences ```
    text = text.replace("```", " <|md_code|> ")

    # Horizontal rule ---
    text = text.replace("---", " <|md_sep|> ")

    # Headings ###
    text = text.replace("###", " <|md_heading|> ")

    # Bold **
    # Use regex to avoid catching "***" or "****"
    text = re.sub(r"\*\*(?!\*)", " <|md_bold|> ", text)

    # Italic *
    # Only replace isolated '*' not part of '**'
    text = re.sub(r"(?<!\*)\*(?!\*)", " <|md_italic|> ", text)

    # Normalize spacing (avoid token collisions)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def build_linked_document(text):
    """ Encode text into bytes and wrap it into a linked-list Document.
    
    Args:
        text (str): raw text string
    
    Returns:
        Document: linked-list representation of the tokenized document
    """

    text = preprocess_special_tokens(text)
    tokens = list(text.encode("utf-8"))
    tokens.append(SPECIAL_TOKENS["<|doc|>"])
    return Document(tokens)


def initialize_pair_stats(token_docs):
    """ Build initial pair_stats and pair_occurrences from linked-list documents. """
    pair_stats = {}         # (a, b) -> frequency
    pair_occurrences = {}   # (a, b) -> SET of Nodes for O(1) lookup

    for doc in token_docs:
        node = doc.head
        while node is not None and node.next is not None:
            pair = (node.token, node.next.token)

            # Update frequency
            pair_stats[pair] = pair_stats.get(pair, 0) + 1

            # Update occurrences (use set for O(1) membership test)
            if pair not in pair_occurrences:
                pair_occurrences[pair] = set()
            pair_occurrences[pair].add(node)

            node = node.next

    return pair_stats, pair_occurrences


def remove_pair_occurrence(p, node, pair_stats, pair_occurrences):
    """ HELPER FOR apply_merge.

    Remove one occurrence of pair p at node.
    """
    if p not in pair_occurrences:
        return

    occ_set = pair_occurrences[p]
    if node not in occ_set:
        return

    occ_set.discard(node)
    pair_stats[p] -= 1

    if pair_stats[p] <= 0:
        pair_stats.pop(p, None)
        pair_occurrences.pop(p, None)


def add_pair_occurrence(p, node, pair_stats, pair_occurrences, heap):
    """ HELPER FOR apply_merge.
    
    Add one occurrence of pair p at node.
    """

    old_freq = pair_stats.get(p, 0)
    new_freq = old_freq + 1
    pair_stats[p] = new_freq
    pair_occurrences.setdefault(p, set()).add(node)

    if new_freq > 0:
        heapq.heappush(heap, (-new_freq, p))


def apply_single_merge(node, a, b, new_token, pair_stats, pair_occurrences, heap):
    """ HELPER FOR apply_merge.

    Apply merge at a single occurrence node. 
    """

    if node is None or node.next is None:
        return
    if node.token != a or node.next.token != b:
        return

    left = node.prev
    right = node.next.next
    old_right_node = node.next

    # Remove old neighboring pairs
    if left is not None:
        remove_pair_occurrence((left.token, a), left, pair_stats, pair_occurrences)
    if right is not None:
        remove_pair_occurrence((b, right.token), old_right_node, pair_stats, pair_occurrences)

    # Perform merge in linked list
    node.token = new_token
    node.next = right
    if right is not None:
        right.prev = node

    # Add new neighboring pairs
    if left is not None:
        add_pair_occurrence((left.token, new_token), left, pair_stats, pair_occurrences, heap)
    if right is not None:
        add_pair_occurrence((new_token, right.token), node, pair_stats, pair_occurrences, heap)


def apply_merge(pair, new_token, pair_stats, pair_occurrences, heap):
    """ Apply the merge (a, b) -> new_token incrementally.

    For each occurrence of (a, b):
        - replace a by new_token
        - remove b from the linked list
        - update pair_stats and pair_occurrences only for neighboring pairs:
            (x, a) -> (x, new_token)
            (b, y) -> (new_token, y)

    This avoids rescanning the corpus and keeps updates local.
    """

    a, b = pair

    # If no occurrences remain, clean up and exit
    if pair not in pair_occurrences:
        pair_stats.pop(pair, None)
        return

    # Retrieve and remove occurrences of (a, b)
    occurrences = pair_occurrences.pop(pair)
    pair_stats.pop(pair, None)

    # Apply merge to each occurrence
    for node in occurrences:
        apply_single_merge(node, a, b, new_token, pair_stats, pair_occurrences, heap)


# =========================
# Tokenizer training
# =========================

def train_bpe(docs, vocab_size=VOCAB_SIZE):
    """
    Train the tokenizer over multiple documents.

    Args:
        docs: list of raw text documents (strings)
        vocab_size: target vocabulary size

    Returns:
        merges: dict (a, b) -> new_token
    """

    # Build linked-list documents
    token_docs = [build_linked_document(doc) for doc in docs]

    # Initialize global pair statistics
    pair_stats, pair_occurrences = initialize_pair_stats(token_docs)

    merges = {}
    num_merges = vocab_size - 256 - len(SPECIAL_TOKENS)

    # Use a max heap (negate for min heap) to avoid max() function (O(n))
    # Heap stores: (-frequency, pair)
    heap = [(-freq, pair) for pair, freq in pair_stats.items()]
    heapq.heapify(heap)

    for i in range(num_merges):

        if i % 500 == 0 or i == num_merges - 1:
            print(f"Training BPE: {i}/{num_merges} merges ({100*i/num_merges:.1f}%)")

        # Find the most frequent pair with valid occurrences
        found = False
        while heap:
            neg_freq, pair = heapq.heappop(heap)
            
            # Skip if pair no longer exists or frequency is stale
            if pair not in pair_stats:
                continue
            if pair_stats[pair] != -neg_freq:
                continue  # Stale entry
            
            # Valid pair found
            found = True
            break
        
        if not found:
            print("No more pairs to merge.")
            break

        new_token = FIRST_MERGE_ID + i

        # Apply merge incrementally 
        apply_merge(pair, new_token, pair_stats, pair_occurrences, heap)

        merges[pair] = new_token

    print("Training complete.")
    return merges


# =========================
# Encoding
# =========================

def encode(text, merges):
    """
    Encode text into BPE tokens:
      - detect special tokens
      - byte-encode normal text
      - apply BPE merges
    """

    # Convert text into initial token list (special tokens + bytes) 
    special_to_id = SPECIAL_TOKENS
    special_tokens = sorted(special_to_id.keys(), key=len, reverse=True)

    tokens = []
    i = 0

    while i < len(text):
        # Try to match a special token
        for sp in special_tokens:
            if text.startswith(sp, i):
                tokens.append(special_to_id[sp])
                i += len(sp)
                break
        else:
            # No special token -> byte encode
            b = text[i].encode("utf-8")
            tokens.extend(b)
            i += 1

    # Apply BPE merges
    merged = True
    while merged:
        merged = False
        new_tokens = []
        i = 0

        while i < len(tokens):
            if i + 1 < len(tokens) and (tokens[i], tokens[i+1]) in merges:
                new_tokens.append(merges[(tokens[i], tokens[i+1])])
                i += 2
                merged = True
            else:
                new_tokens.append(tokens[i])
                i += 1

        tokens = new_tokens

    return tokens


# =========================
# Decoding
# =========================

def decode(tokens, merges):
    """
    Decode BPE tokens back into a UTF-8 string.
    Special tokens are restored textually.
    """

    id_to_special = {v: k for k, v in SPECIAL_TOKENS.items()}

    inverse = {v: k for k, v in merges.items()}

    def expand(t):
        # Special token -> return textual form
        if t in id_to_special:
            return id_to_special[t]

        # Raw byte
        if t < 256:
            return bytes([t])

        # BPE merge token -> recursively expand
        a, b = inverse[t]
        return expand(a) + expand(b)

    # Build output
    out = []
    for t in tokens:
        piece = expand(t)
        if isinstance(piece, bytes):
            out.append(piece.decode("utf-8", errors="replace"))
        else:
            out.append(piece)  # special token string

    return "".join(out)


if __name__ == "__main__":
    """ Minimal test driver for the BPE tokenizer.
    
    Trains the tokenizer on the corpus, sample merges and bijectivity checks.
    """

    # Load corpus documents
    folder = "corpus_docs"
    docs = load_documents(folder)

    # Train BPE on the corpus
    tokenizer = Tokenizer(vocab_size=VOCAB_SIZE)
    tokenizer.train(docs)
    merges = tokenizer.merges

    print(f"Learned {len(merges)} merge rules.")

    # Visualize some learned merges
    print("Sample Learned Merges")
    id_to_special = {v: k for k, v in SPECIAL_TOKENS.items()}
    inverse_merges = {v: k for k, v in merges.items()}

    def decode_token_to_bytes(t):
        """ Recursively decode a token to bytes. """
        if t in id_to_special:
            return id_to_special[t].encode('utf-8')
        if t < 256:
            return bytes([t])
        if t in inverse_merges:
            a, b = inverse_merges[t]
            return decode_token_to_bytes(a) + decode_token_to_bytes(b)
        return b''

    def bytes_to_repr(b):
        """ Show bytes as readable string (with special handling for space). """
        try:
            text = b.decode('utf-8')
            if text == ' ':
                return '[SPACE]'
            elif text.isprintable():
                return repr(text)
            else:
                return f'[byte {b[0]}]'
        except:
            return f'[bytes {b.hex()}]'

    merge_list = list(merges.items())
    for idx, ((a, b), new_token) in enumerate(merge_list[:20]):
        a_bytes = decode_token_to_bytes(a)
        b_bytes = decode_token_to_bytes(b)
        result_bytes = decode_token_to_bytes(new_token)
        
        a_repr = bytes_to_repr(a_bytes)
        b_repr = bytes_to_repr(b_bytes)
        result_repr = bytes_to_repr(result_bytes)
        
        print(f"Merge {idx}: {a_repr} + {b_repr} -> {result_repr}")

    # Bijectivity test
    print("Bijectivity Test")

    sample_texts = [
        "Bonjour le monde",
        "Ceci est un test **important**",
        "### Titre principal",
        "<|doc|> Séparateur explicite",
        "Texte avec --- séparateur Markdown",
        "Bloc de code ```print('hello')```",
        "<|user|> Bonjour <|assistant|> Salut",
    ]

    for text in sample_texts:
        # Preprocess special tokens before encoding
        pre = preprocess_special_tokens(text)

        enc = encode(pre, merges)
        dec = decode(enc, merges)

        ok = (dec == pre)

        print(f"\nText: {text!r}")
        print(f"Preprocessed: {pre!r}")
        print(f"Encoded: {enc[:30]}{'...' if len(enc)>30 else ''}")
        print(f"Decoded: {dec!r}")
        print(f"Bijective: {ok}")