"""
evaluation/hellaswag_eval.py

Contains logic for downloading HellaSwag data, rendering examples,
and evaluating a model's predictions.
"""

import os
import json
import requests
import torch
import torch.nn.functional as F
import tiktoken
from tqdm import tqdm

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "hellaswag")

hellaswags = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val":   "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test":  "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

enc = tiktoken.get_encoding("gpt2")

def download_file(url: str, fname: str, chunk_size=1024):
    """Helper function to download a file from a URL with a progress bar."""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)

def download_hellaswag(split: str):
    """
    Ensures HellaSwag for the specified split is downloaded locally.
    """
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    data_url = hellaswags[split]
    data_filename = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)

def render_example(example: dict):
    """
    Given one HellaSwag example, returns:
    - data: helper dictionary with tokens,
    - tokens: 4xN tensor of token IDs (for 4 endings),
    - mask: 4xN tensor that indicates which positions belong to the ending,
    - label: index of the correct ending.
    """
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]

    ctx_tokens = enc.encode(ctx)
    tok_rows = []
    mask_rows = []

    for end in endings:
        end_tokens = enc.encode(" " + end)  # prepend space
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0] * len(ctx_tokens) + [1] * len(end_tokens))

    max_len = max(len(r) for r in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)

    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)

    data = {
        "label": label,
        "ctx_tokens": ctx_tokens,
        "ending_tokens": [enc.encode(" " + e) for e in endings],
    }
    return data, tokens, mask, label

def iterate_examples(split: str):
    """
    Yields one HellaSwag example at a time (loaded from local .jsonl).
    """
    download_hellaswag(split)
    filename = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")
    with open(filename, "r") as f:
        for line in f:
            yield json.loads(line)

def get_most_likely_row(tokens: torch.Tensor, mask: torch.Tensor, logits: torch.Tensor) -> int:
    """
    Evaluates the autoregressive loss for each row's completion region
    and returns the index of the row with the lowest average loss.
    """
    # shift
    shift_logits = logits[..., :-1, :].contiguous()
    shift_tokens = tokens[..., 1:].contiguous()

    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)

    # shift mask (one position to right)
    shift_mask = mask[..., 1:].contiguous()
    masked_shift_losses = shift_losses * shift_mask

    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    pred_norm = avg_loss.argmin().item()
    return pred_norm