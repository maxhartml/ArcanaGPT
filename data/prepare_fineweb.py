"""
data/prepare_fineweb.py

Downloads the fineweb-edu dataset from Hugging Face,
tokenises it using tiktoken (GPT-2 encoder),
and saves shards as .npy files for training.
"""

import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"
shard_size = int(1e8)  # 100M tokens per shard, total of 100 shards
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']  # GPT-2 end-of-text token

def tokenize_doc(doc):
    """
    Tokenise a single document and return a NumPy array of uint16 tokens.
    Each doc is delimited by <|endoftext|>.
    """
    tokens = [eot]  # delimiter
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens, dtype=np.uint16)
    return tokens_np

def write_shard_file(filename: str, tokens_np: np.ndarray):
    """Saves a .npy shard file."""
    print(f"Saving shard -> {filename}")
    np.save(filename, tokens_np)

if __name__ == "__main__":
    nprocs = max(1, os.cpu_count() // 2)
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        prog_bar = None

        for tokens in pool.imap(tokenize_doc, fw, chunksize=16):
            if token_count + len(tokens) < shard_size:
                # Fits in current shard
                all_tokens_np[token_count:token_count + len(tokens)] = tokens
                token_count += len(tokens)
                if prog_bar is None:
                    prog_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                prog_bar.update(len(tokens))
            else:
                # Write current shard
                remainder = shard_size - token_count
                prog_bar.update(remainder)
                all_tokens_np[token_count:token_count + remainder] = tokens[:remainder]

                split_name = "val" if shard_index == 0 else "train"
                shard_path = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split_name}_{shard_index:06d}")
                write_shard_file(shard_path, all_tokens_np)

                shard_index += 1
                prog_bar = None

                # Start new shard with leftover tokens
                leftover_size = len(tokens) - remainder
                all_tokens_np[0:leftover_size] = tokens[remainder:]
                token_count = leftover_size

        # Write last shard if not empty
        if token_count != 0:
            split_name = "val" if shard_index == 0 else "train"
            shard_path = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split_name}_{shard_index:06d}")
            write_shard_file(shard_path, all_tokens_np[:token_count])