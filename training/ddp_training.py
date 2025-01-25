"""
training/ddp_training.py

Sets up Distributed Data Parallel (DDP) training and runs the main loop.
"""

import os
import math
import time
import torch
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
import tiktoken

# Local imports
from data.data_loader import DataLoaderLite
from evaluation.hellaswag_eval import iterate_examples, render_example, get_most_likely_row
from models.gpt_model import GPT, GPTConfig

# Expose these for usage
ddp_rank = 0
ddp_local_rank = 0
ddp_world_size = 1
master_process = True

def setup_ddp_environment():
    """
    Checks environment variables to see if we are in a DDP run.
    Initialises the process group if so.
    Returns ddp_rank, ddp_local_rank, ddp_world_size, master_process, device.
    """
    ddp_active = int(os.environ.get('RANK', -1)) != -1
    if ddp_active:
        assert torch.cuda.is_available(), "DDP currently requires CUDA."
        init_process_group(backend='nccl')

        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        device_str = f'cuda:{local_rank}'
        torch.cuda.set_device(device_str)
        master = (rank == 0)

        return rank, local_rank, world_size, master, device_str
    else:
        device_str = "cpu"
        if torch.cuda.is_available():
            device_str = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_str = "mps"
        print(f"Using device: {device_str}")

        return 0, 0, 1, True, device_str

def get_learning_rate(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    """
    Implements a learning rate schedule with linear warmup and cosine decay.
    """
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr

    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

def main_training_loop():
    """
    Main entry point for DDP training. 
    Runs the training loop, evaluations (validation + HellaSwag),
    and optionally logs outputs.
    """
    global ddp_rank, ddp_local_rank, ddp_world_size, master_process

    ddp_rank, ddp_local_rank, ddp_world_size, master_process, device = setup_ddp_environment()

    # Set seeds
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    # Hyperparameters
    total_batch_size = 524288  # 2**19 tokens ~ 0.5M
    B = 8                     # micro-batch size
    T = 1024                   # sequence length
    assert total_batch_size % (B * T * ddp_world_size) == 0, \
        "Ensure total_batch_size is divisible by B*T*ddp_world_size"
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

    if master_process:
        print(f"Total desired batch size: {total_batch_size}")
        print(f"Calculated gradient accumulation steps: {grad_accum_steps}")

    # Create data loaders
    train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='train')
    val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='val')

    # Optimise matrix multiplies
    torch.set_float32_matmul_precision('high')

    # Create model (optionally load GPT-2 weights)
    # model = GPT.from_pretrained('gpt2')
    model = GPT(GPTConfig(vocab_size=50304))
    model.to(device)

    # Tiktoken encoder
    enc = tiktoken.get_encoding('gpt2')

    # Optional compile
    use_compile = False  # torch.compile can cause issues with certain eval
    if use_compile:
        print("Compiling model with torch.compile()...")
        model = torch.compile(model)  # require CUDA

    # Wrap model in DDP if needed
    if ddp_world_size > 1:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp_world_size > 1 else model

    # LR schedule params
    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warmup_steps = 715
    max_steps = 19073

    # Create optimiser
    optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device, master_process=master_process)

    # Logging
    log_dir = 'log'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "log.txt")
    with open(log_file, "w") as f:
        pass

    for step in range(max_steps):
        step_start = time.time()
        last_step = (step == max_steps - 1)

        # Periodic validation
        if step % 100 == 0:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 20
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        logits, loss = model(x, y)
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()

            if ddp_world_size > 1:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)

            if master_process:
                val_loss_item = val_loss_accum.item()
                print(f"Validation loss: {val_loss_item:.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} val {val_loss_item:.4f}\n")

                # Checkpointing
                if step > 0 and (step % 5000 == 0 or last_step):
                    ckpt_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'config': raw_model.config,
                        'step': step,
                        'val_loss': val_loss_item,
                    }
                    torch.save(checkpoint, ckpt_path)

        # Periodic HellaSwag eval
        if (step % 250 == 0 or last_step) and (not use_compile):
            model.eval()
            num_correct_norm = 0
            num_total = 0

            for i, example in enumerate(iterate_examples("val")):
                # Partition tasks by rank
                if i % ddp_world_size != ddp_rank:
                    continue
                _, tokens, mask, label = render_example(example)
                tokens, mask = tokens.to(device), mask.to(device)

                with torch.no_grad():
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        logits, _ = model(tokens)
                    pred_norm = get_most_likely_row(tokens, mask, logits)

                num_total += 1
                num_correct_norm += int(pred_norm == label)

            # All-reduce stats
            if ddp_world_size > 1:
                num_total_t = torch.tensor(num_total, dtype=torch.long, device=device)
                num_correct_norm_t = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
                dist.all_reduce(num_total_t, op=dist.ReduceOp.SUM)
                dist.all_reduce(num_correct_norm_t, op=dist.ReduceOp.SUM)
                num_total = num_total_t.item()
                num_correct_norm = num_correct_norm_t.item()

            if master_process:
                acc_norm = num_correct_norm / num_total
                print(f"HellaSwag accuracy: {num_correct_norm}/{num_total} = {acc_norm:.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} hella {acc_norm:.4f}\n")

        # Periodic text generation
        if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile):
            model.eval()
            num_return_sequences = 4
            max_length = 32
            prompt = "Hello, World! I am stuck inside the computer,"
            tokens_init = enc.encode(prompt)
            tokens_init = torch.tensor(tokens_init, dtype=torch.long).unsqueeze(0)
            tokens_init = tokens_init.repeat(num_return_sequences, 1).to(device)

            sample_rng = torch.Generator(device=device)
            sample_rng.manual_seed(42 + ddp_rank)
            xgen = tokens_init

            while xgen.size(1) < max_length:
                with torch.no_grad():
                    logits, _ = model(xgen)
                    logits = logits[:, -1, :]  # last position
                    probs = F.softmax(logits, dim=-1)
                    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                    ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
                    xcol = torch.gather(topk_indices, -1, ix)
                    xgen = torch.cat((xgen, xcol), dim=1)

            # Print the generated text
            for i in range(num_return_sequences):
                out_tokens = xgen[i, :max_length].tolist()
                decoded = enc.decode(out_tokens)
                print(f"rank {ddp_rank} sample {i}: {decoded}")

        # Training step
        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0

        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()

            # Synchronise gradients only on last micro-step in gradient accumulation
            if ddp_world_size > 1:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)

            loss.backward()

        if ddp_world_size > 1:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_learning_rate(step, warmup_steps, max_steps, max_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.step()
        torch.cuda.synchronize()

        step_end = time.time()
        dt = (step_end - step_start)
        tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
        tokens_per_sec = tokens_processed / dt

        if master_process:
            train_loss_value = loss_accum.item()
            print(f"Step: {step:4d} | Loss: {train_loss_value:.6f} | LR: {lr:.4e} "
                  f"| GradNorm: {norm:.4f} | Time: {dt*1000:.2f}ms | Tok/sec: {tokens_per_sec:.2f}")
            with open(log_file, "a") as f:
                f.write(f"{step} train {train_loss_value:.6f}\n")

    # Cleanup
    if ddp_world_size > 1:
        destroy_process_group()