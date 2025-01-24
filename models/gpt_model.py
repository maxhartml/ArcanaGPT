"""
models/gpt_model.py

Implements the GPT (Generative Pre-trained Transformer) architecture and configuration.
"""

from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect

@dataclass
class GPTConfig:
    """
    Holds configuration parameters for the GPT model.
    """
    block_size: int = 1024               # Maximum sequence length
    vocab_size: int = 50257             # Vocabulary size
    n_layer: int = 12                   # Number of transformer blocks
    n_head: int = 6                     # Number of attention heads
    n_embd: int = 384                   # Embedding dimension

class CausalSelfAttention(nn.Module):
    """
    A multi-head causal self-attention module.
    Uses PyTorch's scaled_dot_product_attention with is_causal=True for Flash Attention.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # Projection layers for Q, K, V.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1  # Used in custom initialisation

        # Causal mask buffer
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch size, sequence length, embedding dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape for multi-head attention
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        # Use Flash Attention
        # Equivalent to manually computing QK^T and applying causal mask
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Reassemble heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Final linear projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    """
    A feed-forward network (MLP) used in transformer blocks.
    Increases dimensionality and then projects back down.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')  # Approximate version for speed
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    """
    A single Transformer block containing:
    - LayerNorm
    - Causal Self-Attention
    - MLP
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    """
    The full GPT language model:
    - Token + Position embeddings
    - A sequence of transformer blocks
    - A final layer norm
    - A projection head for language modelling
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config.vocab_size, config.n_embd),
            'wpe': nn.Embedding(config.block_size, config.n_embd),
            'h': nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            'ln_f': nn.LayerNorm(config.n_embd),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # Parameter initialisation
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Custom weight initialisation for better stability."""
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        """
        Forward pass of the model.
        idx: (B, T) input token indices
        targets: (B, T) optional, for language modelling loss
        """
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is max {self.config.block_size}."

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb

        # Pass through blocks
        for block in self.transformer['h']:
            x = block(x)

        # Final layer norm + output projection
        x = self.transformer['ln_f'](x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type: str):
        """
        Load pretrained GPT-2 model weights from Hugging Face.
        """
        from transformers import GPT2LMHeadModel

        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        print(f"Loading weights from pretrained GPT: {model_type}")

        # Map GPT-2 variants to known settings
        config_args_map = {
            'gpt2':        dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large':  dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl':     dict(n_layer=48, n_head=25, n_embd=1600),
        }
        config_args = config_args_map[model_type]
        print("Forcing vocab_size=50257, block_size=1024")
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024

        # Create a new GPT instance
        config = GPTConfig(**config_args)
        model = cls(config)
        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')]

        # Load HF GPT-2
        hf_model = GPT2LMHeadModel.from_pretrained(model_type)
        hf_sd = hf_model.state_dict()
        hf_sd_keys = [k for k in hf_sd.keys() if not k.endswith('.attn.masked_bias') and not k.endswith('.attn.bias')]

        # Some weights need to be transposed
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(hf_sd_keys) == len(sd_keys), f"Mismatched keys: {len(hf_sd_keys)} != {len(sd_keys)}"

        for k in hf_sd_keys:
            if any(k.endswith(w) for w in transposed):
                assert hf_sd[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(hf_sd[k].t())
            else:
                assert hf_sd[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(hf_sd[k])

        return model

    def configure_optimizers(self, weight_decay: float, learning_rate: float, device: str):
        """
        Creates and returns an AdamW optimizer with optional fused support.
        """
        # Gather parameters that require grad
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        # Split into decayed vs. non-decayed
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]

        # Debug prints
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"Decayed param tensors: {len(decay_params)} ({num_decay_params:,} parameters)")
        print(f"Non-decayed param tensors: {len(nodecay_params)} ({num_nodecay_params:,} parameters)")

        # Use fused AdamW if available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and ('cuda' in device)
        print(f"Using fused AdamW: {use_fused}")

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=use_fused
        )
        return optimizer