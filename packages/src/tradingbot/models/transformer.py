"""Decoder-only Transformer models for sequence prediction."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F


OutputType = Literal["classification", "regression"]
TransformerArchitecture = Literal["decoder_only"]


def apply_rope(x: Tensor) -> Tensor:
    """Apply rotary positional embeddings to attention query/key tensors."""
    _, _, time, head_dim = x.shape
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires an even attention head dimension.")

    position = torch.arange(time, device=x.device, dtype=x.dtype)
    inv_freq = 1.0 / (
        10000
        ** (torch.arange(0, head_dim, 2, device=x.device, dtype=x.dtype) / head_dim)
    )
    angles = position[:, None] * inv_freq[None, :]
    cos = angles.cos()[None, None, :, :]
    sin = angles.sin()[None, None, :, :]

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotated = torch.stack(
        [x_even * cos - x_odd * sin, x_even * sin + x_odd * cos],
        dim=-1,
    )
    return rotated.flatten(-2)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention for decoder-only Transformers."""

    def __init__(self, model_dim: int, num_heads: int, dropout: float):
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads.")

        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError(
                "model_dim / num_heads must be even when using rotary embeddings."
            )

        self.qkv = nn.Linear(model_dim, 3 * model_dim)
        self.out = nn.Linear(model_dim, model_dim)
        self.dropout_p = dropout

    def forward(self, x: Tensor) -> Tensor:
        batch, time, channels = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(batch, time, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, time, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, time, self.num_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q)
        k = apply_rope(k)

        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, time, channels)
        return self.out(attended)


class TransformerBlock(nn.Module):
    """Pre-norm decoder block with causal attention and feed-forward network."""

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        dropout: float,
        ffn_multiplier: int = 4,
    ):
        super().__init__()
        self.attn_norm = nn.LayerNorm(model_dim)
        self.attn = CausalSelfAttention(model_dim, num_heads, dropout)
        self.ffn_norm = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, ffn_multiplier * model_dim),
            nn.GELU(),
            nn.Linear(ffn_multiplier * model_dim, model_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class DecoderOnlyTransformer(nn.Module):
    """Decoder-only Transformer for classification or regression over windows."""

    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        output_dim: int,
        output_type: OutputType = "classification",
        ffn_multiplier: int = 4,
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if model_dim <= 0:
            raise ValueError("model_dim must be positive.")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if not 0 <= dropout < 1:
            raise ValueError(
                "dropout must be greater than or equal to 0 and less than 1."
            )
        if output_dim <= 0:
            raise ValueError("output_dim must be positive.")
        if output_type not in {"classification", "regression"}:
            raise ValueError(
                "output_type must be either 'classification' or 'regression'."
            )

        self.input_dim = input_dim
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_dim = output_dim
        self.output_type = output_type

        self.input_projection = nn.Linear(input_dim, model_dim)
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(model_dim, num_heads, dropout, ffn_multiplier)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(model_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(model_dim * 2),
            nn.Linear(model_dim * 2, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_projection(x)
        x = self.blocks(x)
        x = self.final_norm(x)
        last_context_token = x[:, -1, :]
        mean_context_token = x.mean(dim=1)
        pooled = torch.cat([last_context_token, mean_context_token], dim=-1)
        return self.classifier(pooled)

    def predict_proba(self, x: Tensor) -> Tensor:
        """Return class probabilities for classification models."""
        if self.output_type != "classification":
            raise RuntimeError(
                "predict_proba is only available for classification models."
            )
        return torch.softmax(self(x), dim=-1)


class CandleGPT(DecoderOnlyTransformer):
    """Compatibility wrapper for the original candle GPT notebook model."""

    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        output_classes: int,
    ):
        super().__init__(
            input_dim=input_dim,
            model_dim=model_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            output_dim=output_classes,
            output_type="classification",
        )
        self.output_classes = output_classes


def build_transformer(
    *,
    input_dim: int,
    model_dim: int,
    num_heads: int,
    num_layers: int,
    dropout: float,
    output_dim: int | None = None,
    output_classes: int | None = None,
    output_type: OutputType = "classification",
    architecture: TransformerArchitecture = "decoder_only",
    ffn_multiplier: int = 4,
) -> DecoderOnlyTransformer:
    """Build a Transformer model for sequence classification or regression.

    For classification, pass either ``output_classes`` or ``output_dim``. For
    regression, ``output_dim`` is the number of predicted continuous values and
    defaults to ``1``.
    """
    if architecture != "decoder_only":
        raise ValueError("Only decoder_only Transformer architecture is supported.")
    if output_type not in {"classification", "regression"}:
        raise ValueError("output_type must be either 'classification' or 'regression'.")

    if output_type == "classification":
        if (
            output_classes is not None
            and output_dim is not None
            and output_classes != output_dim
        ):
            raise ValueError("output_classes and output_dim must match when both are set.")
        resolved_output_dim = output_classes if output_classes is not None else output_dim
        if resolved_output_dim is None:
            raise ValueError(
                "Classification transformers require output_classes or output_dim."
            )
    else:
        if output_classes is not None:
            raise ValueError(
                "Regression transformers should use output_dim, not output_classes."
            )
        resolved_output_dim = 1 if output_dim is None else output_dim

    return DecoderOnlyTransformer(
        input_dim=input_dim,
        model_dim=model_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        output_dim=resolved_output_dim,
        output_type=output_type,
        ffn_multiplier=ffn_multiplier,
    )
