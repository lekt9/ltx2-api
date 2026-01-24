"""Simplified attention implementation for LTX-2 pipeline."""
import torch
import torch.nn.functional as F

__all__ = ["pay_attention"]


@torch.compiler.disable()
def pay_attention(
    qkv_list,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    version=None,
    force_attention=None,
    attention_mask=None,
    cross_attn=False,
    q_lens=None,
    k_lens=None,
):
    """Simplified attention using PyTorch's scaled_dot_product_attention."""
    q, k, v = qkv_list
    qkv_list.clear()
    out_dtype = q.dtype

    batch = len(q)
    if len(k) != batch:
        k = k.expand(batch, -1, -1, -1)
    if len(v) != batch:
        v = v.expand(batch, -1, -1, -1)

    # Use scaled_dot_product_attention (SDPA)
    q = q.transpose(1, 2)  # B, H, T, D
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    if attention_mask is not None:
        attention_mask = attention_mask.transpose(1, 2)

    x = F.scaled_dot_product_attention(
        q, k, v, attn_mask=attention_mask, is_causal=causal, dropout_p=dropout_p
    )
    x = x.transpose(1, 2)  # B, T, H, D

    return x.type(out_dtype)
