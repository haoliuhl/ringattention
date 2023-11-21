import numpy as np
import flax.linen as nn
import jax
import jax.lax as lax
import jax.numpy as jnp
from einops import rearrange
from functools import partial
from llamabpt.flash_attention_tpu import _flash_attention_fwd, _flash_attention_bwd, BlockSizes


"""
Ring Attention with (a) scan and (b) pallas
"""
def _ring_attention_fwd(q, k, v, attn_bias, axis_name, float32_logits, blockwise_kwargs):
    if float32_logits:
        q, k = q.astype(jnp.float32), k.astype(jnp.float32)
    batch, q_len, num_heads, dim_per_head = q.shape
    batch, kv_len, num_heads, dim_per_head = k.shape
    numerator = jnp.zeros((batch, q_len, num_heads, dim_per_head)).astype(q.dtype)
    denominator = jnp.zeros((batch, num_heads, q_len)).astype(q.dtype)
    axis_size = lax.psum(1, axis_name)
    block_size = q_len # assumes this function is pre-sharded inside shard_map
    query_chunk_size = blockwise_kwargs["query_chunk_size"]
    key_chunk_size = blockwise_kwargs["key_chunk_size"]
    def scan_kv_block(carry, idx):
        prev_max_score, numerator, denominator, k, v = carry
        ''' einsum version
        mask = lax.dynamic_slice_in_dim(attn_mask,
            (lax.axis_index(axis_name) - idx) % axis_size * kv_len, kv_len, axis=-1)
        attn_weights = jnp.einsum("bqhd,bkd->bhqk", q, k) / scale
        attn_weights = jnp.where(mask, -jnp.inf, attn_weights)
        max_score = jnp.maximum(prev_max_score, jnp.max(attn_weights, axis=-1))
        exp_weights = jnp.exp(attn_weights - max_score[..., None])
        correction = rearrange(jnp.exp(prev_max_score - max_score), 'b h q -> b q h')[..., None]
        numerator = numerator * correction + jnp.einsum("bhqk,bkd->bqhd", exp_weights, v)
        denominator = denominator * jnp.exp(prev_max_score - max_score) + jnp.sum(exp_weights, axis=-1)
        '''
        attn_bias_slice = lax.dynamic_slice_in_dim(attn_bias,
            (lax.axis_index(axis_name) - idx) % axis_size * kv_len, kv_len, axis=-1
        )
        q_block_idx = lax.axis_index(axis_name)
        k_block_idx = (lax.axis_index(axis_name) - idx) % axis_size
        q_chunk_idx_start = q_block_idx * (block_size // query_chunk_size)
        k_chunk_idx_start = k_block_idx * (block_size // key_chunk_size)
        numerator, denominator, max_score = _blockwise_attention_fwd(q, k, v, (numerator, denominator, prev_max_score), q_chunk_idx_start, k_chunk_idx_start, bias=attn_bias_slice, **blockwise_kwargs)
        k, v = map(lambda x: lax.ppermute(x, axis_name, perm=[(i, (i + 1) % axis_size) for i in range(axis_size)]), (k, v))
        return (max_score, numerator, denominator, k, v), None
    prev_max_score = jnp.full((batch, num_heads, q_len), -jnp.inf).astype(q.dtype)
    (max_score, numerator, denominator, _, _), _ = lax.scan(scan_kv_block,
        init=(prev_max_score, numerator, denominator, k, v), xs=jnp.arange(0, axis_size))
    output = numerator / rearrange(denominator, 'b h q -> b q h')[..., None]
    return output.astype(v.dtype), (output, q, k, v, attn_bias, denominator, max_score)

def _ring_attention_bwd(axis_name, float32_logits, blockwise_kwargs, res, g):
    del float32_logits
    output, q, k, v, attn_bias, denominator, max_score = res
    batch, q_len, num_heads, dim_per_head = q.shape
    batch, kv_len, num_heads, dim_per_head = k.shape
    axis_size = lax.psum(1, axis_name)
    dq = jnp.zeros_like(q, dtype=q.dtype)
    dk = jnp.zeros_like(k, dtype=k.dtype)
    dv = jnp.zeros_like(v, dtype=k.dtype)
    query_chunk_size = blockwise_kwargs["query_chunk_size"]
    key_chunk_size = blockwise_kwargs["key_chunk_size"]
    block_size = q.shape[1] # assumes this function is pre-sharded inside shard_map
    def scan_kv_block(carry, idx):
        dq, dk, dv, k, v = carry
        ''' einsum version
        # mask = lax.dynamic_slice_in_dim(attn_mask,
        #     (lax.axis_index(axis_name) - idx) % axis_size * kv_len, kv_len, axis=-1)
        # attn_weights = jnp.einsum("bqhd,bkd->bhqk", q, k) / scale
        # attn_weights = jnp.where(mask, -jnp.inf, attn_weights)
        # exp_weights = jnp.exp(attn_weights - max_score[..., None]) / denominator[..., None]
        # ds = jnp.einsum("bqhd,bkd->bhqk", g, v)
        # dl = (ds - jnp.einsum("bqhd,bqhd->bhs", g, output)[..., None]) * exp_weights
        # dq = dq + jnp.einsum("bhqk,bkd->bqhd", dl, k) / scale
        # dk = dk + jnp.einsum("bqhd,bhqk->bkd", q, dl) / scale
        # dv = dv + jnp.einsum("bhqk,bqhd->bkd", exp_weights, g)
        '''
        attn_bias_slice = lax.dynamic_slice_in_dim(attn_bias,
            (lax.axis_index(axis_name) - idx) % axis_size * kv_len, kv_len, axis=-1
        )
        q_block_idx = lax.axis_index(axis_name)
        k_block_idx = (lax.axis_index(axis_name) - idx) % axis_size
        q_chunk_idx_start = q_block_idx * (block_size // query_chunk_size)
        k_chunk_idx_start = k_block_idx * (block_size // key_chunk_size)
        dq, dk, dv = _blockwise_attention_bwd(q, k, v, g, (dq, dk, dv, output, denominator, max_score), q_chunk_idx_start, k_chunk_idx_start, bias=attn_bias_slice, **blockwise_kwargs)
        k, v, dk, dv = map(lambda x: lax.ppermute(x, axis_name, perm=[(i,
            (i + 1) % axis_size) for i in range(axis_size)]), (k, v, dk, dv))
        return (dq, dk, dv, k, v), None
    (dq, dk, dv, k, v), _ = lax.scan(scan_kv_block, init=(dq, dk, dv, k, v), xs=jnp.arange(0, axis_size))
    dq, dk, dv = dq.astype(q.dtype), dk.astype(k.dtype), dv.astype(k.dtype)
    return dq, dk, dv, None

@partial(jax.custom_vjp, nondiff_argnums=[4, 5, 6])
def ring_attention(q, k, v, attn_bias, axis_name, float32_logits, blockwise_kwargs):
    y, _ = _ring_attention_fwd(q, k, v, attn_bias, axis_name, float32_logits, blockwise_kwargs)
    return y

ring_attention.defvjp(_ring_attention_fwd, _ring_attention_bwd)


def _ring_attention_standard_fwd(q, k, v, attn_mask, axis_name, float32_logits):
    if float32_logits:
        q, k = q.astype(jnp.float32), k.astype(jnp.float32)
    batch, q_len, num_heads, _ = q.shape
    batch, kv_len, num_heads, dim_per_head = k.shape
    numerator = jnp.zeros((batch, q_len, num_heads, dim_per_head)).astype(q.dtype)
    denominator = jnp.zeros((batch, num_heads, q_len)).astype(q.dtype)
    axis_size = lax.psum(1, axis_name)
    scale = jnp.sqrt(q.shape[-1])
    def scan_kv_block(carry, idx):
        prev_max_score, numerator, denominator, k, v = carry
        mask = lax.dynamic_slice_in_dim(attn_mask,
            (lax.axis_index(axis_name) - idx) % axis_size * kv_len, kv_len, axis=-1)
        attn_weights = jnp.einsum("bqhd,bkhd->bhqk", q, k) / scale
        # attn_weights = jnp.where(mask, -jnp.inf, attn_weights)
        attn_weights = jnp.where(mask, attn_weights, -jnp.inf)
        max_score = jnp.maximum(prev_max_score, jnp.max(attn_weights, axis=-1))
        exp_weights = jnp.exp(attn_weights - max_score[..., None])
        correction = rearrange(jnp.exp(prev_max_score - max_score), 'b h q -> b q h')[..., None]
        numerator = numerator * correction + jnp.einsum("bhqk,bkhd->bqhd", exp_weights, v)
        denominator = denominator * jnp.exp(prev_max_score - max_score) + jnp.sum(exp_weights, axis=-1)
        k, v = map(lambda x: lax.ppermute(x, axis_name, perm=[(i,
            (i + 1) % axis_size) for i in range(axis_size)]), (k, v))
        return (max_score, numerator, denominator, k, v), None
    prev_max_score = jnp.full((batch, num_heads, q_len), -jnp.inf).astype(q.dtype)
    (max_score, numerator, denominator, _, _), _ = lax.scan(scan_kv_block,
    init=(prev_max_score, numerator, denominator, k, v), xs=jnp.arange(0, axis_size))
    output = numerator / rearrange(denominator, 'b h q -> b q h')[..., None]
    return output.astype(v.dtype), (output, q, k, v, attn_mask, numerator, denominator, max_score)

def _ring_attention_standard_bwd(axis_name, float32_logits, res, g):
    del float32_logits
    axis_size = lax.psum(1, axis_name)
    output, q, k, v, attn_mask, numerator, denominator, max_score = res
    dq = jnp.zeros_like(q, dtype=jnp.float32)
    dk = jnp.zeros_like(k, dtype=jnp.float32)
    dv = jnp.zeros_like(v, dtype=jnp.float32)
    batch, kv_len, num_heads, dim_per_head = k.shape
    scale = jnp.sqrt(q.shape[-1])
    def scan_kv_block(carry, idx):
        dq, dk, dv, k, v = carry
        mask = lax.dynamic_slice_in_dim(attn_mask,
            (lax.axis_index(axis_name) - idx) % axis_size * kv_len, kv_len, axis=-1)
        attn_weights = jnp.einsum("bqhd,bkhd->bhqk", q, k) / scale
        # attn_weights = jnp.where(mask, -jnp.inf, attn_weights)
        attn_weights = jnp.where(mask, attn_weights, -jnp.inf)
        exp_weights = jnp.exp(attn_weights - max_score[..., None]) / denominator[..., None]
        ds = jnp.einsum("bqhd,bkhd->bhqk", g, v)
        dl = (ds - jnp.einsum("bqhd,bqhd->bhq", g, output)[..., None]) * exp_weights
        dq = dq + jnp.einsum("bhqk,bkhd->bqhd", dl, k) / scale
        dk = dk + jnp.einsum("bqhd,bhqk->bkhd", q, dl) / scale
        dv = dv + jnp.einsum("bhqk,bqhd->bkhd", exp_weights, g)
        k, v, dk, dv = map(lambda x: lax.ppermute(x, axis_name, perm=[(i,
            (i + 1) % axis_size) for i in range(axis_size)]), (k, v, dk, dv))
        return (dq, dk, dv, k, v), None
    (dq, dk, dv, k, v), _ = lax.scan(scan_kv_block, init=(dq, dk, dv, k, v), xs=jnp.arange(0, axis_size))
    dq, dk, dv = dq.astype(q.dtype), dk.astype(k.dtype), dv.astype(v.dtype)
    return dq, dk, dv, None

@partial(jax.custom_vjp, nondiff_argnums=[4, 5])
def ring_attention_standard(q, k, v, attn_mask, axis_name, float32_logits=True):
    y, _ = _ring_attention_standard_fwd(q, k, v, attn_mask, axis_name, float32_logits)
    return y

ring_attention_standard.defvjp(_ring_attention_standard_fwd, _ring_attention_standard_bwd)


def _blockwise_attention_fwd(q, k, v, carry, q_chunk_idx_start, k_chunk_idx_start, bias, causal, query_chunk_size,
                             key_chunk_size, deterministic, dropout_rng, attn_pdrop, dtype, policy, precision, prevent_cse):
    batch, q_len, num_heads, dim_per_head = q.shape
    batch, kv_len, num_heads, dim_per_head = k.shape
    batch, kv_len, num_heads, dim_per_head = v.shape
    num_q = q_len // query_chunk_size
    num_kv = kv_len // key_chunk_size
    q = q.reshape((batch, num_q, query_chunk_size, num_heads, dim_per_head))
    k = k.reshape((batch, num_kv, key_chunk_size, num_heads, dim_per_head))
    v = v.reshape((batch, num_kv, key_chunk_size, num_heads, dim_per_head))
    q, k, v = map(lambda x: jnp.moveaxis(x, 1, 0), (q, k, v))

    numerator, denominator, max_score = carry
    numerator = numerator.reshape((batch, num_q, query_chunk_size, num_heads, dim_per_head))
    numerator = jnp.moveaxis(numerator, 1, 0)
    denominator = denominator.reshape((batch, num_heads, num_q, query_chunk_size))
    max_score = max_score.reshape((batch, num_heads, num_q, query_chunk_size))
    denominator, max_score = map(lambda x: rearrange(x, 'b h n c -> n b h c'), (denominator, max_score))

    scale = jnp.sqrt(q.shape[-1])
    if bias is not None:
        for bias_dim, broadcast_dim in zip(bias.shape, (batch, num_heads, q_len, kv_len)):
            assert bias_dim == 1 or bias_dim == broadcast_dim
    if not deterministic and attn_pdrop > 0.0:
        attn_dropout_rng, dropout_rng = jax.random.split(dropout_rng)
        attn_dropout = jax.random.bernoulli(attn_dropout_rng, attn_pdrop, (batch, num_heads, q_len, kv_len))
    else:
        attn_dropout = None
    _chunk_bias_fn = partial(
        _chunk_attention_bias,
        query_chunk_size, key_chunk_size, bias, deterministic,
        attn_dropout, attn_pdrop, causal, dtype)
    def scan_attention(_, scan):
        q_chunk, numerator_chunk, denominator_chunk, max_score_chunk, q_chunk_idx = scan
        @partial(jax.checkpoint, prevent_cse=prevent_cse, policy=policy)
        def scan_kv_block(carry, scan):
            k_chunk, value_chunk, k_chunk_idx = scan
            numerator_chunk, denominator_chunk, prev_max_score_chunk = carry
            # attn_weights = jnp.einsum('bqhd,bkhd->bqhk', q_chunk, k_chunk, precision=precision) / scale
            attn_weights = jnp.einsum('bqhd,bkhd->bhqk', q_chunk, k_chunk, precision=precision) / scale
            bias_chunk = _chunk_bias_fn(q_chunk_idx_start + q_chunk_idx, k_chunk_idx_start + k_chunk_idx)
            # bias_chunk = jnp.moveaxis(bias_chunk, 1, 2)
            attn_weights = attn_weights + bias_chunk

            max_score_chunk = jnp.maximum(prev_max_score_chunk, jnp.max(attn_weights, axis=-1))
            max_score_chunk = lax.stop_gradient(max_score_chunk)
            exp_weights = jnp.exp(attn_weights - max_score_chunk[..., None])
            exp_values = jnp.einsum('bhqk,bkhd->bqhd', exp_weights, value_chunk, precision=precision)
            correction = rearrange(jnp.exp(prev_max_score_chunk - max_score_chunk), 'b h q -> b q h')[..., None]
            # max_score_chunk = jnp.max(attn_weights, axis=-1, keepdims=True)
            # max_score_chunk = jnp.maximum(prev_max_score_chunk, max_score_chunk)
            # exp_weights = jnp.exp(attn_weights - max_score_chunk)
            # exp_values = jnp.einsum('bqhv,bvhd->bqhd', exp_weights, value_chunk, precision=precision)
            # correction = jnp.exp(prev_max_score_chunk - max_score_chunk)
            numerator_chunk = numerator_chunk * correction + exp_values
            # denominator_chunk = denominator_chunk * correction + exp_weights.sum(axis=-1, keepdims=True)
            denominator_chunk = denominator_chunk * jnp.exp(prev_max_score_chunk - max_score_chunk) + exp_weights.sum(axis=-1)
            return (numerator_chunk, denominator_chunk, max_score_chunk), None

        def skip_upper_half(carry, args):
            key_chunk, value_chunk, k_chunk_idx = args
            skip_block = jnp.array(False)
            if causal:
                skip_block = q_chunk_idx_start + q_chunk_idx < k_chunk_idx_start + k_chunk_idx
            return jax.lax.cond(
                skip_block,
                lambda carry, args: (carry, None),
                scan_kv_block,
                carry,
                args
            )

        (numerator_chunk, denominator_chunk, max_score_chunk), _ = lax.scan(
            skip_upper_half, init=(numerator_chunk, denominator_chunk, max_score_chunk), xs=(k, v, jnp.arange(0, num_kv))
        )
        # output_chunk = (numerator_chunk / denominator_chunk).astype(dtype)
        output_chunk = numerator_chunk / rearrange(denominator_chunk, 'b h q -> b q h')[..., None].astype(dtype)
        return (), (output_chunk, numerator_chunk, denominator_chunk, max_score_chunk)
    _, (_, numerator, denominator, max_score) = lax.scan(scan_attention, init=(), xs=(q, numerator, denominator, max_score, jnp.arange(0, num_q)))
    # numerator, denominator, max_score = map(lambda x: rearrange(x, 'n b c h d -> b (n c) h d'), (numerator, denominator, max_score))

    numerator = jnp.moveaxis(numerator, 1, 0)
    numerator = numerator.reshape((batch, q_len, num_heads, dim_per_head))
    denominator, max_score = map(lambda x: rearrange(x, 'n b h c -> b h n c'), (denominator, max_score))
    denominator = denominator.reshape((batch, num_heads, q_len))
    max_score = max_score.reshape((batch, num_heads, q_len))

    return numerator, denominator, max_score

def _blockwise_attention_bwd(q, k, v, g, carry, q_chunk_idx_start, k_chunk_idx_start, bias, causal, query_chunk_size, key_chunk_size, deterministic, dropout_rng, attn_pdrop, dtype, policy, precision, prevent_cse):
    batch, q_len, num_heads, dim_per_head = q.shape
    batch, kv_len, num_heads, dim_per_head = k.shape
    batch, kv_len, num_heads, dim_per_head = v.shape
    num_q = q_len // query_chunk_size
    num_kv = kv_len // key_chunk_size
    dq, dk, dv, output, denominator, max_score = carry

    g = g.reshape((batch, num_q, query_chunk_size, num_heads, dim_per_head))
    dq = dq.reshape((batch, num_q, query_chunk_size, num_heads, dim_per_head))
    dk = dk.reshape((batch, num_kv, key_chunk_size, num_heads, dim_per_head))
    dv = dv.reshape((batch, num_kv, key_chunk_size, num_heads, dim_per_head))
    output = output.reshape((batch, num_q, query_chunk_size, num_heads, dim_per_head))
    g, dq, dk, dv, output = map(lambda x: jnp.moveaxis(x, 1, 0), (g, dq, dk, dv, output))

    denominator = denominator.reshape((batch, num_heads, num_q, query_chunk_size))
    max_score = max_score.reshape((batch, num_heads, num_q, query_chunk_size))
    denominator, max_score = map(lambda x: rearrange(x, 'b h n c -> n b h c'), (denominator, max_score))

    q = q.reshape((batch, num_q, query_chunk_size, num_heads, dim_per_head))
    k = k.reshape((batch, num_kv, key_chunk_size, num_heads, dim_per_head))
    v = v.reshape((batch, num_kv, key_chunk_size, num_heads, dim_per_head))
    q, k, v = map(lambda x: jnp.moveaxis(x, 1, 0), (q, k, v))

    scale = jnp.sqrt(q.shape[-1])
    if bias is not None:
        for bias_dim, broadcast_dim in zip(bias.shape, (batch, num_heads, q_len, kv_len)):
            assert bias_dim == 1 or bias_dim == broadcast_dim
    if not deterministic and attn_pdrop > 0.0:
        attn_dropout_rng, dropout_rng = jax.random.split(dropout_rng)
        attn_dropout = jax.random.bernoulli(attn_dropout_rng, attn_pdrop, (batch, num_heads, q_len, kv_len))
    else:
        attn_dropout = None
    _chunk_bias_fn = partial(
        _chunk_attention_bias,
        query_chunk_size, key_chunk_size, bias, deterministic,
        attn_dropout, attn_pdrop, causal, dtype)
    def scan_attention(carry, scan):
        dk, dv = carry
        q_chunk, dq_chunk, g_chunk, output_chunk, denominator_chunk, max_score_chunk, q_chunk_idx = scan
        dl_part = jnp.einsum("bqhd,bqhd->bhq", g_chunk, output_chunk)[..., None]
        @partial(jax.checkpoint, prevent_cse=prevent_cse, policy=policy)
        def scan_kv_block(carry, scan):
            k_chunk, value_chunk, k_chunk_idx = scan
            dq_chunk = carry
            attn_weights = jnp.einsum('bqhd,bkhd->bhqk', q_chunk, k_chunk, precision=precision) / scale
            bias_chunk = _chunk_bias_fn(q_chunk_idx_start + q_chunk_idx, k_chunk_idx_start + k_chunk_idx)
            # bias_chunk = jnp.moveaxis(bias_chunk, 1, 2)
            attn_weights = attn_weights + bias_chunk
            exp_weights = jnp.exp(attn_weights - max_score_chunk[..., None]) / denominator_chunk[..., None]

            ds = jnp.einsum("bqhd,bkhd->bhqk", g_chunk, value_chunk)
            dl = (ds - dl_part) * exp_weights
            dq_chunk = dq_chunk + jnp.einsum("bhqk,bkhd->bqhd", dl, k_chunk) / scale
            dk_chunk = jnp.einsum("bqhd,bhqk->bkhd", q_chunk, dl) / scale
            dv_chunk = jnp.einsum("bhqk,bqhd->bkhd", exp_weights, g_chunk)
            return dq_chunk, (dk_chunk, dv_chunk)

        def skip_upper_half(carry, args):
            key_chunk, value_chunk, k_chunk_idx = args
            skip_block = jnp.array(False)
            if causal:
                skip_block = q_chunk_idx_start + q_chunk_idx < k_chunk_idx_start + k_chunk_idx
            return lax.cond(
                skip_block,
                lambda carry, args: (
                    carry, (
                        jnp.zeros((batch, key_chunk_size, num_heads, dim_per_head), dtype=dk.dtype),
                        jnp.zeros((batch, key_chunk_size, num_heads, dim_per_head), dtype=dk.dtype),
                    )
                ),
                scan_kv_block,
                carry,
                args
            )

        dq_chunk, (dk_part, dv_part) = lax.scan(
            skip_upper_half, init=dq_chunk, xs=(k, v, jnp.arange(0, num_kv))
        )
        return (dk + dk_part, dv + dv_part), dq_chunk
    (dk, dv), dq = lax.scan(scan_attention, init=(dk, dv), xs=(q, dq, g, output, denominator, max_score, jnp.arange(0, num_q)))

    dq, dk, dv = map(lambda x: jnp.moveaxis(x, 1, 0), (dq, dk, dv))
    dq = dq.reshape((batch, q_len, num_heads, dim_per_head))
    dk = dk.reshape((batch, kv_len, num_heads, dim_per_head))
    dv = dv.reshape((batch, kv_len, num_heads, dim_per_head))

    return dq, dk, dv

'''
Computing ffn blockwise without materializing the large hidden tensor, training 4x longer sequences than the memory-efficient transformer.
Blockwise parallel transformer https://arxiv.org/abs/2305.19370 Liu et al. 2023
'''
def blockwise_ffn(remat_ffn, inputs, chunk_size, deterministic):
    # remat_ffn: a rematerialized ffn with policy jax.checkpoint_policies.nothing_saveable()
    # inputs: (batch, seq_len, dim)
    # chunk_size: the chunk size to split the sequence
    inputs = rearrange(inputs, 'b (c n) d -> b c n d', c=chunk_size)
    def scan_ffn(remat_ffn, carry, hidden_states):
        outputs = remat_ffn(hidden_states, deterministic=deterministic)
        return carry, outputs
    scan_axis = inputs.ndim - 2
    _, output = nn.scan(
        scan_ffn,
        variable_broadcast="params",
        split_rngs={"params": False, "dropout": True},
        in_axes=scan_axis,
        out_axes=scan_axis,
    )(remat_ffn, None, inputs)
    output = rearrange(output, 'b c n d -> b (c n) d')
    return output


def blockwise_attn(query, key, value, bias, deterministic,
        dropout_rng, attn_pdrop, causal, query_chunk_size,
        key_chunk_size, dtype, policy, precision, float32_logits,
        prevent_cse):
    # query, key, value: (batch, seq_len, num_heads, dim_per_head)
    # bias: (batch, seq_len) can be used to mask out attention (e.g. padding)
    # causal: whether to use causal mask
    # policy: one of jax.checkpoint_policies
    query = query / jnp.sqrt(query.shape[-1]).astype(dtype)
    if float32_logits:
        query = query.astype(jnp.float32)
        key = key.astype(jnp.float32)

    batch, q_len, num_heads, dim_per_head = query.shape
    batch, kv_len, num_heads, dim_per_head = key.shape
    batch, kv_len, num_heads, dim_per_head = value.shape

    num_q = q_len // query_chunk_size
    num_kv = kv_len // key_chunk_size
    query = query.reshape((batch, num_q, query_chunk_size, num_heads, dim_per_head))
    key = key.reshape((batch, num_kv, key_chunk_size, num_heads, dim_per_head))
    value = value.reshape((batch, num_kv, key_chunk_size, num_heads, dim_per_head))

    query = jnp.moveaxis(query, 1, 0)
    key = jnp.moveaxis(key, 1, 0)
    value = jnp.moveaxis(value, 1, 0)

    if bias is not None:
        for bias_dim, broadcast_dim in zip(bias.shape, (batch, num_heads, q_len, kv_len)):
            assert bias_dim == 1 or bias_dim == broadcast_dim
    if not deterministic and attn_pdrop > 0.0:
        attn_dropout_rng, dropout_rng = jax.random.split(dropout_rng)
        attn_dropout = jax.random.bernoulli(attn_dropout_rng, attn_pdrop, (batch, num_heads, q_len, kv_len))
    else:
        attn_dropout = None

    _chunk_bias_fn = partial(
        _chunk_attention_bias,
        query_chunk_size, key_chunk_size, bias, deterministic,
        attn_dropout, attn_pdrop, causal, dtype)

    def scan_attention(carry, args):
        del carry
        query_chunk, query_chunk_idx = args

        @partial(jax.checkpoint, prevent_cse=prevent_cse, policy=policy)
        def scan_kv_block(carry, args):
            key_chunk, value_chunk, key_chunk_idx = args
            (numerator, denominator, prev_max_score) = carry
            attn_weights = jnp.einsum('bqhd,bkhd->bqhk', query_chunk, key_chunk, precision=precision)
            bias_chunk = _chunk_bias_fn(query_chunk_idx, key_chunk_idx)
            bias_chunk = jnp.moveaxis(bias_chunk, 1, 2)
            attn_weights = attn_weights + bias_chunk

            max_score = jnp.max(attn_weights, axis=-1, keepdims=True)
            max_score = jnp.maximum(prev_max_score, max_score)
            max_score = lax.stop_gradient(max_score)
            exp_weights = jnp.exp(attn_weights - max_score)
            exp_values = jnp.einsum(
                'bqhv,bvhd->bqhd', exp_weights, value_chunk, precision=precision
            )
            correction = jnp.exp(prev_max_score - max_score)
            numerator = numerator * correction + exp_values
            denominator = denominator * correction + exp_weights.sum(axis=-1, keepdims=True)
            return (numerator, denominator, max_score), None

        def skip_upper_half(carry, args):
            key_chunk, value_chunk, key_chunk_idx = args
            skip_block = jnp.array(False)
            if causal:
                skip_block = query_chunk_idx < key_chunk_idx
            return lax.cond(
                skip_block,
                lambda carry, args: (carry, None),
                scan_kv_block,
                carry,
                args,
            )

        init_carry = (
            jnp.zeros((batch, query_chunk_size, num_heads, dim_per_head), dtype=query.dtype),
            jnp.zeros((batch, query_chunk_size, num_heads, dim_per_head), dtype=query.dtype),
            (-jnp.inf) * jnp.ones((batch, query_chunk_size, num_heads, 1), dtype=query.dtype),
        )
        (numerator, denominator, max_score), _ = lax.scan(
            skip_upper_half, init_carry, xs=(key, value, jnp.arange(0, num_kv))
        )
        output = (numerator / denominator).astype(dtype)
        return (), output

    _, output = lax.scan(scan_attention, (), xs=(query, jnp.arange(0, num_q)))
    output = rearrange(output, 'n b c h d -> b (n c) h d')
    return output

def _chunk_attention_bias(query_chunk_size, key_chunk_size,
            bias, deterministic, attn_dropout, attn_pdrop, causal,
            dtype, query_chunk_idx, key_chunk_idx):
    query_offset = query_chunk_idx * query_chunk_size
    key_offset = key_chunk_idx * key_chunk_size
    chunk_bias = jnp.zeros((1, 1, 1, 1), dtype=dtype)
    if bias is not None:
        chunk_bias = lax.dynamic_slice(
            bias,
            start_indices=(0, 0, query_offset, key_offset),
            slice_sizes=(*bias.shape[:2], min(bias.shape[-2], query_chunk_size), min(bias.shape[-1], key_chunk_size)),
        )

    if causal:
        query_idx = lax.broadcasted_iota(dtype=jnp.int32, shape=(query_chunk_size, 1), dimension=0)
        key_idx = lax.broadcasted_iota(dtype=jnp.int32, shape=(1, key_chunk_size), dimension=1)
        offset = query_offset - key_offset
        query_idx += offset
        causal_mask_value = (query_idx < key_idx) * jnp.finfo(dtype).min
        chunk_bias += causal_mask_value.reshape(1, 1, *causal_mask_value.shape)

    if not deterministic and attn_pdrop > 0.0:
        attn_dropout_slice = lax.dynamic_slice(
            attn_dropout,
            start_indices=(0, 0, query_offset, key_offset),
            slice_sizes=(
                *attn_dropout.shape[:2],
                min(attn_dropout.shape[-2], query_chunk_size),
                min(attn_dropout.shape[-1], key_chunk_size),
            ),
        )
        chunk_bias += attn_dropout_slice * jnp.finfo(dtype).min
    return chunk_bias.astype(dtype)



def ring_flash_dummy(q, k, v, attn_bias, axis_name, float32_logits, blockwise_kwargs):
    from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention, BlockSizes, mha_reference
    if float32_logits:
        q, k = q.astype(jnp.float32), k.astype(jnp.float32)
    q, k, v = map(lambda x: rearrange(x, 'b q h d -> b h q d'), [q, k, v])
    attn_bias = attn_bias[:, 0, 0] # (batch, q_len)

    query_chunk_size = blockwise_kwargs["query_chunk_size"]
    key_chunk_size = blockwise_kwargs["key_chunk_size"]

    block_sizes = BlockSizes(
        block_q=query_chunk_size,
        block_k_major=key_chunk_size,
        block_k=key_chunk_size,
        block_b=1,
        block_q_major_dkv=query_chunk_size,
        block_k_major_dkv=key_chunk_size,
        block_k_dkv=key_chunk_size,
        block_q_dkv=query_chunk_size,
        block_k_major_dq=key_chunk_size,
        block_k_dq=key_chunk_size,
        block_q_dq=query_chunk_size,
    )

    o = flash_attention(
        q, k, v,
        causal=blockwise_kwargs["causal"],
        block_sizes=block_sizes,
        sm_scale=q.shape[-1] ** -0.5,
    )
    output = rearrange(o.astype(v.dtype), 'b h q d -> b q h d')
    return output


def ring_standard_dummy(q, k, v, attn_bias, axis_name, float32_logits, blockwise_kwargs):
    from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention, BlockSizes, mha_reference
    if float32_logits:
        q, k = q.astype(jnp.float32), k.astype(jnp.float32)
    q, k, v = map(lambda x: rearrange(x, 'b q h d -> b h q d'), [q, k, v])
    attn_bias = attn_bias[:, 0, 0] # (batch, q_len)

    o = mha_reference(
        q, k, v,
        ab=None,
        causal=blockwise_kwargs["causal"],
        sm_scale=q.shape[-1] ** -0.5,
    )
    output = rearrange(o.astype(v.dtype), 'b h q d -> b q h d')
    return output


def _ring_flash_attention_fwd_tpu(q, k, v, attn_bias, axis_name, float32_logits, blockwise_kwargs):
    if float32_logits:
        q, k = q.astype(jnp.float32), k.astype(jnp.float32)
    q, k, v = map(lambda x: rearrange(x, 'b q h d -> b h q d'), [q, k, v])
    batch, num_heads, q_len, dim_per_head = q.shape
    batch, num_heads, kv_len, dim_per_head = k.shape
    attn_bias = attn_bias[:, 0, 0] # (batch, q_len)

    o = jnp.zeros((batch, num_heads, q_len, dim_per_head)).astype(q.dtype)
    l = jnp.zeros((batch, num_heads, q_len)).astype(q.dtype)
    m = jnp.full((batch, num_heads, q_len), -jnp.inf).astype(q.dtype)

    axis_size = lax.psum(1, axis_name)
    block_size = q_len # assumes this function is pre-sharded inside shard_map
    query_chunk_size = blockwise_kwargs["query_chunk_size"]
    key_chunk_size = blockwise_kwargs["key_chunk_size"]

    block_sizes = BlockSizes(
        block_q=query_chunk_size,
        block_k_major=key_chunk_size,
        block_k=key_chunk_size,
        block_b=1,
        block_q_major_dkv=query_chunk_size,
        block_k_major_dkv=key_chunk_size,
        block_k_dkv=key_chunk_size,
        block_q_dkv=query_chunk_size,
        block_k_major_dq=key_chunk_size,
        block_k_dq=key_chunk_size,
        block_q_dq=query_chunk_size,
    )

    scale = q.shape[-1] ** -0.5
    def scan_kv_block(carry, idx):
        o, l, m, k, v = carry
        # attn_bias_slice = lax.dynamic_slice_in_dim(attn_bias,
        #     (lax.axis_index(axis_name) - idx) % axis_size * kv_len, kv_len, axis=-1
        # )
        attn_bias_slice = None # TODO
        q_block_idx = lax.axis_index(axis_name)
        k_block_idx = (lax.axis_index(axis_name) - idx) % axis_size
        q_chunk_idx_start = q_block_idx * (block_size // query_chunk_size)
        k_chunk_idx_start = k_block_idx * (block_size // key_chunk_size)
        o, l, m = _flash_attention_fwd(
            q, k, v,
            carry=(o, l, m),
            q_chunk_idx_start=q_chunk_idx_start,
            k_chunk_idx_start=k_chunk_idx_start,
            ab=attn_bias_slice,
            segment_ids=None,
            save_residuals=False,
            causal=blockwise_kwargs["causal"],
            sm_scale=scale,
            block_sizes=block_sizes,
            debug=False
        )
        k, v = map(lambda x: lax.ppermute(x, axis_name, perm=[(i, (i + 1) % axis_size) for i in range(axis_size)]), (k, v))
        return (o, l, m, k, v), None
    (o, l, m, _, _), _ = lax.scan(scan_kv_block,
        init=(o, l, m, k, v), xs=jnp.arange(0, axis_size))
    output = rearrange(o.astype(v.dtype), 'b h q d -> b q h d')
    return output, (o, q, k, v, attn_bias, l, m)

def _ring_flash_attention_bwd_tpu(axis_name, float32_logits, blockwise_kwargs, res, g):
    del float32_logits
    o, q, k, v, attn_bias, l, m = res
    batch, num_heads, kv_len, dim_per_head = k.shape
    axis_size = lax.psum(1, axis_name)
    dq = jnp.zeros_like(q, dtype=jnp.float32)
    dk = jnp.zeros_like(k, dtype=jnp.float32)
    dv = jnp.zeros_like(v, dtype=jnp.float32)
    query_chunk_size = blockwise_kwargs["query_chunk_size"]
    key_chunk_size = blockwise_kwargs["key_chunk_size"]
    block_size = q.shape[2] # assumes this function is pre-sharded inside shard_map
    scale = q.shape[-1] ** -0.5

    g = rearrange(g, 'b q h d -> b h q d')

    block_sizes = BlockSizes(
        block_q=query_chunk_size,
        block_k_major=key_chunk_size,
        block_k=key_chunk_size,
        block_b=1,
        block_q_major_dkv=query_chunk_size,
        block_k_major_dkv=key_chunk_size,
        block_k_dkv=key_chunk_size,
        block_q_dkv=query_chunk_size,
        block_k_major_dq=key_chunk_size,
        block_k_dq=key_chunk_size,
        block_q_dq=query_chunk_size,
    )

    def scan_kv_block(carry, idx):
        dq, dk, dv, k, v = carry
        # attn_bias_slice = lax.dynamic_slice_in_dim(attn_bias,
        #     (lax.axis_index(axis_name) - idx) % axis_size * kv_len, kv_len, axis=-1
        # )
        attn_bias_slice = None # TODO
        q_block_idx = lax.axis_index(axis_name)
        k_block_idx = (lax.axis_index(axis_name) - idx) % axis_size
        q_chunk_idx_start = q_block_idx * (block_size // query_chunk_size)
        k_chunk_idx_start = k_block_idx * (block_size // key_chunk_size)
        dq_i, dk_i, dv_i, = _flash_attention_bwd(
            save_residuals=False,
            causal=blockwise_kwargs["causal"],
            sm_scale=scale,
            block_sizes=block_sizes,
            debug=False,
            q_chunk_idx_start=q_chunk_idx_start,
            k_chunk_idx_start=k_chunk_idx_start,
            residuals=(q, k, v, attn_bias_slice, None, o, l, m),
            do=g
        )
        dq += dq_i
        dk += dk_i
        dv += dv_i
        k, v, dk, dv = map(lambda x: lax.ppermute(x, axis_name, perm=[(i,
            (i + 1) % axis_size) for i in range(axis_size)]), (k, v, dk, dv))
        return (dq, dk, dv, k, v), None
    (dq, dk, dv, k, v), _ = lax.scan(scan_kv_block, init=(dq, dk, dv, k, v), xs=jnp.arange(0, axis_size))
    dq, dk, dv = dq.astype(q.dtype), dk.astype(k.dtype), dv.astype(v.dtype)
    dq, dk, dv = map(lambda x: rearrange(x, 'b h q d -> b q h d'), (dq, dk, dv))
    return dq, dk, dv, None

@partial(jax.custom_vjp, nondiff_argnums=[4, 5, 6])
def ring_flash_attention_tpu(q, k, v, attn_bias, axis_name, float32_logits, blockwise_kwargs):
    y, _ = _ring_flash_attention_fwd_tpu(q, k, v, attn_bias, axis_name, float32_logits, blockwise_kwargs)
    return y

ring_flash_attention_tpu.defvjp(_ring_flash_attention_fwd_tpu, _ring_flash_attention_bwd_tpu)


from llamabpt.flash_attention_gpu import _mha_forward, _mha_backward

def ring_flash_dummy_gpu(q, k, v, attn_bias, axis_name, float32_logits, blockwise_kwargs):
    if float32_logits:
        q, k = q.astype(jnp.float32), k.astype(jnp.float32)
    attn_bias = attn_bias[:, 0, 0] # (batch, q_len)

    query_chunk_size = blockwise_kwargs["query_chunk_size"]
    key_chunk_size = blockwise_kwargs["key_chunk_size"]

    output = mha(q, k, v, None, sm_scale=q.shape[-1] ** -0.5)
    return output


def _ring_flash_attention_fwd_gpu(q, k, v, attn_bias, axis_name, float32_logits, blockwise_kwargs):
    if float32_logits:
        q, k = q.astype(jnp.float32), k.astype(jnp.float32)
    batch, q_len, num_heads, dim_per_head = q.shape
    batch, kv_len, num_heads, dim_per_head = k.shape
    attn_bias = attn_bias[:, 0, 0] # (batch, q_len)

    o = jnp.zeros((batch, q_len, num_heads, dim_per_head)).astype(jnp.bfloat16)
    l = jnp.zeros((batch, num_heads, q_len)).astype(jnp.float32)
    m = jnp.full((batch, num_heads, q_len), -jnp.inf).astype(jnp.float32)

    axis_size = lax.psum(1, axis_name)
    block_size = q_len # assumes this function is pre-sharded inside shard_map
    query_chunk_size = blockwise_kwargs["query_chunk_size"]
    key_chunk_size = blockwise_kwargs["key_chunk_size"]

    scale = q.shape[-1] ** -0.5
    def scan_kv_block(carry, idx):
        o, l, m, k, v = carry
        # attn_bias_slice = lax.dynamic_slice_in_dim(attn_bias,
        #     (lax.axis_index(axis_name) - idx) % axis_size * kv_len, kv_len, axis=-1
        # )
        attn_bias_slice = None # TODO
        q_block_idx = lax.axis_index(axis_name)
        k_block_idx = (lax.axis_index(axis_name) - idx) % axis_size
        q_chunk_idx_start = q_block_idx * (block_size // query_chunk_size)
        k_chunk_idx_start = k_block_idx * (block_size // key_chunk_size)
        o, l, m = _mha_forward(
            q, k, v,
            carry=(o.astype(jnp.float32), l, m),
            q_chunk_idx_start=q_chunk_idx_start, # TODO handle this
            k_chunk_idx_start=k_chunk_idx_start,
            segment_ids=None,
            sm_scale=scale,
            causal=blockwise_kwargs["causal"],
            block_q=query_chunk_size,
            block_k=key_chunk_size,
            backward_pass_impl='triton', # unused
            num_warps=None,
            num_stages=2, # unused
            grid=None,
            interpret=False,
            debug=False
        )
        k, v = map(lambda x: lax.ppermute(x, axis_name, perm=[(i, (i + 1) % axis_size) for i in range(axis_size)]), (k, v))
        return (o, l, m, k, v), None
    (o, l, m, _, _), _ = lax.scan(scan_kv_block,
        init=(o, l, m, k, v), xs=jnp.arange(0, axis_size))
    output = o.astype(v.dtype)
    return output, (o, q, k, v, attn_bias, l, m)


from jax.experimental.pallas.ops.attention import _mha_forward as _mha_forward_d, _mha_backward as _mha_backward_d, mha
def _ring_flash_attention_bwd_gpu(axis_name, float32_logits, blockwise_kwargs, res, g):
    del float32_logits
    o, q, k, v, attn_bias, l, m = res
    batch, kv_len, num_heads, dim_per_head = k.shape
    axis_size = lax.psum(1, axis_name)
    dq = jnp.zeros_like(q, dtype=q.dtype)
    dk = jnp.zeros_like(k, dtype=k.dtype)
    dv = jnp.zeros_like(v, dtype=v.dtype)
    query_chunk_size = blockwise_kwargs["query_chunk_size"]
    key_chunk_size = blockwise_kwargs["key_chunk_size"]
    block_size = q.shape[1] # assumes this function is pre-sharded inside shard_map
    scale = q.shape[-1] ** -0.5

    def scan_kv_block(carry, idx):
        dq, dk, dv, k, v = carry
        # attn_bias_slice = lax.dynamic_slice_in_dim(attn_bias,
        #     (lax.axis_index(axis_name) - idx) % axis_size * kv_len, kv_len, axis=-1
        # )
        attn_bias_slice = None # TODO
        q_block_idx = lax.axis_index(axis_name)
        k_block_idx = (lax.axis_index(axis_name) - idx) % axis_size
        q_chunk_idx_start = q_block_idx * (block_size // query_chunk_size)
        k_chunk_idx_start = k_block_idx * (block_size // key_chunk_size)
        dq_i, dk_i, dv_i = _mha_backward(
            sm_scale=scale,
            causal=blockwise_kwargs["causal"],
            block_q=query_chunk_size,
            block_k=key_chunk_size,
            backward_pass_impl='triton',
            num_warps=None,
            num_stages=2,
            grid=None,
            interpret=False,
            debug=False,
            q_chunk_idx_start=q_chunk_idx_start,
            k_chunk_idx_start=k_chunk_idx_start,
            res=(q, k, v, None, o, l, m),
            do=g,
        )
        dq += dq_i
        dk += dk_i
        dv += dv_i
        k, v, dk, dv = map(lambda x: lax.ppermute(x, axis_name, perm=[(i,
            (i + 1) % axis_size) for i in range(axis_size)]), (k, v, dk, dv))
        return (dq, dk, dv, k, v), None
    (dq, dk, dv, k, v), _ = lax.scan(scan_kv_block, init=(dq, dk, dv, k, v), xs=jnp.arange(0, axis_size))
    dq, dk, dv = dq.astype(q.dtype), dk.astype(k.dtype), dv.astype(v.dtype)
    return dq, dk, dv, None

@partial(jax.custom_vjp, nondiff_argnums=[4, 5, 6])
def ring_flash_attention_gpu(q, k, v, attn_bias, axis_name, float32_logits, blockwise_kwargs):
    y, _ = _ring_flash_attention_fwd_gpu(q, k, v, attn_bias, axis_name, float32_logits, blockwise_kwargs)
    return y

ring_flash_attention_gpu.defvjp(_ring_flash_attention_fwd_gpu, _ring_flash_attention_bwd_gpu)


if __name__ == '__main__':
    # test
    def reference_attn(query, key, value, causal, dtype):
        query = query / jnp.sqrt(query.shape[-1]).astype(dtype)
        logits = jnp.einsum("bqhc,bkhc->bhqk", query, key)
        if causal:
            mask_value = jnp.finfo(logits.dtype).min
            _, q_seq_len, _, _ = query.shape
            _, kv_seq_len, _, _ = key.shape
            mask_shape = (q_seq_len, kv_seq_len)
            row_ids = lax.broadcasted_iota(jnp.int32, mask_shape, 0)
            col_ids = lax.broadcasted_iota(jnp.int32, mask_shape, 1)
            causal_mask = (row_ids < col_ids)[None, None, :, :]
            logits = logits + jnp.where(causal_mask, mask_value, 0.0)
        weights = jax.nn.softmax(logits, axis=-1)
        out = jnp.einsum("bhqk,bkhc->bqhc", weights, value)
        return out

    # random inputs
    shape = (1, 32, 8, 64)
    query = jax.random.normal(jax.random.PRNGKey(0), shape)
    key = jax.random.normal(jax.random.PRNGKey(1), shape)
    value = jax.random.normal(jax.random.PRNGKey(2), shape)

    causal = True
    chunk_size = 4
    policy = jax.checkpoint_policies.nothing_saveable()

    blockwise = blockwise_attn(query, key, value, None, False, None, 0.0, causal, chunk_size, chunk_size, jnp.float32, policy, 'float32', True, False)
    reference = reference_attn(query, key, value, causal, 'float32')

    print('max diff sum:', jnp.abs(reference - blockwise).sum())
    print('max diff ele:', jnp.abs(reference - blockwise).max())
