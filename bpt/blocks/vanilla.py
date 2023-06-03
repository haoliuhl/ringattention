import functools
import json
import math
from functools import partial
from typing import Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange
from flax.linen import combine_masks, make_causal_mask
from jax import lax
from jax import numpy as jnp


def quick_gelu(x):
    return x * jax.nn.sigmoid(1.702 * x)

ACT2FN = {
    "gelu": partial(nn.gelu, approximate=False),
    "relu": nn.relu,
    "silu": nn.swish,
    "swish": nn.swish,
    "gelu_new": partial(nn.gelu, approximate=True),
    "quick_gelu": quick_gelu,
}


MASK_VALUE = -1e10

Q_CHUNK_SIZE = 1024
K_CHUNK_SIZE = 1024

def create_sinusoidal_positions(num_pos, dim):
    inv_freq = 1.0 / (10000 ** (np.arange(0, dim, 2) / dim))
    sinusoid_inp = np.einsum("i , j -> i j", np.arange(num_pos), inv_freq).astype("float32")
    sin, cos = np.sin(sinusoid_inp), np.cos(sinusoid_inp)

    sentinel = dim // 2 + dim % 2
    out = np.zeros((num_pos, dim))
    out[:, 0:sentinel] = sin
    out[:, sentinel:] = cos

    return jnp.array(out)


def rotate_every_two(tensor):
    rotate_half_tensor = jnp.stack((-tensor[:, :, :, 1::2], tensor[:, :, :, ::2]), axis=-1)
    rotate_half_tensor = rotate_half_tensor.reshape(rotate_half_tensor.shape[:-2] + (-1,))
    return rotate_half_tensor


def apply_rotary_pos_emb(tensor, sincos):
    sin_pos, cos_pos = sincos
    sin_pos = sin_pos[:, :, None, :].repeat(2, 3)
    cos_pos = cos_pos[:, :, None, :].repeat(2, 3)
    return (tensor * cos_pos) + (rotate_every_two(tensor) * sin_pos)


class _AttentionBlock(nn.Module):
    hidden_size: int
    num_heads: int
    rotary_dim: Optional[int]
    intermediate_size: int
    layer_norm_epsilon: float = 1e-5
    activation_function: str = "gelu"
    resid_pdrop: float = 0.0
    max_position_embeddings: int = 1024
    dtype: jnp.dtype = jnp.float32
    causal: bool = True
    float32_logits: bool = False

    def setup(self):
        self.embed_dim = self.hidden_size
        self.head_dim = self.embed_dim // self.num_heads
        dense = partial(
            nn.Dense,
            self.embed_dim,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.variance_scaling(
                scale=1.0, mode='fan_in',
                distribution='normal',
            )
        )
        self.q_proj, self.k_proj, self.v_proj = dense(), dense(), dense()
        self.out_proj = dense()
        self.ln_1 = nn.LayerNorm(epsilon=self.layer_norm_epsilon, dtype=self.dtype)

        self.ln_2 = nn.LayerNorm(epsilon=self.layer_norm_epsilon, dtype=self.dtype)
        self.fc_in = nn.Dense(self.intermediate_size,
                            dtype=self.dtype,
                            kernel_init=jax.nn.initializers.variance_scaling(
                            scale=1.0, mode='fan_in',
                            distribution='normal',
            )
        )
        self.fc_out = nn.Dense(self.embed_dim,
                            dtype=self.dtype,
                            kernel_init=jax.nn.initializers.variance_scaling(
                            scale=1.0, mode='fan_in',
                            distribution='normal',
            )
        )
        self.act = ACT2FN[self.activation_function]
        self.resid_dropout = nn.Dropout(rate=self.resid_pdrop)

        if self.rotary_dim is not None and self.rotary_dim > 0:
            pos_embd_dim = self.rotary_dim
        else:
            pos_embd_dim = self.embed_dim // self.num_heads
        self.embed_positions = create_sinusoidal_positions(self.max_position_embeddings, pos_embd_dim)

    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    def attn_out_proj(self, attn_output, deterministic):
        attn_output = self._merge_heads(attn_output)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output, deterministic=deterministic)
        return attn_output

    def forward_qkv(
        self,
        hidden_states,
        position_ids,
        deterministic: bool = True,
    ):
        hidden_states = self.ln_1(hidden_states)
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        sincos = jnp.take(self.embed_positions, position_ids, axis=0)
        sincos = jnp.split(sincos, 2, axis=-1)
        if self.rotary_dim is not None and self.rotary_dim > 0:
            k_rot = key[:, :, :, : self.rotary_dim]
            k_pass = key[:, :, :, self.rotary_dim :]

            q_rot = query[:, :, :, : self.rotary_dim]
            q_pass = query[:, :, :, self.rotary_dim :]

            k_rot = apply_rotary_pos_emb(k_rot, sincos)
            q_rot = apply_rotary_pos_emb(q_rot, sincos)

            key = jnp.concatenate([k_rot, k_pass], axis=-1)
            query = jnp.concatenate([q_rot, q_pass], axis=-1)
        else:
            key = apply_rotary_pos_emb(key, sincos)
            query = apply_rotary_pos_emb(query, sincos)

        if self.float32_logits:
            query = query.astype(jnp.float32)
            key = key.astype(jnp.float32)

        return query, key, value

    def forward_ffn(
        self,
        hidden_states,
        deterministic: bool = True,
    ):
        hidden_states = self.ln_2(hidden_states)
        hidden_states = self.fc_in(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc_out(hidden_states)
        hidden_states = self.resid_dropout(hidden_states, deterministic=deterministic)

        return hidden_states



class AttentionBlock(nn.Module):
    q_chunk_size: int # not used
    k_chunk_size: int # not used
    hidden_size: int
    num_heads: int
    rotary_dim: Optional[int]
    intermediate_size: int
    layer_norm_epsilon: float = 1e-5
    activation_function: str = "gelu"
    attn_pdrop: float = 0.0
    resid_pdrop: float = 0.0
    max_position_embeddings: int = 1024
    dtype: jnp.dtype = jnp.float32
    causal: bool = True
    policy: str = None # not used
    prevent_cse: bool = False # not used
    float32_logits: bool = False

    def setup(self):
        self.attn = _AttentionBlock(
            self.hidden_size,
            self.num_heads,
            self.rotary_dim,
            self.intermediate_size,
            self.layer_norm_epsilon,
            self.activation_function,
            self.resid_pdrop,
            self.max_position_embeddings,
            self.dtype,
            self.causal,
            self.float32_logits,
        )
        self.causal_mask = make_causal_mask(jnp.ones((1, self.max_position_embeddings), dtype="bool"), dtype="bool")

    @nn.compact
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        """
        This function takes projected key, value states from a single input token and concatenates the states to cached
        states from previous steps. This function is slighly adapted from the official Flax repository:
        https://github.com/google/flax/blob/491ce18759622506588784b4fca0e4bf05f8c8cd/flax/linen/attention.py#L252
        """
        # detect if we're initializing by absence of existing cache data.
        is_initialized = self.has_variable("cache", "cached_key")
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

        if is_initialized:
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # update key, value caches with our new 1d spatial slices
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = lax.dynamic_update_slice(cached_key.value, key, indices)
            value = lax.dynamic_update_slice(cached_value.value, value, indices)
            cached_key.value = key
            cached_value.value = value
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
            # causal mask for cached decoder self-attention: our single query position should only attend to those key positions that have already been generated and cached, not the remaining zero elements.
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            attention_mask = combine_masks(pad_mask, attention_mask)
        return key, value, attention_mask

    def __call__(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        deterministic: bool = True,
        init_cache: bool = False,
    ):

        query, key, value = self.attn.forward_qkv(
            hidden_states,
            position_ids,
            deterministic=deterministic,
        )

        query_length, key_length = query.shape[1], key.shape[1]
        if self.has_variable("cache", "cached_key"):
            mask_shift = self.variables["cache"]["cache_index"]
            max_decoder_length = self.variables["cache"]["cached_key"].shape[1]
            causal_mask = lax.dynamic_slice(
                self.causal_mask, (0, 0, mask_shift, 0), (1, 1, query_length, max_decoder_length)
            )
        else:
            causal_mask = self.causal_mask[:, :, :query_length, :key_length]

        batch_size = hidden_states.shape[0]
        causal_mask = jnp.broadcast_to(causal_mask, (batch_size,) + causal_mask.shape[1:])

        attention_mask = jnp.broadcast_to(jnp.expand_dims(attention_mask, axis=(-3, -2)), causal_mask.shape)
        if self.causal:
            attention_mask = combine_masks(attention_mask, causal_mask)
        else:
            attention_mask = attention_mask

        dropout_rng = None
        if not deterministic and self.attn_pdrop > 0.0:
            dropout_rng = self.make_rng("dropout")

        # During fast autoregressive decoding, we feed one position at a time,
        # and cache the keys and values step by step.
        if self.has_variable("cache", "cached_key") or init_cache:
            key, value, attention_mask = self._concatenate_to_cache(key, value, query, attention_mask)

        attention_bias = lax.select(
            attention_mask > 0,
            jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
            jnp.full(attention_mask.shape, -1e9).astype(self.dtype),
        )
        attn_weights = nn.dot_product_attention_weights(
            query,
            key,
            bias=attention_bias,
            dropout_rng=dropout_rng,
            dropout_rate=self.attn_pdrop,
            deterministic=deterministic,
            dtype=self.dtype,
            precision=None,
        )
        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value)
        attn_output = self.attn.attn_out_proj(attn_output, deterministic=deterministic)
        ffn_output = self.attn.forward_ffn(hidden_states + attn_output, deterministic=deterministic)
        outputs = attn_output + ffn_output + hidden_states
        return outputs


if __name__ == '__main__':
    with jax.profiler.trace('/tmp/prof/vanilla'):
        class Model(nn.Module):
            def setup(self):
                self.blocks = [
                    AttentionBlock(
                        q_chunk_size=256,
                        k_chunk_size=256,
                        hidden_size=2048,
                        num_heads=16,
                        rotary_dim=128,
                        intermediate_size=8192,
                        layer_norm_epsilon=1e-5,
                        activation_function="gelu",
                        resid_pdrop=0.0,
                        max_position_embeddings=2048,
                        dtype=jnp.float32,
                        causal=True,
                )
                for _ in range(2)
                ]
            def __call__(self, hidden_states, attention_mask, position_ids):
                for block in self.blocks:
                    hidden_states = block(hidden_states, attention_mask, position_ids)
                return hidden_states

        hidden_states = jnp.zeros((2, 1024, 2048))
        attention_mask = jnp.zeros((2, 1024), dtype=jnp.int32)
        position_ids = jnp.zeros((2, 1024), dtype=jnp.int32)
        model = Model()
        variables = model.init(jax.random.PRNGKey(0), hidden_states, attention_mask, position_ids)
        output = model.apply(variables, hidden_states, attention_mask, position_ids)
        output = output.block_until_ready()

