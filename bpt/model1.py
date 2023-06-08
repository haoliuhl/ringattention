# coding=utf-8
# Copyright 2021 The EleutherAI and The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from functools import partial
from typing import Optional, Tuple
import json

import numpy as np

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from flax.linen import partitioning as nn_partitioning

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutput
from transformers.modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging
from transformers.generation.flax_logits_process import FlaxLogitsProcessorList
from transformers import AutoTokenizer
from jax.sharding import PartitionSpec as PS

from ml_collections import ConfigDict
from ml_collections.config_dict import config_dict
from bpt.tools.utils import function_args_to_config, load_pickle, open_file

from bpt.tools.jax_utils import (
    with_sharding_constraint, get_jax_mesh, get_gradient_checkpoint_policy
)
from bpt.blocks.memeff import AttentionBlock as MemEffAttentionBlock
from bpt.blocks.blockwise_parallel_v1 import AttentionBlock as BPAttentionBlock_v1
from bpt.blocks.blockwise_parallel import AttentionBlock as BPAttentionBlock, Blockwise_LM_Head
from bpt.blocks.vanilla import AttentionBlock as VanillaAttentionBlock


GPT_STANDARD_CONFIGS = {
    # 1.3B
    '1b': {
        'vocab_size': 50432,
        'n_embd': 2048,
        'n_inner': 8192,
        'n_layer': 24,
        'n_head': 16,
        'n_positions': 16384,
        'initializer_range': 0.02,
        'layer_norm_epsilon': 1e-5,
        'use_cache': True,
        'tie_word_embeddings': False,
        'rotary_dim': 128,
        'bos_token_id': 50256,
        'eos_token_id': 50256,
        'n_real_tokens': 50257,
    },
    # 2.7B
    '3b': {
        'vocab_size': 50432,
        'n_embd': 2560,
        'n_inner': 10240,
        'n_layer': 32,
        'n_head': 32,
        'n_positions': 16384,
        'initializer_range': 0.02,
        'layer_norm_epsilon': 1e-5,
        'use_cache': True,
        'tie_word_embeddings': False,
        'rotary_dim': 80,
        'bos_token_id': 50256,
        'eos_token_id': 50256,
        'n_real_tokens': 50257,
    },
    # 6.7B
    '7b': {
        'vocab_size': 50432,
        'n_embd': 4096,
        'n_inner': 16384,
        'n_layer': 32,
        'n_head': 32,
        'n_positions': 16384,
        'initializer_range': 0.02,
        'layer_norm_epsilon': 1e-5,
        'use_cache': True,
        'tie_word_embeddings': False,
        'rotary_dim': 128,
        'bos_token_id': 50256,
        'eos_token_id': 50256,
        'n_real_tokens': 50257,
    },
    # 13B
    '13b': {
        'vocab_size': 50432,
        'n_embd': 5120,
        'n_inner': 20480,
        'n_layer': 40,
        'n_head': 40,
        'n_positions': 16384,
        'initializer_range': 0.02,
        'layer_norm_epsilon': 1e-5,
        'use_cache': True,
        'tie_word_embeddings': False,
        'rotary_dim': 128,
        'bos_token_id': 50256,
        'eos_token_id': 50256,
        'n_real_tokens': 50257,
    },
    # 30B
    '30b': {
        'vocab_size': 50432,
        'n_embd': 7168,
        'n_inner': 28672,
        'n_layer': 48,
        'n_head': 56,
        'n_positions': 16384,
        'initializer_range': 0.02,
        'layer_norm_epsilon': 1e-5,
        'use_cache': True,
        'tie_word_embeddings': False,
        'rotary_dim': 128,
        'bos_token_id': 50256,
        'eos_token_id': 50256,
        'n_real_tokens': 50257,
    },
    # 70B
    '70b': {
        'vocab_size': 50432,
        'n_embd': 8192,
        'n_inner': 32768,
        'n_layer': 80,
        'n_head': 64,
        'n_positions': 16384,
        'initializer_range': 0.02,
        'layer_norm_epsilon': 1e-5,
        'use_cache': True,
        'tie_word_embeddings': False,
        'rotary_dim': 128,
        'bos_token_id': 50256,
        'eos_token_id': 50256,
        'n_real_tokens': 50257,
    },
    'debug': { # A small model for debugging
        'vocab_size': 50432,
        'n_embd': 128,
        'n_inner': 256,
        'n_layer': 2,
        'n_head': 4,
        'n_positions': 16384,
        'initializer_range': 0.02,
        'layer_norm_epsilon': 1e-5,
        'use_cache': True,
        'tie_word_embeddings': False,
        'rotary_dim': 32,
        'bos_token_id': 50256,
        'eos_token_id': 50256,
        'n_real_tokens': 50257,
    },
}

class GPTConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`GPTModel`]. It is used to instantiate a GPT-J
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the GPT-J
    [EleutherAI/gpt-j-6B](https://huggingface.co/EleutherAI/gpt-j-6B) architecture. Configuration objects inherit from
    [`PretrainedConfig`] and can be used to control the model outputs. Read the documentation from [`PretrainedConfig`]
    for more information.
    Args:
        vocab_size (`int`, *optional*, defaults to 50432):
            Vocabulary size of the GPT-J model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`GPTModel`].
        n_positions (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        n_embd (`int`, *optional*, defaults to 4096):
            Dimensionality of the embeddings and hidden states.
        n_layer (`int`, *optional*, defaults to 28):
            Number of hidden layers in the Transformer encoder.
        n_head (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        rotary_dim (`int`, *optional*, defaults to 64):
            Number of dimensions in the embedding that Rotary Position Embedding is applied to.
        n_inner (`int`, *optional*, defaults to 0):
            Dimensionality of the inner feed-forward layers. 0 will set it to 4 times n_embd
        activation_function (`str`, *optional*, defaults to `"gelu_new"`):
            Activation function, to be selected in the list `["relu", "silu", "gelu", "tanh", "gelu_new"]`.
        resid_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        embd_pdrop (`int`, *optional*, defaults to 0.1):
            The dropout ratio for the embeddings.
        attn_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-5):
            The epsilon to use in the layer normalization layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        scale_attn_weights (`bool`, *optional*, defaults to `True`):
            Scale attention weights by dividing by sqrt(hidden_size).
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
    Example:
    ```python
    >>> from transformers import GPTModel, GPTConfig
    >>> # Initializing a GPT-J 6B configuration
    >>> configuration = GPTConfig()
    >>> # Initializing a model from the configuration
    >>> model = GPTModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "gpt"
    attribute_map = {
        "max_position_embeddings": "n_positions",
        "hidden_size": "n_embd",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    def __init__(
        self,
        vocab_size=50432,
        n_positions=2048,
        n_embd=4096,
        n_layer=28,
        n_head=16,
        rotary_dim=64,
        n_inner=None,
        activation_function="gelu_new",
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        scale_attn_weights=True,
        use_cache=True,
        bos_token_id=50256,
        eos_token_id=50256,
        tie_word_embeddings=False,
        gradient_checkpointing='nothing_saveable',
        n_real_tokens=50257,
        fcm_min_ratio=0.0,
        fcm_max_ratio=0.0,
        causal=True,
        attn_type='dot',
        q_chunk_size=1024,
        k_chunk_size=2048,
        scan_layers=True,
        param_scan_axis=0,
        float32_logits=False,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.rotary_dim = rotary_dim
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.scale_attn_weights = scale_attn_weights
        self.use_cache = use_cache
        self.gradient_checkpointing = gradient_checkpointing
        self.n_real_tokens = n_real_tokens
        self.fcm_min_ratio = fcm_min_ratio
        self.fcm_max_ratio = fcm_max_ratio
        self.causal = causal
        self.attn_type = attn_type
        self.q_chunk_size = q_chunk_size
        self.k_chunk_size = k_chunk_size
        self.scan_layers = scan_layers
        self.param_scan_axis = param_scan_axis
        self.float32_logits = float32_logits
        if self.n_real_tokens is None:
            self.n_real_tokens = self.vocab_size

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        super().__init__(
            bos_token_id=bos_token_id, eos_token_id=eos_token_id, tie_word_embeddings=tie_word_embeddings, **kwargs
        )

    @classmethod
    def get_default_config(cls, updates=None):
        none_arg_types = dict(
            n_inner=int,
            rotary_dim=int,
        )
        config = function_args_to_config(cls.__init__, none_arg_types=none_arg_types)

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())

        return config

    @staticmethod
    def get_jax_mesh(axis_dims):
        return get_jax_mesh(axis_dims, ('dp', 'fsdp', 'mp'))

    @staticmethod
    def get_partition_rules(scan_layers=False):
        """ Parition rules for GPT. Note that these rules are orderd, so that
            the beginning rules match first. It is important to use
            PartitionSpec() instead of None here because JAX does not treat
            None as a pytree leaf.
        """
        if scan_layers:
            return (
                ('transformer/wte/embedding', PS('mp', 'fsdp')),
                ('attn/(k_proj|q_proj|v_proj)/kernel', PS(None, 'fsdp', 'mp')),
                ('attn/out_proj/kernel', PS(None, 'mp', 'fsdp')),
                ('attn/fc_in/kernel', PS(None, 'fsdp', 'mp')),
                ('attn/fc_in/bias', PS(None, 'mp')),
                ('attn/fc_out/kernel', PS(None, 'mp', 'fsdp')),
                ('attn/fc_out/bias', PS(None, None)),
                ('ln_[0-9]+/bias', PS(None, None)),
                ('[0-9]+/ln_[0-9]+/scale', PS(None, None)),
                ('ln_f/bias', PS(None)),
                ('ln_f/scale', PS(None)),
                ('lm_head/kernel', PS('fsdp', 'mp')),
                ('lm_head/bias', PS('mp')),
                ('.*', PS(None)),
            )
        else:
            return (
                ('transformer/wte/embedding', PS('mp', 'fsdp')),
                ('attn/(k_proj|q_proj|v_proj)/kernel', PS('fsdp', 'mp')),
                ('attn/out_proj/kernel', PS('mp', 'fsdp')),
                ('attn/fc_in/kernel', PS('fsdp', 'mp')),
                ('attn/fc_in/bias', PS('mp')),
                ('attn/fc_out/kernel', PS('mp', 'fsdp')),
                ('attn/fc_out/bias', PS(None)),
                ('ln_[0-9]+/bias', PS(None)),
                ('[0-9]+/ln_[0-9]+/scale', PS(None)),
                ('ln_f/bias', PS(None)),
                ('ln_f/scale', PS(None)),
                ('lm_head/kernel', PS('fsdp', 'mp')),
                ('lm_head/bias', PS('mp')),
                ('.*', PS(None)),
            )

    @staticmethod
    def get_weight_decay_exclusions():
        return (
            'ln_[0-9]+/bias', 'ln_[0-9]+/scale', 'ln_f/bias', 'ln_f/scale',
            'bias'
        )

    @staticmethod
    def rng_keys():
        return ('params', 'dropout', 'fcm')

    @staticmethod
    def get_tokenizer_config(updates=None):
        config = ConfigDict()
        config.name = 'EleutherAI/gpt-j-6B'
        config.bos_token = '<|endoftext|>'
        config.eos_token = '<|endoftext|>'
        config.pad_token = '<|extratoken_40|>'
        config.cls_token = '<|extratoken_41|>'
        config.mask_token = '<|extratoken_42|>'

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())

        return config

    @classmethod
    def get_tokenizer(cls, config, padding_side='left', truncation_side='right'):
        config = cls.get_tokenizer_config(config)
        return AutoTokenizer.from_pretrained(
            config.name,
            bos_token=config.bos_token,
            eos_token=config.eos_token,
            pad_token=config.pad_token,
            cls_token=config.cls_token,
            mask_token=config.mask_token,
            padding_side=padding_side,
            truncation_side=truncation_side,
        )

    @staticmethod
    def load_pretrained(name, dtype=jnp.float32):
        with jax.default_device(jax.devices("cpu")[0]):
            params = FlaxGPTForCausalLM.from_pretrained(
                name, _do_init=False, dtype=dtype
            )[1]
            params = freeze({'params': params})
        return jax.device_get(params)

    @classmethod
    def load_config(cls, path):
        if path in GPT_STANDARD_CONFIGS:
            return cls.from_dict(GPT_STANDARD_CONFIGS[path])
        load_type, load_path = path.split('::', 1)
        if load_type == 'pickle':
            return cls.from_dict(load_pickle(load_path)['gpt_config'])
        elif load_type == 'json':
            with open_file(load_path, 'r') as fin:
                raw_config = fin.read()
            return cls.from_dict(json.loads(raw_config))
        elif load_type == 'huggingface':
            return cls.from_pretrained(load_path)
        else:
            raise ValueError(f'Unsupported load config type: {load_type}')


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "gpt"
_CONFIG_FOR_DOC = "GPTConfig"

GPT_START_DOCSTRING = r"""
    This model inherits from [`FlaxPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)
    This model is also a Flax Linen
    [flax.nn.Module](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html) subclass. Use it as a
    regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior.
    Finally, this model supports inherent JAX features such as:
    - [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
    - [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
    - [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
    - [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)
    Parameters:
        config ([`GPTConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~FlaxPreTrainedModel.from_pretrained`] method to load the model weights.
        dtype (`jax.numpy.dtype`, *optional*, defaults to `jax.numpy.float32`):
            The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and
            `jax.numpy.bfloat16` (on TPUs).
            This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If
            specified all the computation will be performed with the given `dtype`.
            **Note that this only specifies the dtype of the computation and does not influence the dtype of model
            parameters.**
            If you wish to change the dtype of the model parameters, see [`~FlaxPreTrainedModel.to_fp16`] and
            [`~FlaxPreTrainedModel.to_bf16`].
"""

GPT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`numpy.ndarray` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length`. Indices of input sequence tokens in the vocabulary.
            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
            [What are input IDs?](../glossary#input-ids)
        attention_mask (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
        position_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.
        past_key_values (`Dict[str, np.ndarray]`, *optional*, returned by `init_cache` or when passing previous `past_key_values`):
            Dictionary of pre-computed hidden-states (key and values in the attention blocks) that can be used for fast
            auto-regressive decoding. Pre-computed key and value hidden-states are of shape *[batch_size, max_length]*.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""



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


class FlaxGPTBlock(nn.Module):
    config: GPTConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        hidden_size = self.config.hidden_size
        inner_dim = self.config.n_inner if self.config.n_inner is not None else 4 * hidden_size

        attention_blocks = {
            # default vanilla transformer (Vaswani et al).
            'vanilla': VanillaAttentionBlock,
            # default memory efficient transformer (Rabe et al and Dao et al).
            'memeff': MemEffAttentionBlock,
            # default blockwise parallel transformer (Liu et al).
            'blockwise_parallel': BPAttentionBlock,
            # less cleaner blockwise parallel transformer used in the paper.
            'blockwise_parallel_v1': BPAttentionBlock_v1,
        }

        if self.config.attn_type in attention_blocks:
            Block = attention_blocks[self.config.attn_type]
        else:
            raise ValueError(f"Unknown attention type {self.config.attn_type}")

        self.attn = Block(
            self.config.q_chunk_size,
            self.config.k_chunk_size,
            self.config.hidden_size,
            self.config.num_attention_heads,
            self.config.rotary_dim,
            inner_dim,
            self.config.layer_norm_epsilon,
            self.config.activation_function,
            self.config.attn_pdrop,
            self.config.resid_pdrop,
            self.config.max_position_embeddings,
            self.dtype,
            self.config.causal,
            policy=self.config.gradient_checkpointing,
            prevent_cse=not self.config.scan_layers,
            float32_logits=self.config.float32_logits,
        )

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        fcm_mask=None,
    ):
        attn_outputs = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            deterministic=deterministic,
            init_cache=init_cache,
        )
        attn_weights = None
        if self.config.scan_layers: # NOTE: this is a hack to work with scan_layers
            outputs = attn_outputs, None
        else:
            outputs = (attn_outputs, attn_weights) if output_attentions else (attn_outputs,)
        return outputs


class FlaxGPTPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = GPTConfig
    base_model_prefix = "transformer"
    module_class: nn.Module = None

    def __init__(
        self,
        config: GPTConfig,
        input_shape: Tuple = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # init input tensors
        input_ids = jnp.zeros(input_shape, dtype="i4")
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape)
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        if self.config.add_cross_attention:
            encoder_hidden_states = jnp.zeros(input_shape + (self.config.n_embd,))
            encoder_attention_mask = attention_mask
            module_init_outputs = self.module.init(
                rngs,
                input_ids,
                attention_mask,
                position_ids,
                encoder_hidden_states,
                encoder_attention_mask,
                return_dict=False,
            )
        else:
            module_init_outputs = self.module.init(rngs, input_ids, attention_mask, position_ids, return_dict=False)

        random_params = module_init_outputs["params"]

        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    def init_cache(self, batch_size, max_length):
        r"""
        Args:
            batch_size (`int`):
                batch_size used for fast auto-regressive decoding. Defines the batch size of the initialized cache.
            max_length (`int`):
                maximum possible length for auto-regressive decoding. Defines the sequence length of the initialized
                cache.
        """
        # init input variables to retrieve cache
        input_ids = jnp.ones((batch_size, max_length))
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        init_variables = self.module.init(
            jax.random.PRNGKey(0), input_ids, attention_mask, position_ids, return_dict=False, init_cache=True
        )
        return init_variables["cache"]

    def _get_logits_processor(self,*args, **kwargs) -> FlaxLogitsProcessorList:
        processors = super()._get_logits_processor(*args, **kwargs)
        def squash_extra_tokens(input_ids, scores, cur_len):
            return scores.at[:, self.config.n_real_tokens:].set(-float('inf'))

        processors.append(squash_extra_tokens)
        return processors

    @add_start_docstrings_to_model_forward(GPT_INPUTS_DOCSTRING)
    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        params: dict = None,
        past_key_values: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        batch_size, sequence_length = input_ids.shape

        if position_ids is None:
            if past_key_values is not None:
                raise ValueError("Make sure to provide `position_ids` when passing `past_key_values`.")

            position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, sequence_length))

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        inputs = {"params": params or self.params}

        # if past_key_values are passed then cache is already initialized a private flag init_cache has to be passed down to ensure cache is used. It has to be made sure that cache is marked as mutable so that it can be changed by FlaxGPTAttention module
        if past_key_values:
            inputs["cache"] = past_key_values
            mutable = ["cache"]
        else:
            mutable = False

        outputs = self.module.apply(
            inputs,
            jnp.array(input_ids, dtype="i4"),
            jnp.array(attention_mask, dtype="i4"),
            jnp.array(position_ids, dtype="i4"),
            not train,
            False,
            output_attentions,
            output_hidden_states,
            return_dict,
            rngs=rngs,
            mutable=mutable,
        )

        # add updated cache to model output
        if past_key_values is not None and return_dict:
            outputs, past_key_values = outputs
            outputs["past_key_values"] = unfreeze(past_key_values["cache"])
            return outputs
        elif past_key_values is not None and not return_dict:
            outputs, past_key_values = outputs
            outputs = outputs[:1] + (unfreeze(past_key_values["cache"]),) + outputs[1:]

        return outputs


class FlaxGPTBlockCollection(nn.Module):
    config: GPTConfig
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        if not deterministic and self.config.fcm_max_ratio > 0:
            # Apply forgetful causal mask
            batch_size, seq_length = hidden_states.shape[0], hidden_states.shape[1]
            fcm_ratio = jax.random.uniform(
                self.make_rng('fcm'), shape=(batch_size, 1, 1, 1),
                minval=self.config.fcm_min_ratio,
                maxval=self.config.fcm_max_ratio
            )
            fcm_mask = jax.random.uniform(
                self.make_rng('fcm'),
                shape=(batch_size, 1, seq_length, seq_length)
            ) > fcm_ratio
            fcm_mask = fcm_mask.at[:, :, :, 0].set(True)
            fcm_mask = fcm_mask.astype('bool')
        else:
            fcm_mask = None

        block = FlaxGPTBlock
        if self.config.gradient_checkpointing != '':
            FlaxGPT2CheckpointBlock = nn.remat(
                block, static_argnums=(3, 4, 5, 6),
                prevent_cse=not self.config.scan_layers,
                policy=get_gradient_checkpoint_policy(self.config.gradient_checkpointing)
            )
            block = FlaxGPT2CheckpointBlock
        if self.config.scan_layers:
            initializing = self.is_mutable_collection('params')
            params_spec = (
                self.config.param_scan_axis if initializing else
                nn_partitioning.ScanIn(self.config.param_scan_axis))
            cache_spec = 0
            hidden_states, _ = nn.scan(
                block,
                variable_axes={
                    'params': params_spec,
                    'cache': cache_spec,
                    'intermediates': 0
                },
                split_rngs={
                    'params': True,
                    'dropout': True
                },
                in_axes=(nn.broadcast, nn.broadcast, nn.broadcast, nn.broadcast, nn.broadcast, nn.broadcast),
                length=self.config.num_hidden_layers,
                metadata_params={nn.PARTITION_NAME: 'scan_decoder_layer'},
                )(config=self.config, name='scan_decoder', dtype=self.dtype)(
                    hidden_states,
                    attention_mask,
                    position_ids,
                    deterministic,
                    init_cache,
                    output_attentions,
                    fcm_mask,
                )
        else:
            blocks = [
                block(self.config, name=str(i), dtype=self.dtype) for i in range(self.config.num_hidden_layers)
            ]
            for block in blocks:
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                layer_outputs = block(
                    hidden_states,
                    attention_mask,
                    position_ids,
                    deterministic,
                    init_cache,
                    output_attentions,
                    fcm_mask,
                )
                hidden_states = layer_outputs[0]

                if output_attentions:
                    all_attentions += (layer_outputs[1],)

        # this contains possible `None` values - `FlaxGPTModule` will filter them out
        outputs = (hidden_states, all_hidden_states, all_attentions)

        return outputs


class FlaxGPTModule(nn.Module):
    config: GPTConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.embed_dim = self.config.hidden_size

        self.wte = nn.Embed(
            self.config.vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )
        self.dropout = nn.Dropout(rate=self.config.embd_pdrop)
        self.h = FlaxGPTBlockCollection(self.config, dtype=self.dtype)
        self.ln_f = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon, dtype=self.dtype)

    def __call__(
        self,
        input_ids,
        attention_mask,
        position_ids,
        deterministic=True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        input_embeds = self.wte(input_ids.astype("i4"))

        hidden_states = self.dropout(input_embeds, deterministic=deterministic)

        outputs = self.h(
            hidden_states,
            attention_mask,
            position_ids=position_ids,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.ln_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = outputs[1] + (hidden_states,)
            outputs = (hidden_states, all_hidden_states) + outputs[2:]
        else:
            outputs = (hidden_states,) + outputs[1:]

        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=outputs[1],
            attentions=outputs[-1],
        )


@add_start_docstrings(
    "The bare GPT Model transformer outputting raw hidden-states without any specific head on top.",
    GPT_START_DOCSTRING,
)
class FlaxGPTModel(FlaxGPTPreTrainedModel):
    module_class = FlaxGPTModule


append_call_sample_docstring(
    FlaxGPTModel,
    _CHECKPOINT_FOR_DOC,
    FlaxCausalLMOutput,
    _CONFIG_FOR_DOC,
)


class FlaxGPTForCausalLMModule(nn.Module):
    config: GPTConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.transformer = FlaxGPTModule(self.config, dtype=self.dtype)
        if self.config.attn_type == 'blockwise_parallel' or self.config.attn_type == 'blockwise_parallel_v1':
            self.lm_head = Blockwise_LM_Head(self.config.vocab_size,
                                      self.config.q_chunk_size, dtype=self.dtype,
                                      prevent_cse=not self.config.scan_layers)
        else:
            self.lm_head = nn.Dense(
                self.config.vocab_size,
                dtype=self.dtype,
                kernel_init=jax.nn.initializers.variance_scaling(
                    scale=1.0, mode='fan_in',
                    distribution='normal',
                )
            )

    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        batch_size, seq_length = input_ids.shape
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)
        if position_ids is None:
            position_ids = jnp.broadcast_to(
                jnp.clip(jnp.cumsum(attention_mask, axis=-1) - 1, a_min=0),
                (batch_size, seq_length)
            )

        outputs = self.transformer(
            input_ids,
            attention_mask,
            position_ids,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        if self.config.tie_word_embeddings:
            shared_kernel = self.transformer.variables["params"]["wte"]["embedding"].T
            lm_logits = self.lm_head.apply({"params": {"kernel": shared_kernel}}, hidden_states)
        else:
            lm_logits = self.lm_head(hidden_states)

        if not return_dict:
            return (lm_logits,) + outputs[1:]

        return FlaxCausalLMOutput(logits=lm_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)


@add_start_docstrings(
    """
    The GPT Model transformer with a language modeling head on top.
    """,
    GPT_START_DOCSTRING,
)
class FlaxGPTForCausalLM(FlaxGPTPreTrainedModel):
    module_class = FlaxGPTForCausalLMModule

    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[jnp.DeviceArray] = None):
        # initializing the cache
        batch_size, seq_length = input_ids.shape

        past_key_values = self.init_cache(batch_size, max_length)
        # Note that usually one would have to put 0's in the attention_mask for x > input_ids.shape[-1] and x < cache_length.
        # But since GPT uses a causal mask, those positions are masked anyways.
        # Thus we can create a single static attention_mask here, which is more efficient for compilation
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        if attention_mask is not None:
            position_ids = attention_mask.cumsum(axis=-1) - 1
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))
        else:
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length))

        return {
            "past_key_values": past_key_values,
            "attention_mask": extended_attention_mask,
            "position_ids": position_ids,
        }

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
        return model_kwargs


append_call_sample_docstring(
    FlaxGPTForCausalLM,
    _CHECKPOINT_FOR_DOC,
    FlaxCausalLMOutput,
    _CONFIG_FOR_DOC,
)
