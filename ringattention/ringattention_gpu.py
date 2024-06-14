"""This module contains ring attention forward and backward pass, supporting both blockwise computation and TPU-compatible fused attention.
It features blockwise computation for feedforward networks to reduce memory cost.
For more details, refer to 'RingAttention' at https://arxiv.org/abs/2310.01889 and 'Blockwise Parallel Transformers' at https://arxiv.org/abs/2305.19370.
"""

import numpy as np
import flax.linen as nn
import jax
import jax.lax as lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from einops import rearrange
from functools import partial
import dataclasses
import functools
from typing import Any, NamedTuple, Optional

from ringattention.ringattention_base import below_or_on_diag, _chunk_attention_bias
from ringattention.ringattention_base import ring_attention


ring_flash_attention_gpu = ring_attention # fused attention is not yet supported on GPU
