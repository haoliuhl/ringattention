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

from ringattention.ringattention_base import below_or_on_diag, segment_ids_ops
from ringattention.ringattention_base import ring_attention


ring_flash_attention_gpu = ring_attention # fused attention is not yet supported on GPU
