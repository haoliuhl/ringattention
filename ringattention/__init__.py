"""This module contains ring attention forward and backward pass, supporting both blockwise computation and TPU-compatible fused attention.
It features blockwise computation for feedforward networks to reduce memory cost.
For more details, refer to 'RingAttention' at https://arxiv.org/abs/2310.01889 and 'Blockwise Parallel Transformers' at https://arxiv.org/abs/2305.19370.
"""

from .ringattention_base import ring_attention as ring_attention_base
from .ringattention_base import blockwise_feedforward
from .ringattention_gpu import ring_flash_attention_gpu
from .ringattention_tpu import ring_flash_attention_tpu
import jax

platform = jax.lib.xla_bridge.get_backend().platform
if platform == "tpu":
    ringattention = ring_flash_attention_tpu
elif platform == "gpu":
    ringattention = ring_flash_attention_gpu
else:
    ringattention = ring_attention_base

__all__ = ["ringattention", "blockwise_feedforward"]
