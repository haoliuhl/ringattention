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
