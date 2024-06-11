## GPU/TPU Jax implementation of RingAttention

This codebase provides the implementation of the Ring Attention with Blockwise Transformers. The model is described in the paper [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/pdf/2310.01889.pdf) and [Blockwise Parallel Transformer for Large Context Models](https://arxiv.org/pdf/2305.19370.pdf).

Blockwise Parallel Transformers (BPT) compute attention and feedforward in a blockwise manner, allowing for the training and inference of sequences up to four times longer than those manageable by standard memory-efficient attention methods, such as flash attention.

Ring Attention with Blockwise Parallel Transformers enables training sequences up to a length of 'number of devices' times longer than those possible with BPT. This is achieved by distributing the attention and feedforward computation across multiple devices and overlapping the communication with computation. Thanks to the blockwise computing of the attention and feedforward network, it is possible to train with tens of millions of tokens in context size without adding any communication or computation overhead.


### Example usage here with code snippets
```python
platform = xla_bridge.get_backend().platform
if platform == "tpu":
    ring_attention_fn = ring_flash_attention_tpu
elif platform == "gpu":
    ring_attention_fn = ring_flash_attention_gpu
else:
    raise ValueError(f"Unsupported platform: {platform}")
ring_attention_sharded = shard_map(
    partial(
        ring_attention_fn,
        axis_name="sp", # if 'sp' is used, the function will be sharded along context size
        float32_logits=True,
        cache_idx=None, # no cache
        blockwise_kwargs=dict(
            causal_block_size=1, # equivalent to being causal
            deterministic=deterministic,
            dropout_rng=dropout_rng,
            attn_pdrop=self.config.attn_pdrop,
            query_chunk_size=self.config.scan_query_chunk_size,
            key_chunk_size=self.config.scan_key_chunk_size,
            dtype=self.dtype,
            policy=get_gradient_checkpoint_policy('nothing_saveable'),
            precision=self.precision,
            prevent_cse=not self.config.scan_layers,
        )
    ),
    mesh=LLaMAConfig.get_jax_mesh(self.config.mesh_dim),
    in_specs=(
        PS(("dp", "fsdp"), "sp", "tp", None),
        PS(("dp", "fsdp"), "sp", "tp", None),
        PS(("dp", "fsdp"), "sp", "tp", None),
        PS(("dp", "fsdp"), None, None, None),
        PS(("dp", "fsdp"), None),
    ),
    out_specs=PS(("dp", "fsdp"), "sp", "tp", None),
    check_rep=False
)
attn_output = ring_attention_sharded(xq, xk, xv, attention_bias, segment_ids)
```

This codebase is utilized to train the Large World Model (LWM) whose project page is [LWM project](https://largeworldmodel.github.io/) and codebase with features for million-length vision-language training is [LWM codebase](https://github.com/LargeWorldModel/LWM).


## Reference
If you find our work relevant to your research, please cite:
```bibtex
@article{liu2023blockwise,
    title={Blockwise Parallel Transformer for Large Context Models},
    author={Liu, Hao and Abbeel, Pieter},
    journal={Advances in neural information processing systems},
    year={2023}
}
```
```bibtex
@article{liu2023ring,
    title={Ring Attention with Blockwise Transformers for Near-Infinite Context},
    author={Liu, Hao and Zaharia, Matei and Abbeel, Pieter},
    journal={arXiv preprint arXiv:2310.01889},
    year={2023}
}
```
