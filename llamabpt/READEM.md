An implementation of blockwise parallel transformer for the Llama architecture.

## Usage

Use `q_chunk_size` and `k_chunk_size` to control the number of chunks in the self-attention.
Use `ffn_chunk_size` to control the number of chunks in the feed-forward network.
Use `head_chunk_size` to control the number of chunks in the lm head.
Use `loss_chunk_size` to control the number of chunks in the final loss computation.

For these arguments, set to -1 to disable chunking. For example, `--llama.ffn_chunk_size=-1` disables blockwise parallelism in the feed-forward network, which reduces BPT to the memory-efficient transformer.

Use `remat_policy` to control the rematerialization policy, recommended is `nothing_saveable`.

Use `float32_logits` to control whether to use float32 for logits for numerical stability, recommended for large models.
