## Ring Attention with Blockwise Transformers for Near-Infinite Context

Hao Liu, Matei Zaharia, Pieter Abbeel

Paper: https://arxiv.org/abs/2310.01889

## Blockwise Parallel Transformer for Large Context Models

Hao Liu, Pieter Abbeel

Paper: https://arxiv.org/abs/2305.19370

---
This codebase provides the implementation of the Ring Attention with Blockwise Transformers. The model is described in the paper [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/pdf/2310.01889.pdf) and [Blockwise Parallel Transformer for Large Context Models](https://arxiv.org/pdf/2305.19370.pdf).

Blockwise Parallel Transformers (BPT) compute attention and feedforward in a blockwise manner, allowing for the training and inference of sequences up to four times longer than those manageable by standard memory-efficient attention methods, such as flash attention.

Ring Attention with Blockwise Parallel Transformers enables training sequences up to a length of 'number of devices' times longer than those possible with BPT. This is achieved by distributing the attention and feedforward computation across multiple devices and overlapping the communication with computation. Thanks to the blockwise computing of the attention and feedforward network, it is possible to train with tens of millions of tokens in context size without adding any communication or computation overhead.

This codebase is utilized to train the Large World Model (LWM) whose project page is [LWM project](https://largeworldmodel.github.io/) and codebase with features for million-length vision-language training is [LWM codebase](https://github.com/LargeWorldModel/LWM).


## Requirements
Install the requirements with:
```
conda env create -f gpu_requirements.yml
```
or set up TPU VM (`tpu-ubuntu2204-base` image required) with:
```
sh tpu_requirements.sh
```

## Code structure

The code is organized as follows:
- `scripts/` contains the requirements and scripts for preparing the data.
- `bpt/` contains the example of applying BPT and RingAttention to LLaMA.

The implementation optimized sharding annotations for distributed FSDP training. It also supports RingAttention, BPT, memeff/flashattention, and vanilla transformers.

## Usage

Use `scan_query_chunk_size` and `scan_key_chunk_size` to control the block size in blockwise compute of the self-attention.
Use `scan_mlp_chunk_size` to control the block size in blockwise compute of the feedforward network.

Use `scan_attention=True` and `scan_mlp=True` to enable/disable blockwise compute in the self-attention and feed-forward network.

Use `remat_attention` and `remat_mlp` to control the rematerialization policy, recommended is `nothing_saveable`.

For the LLaMA tokenizer, you can use OpenLLaMAv2 tokenizer or the official LLaMA tokenizer.

For the training dataset, you can use `scripts/prepare_data.py` to download OpenWebText dataset and prepare the dataset for training.

You can use `mesh_dim` to control the degree of parallelism and Ring Attention.
For example, `mesh_dim='1,64,4,1'` means 1 data parallelism, 64 fully sharded data parallelism, 4 tensor parallelism, and 1 sequence parallelism. `mesh_dim='1,1,4,64'` means 1 data parallelism, 1 fully sharded data parallelism, 4 tensor parallelism, and 64 sequence parallelism.

Ring Attention use the last dimension of `mesh_dim` to control how many devices to use for Ring Attention, ie, `mesh_dim='1,1,4,64'` means 64 devices are used for Ring Attention, meaning that context length can be expanded 64 times.

#### Blockwise Transformers

An example of using BPT to train 13B LLaMA model with 32K context length and 2M batch size on TPU v4-512 is as follows:

```bash
python3 -m bpt.train \
    --mesh_dim='1,64,4,1' \
    --dtype='bf16' \
    --total_steps=480000 \
    --log_freq=200 \
    --save_model_freq=0 \
    --save_milestone_freq=1000 \
    --load_llama_config='13b' \
    --update_llama_config="dict(max_sequence_length=32768,scan_attention=True,scan_query_chunk_size=2048,scan_key_chunk_size=4096,remat_attention='nothing_saveable',scan_mlp=True,scan_mlp_chunk_size=2048,remat_mlp='nothing_saveable',remat_block='nothing_saveable',scan_layers=True,attention_type='blockwise',param_scan_axis=0,mesh_dim='1,64,4,1')" \
    --load_dataset_state='' \
    --load_checkpoint='' \
    --tokenizer.vocab_file="<path to your llama tokenizer>" \
    --optimizer.type='adamw' \
    --optimizer.adamw_optimizer.weight_decay=0.1 \
    --optimizer.adamw_optimizer.lr=1.5e-4 \
    --optimizer.adamw_optimizer.end_lr=1.5e-5 \
    --optimizer.adamw_optimizer.lr_warmup_steps=2000 \
    --optimizer.adamw_optimizer.lr_decay_steps=480000 \
    --train_dataset.type='json' \
    --train_dataset.text_processor.fields='text' \
    --train_dataset.json_dataset.path="<path to your training dataset>" \
    --train_dataset.json_dataset.seq_length=32768 \
    --train_dataset.json_dataset.batch_size=64 \
    --train_dataset.json_dataset.tokenizer_processes=16 \
    --checkpointer.save_optimizer_state=True
```

#### Ring Attention
Similarly, an example of using Ring Attention to train 13B LLaMA model with 2M context length and 2M batch size on TPU v4-512 is as follows:

```bash
python3 -m bpt.train \
    --mesh_dim='1,1,4,64' \
    --dtype='bf16' \
    --total_steps=480000 \
    --log_freq=200 \
    --save_model_freq=0 \
    --save_milestone_freq=1000 \
    --load_llama_config='7b' \
    --update_llama_config="dict(max_sequence_length=2097152,scan_attention=True,scan_query_chunk_size=2048,scan_key_chunk_size=4096,remat_attention='nothing_saveable',scan_mlp=True,scan_mlp_chunk_size=2048,remat_mlp='nothing_saveable',remat_block='nothing_saveable',scan_layers=True,attention_type='ring_blockwise',param_scan_axis=0,mesh_dim='1,1,4,64')" \
    --load_dataset_state='' \
    --load_checkpoint='' \
    --tokenizer.vocab_file="<path to your llama tokenizer>" \
    --optimizer.type='adamw' \
    --optimizer.adamw_optimizer.weight_decay=0.1 \
    --optimizer.adamw_optimizer.lr=1.5e-4 \
    --optimizer.adamw_optimizer.end_lr=1.5e-5 \
    --optimizer.adamw_optimizer.lr_warmup_steps=2000 \
    --optimizer.adamw_optimizer.lr_decay_steps=480000 \
    --train_dataset.type='json' \
    --train_dataset.text_processor.fields='text' \
    --train_dataset.json_dataset.path="<path to your training dataset>" \
    --train_dataset.json_dataset.seq_length=2097152 \
    --train_dataset.json_dataset.batch_size=1 \
    --train_dataset.json_dataset.tokenizer_processes=16 \
    --checkpointer.save_optimizer_state=True
```

Switching between BPT and Ring Attention is as simple as changing the `attention_type` parameter, and the `mesh_dim` parameter. `attention_type='blockwise'` means BPT, and `attention_type='ring_blockwise'` means Ring Attention. Use `mesh_dim` to control how many devices for FSDP/TP/DP, and how many devices for Ring Attention.

For large scale end-to-end training on TPU or on GPU cluster with high bandwidth inter connection, we recommend using FSDP to shard large models and using \ours to achieve large context. If total batch size is too large, add tensor parallelism to reduce the global batch size. The degree of parallelism can be adjusted using the \texttt{mesh\_dim} parameter within the codebase.
To illustrate, consider a setup with 512 devices, such as 512x A100. If the model size is 30B, you can shard it across 8 devices and allocate the remaining 32 devices for \ours. This setup allows the context size to be expanded 32 times more than if you didn't use \ours. Conversely, for models sized 7B or 3B, there is no need for FSDP. This means you can utilize all 512 devices exclusively to expand the context using \ours by 512 times. Building upon the result that our approach allows for a 256K context size when using 8x A100 GPUs, it suggests that by employing 512 A100 GPUs, the potential context size can be expanded to 16 million.

For finetuning purposes, e.g., finetuning a huggingface hosted model. We provide a script to convert huggingface model to our format. The script is in `scripts/hf2jax.py`. The script takes in a downloaded huggingface model path and outputs a jax format.
The usage is as follows:
```bash
python hf2jax.py  \
       --checkpoint_dir /path/hf_format_dir/    \
       --output_file /path/output   \
       --model_size 7b \
       --streaming
```
Then you can load the model using the `--load_checkpoint` flag:
```bash
--load_checkpoint='params::/path/output'
```


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
