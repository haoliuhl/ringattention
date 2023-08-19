# Blockwise Parallel Transformer for Long Context Large Models

Hao Liu, Pieter Abbeel

Paper: https://arxiv.org/abs/2305.19370

This is the implementation of the Blockwise Parallel Transformer (BPT) model. The model is described in the paper [Blockwise Parallel Transformer for Long Context Large Models](https://arxiv.org/pdf/2305.19370.pdf).

BPT allows training 32x longer sequence length than vanilla transformer with the same memory cost and 4x longer than memeff / flashattention.

## Requirements
Install the requirements with:
```
conda env create -f gpu_requirements.yml
```
or set up TPU VM with:
```
sh tpu_requirements.sh
```

## Code structure

The code is organized as follows:
- `scripts/` contains the requirements and scripts for preparing the data.
- `llamabpt/` contains the example of applying BPT to LLaMA.

The implementation optimized sharding annotations for distributed FSDP training. It also supports BPT, memeff/flashattention, and vanilla transformer.

## Usage

Use `scan_query_chunk_size` and `scan_key_chunk_size` to control the block size in blockwise compute of the self-attention.
Use `scan_mlp_chunk_size` to control the block size in blockwise compute of the feedforward network.

Use `scan_attention=True` and `scan_mlp=True` to enable/disable blockwise compute in the self-attention and feed-forward network.

Use `remat_attention` and `remat_mlp` to control the rematerialization policy, recommended is `nothing_saveable`.

For the LLaMA tokenizer, you can use OpenLLaMAv2 tokenizer or the official LLaMA tokenizer.

For the training dataset, you can use `scripts/prepare_data.py` to download OpenWebText dataset and prepare the dataset for training.

An example of training 13B LLaMA model with 32K context length and 2M batch size on TPU v4-512 is as follows:

```bash
python3 -m llamabpt.train \
    --mesh_dim='1,64,4' \
    --dtype='bf16' \
    --total_steps=480000 \
    --log_freq=200 \
    --save_model_freq=0 \
    --save_milestone_freq=1000 \
    --load_llama_config='7b' \
    --update_llama_config='' \
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
    --checkpointer.save_optimizer_state=True \
    --llama.scan_attention=True \
    --llama.remat_attention='nothing_saveable' \
    --llama.scan_query_chunk_size=2048 \
    --llama.scan_key_chunk_size=4096 \
    --llama.scan_mlp=True \
    --llama.remat_mlp='nothing_saveable' \
    --llama.scan_mlp_chunk_size=2048 \
    --llama.remat_block='' \
    --llama.scan_layers=True \
    --llama.param_scan_axis=0 \
    --llama.max_sequence_length=32768
```

For large scale distributed training on TPU or on GPU cluster with good inter connection, we recommend first using FSDP, and add tensor parallelism when the model is very large or the global batch size is too large. You can control parallelism by using mesh_dim.

*Note*: This code relies on compiler to fuse blockwise attention, ffn and loss computation, while this is enough for cutting down the memory cost (BPT can train 32x longer sequence length than vanilla transformer with the same memory cost and 4x longer than memeff / flashattention), it is not enough for achieving optimal speed up due to compiler limitation.
The ideal way is to fuse manually, which is not supported by the current code.

The repo for the original BPT release is in the [initial release](https://github.com/lhao499/blockwise-parallel-transformer/tree/bpt_init_v1) branch.

## Reference
If you find our work relevant to your research, please cite:
```bibtex
@article{liu2023blockwise,
    title={Blockwise Parallel Transformer for Long Context Large Models},
    author={Hao Liu and Pieter Abbeel},
    year={2023},
    journal={arXiv preprint arxiv:2305.19370}
}
```
