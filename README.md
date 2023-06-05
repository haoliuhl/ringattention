# Blockwise Parallel Transformer for Long Context Large Models

Hao Liu, Pieter Abbeel

Paper: https://arxiv.org/abs/2305.19370

This is the implementation of the Blockwise Parallel Transformer (BPT) model. The model is described in the paper [Blockwise Parallel Transformer for Long Context Large Models](https://arxiv.org/pdf/2305.19370.pdf).

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
- `bpt/` contains the implementation of the BPT model.
- `bpt/blocks/` contains the implementation of the vanilla transformer, memory efficient transformer, and blockwise parallel transformer.
- `data.py` contains the implementation of the data loader.
- `train.py` contains the training loop.
- `model.py` contains the model implementation.
- `bpt/tools/` contains utility functions for training, logging, profiling, etc.

## Usage
An example script for training a 3B Transformers with 65536 context window length on 1 A100 80GB using blockwise parallel transformer is as follows:
```bash
#! /bin/bash

export PYTHONPATH="$PYTHONPATH:$PWD"
export WANDB_API_KEY=''
export project_id='bpt'
export experiment_note=''
export XLA_PYTHON_CLIENT_MEM_FRACTION=.95
export experiment_id='3b,blockwise_parallel,65536,2048,4096'

python3 -m bpt.train \
    --mesh_dim='1,1,-1' \
    --total_steps=1000 \
    --optimizer.type='adamw' \
    --optimizer.accumulate_gradient_steps=1 \
    --log_freq=2 \
    --save_model_freq=-1 \
    --load_gpt_config='3b' \
    --profile_steps=0 \
    --stop_after_profile=True \
    --gpt.scan_layers=True \
    --gpt.param_scan_axis=0 \
    --gpt.q_chunk_size=2048 \
    --gpt.k_chunk_size=4096 \
    --gpt.attn_type='blockwise_parallel' \
    --gpt.gradient_checkpointing='nothing_saveable' \
    --gpt.n_positions=65536 \
    --text_processor.fields='text' \
    --train_dataset.path='./local/owt/openwebtext_train.jsonl' \
    --train_dataset.seq_length=65536 \
    --train_dataset.batch_size=1 \
    --eval_steps=0 \
    --logger.online=False \
    --logger.project_id="$project_id" \
    --logger.experiment_id="$experiment_id" \
    --logger.append_uuid=False \
    --logger.experiment_note="$experiment_note" \
    --logger.output_dir="$HOME/experiment_output/$project_id" \
    --logger.wandb_dir="$HOME/experiment_output/$project_id" \
    --logger.profile_dir="$HOME/experiment_output/$project_id/jax_profile"
```
Explanation of the arguments:
- `mesh_dim`: the mesh dimension, the initial dimension of the mesh is usually for data parallelism, the second dimension is utilized for fully sharded data parallelism (FSDP), and the third dimension is allocated for tensor parallelism.
- `total_steps`: the total number of training steps.
- `optimizer.accumulate_gradient_steps`: the number of gradient accumulation steps.
- `load_gpt_config`: the GPT model configuration, can be one of the configs in `GPT_STANDARD_CONFIGS` in `bpt/model.py`.
- `gpt.scan_layers`: whether to scan the layers in the GPT model. If `True`, the compiling time will be significantly reduced.
- `gpt.param_scan_axis`: the axis to scan the parameters. If `0`, the parameters will be scanned along the first axis, which is the default setting.
- `gpt.q_chunk_size`: the chunk size for the query sequence length.
- `gpt.k_chunk_size`: the chunk size for the key sequence length. Both `q_chunk_size` and `k_chunk_size` are used to control the memory usage and computation speed. We found memory usage is not very sensitive to the values of `q_chunk_size` and `k_chunk_size`, and computation speed can be accelerated by using larger `q_chunk_size` and `k_chunk_size`. Which values perform best depends on the specific hardware and model size, and we leave it as a hyperparameter to be tuned.
- `gpt.attn_type`: the type of Transformer model, can be one of `bp`, `vanilla`, and `mem_efficient`. `bp` is the blockwise parallel transformer, `vanilla` is the vanilla transformer, and `mem_efficient` is the memory efficient transformer.
- `gpt.gradient_checkpointing`: the gradient checkpointing strategy, both memory efficient transformer and blockwise parallel transformer require gradient checkpointing to be enabled, otherwise memory saving from reorganizing the compute will be lost due to storing all those intermediate results. See `get_gradient_checkpoint_policy` for availale options, use `nothing_saveable` for lowest memory usage.
- `gpt.n_positions`: the maximum sequence length.
- `gpt.seq_length`: the sequence length. The sequence length should be smaller than or equal to `n_positions`.
- `gpt.float32_logits`: whether to use float32 for logits. If `True` then compute logits in float32 to avoid numerical issues with bfloat16.


For large scale distributed training on TPU or on GPU cluster with good inter connection, we recommend first using FSDP, and add tensor parallelism when the model is very large or the global batch size is too large. You can control parallelism by using mesh_dim.

This code uses autodiff and rematerialization. In order to achieve optimal performance especially for achieving speed up, the backward pass should be written in low level kernels such as Triton/CUDA etc. This is on our TODO list and PRs are welcome.


## Reference
If you find our work relevant to your research, please cite:
```
@article{liu2023blockwise,
    title={Blockwise Parallel Transformer for Long Context Large Models},
    author={Hao Liu and Pieter Abbeel},
    year={2023},
    journal={arXiv preprint arxiv:2305.19370}
}
```
