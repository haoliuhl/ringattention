An implementation of blockwise parallel transformer for the Llama architecture.

## Usage

Use `q_chunk_size` and `k_chunk_size` to control the number of chunks in the self-attention.
Use `ffn_chunk_size` to control the number of chunks in the feed-forward network.
Use `head_chunk_size` to control the number of chunks in the lm head.
Use `loss_chunk_size` to control the number of chunks in the final loss computation.

For these arguments, set to -1 to disable chunking. For example, `--llama.ffn_chunk_size=-1` disables blockwise parallelism in the feed-forward network, which reduces BPT to the memory-efficient transformer.

Use `remat_policy` to control the rematerialization policy, recommended is `nothing_saveable`.

Use `float32_logits` to control whether to use float32 for logits for numerical stability, recommended for large models.

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
    --llama.scan_layers=True \
    --llama.param_scan_axis=0 \
    --llama.q_chunk_size=2048 \
    --llama.k_chunk_size=4096 \
    --llama.ffn_chunk_size=2048 \
    --llama.head_chunk_size=-1 \
    --llama.loss_chunk_size=-1 \
    --llama.remat_policy='nothing_saveable' \
    --llama.max_sequence_length=32768 \
    --llama.float32_logits=True \
    --autoresume=True
```

For the LLaMA tokenizer, you can use OpenLLaMAv2 tokenizer which can be downloaded from [here](https://drive.google.com/file/d/1p9KAUxtAEOgJhUvOJVhUjHp8oOf7d1wW/view?usp=sharing) or the official LLaMA tokenizer.

For the training dataset, you can use `scripts/prepare_data.py` to download OpenWebText dataset and prepare the dataset for training.
