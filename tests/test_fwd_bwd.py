import copy
from absl.app import run
import tux

import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as PS
from flax.training.train_state import TrainState

from llamabpt.data import DatasetFactory
from tux import (
    JaxRNG, JaxDistributedConfig, next_rng, init_rng, match_partition_rules,
    cross_entropy_loss_and_accuracy, global_norm, get_float_dtype_by_name,
    set_random_seed, average_metrics, get_weight_decay_mask,
    make_shard_and_gather_fns, with_sharding_constraint, define_flags_with_default,
    OptimizerFactory, StreamingCheckpointer
)
from llamabpt.llama import LLaMAConfig, FlaxLLaMAForCausalLMModule


FLAGS, FLAGS_DEF = define_flags_with_default(
    seed=42,
    mesh_dim='1,-1,1,2',
    dtype='fp32',
    total_steps=10000,
    load_llama_config='',
    update_llama_config='',
    load_checkpoint='',
    load_dataset_state='',
    log_freq=50,
    save_model_freq=0,
    save_milestone_freq=0,
    eval_steps=0,
    tokenizer=LLaMAConfig.get_tokenizer_config(),
    train_dataset=DatasetFactory.get_default_config(),
    eval_dataset=DatasetFactory.get_default_config(),
    optimizer=OptimizerFactory.get_default_config(),
    checkpointer=StreamingCheckpointer.get_default_config(),
    llama=LLaMAConfig.get_default_config(),
    logger=tux.WandBLogger.get_default_config(),
    log_all_worker=False,
    jax_distributed=JaxDistributedConfig.get_default_config(),
    autoresume=False,
)


def main(argv):
    JaxDistributedConfig.initialize(FLAGS.jax_distributed)
    set_random_seed(FLAGS.seed)
    # jax.config.update("jax_debug_nans", True)

    tokenizer = LLaMAConfig.get_tokenizer(FLAGS.tokenizer)
    dataset = DatasetFactory.load_dataset(FLAGS.train_dataset, tokenizer)
    if FLAGS.eval_steps > 0:
        eval_dataset = DatasetFactory.load_dataset(
            FLAGS.eval_dataset, dataset.tokenizer
        )
        eval_iterator = iter(eval_dataset)
    seq_length = dataset.seq_length

    if FLAGS.load_llama_config != '':
        llama_config = LLaMAConfig.load_config(FLAGS.load_llama_config)
        updates = LLaMAConfig(**FLAGS.llama)
        llama_config.update(dict(
            remat_block=updates.remat_block,
            remat_attention=updates.remat_attention,
            remat_mlp=updates.remat_mlp,
            scan_attention=updates.scan_attention,
            scan_mlp=updates.scan_mlp,
            scan_query_chunk_size=updates.scan_query_chunk_size,
            scan_key_chunk_size=updates.scan_key_chunk_size,
            scan_mlp_chunk_size=updates.scan_mlp_chunk_size,
            scan_layers=updates.scan_layers,
            param_scan_axis=updates.param_scan_axis,
        ))
    else:
        llama_config = LLaMAConfig(**FLAGS.llama)

    if FLAGS.update_llama_config != '':
        llama_config.update(dict(eval(FLAGS.update_llama_config)))

    llama_config.update(dict(
        bos_token_id=dataset.tokenizer.bos_token_id,
        eos_token_id=dataset.tokenizer.eos_token_id,
    ))
    if llama_config.vocab_size < dataset.vocab_size:
        llama_config.update(dict(vocab_size=dataset.vocab_size))
    llama_config.update(dict(mesh_dim=FLAGS.mesh_dim))

    attention_types = ['standard', 'ring_standard', 'ring_blockwise', 'blockwise', 'blockwise_custom']
    models = []
    for attention_type in attention_types:
        llama_config_copy = copy.deepcopy(llama_config)
        llama_config_copy.update(dict(attention_type=attention_type))
        model = FlaxLLaMAForCausalLMModule(
            llama_config_copy, dtype=get_float_dtype_by_name(FLAGS.dtype)
        )
        models.append(model)
    model = models[0]

    optimizer, optimizer_info = OptimizerFactory.get_optimizer(
        FLAGS.optimizer,
        get_weight_decay_mask(LLaMAConfig.get_weight_decay_exclusions())
    )

    def init_fn(rng):
        rng_generator = JaxRNG(rng)
        params = model.init(
            input_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            position_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            attention_mask=jnp.ones((4, seq_length), dtype=jnp.int32),
            rngs=rng_generator(llama_config.rng_keys()),
        )
        return TrainState.create(params=params, tx=optimizer, apply_fn=None)

    train_state_shapes = jax.eval_shape(init_fn, next_rng())
    train_state_partition = match_partition_rules(
        LLaMAConfig.get_partition_rules(FLAGS.llama.scan_layers, FLAGS.llama.param_scan_axis), train_state_shapes
    )

    sharded_init_fn = pjit(
        init_fn,
        in_shardings=PS(),
        out_shardings=train_state_partition,
    )


    mesh = LLaMAConfig.get_jax_mesh(FLAGS.mesh_dim)
    with mesh:
        train_state = sharded_init_fn(next_rng())
        batch, _ = next(iter(dataset))
        all_logits, all_grads = [], []
        for model in models:
            def train_step(train_state, rng, batch):
                rng_generator = JaxRNG(rng)
                batch = with_sharding_constraint(batch, PS(('dp', 'fsdp'), 'sp'))
                def loss_and_accuracy(params):
                    logits = model.apply(
                        params, batch['input_tokens'], deterministic=True,
                        rngs=rng_generator(llama_config.rng_keys()),
                    ).logits
                    loss, _ = cross_entropy_loss_and_accuracy(
                        logits, batch['target_tokens'], batch['loss_masks']
                    )
                    return loss, logits
                grad_fn = jax.value_and_grad(loss_and_accuracy, has_aux=True)
                (_, logits), grads = grad_fn(train_state.params)
                return logits, grads
            sharded_train_step = pjit(
                train_step,
                in_shardings=(train_state_partition, PS(), PS()),
                out_shardings=(PS(), PS()),
            )

            init_rng(FLAGS.seed)
            sharded_rng = next_rng()
            logits, grads = sharded_train_step(
                train_state, sharded_rng, batch
            )
            all_logits.append(jax.device_get(logits))
            all_grads.append(jax.device_get(grads.unfreeze()))
        for i in range(len(models)):
            print(attention_types[i])
            l1, l2 = all_logits[i], all_logits[0]
            print('logits:', np.max(np.abs(l1 - l2)), np.max(l1), np.max(l2))
            g1, g2 = all_grads[i], all_grads[0]
            g_diff = jax.tree_map(lambda x, y: np.max(np.abs(x - y)), g1, g2)
            g_diff = np.max(jax.tree_util.tree_flatten(g_diff)[0])
            g1_max = jax.tree_map(lambda x: np.max(x), g1)
            g1_max = np.max(jax.tree_util.tree_flatten(g1_max)[0])
            g2_max = jax.tree_map(lambda x: np.max(x), g2)
            g2_max = np.max(jax.tree_util.tree_flatten(g2_max)[0])
            print('grads:', g_diff, g1_max, g2_max)
            print()

if __name__ == "__main__":
    run(main)
