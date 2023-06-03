import dataclasses
import pprint
from functools import partial
import re

from tqdm import tqdm, trange
import numpy as np
import bpt.tools.utils as utils

import jax
import jax.numpy as jnp
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as PS
import flax
from flax import linen as nn
from flax.jax_utils import prefetch_to_device
from flax.training.train_state import TrainState
import optax

from bpt.data import Dataset, TextProcessor
from bpt.tools.checkpoint import StreamingCheckpointer
from bpt.tools.optimizers import OptimizerFactory
from bpt.tools.jax_utils import (
    JaxRNG, next_rng, match_partition_rules,
    cross_entropy_loss_and_accuracy, named_tree_map, global_norm,
    set_random_seed, average_metrics, get_weight_decay_mask,
    make_shard_and_gather_fns, with_sharding_constraint, tree_apply, get_metrics,
)
from bpt.model import GPTConfig, FlaxGPTForCausalLMModule
from bpt.blocks.blockwise_parallel import blockwise_cross_entropy


FLAGS, FLAGS_DEF = utils.define_flags_with_default(
    seed=42,
    initialize_jax_distributed=False,
    mesh_dim='1,-1,1',
    total_steps=10000,
    load_gpt_config='',
    update_gpt_config='',
    load_checkpoint='',
    load_dataset_state='',
    log_freq=50,
    save_model_freq=0,
    save_milestone_freq=0,
    eval_steps=0,
    tokenizer=GPTConfig.get_tokenizer_config(),
    text_processor=TextProcessor.get_default_config(),
    train_dataset=Dataset.get_default_config(),
    eval_dataset=Dataset.get_default_config(),
    optimizer=OptimizerFactory.get_default_config(),
    checkpointer=StreamingCheckpointer.get_default_config(),
    gpt=GPTConfig.get_default_config(),
    logger=utils.WandBLogger.get_default_config(),
    log_all_worker=False,
    profile_steps=0,
    stop_after_profile=True,
)


def main(argv):
    if FLAGS.initialize_jax_distributed:
        jax.distributed.initialize()

    variant = utils.get_user_flags(FLAGS, FLAGS_DEF)
    flags_config_dict = utils.user_flags_to_config_dict(FLAGS, FLAGS_DEF)
    logger = utils.WandBLogger(
        config=FLAGS.logger,
        variant=variant,
        enable=FLAGS.log_all_worker or (jax.process_index() == 0),
    )
    set_random_seed(FLAGS.seed)

    if FLAGS.load_dataset_state != '':
        dataset = utils.load_pickle(FLAGS.load_dataset_state)
    else:
        tokenizer = GPTConfig.get_tokenizer(FLAGS.tokenizer)
        text_processor = TextProcessor(FLAGS.text_processor, tokenizer)
        dataset = Dataset(FLAGS.train_dataset, tokenizer, text_processor)

    if FLAGS.eval_steps > 0:
        eval_dataset = Dataset(
            FLAGS.eval_dataset, dataset.tokenizer, dataset.text_processor,
        )
        eval_iterator = iter(eval_dataset.val_iter())

    seq_length = dataset.seq_length

    if FLAGS.load_gpt_config != '':
        gpt_config = GPTConfig.load_config(FLAGS.load_gpt_config)
        update_gpt_config = GPTConfig(**FLAGS.gpt)
        gpt_config.update(dict(
            q_chunk_size=update_gpt_config.q_chunk_size,
            k_chunk_size=update_gpt_config.k_chunk_size,
            attn_type=update_gpt_config.attn_type,
            n_positions=update_gpt_config.n_positions,
            gradient_checkpointing=update_gpt_config.gradient_checkpointing,
            scan_layers=update_gpt_config.scan_layers,
            param_scan_axis=update_gpt_config.param_scan_axis,
        ))
    else:
        gpt_config = GPTConfig(**FLAGS.gpt)

    if FLAGS.update_gpt_config != '':
        gpt_config.update(dict(eval(FLAGS.update_gpt_config)))

    gpt_config.update(dict(
        bos_token_id=dataset.tokenizer.bos_token_id,
        eos_token_id=dataset.tokenizer.eos_token_id,
    ))
    if gpt_config.vocab_size < dataset.vocab_size:
        gpt_config.update(dict(vocab_size=dataset.vocab_size))
    model = FlaxGPTForCausalLMModule(gpt_config)

    optimizer, optimizer_info = OptimizerFactory.get_optimizer(
        FLAGS.optimizer,
        get_weight_decay_mask(GPTConfig.get_weight_decay_exclusions()),
    )

    def create_trainstate_from_params(params):
        return TrainState.create(params=params, tx=optimizer, apply_fn=None)

    def init_fn(rng):
        rng_generator = JaxRNG(rng)
        params = model.init(
            input_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            position_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            attention_mask=jnp.ones((4, seq_length), dtype=jnp.int32),
            rngs=rng_generator(gpt_config.rng_keys()),
        )
        return TrainState.create(params=params, tx=optimizer, apply_fn=None)

    if FLAGS.gpt.attn_type == 'blockwise_parallel' or FLAGS.gpt.attn_type == 'blockwise_parallel_v1':
        cross_entropy_loss_and_accuracy_fn = partial(blockwise_cross_entropy,
                                                     policy=FLAGS.gpt.gradient_checkpointing,
                                                     chunk_size=FLAGS.gpt.q_chunk_size,
                                                     prevent_cse=not FLAGS.gpt.scan_layers,)
    else:
        cross_entropy_loss_and_accuracy_fn = cross_entropy_loss_and_accuracy

    def train_step(train_state, rng, batch):
        rng_generator = JaxRNG(rng)
        input_tokens = with_sharding_constraint(batch['input_tokens'], PS(('dp', 'fsdp')))
        output_tokens = with_sharding_constraint(batch['output_tokens'], PS(('dp', 'fsdp')))
        loss_masks = with_sharding_constraint(batch['loss_masks'], PS(('dp', 'fsdp')))
        def loss_and_accuracy(params):
            logits = model.apply(
                params,
                input_tokens,
                deterministic=False,
                rngs=rng_generator(gpt_config.rng_keys()),
            ).logits
            return cross_entropy_loss_and_accuracy_fn(logits, output_tokens, loss_masks)

        grad_fn = jax.value_and_grad(loss_and_accuracy, has_aux=True)
        (loss, accuracy), grads = grad_fn(train_state.params)
        train_state = train_state.apply_gradients(grads=grads)
        metrics = dict(
            loss=loss,
            accuracy=accuracy,
            learning_rate=optimizer_info['learning_rate_schedule'](train_state.step),
            gradient_norm=global_norm(grads),
            param_norm=global_norm(train_state.params),
        )
        return train_state, rng_generator(), metrics

    def eval_step(train_state, rng, batch):
        rng_generator = JaxRNG(rng)
        input_tokens = with_sharding_constraint(batch['input_tokens'], PS(('dp', 'fsdp')))
        output_tokens = with_sharding_constraint(batch['output_tokens'], PS(('dp', 'fsdp')))
        loss_masks = with_sharding_constraint(batch['loss_masks'], PS(('dp', 'fsdp')))
        logits = model.apply(
            train_state.params,
            input_tokens,
            deterministic=True,
            rngs=rng_generator(gpt_config.rng_keys()),
        ).logits
        loss, accuracy = cross_entropy_loss_and_accuracy_fn(logits, output_tokens, loss_masks)
        metrics = dict(
            loss=loss,
            accuracy=accuracy,
        )
        return rng_generator(), metrics

    train_state_shapes = jax.eval_shape(init_fn, next_rng())
    train_state_partition = match_partition_rules(
        GPTConfig.get_partition_rules(FLAGS.gpt.scan_layers), train_state_shapes
    )

    num_params = sum(x.size for x in jax.tree_leaves(train_state_shapes.params))
    num_nonembed_params = num_params - gpt_config.vocab_size * gpt_config.n_embd
    param_stats = {"num_params": num_params,"num_nonembed_params": num_nonembed_params}
    logger.log(param_stats)
    tqdm.write("\n" + pprint.pformat(param_stats) + "\n")

    shard_fns, gather_fns = make_shard_and_gather_fns(
        train_state_partition, train_state_shapes
    )
    checkpointer = StreamingCheckpointer(
        FLAGS.checkpointer, logger.output_dir,
        enable=jax.process_index() == 0,
    )

    sharded_init_fn = pjit(
        init_fn,
        in_shardings=PS(),
        out_shardings=train_state_partition
    )

    sharded_create_trainstate_from_params = pjit(
        create_trainstate_from_params,
        in_shardings=(train_state_partition.params, ),
        out_shardings=train_state_partition,
        donate_argnums=(0, ),
    )

    sharded_train_step = pjit(
        train_step,
        in_shardings=(train_state_partition, PS(), PS()),
        out_shardings=(train_state_partition, PS(), PS()),
        donate_argnums=(0, 1),
    )

    sharded_eval_step = pjit(
        eval_step,
        in_shardings=(train_state_partition, PS(), PS()),
        out_shardings=(PS(), PS()),
        donate_argnums=(1,),
    )

    def save_checkpoint(train_state, milestone=False):
        step = int(jax.device_get(train_state.step))
        metadata = dict(
            step=step,
            variant=variant,
            flags=flags_config_dict,
            gpt_config=gpt_config.to_dict(),
        )
        checkpointer.save_all(
            train_state=train_state,
            gather_fns=gather_fns,
            metadata=metadata,
            dataset=dataset.get_state_dict(),
            milestone=milestone,
        )

    if FLAGS.profile_steps > 0:
        import os
        os.makedirs(logger.profile_dir, exist_ok=True)
        mesh = GPTConfig.get_jax_mesh(FLAGS.mesh_dim)
        with mesh:
            train_state, restored_params = None, None
            if train_state is None and restored_params is None:
                # Initialize from scratch
                train_state = sharded_init_fn(next_rng())
            elif train_state is None and restored_params is not None:
                # Restore from params but initialize train_state
                train_state = sharded_create_trainstate_from_params(restored_params)
                del restored_params
            sharded_rng = next_rng()
            # warmup
            for batch, dataset_metrics in dataset:
                train_state, sharded_rng, metrics = sharded_train_step(
                    train_state, sharded_rng, batch
                )
                break
            # profile
            jax.profiler.start_trace(logger.profile_dir)
            for step, (batch, dataset_metrics) in zip(trange(FLAGS.profile_steps), dataset):
                train_state, sharded_rng, metrics = sharded_train_step(
                    train_state, sharded_rng, batch
                )
                jax.block_until_ready(train_state)
                jax.profiler.save_device_memory_profile(f'{logger.profile_dir}/memory{step}.prof')
            jax.profiler.stop_trace()
        if FLAGS.stop_after_profile:
            exit()

    mesh = GPTConfig.get_jax_mesh(FLAGS.mesh_dim)
    with mesh:
        train_state, restored_params = None, None
        if FLAGS.load_checkpoint != '':
            load_type, load_path = FLAGS.load_checkpoint.split('::', 1)
            if load_type == 'huggingface':
                restored_params = tree_apply(
                    shard_fns.params, gpt_config.load_pretrained(load_path)
                )
                train_state = None
            else:
                train_state, restored_params = checkpointer.load_trainstate_checkpoint(
                    FLAGS.load_checkpoint, train_state_shapes, shard_fns
                )

        if train_state is None and restored_params is None:
            # Initialize from scratch
            train_state = sharded_init_fn(next_rng())
        elif train_state is None and restored_params is not None:
            # Restore from params but initialize train_state
            train_state = sharded_create_trainstate_from_params(restored_params)
            del restored_params

        start_step = int(jax.device_get(train_state.step))

        if FLAGS.save_model_freq > 0:
            save_checkpoint(train_state)

        sharded_rng = next_rng()

        step_counter = trange(start_step, FLAGS.total_steps, ncols=0)

        def run_eval(sharded_rng, eval_fn, batch, eval_steps, eval_name):
            eval_metric_list = []
            for _ in range(eval_steps):
                sharded_rng, eval_metrics = eval_fn(
                    train_state, sharded_rng, batch
                )
                eval_metric_list.append(eval_metrics)
            log_metrics = get_metrics(eval_metric_list, stack=True)
            mean_metrics = {
                f"{eval_name}/{k}": np.mean(v)
                for k, v in log_metrics.items()
            }
            mean_metrics["step"] = step
            logger.log(mean_metrics)
            tqdm.write("\n" + pprint.pformat(mean_metrics) + "\n")
            return sharded_rng

        for step, (batch, dataset_metrics) in zip(step_counter, dataset):
            train_state, sharded_rng, metrics = sharded_train_step(
                train_state, sharded_rng, batch
            )

            if step % FLAGS.log_freq == 0:
                if FLAGS.eval_steps > 0:
                    batch, _ = next(eval_iterator)
                    sharded_rng = run_eval(sharded_rng, sharded_eval_step,
                                           batch, FLAGS.eval_steps, "val")
                log_metrics = {"step": step}
                log_metrics.update(metrics)
                log_metrics.update(dataset_metrics)
                log_metrics = jax.device_get(log_metrics)
                logger.log(log_metrics)
                tqdm.write("\n" + pprint.pformat(log_metrics) + "\n")

            if FLAGS.save_milestone_freq > 0 and (step + 1) % FLAGS.save_milestone_freq == 0:
                save_checkpoint(train_state, milestone=True)
            elif FLAGS.save_model_freq > 0 and (step + 1) % FLAGS.save_model_freq == 0:
                save_checkpoint(train_state)

        if FLAGS.save_model_freq > 0:
            save_checkpoint(train_state)


if __name__ == "__main__":
    utils.run(main)
