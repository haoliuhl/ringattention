import inspect
import logging
import os
import pprint
import random
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from copy import copy
from io import BytesIO
from socket import gethostname
import dataclasses

import absl.flags
import absl.logging
import cloudpickle as pickle
import flax
import gcsfs
import jax
import jax.numpy as jnp
import msgpack
import numpy as np
import wandb
from flax.serialization import from_bytes, to_bytes
from ml_collections import ConfigDict
from ml_collections.config_dict.config_dict import placeholder
from ml_collections.config_flags import config_flags
from flax.training.train_state import TrainState
from flax.core import FrozenDict
from absl.app import run


class WandBLogger(object):
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.project_id = ""
        config.project_entity = placeholder(str)
        config.experiment_id = placeholder(str)
        config.append_uuid = True
        config.experiment_note = placeholder(str)

        config.output_dir = "/tmp/"
        config.wandb_dir = ""
        config.profile_dir = ""

        config.online = False

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, variant, enable=True):
        self.enable = enable
        self.config = self.get_default_config(config)

        if self.config.experiment_id is None or self.config.experiment_id == "":
            self.config.experiment_id = uuid.uuid4().hex
        else:
            if self.config.append_uuid:
                self.config.experiment_id = (
                    str(self.config.experiment_id) + "_" + uuid.uuid4().hex
                )
            else:
                self.config.experiment_id = str(self.config.experiment_id)

        if self.enable:
            if self.config.output_dir == "":
                self.config.output_dir = tempfile.mkdtemp()
            else:
                self.config.output_dir = os.path.join(
                    self.config.output_dir, self.config.experiment_id
                )
                if not self.config.output_dir.startswith("gs://"):
                    os.makedirs(self.config.output_dir, exist_ok=True)

            if self.config.wandb_dir == "":
                if not self.config.output_dir.startswith("gs://"):
                    # Use the same directory as output_dir if it is not a GCS path.
                    self.config.wandb_dir = self.config.output_dir
                else:
                    # Otherwise, use a temporary directory.
                    self.config.wandb_dir = tempfile.mkdtemp()
            else:
                # Join the wandb_dir with the experiment_id.
                self.config.wandb_dir = os.path.join(
                    self.config.wandb_dir, self.config.experiment_id
                )
                os.makedirs(self.config.wandb_dir, exist_ok=True)

            if self.config.profile_dir == "":
                if not self.config.output_dir.startswith("gs://"):
                    # Use the same directory as output_dir if it is not a GCS path.
                    self.config.profile_dir = self.config.output_dir
                else:
                    # Otherwise, use a temporary directory.
                    self.config.profile_dir = tempfile.mkdtemp()
            else:
                # Join the profile_dir with the experiment_id.
                self.config.profile_dir = os.path.join(
                    self.config.profile_dir, self.config.experiment_id
                )
                os.makedirs(self.config.profile_dir, exist_ok=True)

        self._variant = flatten_config_dict(variant)

        if "hostname" not in self._variant:
            self._variant["hostname"] = gethostname()

        if self.enable:
            self.run = wandb.init(
                reinit=True,
                config=self._variant,
                project=self.config.project_id,
                dir=self.config.wandb_dir,
                id=self.config.experiment_id,
                resume="allow",
                notes=self.config.experiment_note,
                entity=self.config.project_entity,
                settings=wandb.Settings(
                    start_method="thread",
                    _disable_stats=True,
                ),
                mode="online" if self.config.online else "offline",
            )
        else:
            self.run = None

    def log(self, *args, **kwargs):
        if self.enable:
            self.run.log(*args, **kwargs)

    def save_pickle(self, obj, filename):
        if self.enable:
            save_pickle(obj, os.path.join(self.config.output_dir, filename))

    @property
    def experiment_id(self):
        return self.config.experiment_id

    @property
    def variant(self):
        return self.config.variant

    @property
    def output_dir(self):
        return self.config.output_dir

    @property
    def wandb_dir(self):
        return self.config.wandb_dir

    @property
    def profile_dir(self):
        return self.config.profile_dir

def config_dict(*args, **kwargs):
    return ConfigDict(dict(*args, **kwargs))


def define_flags_with_default(**kwargs):
    for key, val in kwargs.items():
        if isinstance(val, tuple):
            val, help_str = val
        else:
            help_str = ""

        if isinstance(val, ConfigDict):
            config_flags.DEFINE_config_dict(key, val)
        elif isinstance(val, bool):
            # Note that True and False are instances of int.
            absl.flags.DEFINE_bool(key, val, help_str)
        elif isinstance(val, int):
            absl.flags.DEFINE_integer(key, val, help_str)
        elif isinstance(val, float):
            absl.flags.DEFINE_float(key, val, help_str)
        elif isinstance(val, str):
            absl.flags.DEFINE_string(key, val, help_str)
        else:
            raise ValueError("Incorrect value type")
    return absl.flags.FLAGS, kwargs


def print_flags(flags, flags_def):
    flag_srings = [
        "{}: {}".format(key, val)
        for key, val in get_user_flags(flags, flags_def).items()
    ]
    logging.info(
        "Hyperparameter configs: \n{}".format(
            pprint.pformat(flag_srings)
        )
    )


def get_user_flags(flags, flags_def):
    output = {}
    for key in flags_def:
        val = getattr(flags, key)
        if isinstance(val, ConfigDict):
            output.update(flatten_config_dict(val, prefix=key))
        else:
            output[key] = val

    return output


def user_flags_to_config_dict(flags, flags_def):
    output = ConfigDict()
    for key in flags_def:
        output[key] = getattr(flags, key)

    return output


def flatten_config_dict(config, prefix=None):
    output = {}
    for key, val in config.items():
        if isinstance(val, ConfigDict) or isinstance(val, dict):
            output.update(flatten_config_dict(val, prefix=key))
        else:
            if prefix is not None:
                output["{}.{}".format(prefix, key)] = val
            else:
                output[key] = val
    return output


def function_args_to_config(fn, none_arg_types=None, exclude_args=None, override_args=None):
    config = ConfigDict()
    arg_spec = inspect.getargspec(fn)
    n_args = len(arg_spec.defaults)
    arg_names = arg_spec.args[-n_args:]
    default_values = arg_spec.defaults
    for name, value in zip(arg_names, default_values):
        if exclude_args is not None and name in exclude_args:
            continue
        elif override_args is not None and name in override_args:
            config[name] = override_args[name]
        elif none_arg_types is not None and value is None and name in none_arg_types:
            config[name] = placeholder(none_arg_types[name])
        else:
            config[name] = value

    return config


def prefix_metrics(metrics, prefix):
    return {"{}/{}".format(prefix, key): value for key, value in metrics.items()}


def open_file(path, mode='rb', cache_type='readahead'):
    if path.startswith("gs://"):
        logging.getLogger("fsspec").setLevel(logging.WARNING)
        return gcsfs.GCSFileSystem().open(path, mode, cache_type=cache_type)
    else:
        return open(path, mode)


def save_pickle(obj, path):
    with open_file(path, "wb") as fout:
        pickle.dump(obj, fout)


def load_pickle(path):
    with open_file(path, "rb") as fin:
        data = pickle.load(fin)
    return data


def text_to_array(text, encoding="utf-8"):
    return np.frombuffer(text.encode(encoding), dtype="uint8")


def array_to_text(array, encoding="utf-8"):
    with BytesIO(array) as fin:
        text = fin.read().decode(encoding)
    return text


class JaxRNG(object):
    """ A convenient stateful Jax RNG wrapper. Can be used to wrap RNG inside
        pure function.
    """

    @classmethod
    def from_seed(cls, seed):
        return cls(jax.random.PRNGKey(seed))

    def __init__(self, rng):
        self.rng = rng

    def __call__(self, keys=None):
        if keys is None:
            self.rng, split_rng = jax.random.split(self.rng)
            return split_rng
        elif isinstance(keys, int):
            split_rngs = jax.random.split(self.rng, num=keys + 1)
            self.rng = split_rngs[0]
            return tuple(split_rngs[1:])
        else:
            split_rngs = jax.random.split(self.rng, num=len(keys) + 1)
            self.rng = split_rngs[0]
            return {key: val for key, val in zip(keys, split_rngs[1:])}


def wrap_function_with_rng(rng):
    """ To be used as decorator, automatically bookkeep a RNG for the wrapped function. """
    def wrap_function(function):
        def wrapped(*args, **kwargs):
            nonlocal rng
            rng, split_rng = jax.random.split(rng)
            return function(split_rng, *args, **kwargs)
        return wrapped
    return wrap_function


def init_rng(seed):
    global jax_utils_rng
    jax_utils_rng = JaxRNG.from_seed(seed)


def next_rng(*args, **kwargs):
    global jax_utils_rng
    return jax_utils_rng(*args, **kwargs)


def flatten_tree(xs, is_leaf=None, sep=None):
    """ A stronger version of flax.traverse_util.flatten_dict, supports
        dict, tuple, list and TrainState. Tuple and list indices will be
        converted to strings.
    """
    tree_node_classes = (FrozenDict, dict, tuple, list, TrainState)
    if not isinstance(xs, tree_node_classes):
        ValueError('fUnsupported node type: {type(xs)}')

    def _is_leaf(prefix, fx):
        if is_leaf is not None:
            return is_leaf(prefix, xs)
        return False

    def _key(path):
        if sep is None:
            return path
        return sep.join(path)

    def _convert_to_dict(xs):
        if isinstance(xs, (FrozenDict, dict)):
            return xs
        elif isinstance(xs, (tuple, list)):
            return {f'{i}': v for i, v in enumerate(xs)}
        elif isinstance(xs, TrainState):
            output = {}
            for field in dataclasses.fields(xs):
                if 'pytree_node' not in field.metadata or field.metadata['pytree_node']:
                    output[field.name] = getattr(xs, field.name)
            return output
        else:
            raise ValueError('fUnsupported node type: {type(xs)}')

    def _flatten(xs, prefix):
        if not isinstance(xs, tree_node_classes) or _is_leaf(prefix, xs):
            return {_key(prefix): xs}

        result = {}
        is_empty = True
        for (key, value) in _convert_to_dict(xs).items():
            is_empty = False
            path = prefix + (key, )
            result.update(_flatten(value, path))
        return result

    return _flatten(xs, ())


def named_tree_map(f, tree, is_leaf=None, sep=None):
    """ An extended version of jax.tree_util.tree_map, where the mapped function
        f takes both the name (path) and the tree leaf as input.
    """
    flattened_tree = flatten_tree(tree, is_leaf=is_leaf, sep=sep)
    id_to_name = {id(val): key for key, val in flattened_tree.items()}
    def map_fn(leaf):
        name = id_to_name[id(leaf)]
        return f(name, leaf)
    return jax.tree_util.tree_map(map_fn, tree)


def get_pytree_shape_info(tree):
    flattend_tree = flatten_tree(tree, sep='/')
    shapes = []
    for key in sorted(list(flattend_tree.keys())):
        val = flattend_tree[key]
        shapes.append(f'{key}: {val.dtype}, {val.shape}')
    return '\n'.join(shapes)


def collect_metrics(metrics, names, prefix=None):
    collected = {}
    for name in names:
        if name in metrics:
            collected[name] = jnp.mean(metrics[name])
    if prefix is not None:
        collected = {
            '{}/{}'.format(prefix, key): value for key, value in collected.items()
        }
    return collected


def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    init_rng(seed)
