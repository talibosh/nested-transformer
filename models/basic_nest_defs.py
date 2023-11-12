import functools
import ml_collections
import sys
from models import nest_modules
MODELS = {}


def register(f):
  MODELS[f.__name__] = f
  return f


def default_config():
  """Shared configs for models."""
  nest = ml_collections.ConfigDict()
  nest.norm_type = "LN"
  nest.attn_type = "local_multi_head"
  nest.mlp_ratio = 4
  nest.qkv_bias = True
  nest.attn_drop = 0.0
  nest.proj_drop = 0.0
  nest.stochastic_depth_drop = 0.1
  return nest


@register
def nest_tiny_s16_32(config):
  """NesT tiny version with sequence length 16 for 32x32 inputs."""
  nest = default_config()
  # Encode one pixel as a word vector.
  nest.init_patch_embed_size = 1
  # Default max sequencee length is 4x4=16, so it has 4 layers.
  nest.patch_size = 4
  nest.num_layers_per_block = [3, 3, 3, 3]
  nest.embedding_dim = 192
  nest.num_heads = 3
  nest.attn_type = "local_multi_query"

  if config.get("nest"):
    nest.update(config.nest)
  return functools.partial(getattr(sys.modules[__name__], config.classname), config=nest)


@register
def nest_small_s16_32(config):
  """NesT small version with sequence length 16 for 32x32 inputs."""
  nest = default_config()
  nest.init_patch_embed_size = 1
  nest.patch_size = 4
  nest.num_layers_per_block = [3, 3, 3, 3]
  nest.embedding_dim = 384
  nest.num_heads = 6
  nest.attn_type = "local_multi_query"

  if config.get("nest"):
    nest.update(config.nest)
  return functools.partial(getattr(sys.modules[__name__], config.classname), config=nest)


@register
def nest_base_s16_32(config):
  """NesT base version with sequence length 16 for 32x32 inputs."""
  nest = default_config()
  nest.init_patch_embed_size = 1
  nest.patch_size = 4
  nest.num_layers_per_block = [3, 3, 3, 3]
  nest.embedding_dim = 768
  nest.num_heads = 12
  nest.attn_type = "local_multi_query"

  if config.get("nest"):
    nest.update(config.nest)
  return functools.partial(getattr(sys.modules[__name__], config.classname), config=nest)


@register
def nest_tiny_s196_224(config):
  """NesT tiny version with sequence length 49 for 224x224 inputs."""
  nest = default_config()
  # Encode 4x4 pixel as a word vector.
  nest.init_patch_embed_size = 4
  # Default max sequencee length is 14x14=196, so it has 3 layers:
  # Spatial image size: [56, 28, 14]
  nest.patch_size = 14
  nest.num_layers_per_block = [2, 2, 8]
  nest.embedding_dim = 96
  nest.num_heads = 3
  nest.scale_hidden_dims = 2
  nest.stochastic_depth_drop = 0.2
  nest.attn_type = "local_multi_head"

  if config.get("nest"):
    nest.update(config.nest)
  return functools.partial(getattr(sys.modules[__name__], config.classname), config=nest)


@register
def nest_small_s196_224(config):
  """NesT small version with sequence length 196 for 224x224 inputs."""
  nest = default_config()
  nest.init_patch_embed_size = 4
  nest.patch_size = 14
  nest.num_layers_per_block = [2, 2, 20]
  nest.embedding_dim = 96
  nest.num_heads = 3
  nest.scale_hidden_dims = 2
  nest.stochastic_depth_drop = 0.3
  nest.attn_type = "local_multi_head"

  if config.get("nest"):
    nest.update(config.nest)
  return functools.partial(getattr(sys.modules[__name__], config.classname), config=nest)


@register
def nest_base_s196_224(config):
  """NesT base version with sequence length 196 for 224x224 inputs."""
  nest = default_config()
  nest.init_patch_embed_size = 4
  nest.patch_size = 14
  nest.num_layers_per_block = [2, 2, 20]
  nest.embedding_dim = 128
  nest.num_heads = 4
  nest.scale_hidden_dims = 2
  nest.stochastic_depth_drop = 0.5
  nest.attn_type = "local_multi_head"

  if config.get("nest"):
    nest.update(config.nest)
  return functools.partial(eval(config.classname), config=nest)


def create_model(name, config):
  """Creates model partial function."""
  if name not in MODELS:
    raise ValueError(f"Model {name} does not exist.")
  return MODELS[name](config)
