import ml_collections
import math
from typing import Any
import functools
import flax.linen as nn
import jax
import jax.numpy as jnp
from libml import attn_utils
from libml import self_attention
import numpy as np

default_kernel_init = attn_utils.trunc_normal(stddev=0.02)
default_bias_init = jax.nn.initializers.zeros

class NesTTransformerBrick(nn.Module):
    """Nested Transformer Net."""
    num_classes: int
    config: ml_collections.ConfigDict
    train: bool = False
    dtype: int = jnp.float32
    activation_fn: Any = nn.gelu

    def setup(self):
        config = self.config
        self.num_layers_per_block = config.num_layers_per_block
        self.num_blocks = len(self.num_layers_per_block)
        self.levels = self.num_blocks
        # save_gradCAT_data = config.save_gradCAT_data
        # Here we just assume image/patch size are squared.
        #assert inputs.shape[1] == inputs.shape[2]
        #assert inputs.shape[1] % config.init_patch_embed_size == 0
        #input_size_after_patch = inputs.shape[1] // config.init_patch_embed_size
        #assert input_size_after_patch % config.patch_size == 0
        #down_sample_ratio = input_size_after_patch // config.patch_size
        # There are 4 child nodes for each node.
        #assert num_blocks == int(np.log(down_sample_ratio) / np.log(2) + 1)

        # If `scale_hidden_dims` is provided, at every block, it increases hidden
        # dimension and num_heads by `scale_hidden_dims`. Set `scale_hidden_dims=2`
        # overall is a common design, so we do not gives the flexibility to control
        # layer-wise `scale_hidden_dims` to simplify the architecture.
        self.scale_hidden_dims = config.get("scale_hidden_dims", None)

        self.total_block_num = np.sum(self.num_layers_per_block)
        self.path_drop = np.linspace(0, config.stochastic_depth_drop, self.total_block_num)

        self.norm_fn = attn_utils.get_norm_layer(
            self.train, self.dtype, norm_type=config.norm_type)
        self.conv_fn = functools.partial(
            nn.Conv, dtype=self.dtype, kernel_init=default_kernel_init)
        self.dense_fn = functools.partial(
            nn.Dense, dtype=self.dtype, kernel_init=default_kernel_init)
        self.encoder_dict = dict(
            num_heads=config.num_heads,
            norm_fn=self.norm_fn,
            mlp_ratio=config.mlp_ratio,
            attn_type=config.attn_type,
            dense_fn=self.dense_fn,
            activation_fn=self.activation_fn,
            qkv_bias=config.qkv_bias,
            attn_drop=config.attn_drop,
            proj_drop=config.proj_drop,
            train=self.train,
            dtype=self.dtype)

        def __call__(self, inputs, level):
            if level == 0:
                x = self_attention.PatchEmbedding(
                    conv_fn=self.conv_fn,
                    patch_size=(config.init_patch_embed_size, config.init_patch_embed_size),
                    embedding_dim=config.embedding_dim)(
                    inputs)
            else:
                x = inputs
            x = attn_utils.block_images(x, (config.patch_size, config.patch_size))
            block_idx = 0 #TODO need to update TALI!!
            #total_block_num = np.sum(num_layers_per_block)
            #path_drop = np.linspace(0, config.stochastic_depth_drop, total_block_num)
            i = level
            for j in range(i):
                block_idx += self.num_layers_per_block(j)
            #for i in range(num_blocks):
            x = self_attention.PositionEmbedding()(x)
            if self.scale_hidden_dims and i != 0:
                # Overwrite the original num_heads value in encoder_dict so num_heads
                # multipled by scale_hidden_dims continueously.
                self.encoder_dict.update(
                    {"num_heads": self.encoder_dict["num_heads"] * self.scale_hidden_dims})
            for _ in range(self.num_layers_per_block[i]):
                x = self_attention.EncoderNDBlock(
                     **self.encoder_dict, path_drop=self.path_drop[block_idx])(
                        x)
                block_idx = block_idx + 1

            # x is now the features map
            return x



class NesTAggregateBrick(nn.Module):
    """Nested Transformer Aggregate."""
    num_classes: int
    config: ml_collections.ConfigDict
    train: bool = False
    dtype: int = jnp.float32
    activation_fn: Any = nn.gelu
    def setup(self):
        config = self.config
        self.num_layers_per_block = config.num_layers_per_block
        self.num_blocks = len(self.num_layers_per_block)
        self.levels = self.num_blocks
        self.norm_fn = attn_utils.get_norm_layer(
            self.train, self.dtype, norm_type=config.norm_type)
        self.conv_fn = functools.partial(
            nn.Conv, dtype=self.dtype, kernel_init=default_kernel_init)
        self.dense_fn = functools.partial(
            nn.Dense, dtype=self.dtype, kernel_init=default_kernel_init)


    def __call__(self, x, level: int):
        i = level
        if i < self.num_blocks - 1:
            grid_size = int(math.sqrt(x.shape[1]))
            if self.scale_hidden_dims:
                output_dim = x.shape[-1] * self.scale_hidden_dims
            else:
                output_dim = None

        x = self_attention.ConvPool(
            grid_size=(grid_size, grid_size),
            patch_size=(self.config.patch_size, self.config.patch_size),
            conv_fn=self.conv_fn,
            dtype=self.dtype,
            output_dim=output_dim)(
            x)

        return x # this is aggregate results

class NesTDenseBlock(nn.Module):
    """Nested Transformer Net."""
    num_classes: int
    config: ml_collections.ConfigDict
    train: bool = False
    dtype: int = jnp.float32
    activation_fn: Any = nn.gelu

    def setup(self):
        config = self.config
        self.num_layers_per_block = config.num_layers_per_block
        self.num_blocks = len(self.num_layers_per_block)
        self.levels = self.num_blocks
        self.norm_fn = attn_utils.get_norm_layer(
            self.train, self.dtype, norm_type=config.norm_type)
        self.conv_fn = functools.partial(
            nn.Conv, dtype=self.dtype, kernel_init=default_kernel_init)
        self.dense_fn = functools.partial(
            nn.Dense, dtype=self.dtype, kernel_init=default_kernel_init)

    def __call__(self, x):

        assert x.shape[1] == 1
        assert x.shape[2] == self.config.patch_size ** 2

        x = self.norm_fn()(x)
        x_pool = jnp.mean(x, axis=(1, 2))
        logits = self.dense_fn(self.num_classes)(x_pool)
        return logits

