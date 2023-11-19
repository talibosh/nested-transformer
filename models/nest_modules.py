import functools

import flax.linen
import math
from typing import Any
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
from libml import attn_utils
from libml import self_attention
import numpy as np

default_kernel_init = attn_utils.trunc_normal(stddev=0.02)
default_bias_init = jax.nn.initializers.zeros


class NesTA(nn.Module):
    """Nested Transformer Net."""
    num_classes: int
    config: ml_collections.ConfigDict
    train: bool = False
    dtype: int = jnp.float32
    activation_fn: Any = nn.gelu

    @nn.compact
    def __call__(self, inputs):
        config = self.config
        num_layers_per_block = config.num_layers_per_block
        num_blocks = len(num_layers_per_block)
        # save_gradCAT_data = config.save_gradCAT_data
        # Here we just assume image/patch size are squared.
        assert inputs.shape[1] == inputs.shape[2]
        assert inputs.shape[1] % config.init_patch_embed_size == 0
        input_size_after_patch = inputs.shape[1] // config.init_patch_embed_size
        assert input_size_after_patch % config.patch_size == 0
        down_sample_ratio = input_size_after_patch // config.patch_size
        # There are 4 child nodes for each node.
        assert num_blocks == int(np.log(down_sample_ratio) / np.log(2) + 1)

        # If `scale_hidden_dims` is provided, at every block, it increases hidden
        # dimension and num_heads by `scale_hidden_dims`. Set `scale_hidden_dims=2`
        # overall is a common design, so we do not gives the flexibility to control
        # layer-wise `scale_hidden_dims` to simplify the architecture.
        scale_hidden_dims = config.get("scale_hidden_dims", None)

        norm_fn = attn_utils.get_norm_layer(
            self.train, self.dtype, norm_type=config.norm_type)
        conv_fn = functools.partial(
            nn.Conv, dtype=self.dtype, kernel_init=default_kernel_init)
        dense_fn = functools.partial(
            nn.Dense, dtype=self.dtype, kernel_init=default_kernel_init)
        encoder_dict = dict(
            num_heads=config.num_heads,
            norm_fn=norm_fn,
            mlp_ratio=config.mlp_ratio,
            attn_type=config.attn_type,
            dense_fn=dense_fn,
            activation_fn=self.activation_fn,
            qkv_bias=config.qkv_bias,
            attn_drop=config.attn_drop,
            proj_drop=config.proj_drop,
            train=self.train,
            dtype=self.dtype)
        x = self_attention.PatchEmbedding(
            conv_fn=conv_fn,
            patch_size=(config.init_patch_embed_size, config.init_patch_embed_size),
            embedding_dim=config.embedding_dim)(
            inputs)
        x = attn_utils.block_images(x, (config.patch_size, config.patch_size))
        block_idx = 0
        total_block_num = np.sum(num_layers_per_block)
        path_drop = np.linspace(0, config.stochastic_depth_drop, total_block_num)
        for i in range(num_blocks):
            x = self_attention.PositionEmbedding()(x)
            if scale_hidden_dims and i != 0:
                # Overwrite the original num_heads value in encoder_dict so num_heads
                # multipled by scale_hidden_dims continueously.
                encoder_dict.update(
                    {"num_heads": encoder_dict["num_heads"] * scale_hidden_dims})
            for _ in range(num_layers_per_block[i]):
                x = self_attention.EncoderNDBlock(
                    **encoder_dict, path_drop=path_drop[block_idx])(
                    x)
                block_idx = block_idx + 1
            # add here saving of feature maps for later use in GradCAT alg
            # if save_gradCAT_data:
            #  uu=0

            if i < num_blocks - 1:
                grid_size = int(math.sqrt(x.shape[1]))
                if scale_hidden_dims:
                    output_dim = x.shape[-1] * scale_hidden_dims
                else:
                    output_dim = None

                x = self_attention.ConvPool(
                    grid_size=(grid_size, grid_size),
                    patch_size=(config.patch_size, config.patch_size),
                    conv_fn=conv_fn,
                    dtype=self.dtype,
                    output_dim=output_dim)(
                    x)
        x = norm_fn()(x)
        self.sow('intermediates', 'last_layer', x)
        return x

class NesTB(nn.Module):
    """Nested Transformer Net."""
    num_classes: int
    config: ml_collections.ConfigDict
    train: bool = False
    dtype: int = jnp.float32
    activation_fn: Any = nn.gelu
    @nn.compact
    def __call__(self, inputs):
        #config = self.config
        #num_layers_per_block = config.num_layers_per_block
        #num_blocks = len(num_layers_per_block)

        # If `scale_hidden_dims` is provided, at every block, it increases hidden
        # dimension and num_heads by `scale_hidden_dims`. Set `scale_hidden_dims=2`
        # overall is a common design, so we do not gives the flexibility to control
        # layer-wise `scale_hidden_dims` to simplify the architecture.
        #scale_hidden_dims = config.get("scale_hidden_dims", None)

        #norm_fn = attn_utils.get_norm_layer(
        #    self.train, self.dtype, norm_type=config.norm_type)

        dense_fn = functools.partial(
            nn.Dense, dtype=self.dtype, kernel_init=default_kernel_init)

        #assert inputs.shape[1] == 1
        #assert inputs.shape[2] == config.patch_size ** 2
        x=inputs
        #x = norm_fn()(inputs)
        x_pool = jnp.mean(x, axis=(1, 2))
        #self.sow('intermediates', 'h', x)
        out = dense_fn(self.num_classes)(x_pool)
        logits = flax.linen.softmax(out)
        res=logits.max()
        return res
        #return out

class PatchEmbaddingBlock(nn.Module):
    num_classes: int
    config: ml_collections.ConfigDict
    train: bool = False
    dtype: int = jnp.float32
    activation_fn: Any = nn.gelu

    @nn.compact
    def __call__(self, inputs):
        config = self.config
        num_layers_per_block = config.num_layers_per_block
        num_blocks = len(num_layers_per_block)
        # save_gradCAT_data = config.save_gradCAT_data
        # Here we just assume image/patch size are squared.
        assert inputs.shape[1] == inputs.shape[2]
        assert inputs.shape[1] % config.init_patch_embed_size == 0
        input_size_after_patch = inputs.shape[1] // config.init_patch_embed_size
        assert input_size_after_patch % config.patch_size == 0
        down_sample_ratio = input_size_after_patch // config.patch_size
        # There are 4 child nodes for each node.
        assert num_blocks == int(np.log(down_sample_ratio) / np.log(2) + 1)

        # If `scale_hidden_dims` is provided, at every block, it increases hidden
        # dimension and num_heads by `scale_hidden_dims`. Set `scale_hidden_dims=2`
        # overall is a common design, so we do not gives the flexibility to control
        # layer-wise `scale_hidden_dims` to simplify the architecture.
        scale_hidden_dims = config.get("scale_hidden_dims", None)

        norm_fn = attn_utils.get_norm_layer(
            self.train, self.dtype, norm_type=config.norm_type)
        conv_fn = functools.partial(
            nn.Conv, dtype=self.dtype, kernel_init=default_kernel_init)
        dense_fn = functools.partial(
            nn.Dense, dtype=self.dtype, kernel_init=default_kernel_init)
        encoder_dict = dict(
            num_heads=config.num_heads,
            norm_fn=norm_fn,
            mlp_ratio=config.mlp_ratio,
            attn_type=config.attn_type,
            dense_fn=dense_fn,
            activation_fn=self.activation_fn,
            qkv_bias=config.qkv_bias,
            attn_drop=config.attn_drop,
            proj_drop=config.proj_drop,
            train=self.train,
            dtype=self.dtype)
        x = self_attention.PatchEmbedding(
            conv_fn=conv_fn,
            patch_size=(config.init_patch_embed_size, config.init_patch_embed_size),
            embedding_dim=config.embedding_dim)(
            inputs)
        return x

class BlockImages(nn.Module):
    num_classes: int
    config: ml_collections.ConfigDict
    train: bool = False
    dtype: int = jnp.float32
    activation_fn: Any = nn.gelu

    @nn.compact
    def __call__(self, inputs):
        config = self.config
        num_layers_per_block = config.num_layers_per_block
        num_blocks = len(num_layers_per_block)
        x = attn_utils.block_images(inputs, (config.patch_size, config.patch_size))
        return x


class PosEmbedAndEncodeBlock(nn.Module):
    level: int
    num_classes: int
    config: ml_collections.ConfigDict
    train: bool = False
    dtype: int = jnp.float32
    activation_fn: Any = nn.gelu

    @nn.compact
    def __call__(self, inputs):
        config = self.config
        num_layers_per_block = config.num_layers_per_block

        # If `scale_hidden_dims` is provided, at every block, it increases hidden
        # dimension and num_heads by `scale_hidden_dims`. Set `scale_hidden_dims=2`
        # overall is a common design, so we do not gives the flexibility to control
        # layer-wise `scale_hidden_dims` to simplify the architecture.
        scale_hidden_dims = config.get("scale_hidden_dims", None)

        norm_fn = attn_utils.get_norm_layer(
            self.train, self.dtype, norm_type=config.norm_type)
        conv_fn = functools.partial(
            nn.Conv, dtype=self.dtype, kernel_init=default_kernel_init)
        dense_fn = functools.partial(
            nn.Dense, dtype=self.dtype, kernel_init=default_kernel_init)
        encoder_dict = dict(
            num_heads=config.num_heads,
            norm_fn=norm_fn,
            mlp_ratio=config.mlp_ratio,
            attn_type=config.attn_type,
            dense_fn=dense_fn,
            activation_fn=self.activation_fn,
            qkv_bias=config.qkv_bias,
            attn_drop=config.attn_drop,
            proj_drop=config.proj_drop,
            train=self.train,
            dtype=self.dtype)

        total_block_num = np.sum(num_layers_per_block)
        path_drop = np.linspace(0, config.stochastic_depth_drop, total_block_num)
        block_idx = 0
        i = self.level
        for j in range(i):
            block_idx += self.config.num_layers_per_block[j]
        name = 'PositionEmbedding_'+str(self.level)
        x = self_attention.PositionEmbedding(name=name)(inputs)
        if scale_hidden_dims and self.level != 0:
            # Overwrite the original num_heads value in encoder_dict so num_heads
            # multipled by scale_hidden_dims continueously.
            encoder_dict.update(
                {"num_heads": encoder_dict["num_heads"] * scale_hidden_dims * self.level})
        for _ in range(num_layers_per_block[self.level]):
            name = 'EncoderNDBlock_' + str(block_idx)
            x = self_attention.EncoderNDBlock(name=name,
                **encoder_dict, path_drop=path_drop[block_idx])(
                x)
            block_idx = block_idx + 1
        self.sow('intermediates', 'features_maps', x)
        return x

class AggregateBlock(nn.Module):
    level: int
    num_classes: int
    config: ml_collections.ConfigDict
    train: bool = False
    dtype: int = jnp.float32
    activation_fn: Any = nn.gelu

    @nn.compact
    def __call__(self, inputs):
        config = self.config
        num_layers_per_block = config.num_layers_per_block
        num_blocks = len(num_layers_per_block)
        # If `scale_hidden_dims` is provided, at every block, it increases hidden
        # dimension and num_heads by `scale_hidden_dims`. Set `scale_hidden_dims=2`
        # overall is a common design, so we do not gives the flexibility to control
        # layer-wise `scale_hidden_dims` to simplify the architecture.
        scale_hidden_dims = config.get("scale_hidden_dims", None)

        norm_fn = attn_utils.get_norm_layer(
            self.train, self.dtype, norm_type=config.norm_type)
        conv_fn = functools.partial(
            nn.Conv, dtype=self.dtype, kernel_init=default_kernel_init)
        dense_fn = functools.partial(
            nn.Dense, dtype=self.dtype, kernel_init=default_kernel_init)
        encoder_dict = dict(
            num_heads=config.num_heads,
            norm_fn=norm_fn,
            mlp_ratio=config.mlp_ratio,
            attn_type=config.attn_type,
            dense_fn=dense_fn,
            activation_fn=self.activation_fn,
            qkv_bias=config.qkv_bias,
            attn_drop=config.attn_drop,
            proj_drop=config.proj_drop,
            train=self.train,
            dtype=self.dtype)

        if self.level < num_blocks - 1:
            grid_size = int(math.sqrt(inputs.shape[1]))
            if scale_hidden_dims:
                output_dim = inputs.shape[-1] * scale_hidden_dims
            else:
                output_dim = None

            name = 'ConvPool_' + str(self.level)
            x = self_attention.ConvPool_no_unblock(name=name,
                grid_size=(grid_size, grid_size),
                patch_size=(config.patch_size, config.patch_size),
                conv_fn=conv_fn,
                dtype=self.dtype,
                output_dim=output_dim)(
                inputs)
            self.sow('intermediates', 'features_maps', x)

        return x

class DenseBlock(nn.Module):
    level: int
    num_classes: int
    config: ml_collections.ConfigDict
    train: bool = False
    dtype: int = jnp.float32
    activation_fn: Any = nn.gelu

    @nn.compact
    def __call__(self, inputs):
        config = self.config
        num_layers_per_block = config.num_layers_per_block
        num_blocks = len(num_layers_per_block)
        # If `scale_hidden_dims` is provided, at every block, it increases hidden
        # dimension and num_heads by `scale_hidden_dims`. Set `scale_hidden_dims=2`
        # overall is a common design, so we do not gives the flexibility to control
        # layer-wise `scale_hidden_dims` to simplify the architecture.
        scale_hidden_dims = config.get("scale_hidden_dims", None)
        assert inputs.shape[1] == 1
        assert inputs.shape[2] == config.patch_size ** 2
        config.classname = 'nest_modules.DenseBlock'
        norm_fn = attn_utils.get_norm_layer(
            self.train, self.dtype, norm_type=config.norm_type)

        dense_fn = functools.partial(
            nn.Dense, dtype=self.dtype, kernel_init=default_kernel_init)

        x = norm_fn()(inputs)
        x_pool = jnp.mean(x, axis=(1, 2))
        out = dense_fn(self.num_classes)(x_pool)
        logits = flax.linen.softmax(out)
        #res = logits.max()
        res = logits.max()
        return res
