import flax
import jax
from models import nest_modules
import functools
from absl import logging
import os
import numpy as np
import tensorflow as tf
tf.config.experimental.set_visible_devices([], "GPU")



from configs import imagenet_nest
import train
from models import basic_nest_defs
from libml import preprocess
from libml import attn_utils
import jax.numpy as jnp
import math


from configs import dog_breeds
from configs import cats_pain

tf.config.experimental.set_visible_devices([], "GPU")
logging.set_verbosity(logging.INFO)

print("JAX devices:\n" + "\n".join([repr(d) for d in jax.devices()]))
print('Current folder content', os.listdir())


#checkpoint_dir = "../checkpoints/"
#remote_checkpoint_dir = "gs://gresearch/nest-checkpoints/nest-b_imagenet"

checkpoint_dir = "checkpoints/nest_cats/3/checkpoints-0/"
print('List checkpoints: ')

# Use checkpoint of host 0.
#imagenet_config = imagenet_nest.get_config()
imagenet_config = cats_pain.get_config()


#state_dict = train.checkpoint.load_state_dict(
#    os.path.join(checkpoint_dir, os.path.basename(remote_checkpoint_dir)))

state_dict = train.checkpoint.load_state_dict(checkpoint_dir)

variables = {
    "params": state_dict["optimizer"]["target"],
}
config = imagenet_config
variables.update(state_dict["model_state"])
config.classname = 'nest_modules.PosEmbedAndEncodeBlock'
model_cls_posembed_encode0 = basic_nest_defs.create_model(imagenet_config.model_name, config)
model_posembed_encode0 = functools.partial(model_cls_posembed_encode0, num_classes=2)
config.classname = 'nest_modules.AggregateBlock'
model_cls_aggregate0 = basic_nest_defs.create_model(imagenet_config.model_name, config)
model_aggregate0 = functools.partial(model_cls_aggregate0, num_classes=2)
MAX_LEVEL = 3

import PIL

#img = PIL.Image.open("../imgs/n02100877_7560.jpg")#irish setter
#img = PIL.Image.open("../imgs/n02088094_60.jpg")#afghan hound
#img = PIL.Image.open("../imgs/n02110958_8627.jpg")#pug

############################################3
def do_bef_grad_level_transformers_3(inputs):
    x, state = model_posembed_encode0(train=False, level=0).apply(variables, inputs, mutable='intermediates')
    grid_size = int(math.sqrt(x.shape[1]))
    x = attn_utils.unblock_images(
        x, grid_size=(grid_size, grid_size), patch_size=(14, 14))

    x, agg_state = model_aggregate0(train=False, level=0).apply(variables, x, mutable='intermediates')
    x, state = model_posembed_encode0(train=False, level=1).apply(variables, x, mutable='intermediates')
    grid_size = int(math.sqrt(x.shape[1]))
    x = attn_utils.unblock_images(
        x, grid_size=(grid_size, grid_size), patch_size=(14, 14))

    x, agg_state = model_aggregate0(train=False, level=1).apply(variables, x, mutable='intermediates')
    x, state = model_posembed_encode0(train=False, level=2).apply(variables, x, mutable='intermediates')

    return x, state, agg_state

def do_post_grad_level_3(inputs):
    #for l in range(int(level)+1, MAX_LEVEL-1):
    #    x, state = model_posembed_encode0(train=False, level=l).apply(variables, inputs, mutable='intermediates')
    #    x = model_aggregate0(train=False, level=l).apply(variables, x, mutable=False)
    #l = l + 1
        #agg_state = state
    x=inputs
    config.classname = 'nest_modules.DenseBlock'
    model_cls_dense = basic_nest_defs.create_model(imagenet_config.model_name, config)
    model_dense = functools.partial(model_cls_dense, num_classes=2)
    prob = model_dense(train=False, level=2).apply(variables, x, mutable=False)
    return prob

def do_bef_grad_level_transformers_2(inputs):
    x, state = model_posembed_encode0(train=False, level=0).apply(variables, inputs, mutable='intermediates')
    grid_size = int(math.sqrt(x.shape[1]))
    x = attn_utils.unblock_images(
        x, grid_size=(grid_size, grid_size), patch_size=(14, 14))

    x, agg_state = model_aggregate0(train=False, level=0).apply(variables, x, mutable='intermediates')

    x, state = model_posembed_encode0(train=False, level=1).apply(variables, x, mutable='intermediates')
    grid_size = int(math.sqrt(x.shape[1]))
    x = attn_utils.unblock_images(
        x, grid_size=(grid_size, grid_size), patch_size=(14, 14))


    return x, state, agg_state

def do_post_grad_level_2(inputs):
    x, agg_state = model_aggregate0(train=False, level=1).apply(variables, inputs, mutable='intermediates')
    x, state = model_posembed_encode0(train=False, level=2).apply(variables, x, mutable='intermediates')
    config.classname = 'nest_modules.DenseBlock'
    model_cls_dense = basic_nest_defs.create_model(imagenet_config.model_name, config)
    model_dense = functools.partial(model_cls_dense, num_classes=2)
    prob = model_dense(train=False, level=2).apply(variables, x, mutable=False)
    return prob



def do_bef_grad_level_transformers_1(inputs):
    x, state = model_posembed_encode0(train=False, level=0).apply(variables, inputs, mutable='intermediates')
    grid_size = int(math.sqrt(x.shape[1]))
    x = attn_utils.unblock_images(
        x, grid_size=(grid_size, grid_size), patch_size=(14, 14))

    x, agg_state = model_aggregate0(train=False, level=0).apply(variables, x, mutable='intermediates')

    x, state = model_posembed_encode0(train=False, level=1).apply(variables, x, mutable='intermediates')
    grid_size = int(math.sqrt(x.shape[1]))
    x = attn_utils.unblock_images(
        x, grid_size=(grid_size, grid_size), patch_size=(14, 14))


    return x, state, agg_state

def do_post_grad_level_1(inputs):
    x, agg_state = model_aggregate0(train=False, level=1).apply(variables, inputs, mutable='intermediates')
    x, state = model_posembed_encode0(train=False, level=2).apply(variables, x, mutable='intermediates')
    config.classname = 'nest_modules.DenseBlock'
    model_cls_dense = basic_nest_defs.create_model(imagenet_config.model_name, config)
    model_dense = functools.partial(model_cls_dense, num_classes=2)
    prob = model_dense(train=False, level=2).apply(variables, x, mutable=False)
    return prob
def calc_maps_and_grads_part__(inputs):
    x, state3, agg_state3= do_bef_grad_level_transformers_3(inputs)
    grad_func3 = jax.grad(do_post_grad_level_3)
    grads3 = grad_func3(x)
    x, state2, agg_state2 = do_bef_grad_level_transformers_2(inputs)
    grad_func2 = jax.grad(do_post_grad_level_2)
    grads2 = grad_func2(x)
    x, state1, agg_state1 = do_bef_grad_level_transformers_1(inputs)
    grad_func1 = jax.grad(do_post_grad_level_1)
    grads1 = grad_func1(x)
    return grads3, grads2, grads1, state3, state2, state1,agg_state3, agg_state2, agg_state1

def calc_maps_and_grads__(inputs):
    config.classname = 'nest_modules.PatchEmbaddingBlock'
    model_cls_patch_embed = basic_nest_defs.create_model(imagenet_config.model_name, config)
    model_patch_embed = functools.partial(model_cls_patch_embed, num_classes=2)
    x = model_patch_embed(train=False).apply(variables, inputs, mutable=False)
    config.classname = 'nest_modules.BlockImages'
    model_cls_block_images = basic_nest_defs.create_model(imagenet_config.model_name, config)
    model_block_images = functools.partial(model_cls_block_images, num_classes=2)
    x = model_block_images(train=False).apply(variables, x, mutable=False)
    grads3, grads2, grads1, state3, state2, state1, agg_state3, agg_state2, agg_state1= calc_maps_and_grads_part__( x)
    return grads3, grads2, grads1, state3, state2, state1, agg_state3, agg_state2, agg_state1

def grad_cat_level3(feature_map, grads, grid_size, patch_size, win_part):
    ftrs_shaped = attn_utils.unblock_images(
        feature_map, grid_size=grid_size, patch_size=patch_size)
        #now x is 1,14,14,512
    if grid_size[0] is 1:
        grads_shaped = attn_utils.unblock_images(
            grads, grid_size=grid_size, patch_size=patch_size)
    else:
        grads_shaped = grads

    pooled_grads = grads_shaped.squeeze().mean((0, 1))
    conv_output = ftrs_shaped.squeeze()

    for i in range(len(pooled_grads)):
        conv_output = conv_output.at[:, :, i].set(conv_output[:, :, i] * pooled_grads[i])

    heatmap = conv_output.mean(axis=-1)

    heatmap1 = flax.linen.relu(heatmap) / heatmap.max()
    heatmap2= heatmap / heatmap.max()
    heatmap3 = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    heatmap3 = heatmap3.reshape(1, heatmap3.shape[0], heatmap3.shape[1], 1)
    h11_ = flax.linen.avg_pool(heatmap3, window_shape=(heatmap3.shape[1]//win_part, heatmap3.shape[2]//win_part),
                               strides=(heatmap3.shape[1]//win_part, heatmap3.shape[2]//win_part))
    heatmap1 = heatmap1.reshape(1, heatmap1.shape[0], heatmap1.shape[1], 1)
    h11_ = flax.linen.avg_pool(heatmap1, window_shape=(heatmap1.shape[1] // win_part, heatmap1.shape[2] //win_part),
                               strides=(heatmap1.shape[1] // win_part, heatmap1.shape[2] // win_part))

    return h11_


def grad_cat_level2(features_maps, grads, patch_size):
    ftrs_shaped = attn_utils.unblock_images(
        features_maps, grid_size=(2, 2), patch_size=patch_size)

    batch, height, width, depth = grads.shape
    for d in range(0, features_maps.shape[1]):
        curr_grads = grads[:, :, :, d]
        pooled_grads = curr_grads.squeeze().mean((0, 1))
        curr_output = ftrs_shaped[:, :, :, d]
        curr_output = curr_output.squeeze()

        for i in range(len(pooled_grads)):
            conv_output = conv_output.at[:, :, i].set(conv_output[:, :, i] * pooled_grads[i])

        heatmap = conv_output.mean(axis=-1)

        heatmap1 = flax.linen.relu(heatmap) / heatmap.max()
        heatmap2 = heatmap / heatmap.max()
        heatmap3 = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

        heatmap3 = heatmap3.reshape(1, heatmap3.shape[0], heatmap3.shape[1], 1)
        h11_ = flax.linen.avg_pool(heatmap3, window_shape=(heatmap3.shape[1] // 2, heatmap3.shape[2] // 2),
                                   strides=(heatmap3.shape[1] // 2, heatmap3.shape[2] // 2))
    return 0


def reshape_last_ftr_map(feature_map, grid_size,patch_size):
    batch, grid_length, patch_length, depth = feature_map.shape
    #(grid_size is 1,1)
    assert np.prod(grid_size) == grid_length
    assert np.prod(patch_size) == patch_length
    new_shape = (batch, grid_size[0], grid_size[1], patch_size[0], patch_size[1],
                 depth)
    height = grid_size[0] * patch_size[0]
    width = grid_size[1] * patch_size[1]
    x = jnp.reshape(feature_map, new_shape)
    x = jnp.transpose(x, (0, 1, 3, 2, 4, 5))
    x = jnp.reshape(x, (batch, height, width, depth))
    return x

#############################################################################
def do_bef_grad_level_transformers(level, inputs):
    #for l in range(0, level+1):
    l = 0
    x, state = model_posembed_encode0(train=False, level=l).apply(variables, inputs, mutable='intermediates')
    x, agg_state = model_aggregate0(train=False, level=l).apply(variables, x, mutable='intermediates')
    if level > 0 :
        l = l + 1
        x, state = model_posembed_encode0(train=False, level=l).apply(variables, x, mutable='intermediates')
        x, agg_state = model_aggregate0(train=False, level=l).apply(variables, x, mutable='intermediates')
    if level > 1:
        l = l + 1
        x, state = model_posembed_encode0(train=False, level=l).apply(variables, x, mutable='intermediates')
        agg_state = state

    return x, state, agg_state, l

def do_post_grad_level(level,inputs):
    #for l in range(int(level)+1, MAX_LEVEL-1):
    #    x, state = model_posembed_encode0(train=False, level=l).apply(variables, inputs, mutable='intermediates')
    #    x = model_aggregate0(train=False, level=l).apply(variables, x, mutable=False)
    #l = l + 1
    level_int= int(level) # can be 0, 1 or 2
    l = level_int
    x = inputs
    if level_int == 0:
        l = l+1
        x, state = model_posembed_encode0(train=False, level=l).apply(variables, x, mutable='intermediates')
        x, agg_state = model_aggregate0(train=False, level=l).apply(variables, x, mutable='intermediates')
    if level_int == 1:
        l = l+1
        x, state = model_posembed_encode0(train=False, level=l).apply(variables, x, mutable='intermediates')
        agg_state=state

    config.classname = 'nest_modules.DenseBlock'
    model_cls_dense = basic_nest_defs.create_model(imagenet_config.model_name, config)
    model_dense = functools.partial(model_cls_dense, num_classes=2)
    prob = model_dense(train=False, level=l).apply(variables, x, mutable=False)
    return prob

def calc_maps_and_grads_for_level(level, inputs):
    x, state, agg_state, l = do_bef_grad_level_transformers(level, inputs)
    grad_func = jax.grad(do_post_grad_level,argnums=1)
    grads = grad_func(float(level), x)
    return grads, state, agg_state, grads*agg_state['intermediates']['features_maps'][0]

def _preprocess(image):
  image = np.array(image.resize((224, 224))).astype(np.float32) / 255
  mean = np.array(preprocess.IMAGENET_DEFAULT_MEAN).reshape(1, 1, 3)
  std = np.array(preprocess.IMAGENET_DEFAULT_STD).reshape(1, 1, 3)
  image = (image - mean) / std


  return image[np.newaxis,...]

def calc_maps_and_grads(image):
    inputs = _preprocess(image)
    config.classname = 'nest_modules.PatchEmbaddingBlock'
    model_cls_patch_embed = basic_nest_defs.create_model(imagenet_config.model_name, config)
    model_patch_embed = functools.partial(model_cls_patch_embed, num_classes=2)
    x = model_patch_embed(train=False).apply(variables, inputs, mutable=False)
    config.classname = 'nest_modules.BlockImages'
    model_cls_block_images = basic_nest_defs.create_model(imagenet_config.model_name, config)
    model_block_images = functools.partial(model_cls_block_images, num_classes=2)
    x = model_block_images(train=False).apply(variables, x, mutable=False)
    grads0, state0, agg_state0, mult0 = calc_maps_and_grads_for_level(0, x)
    grads1, state1, agg_state1, mult1 = calc_maps_and_grads_for_level(1, x)
    grads2, state2, agg_state2, mult2 = calc_maps_and_grads_for_level(2, x)
    return mult0, mult1, mult2


#grads3, grads2,grads1, state3, state2, state1, agg_state3, agg_state2, agg_state1 = calc_maps_and_grads__(img)
#hh=grad_cat_level3(state3['intermediates']['features_maps'][0], grads3, grid_size=(1,1), patch_size=(14,14), win_part = 2)
#hhh=grad_cat_level3(state2['intermediates']['features_maps'][0], grads2, grid_size=(2,2), patch_size=(14,14), win_part = 4)
#mult0, mult1, mult2 = calc_maps_and_grads(img)

#h1_0 = -1*mult0
#h1_1 = -1*mult1
#h2_2 = -1*mult2
#h11=jnp.transpose(h2_2,(0,2,3,1))
#h11_ = flax.linen.avg_pool(h11, window_shape=(98, 256),strides=(98,256))