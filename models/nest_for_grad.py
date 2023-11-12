import flax
import jax
from models import nest_modules
import functools
from absl import logging
import os
import numpy as np
import tensorflow as tf
from configs import imagenet_nest
import train
from models import basic_nest_defs
from libml import preprocess

tf.config.experimental.set_visible_devices([], "GPU")
logging.set_verbosity(logging.INFO)

print("JAX devices:\n" + "\n".join([repr(d) for d in jax.devices()]))
print('Current folder content', os.listdir())


#checkpoint_dir = "./nested-transformer/checkpoints/"
checkpoint_dir = "../checkpoints/"
remote_checkpoint_dir = "gs://gresearch/nest-checkpoints/nest-b_imagenet"
print('List checkpoints: ')

# Use checkpoint of host 0.
imagenet_config = imagenet_nest.get_config()

state_dict = train.checkpoint.load_state_dict(
    os.path.join(checkpoint_dir, os.path.basename(remote_checkpoint_dir)))
variables = {
    "params": state_dict["optimizer"]["target"],
}
config = imagenet_config
variables.update(state_dict["model_state"])
config.classname = 'nest_modules.PosEmbedAndEncodeBlock'
model_cls_posembed_encode0 = basic_nest_defs.create_model(imagenet_config.model_name, config)
model_posembed_encode0 = functools.partial(model_cls_posembed_encode0, num_classes=1000)
config.classname = 'nest_modules.AggregateBlock'
model_cls_aggregate0 = basic_nest_defs.create_model(imagenet_config.model_name, config)
model_aggregate0 = functools.partial(model_cls_aggregate0, num_classes=1000)
MAX_LEVEL = 3

import PIL

img = PIL.Image.open("../spoon.jpg")
def do_bef_grad_level_transformers(level, inputs):
    #for l in range(0, level+1):
    l = 0
    x, state = model_posembed_encode0(train=False, level=l).apply(variables, inputs, mutable='intermediates')
    x, agg_state = model_aggregate0(train=False, level=l).apply(variables, x, mutable=False)
    if level > 0 :
        l = l + 1
        x, state = model_posembed_encode0(train=False, level=l).apply(variables, x, mutable='intermediates')
        x, agg_state = model_aggregate0(train=False, level=l).apply(variables, x, mutable=False)
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
        x, agg_state = model_aggregate0(train=False, level=l).apply(variables, x, mutable=False)
    if level_int == 1:
        l = l+1
        x, state = model_posembed_encode0(train=False, level=l).apply(variables, x, mutable='intermediates')
        agg_state=state

    config.classname = 'nest_modules.DenseBlock'
    model_cls_dense = basic_nest_defs.create_model(imagenet_config.model_name, config)
    model_dense = functools.partial(model_cls_dense, num_classes=1000)
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
    model_patch_embed = functools.partial(model_cls_patch_embed, num_classes=1000)
    x = model_patch_embed(train=False).apply(variables, inputs, mutable=False)
    config.classname = 'nest_modules.BlockImages'
    model_cls_block_images = basic_nest_defs.create_model(imagenet_config.model_name, config)
    model_block_images = functools.partial(model_cls_block_images, num_classes=1000)
    x = model_block_images(train=False).apply(variables, x, mutable=False)
    grads0, state0, mult0 = calc_maps_and_grads_for_level(0, x)
    grads1, state1, mult1 = calc_maps_and_grads_for_level(1, x)
    grads2, state2, mult2 = calc_maps_and_grads_for_level(2, x)
    return mult0, mult1, mult2



grads0, state0 = calc_maps_and_grads(img)