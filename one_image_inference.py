import sys
sys.path.append('./nested-transformer')

import os
import time
import flax
import flax.linen as nn
#from flax import nn
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
tf.config.experimental.set_visible_devices([], "GPU")
import functools
from absl import logging

from libml import input_pipeline
from libml import preprocess
from models import nest_net
import train
from configs import cifar_nest
from configs import imagenet_nest
from configs import cats_pain
from jax import grad  # for gradCAT

from models import nest_modules
from libml import self_attention
from models import basic_nest_defs

# Hide any GPUs form TensorFlow. Otherwise TF might reserve memory and make
# it unavailable to JAX.
tf.config.experimental.set_visible_devices([], "GPU")
logging.set_verbosity(logging.INFO)

print("JAX devices:\n" + "\n".join([repr(d) for d in jax.devices()]))
print('Current folder content', os.listdir())


#checkpoint_dir = "./nested-transformer/checkpoints/"
checkpoint_dir = "./checkpoints/"
#checkpoint_dir = './checkpoints/nest_cats_norm1_60/12/checkpoints-0/'#"./checkpoints/nest_dogs/checkpoints-0/"
remote_checkpoint_dir = "gs://gresearch/nest-checkpoints/nest-b_imagenet"
print('List checkpoints: ')

# Use checkpoint of host 0.
imagenet_config = imagenet_nest.get_config()
#imagenet_config =cats_pain.get_config()
state_dict = train.checkpoint.load_state_dict(
    os.path.join(checkpoint_dir, os.path.basename(remote_checkpoint_dir)))
#state_dict = train.checkpoint.load_state_dict(checkpoint_dir)

variables = {
    "params": state_dict["optimizer"]["target"],
}
variables.update(state_dict["model_state"])
model_cls = nest_net.create_model(imagenet_config.model_name, imagenet_config)
#model = functools.partial(model_cls, num_classes=1000)

model = functools.partial(model_cls, num_classes=1000)
import PIL

#img = PIL.Image.open('dog.jpg')
#img = PIL.Image.open('/home/tali/cats_pain_proj/face_images/pain/cat_12_video_3.11.jpg')#afghan hound
img = PIL.Image.open('n02086079_499.jpg')
img
#%%
def predict(image):
  #logits = model(train=False).apply(variables, image, mutable=False)

  logits, state = model(train=False).apply(variables, image, mutable=['intermediates'])

  # Return predicted class and confidence.
  return logits.argmax(axis=-1), nn.softmax(logits, axis=-1).max(axis=-1)

def _preprocess(image):
  image = np.array(image.resize((224, 224))).astype(np.float32) / 255
  mean = np.array(preprocess.IMAGENET_DEFAULT_MEAN).reshape(1, 1, 3)
  std = np.array(preprocess.IMAGENET_DEFAULT_STD).reshape(1, 1, 3)
  image = (image - mean) / std
  return image[np.newaxis,...]

#save data for gradCAT()
def _try_conv(inputs):
    from models import basic_nest_defs
    config = imagenet_config
    config.classname = 'nest_modules.PatchEmbaddingBlock'
    model_cls_patch_embed = basic_nest_defs.create_model(imagenet_config.model_name, config)
    model_patch_embed = functools.partial(model_cls_patch_embed, num_classes=1000)
    x = model_patch_embed(train=False).apply(variables, inputs, mutable=False)
    config.classname = 'nest_modules.BlockImages'
    model_cls_block_images = basic_nest_defs.create_model(imagenet_config.model_name, config)
    model_block_images = functools.partial(model_cls_block_images, num_classes=1000)
    x = model_block_images(train=False).apply(variables, x, mutable=False)
    config.classname = 'nest_modules.PosEmbedAndEncodeBlock'
    model_cls_posembed_encode0 = basic_nest_defs.create_model(imagenet_config.model_name, config)
    model_posembed_encode0 = functools.partial(model_cls_posembed_encode0, num_classes=1000)
    config.classname = 'nest_modules.AggregateBlock'
    model_cls_aggregate0 = basic_nest_defs.create_model(imagenet_config.model_name, config)
    model_aggregate0 = functools.partial(model_cls_aggregate0, num_classes=1000)

    for level in range(0,2):
        x = model_posembed_encode0(train=False, level=level).apply(variables, x, mutable=False)
        x = model_aggregate0(train=False, level=level).apply(variables, x, mutable=False)
    level = level+1
    x = model_posembed_encode0(train=False, level=level).apply(variables, x, mutable=False)
    config.classname = 'nest_modules.DenseBlock'
    model_cls_dense = basic_nest_defs.create_model(imagenet_config.model_name, config)
    model_dense = functools.partial(model_cls_dense, num_classes=1000)
    logits = model_dense(train=False, level=level).apply(variables, x, mutable=False)
    return logits.argmax(axis=-1), nn.softmax(logits, axis=-1).max(axis=-1)

def InsForGrad(x):
    config=imagenet_config
    config.classname = 'nest_modules.NesTB'
    model_cls_B= basic_nest_defs.create_model(imagenet_config.model_name, config)
    #model_B = functools.partial(model_cls_B, num_classes=1000)
    prob, state = model_cls_B(train=False, num_classes=2).apply(variables, x, mutable='intermediates')
    #prob = nn.softmax(logits, axis=-1).max(axis=-1)
    return prob

def try_grad(inputs):
    config = imagenet_config
    config.classname = 'nest_modules.NesTA'
    model_cls_A = basic_nest_defs.create_model(imagenet_config.model_name, config)
    #model_A = functools.partial(model_cls_A, num_classes=1000)
    x, state = model_cls_A(train=False, num_classes=2).apply(variables, inputs, mutable='intermediates')
    grad_func = jax.grad(InsForGrad)
    grads = grad_func(x)
    #norm_fn = functools.partial(nn.LayerNorm, epsilon=1e-6, dtype=dtype)
    feature_map = state['intermediates']['last_layer']
    h1 = -1 * x * grads
    h11=jnp.transpose(h1,(0,2,3,1))
    h11_ = nn.avg_pool(h11, window_shape=(98, 256),strides=(98,256))
    return grads
    #config.classname = 'nest_modules.nesTB'
    #model_cls_B= basic_nest_defs.create_model(imagenet_config.model_name, config)
    #model_B = functools.partial(model_cls_B, num_classes=1000)
    #x = model_B(train=False).apply(variables, x, mutable='intermediates')




input = _preprocess(img)
#x=_try_conv(input)
#my_grads = try_grad(input)
cls, prob = predict(input)
print(f'ImageNet class id: {cls[0]}, prob: {prob[0]}')