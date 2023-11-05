import sys
sys.path.append('./nested-transformer')

import os
import time
import flax
from flax import nn
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import functools
from absl import logging

from libml import input_pipeline
from libml import preprocess
from models import nest_net
import train
from configs import cifar_nest
from configs import imagenet_nest
from jax import grad  # for gradCAT

from models import conv_trials

# Hide any GPUs form TensorFlow. Otherwise TF might reserve memory and make
# it unavailable to JAX.
tf.config.experimental.set_visible_devices([], "GPU")
logging.set_verbosity(logging.INFO)

print("JAX devices:\n" + "\n".join([repr(d) for d in jax.devices()]))
print('Current folder content', os.listdir())


#checkpoint_dir = "./nested-transformer/checkpoints/"
checkpoint_dir = "./checkpoints/"
remote_checkpoint_dir = "gs://gresearch/nest-checkpoints/nest-b_imagenet"
print('List checkpoints: ')

# Use checkpoint of host 0.
imagenet_config = imagenet_nest.get_config()

state_dict = train.checkpoint.load_state_dict(
    os.path.join(checkpoint_dir, os.path.basename(remote_checkpoint_dir)))
variables = {
    "params": state_dict["optimizer"]["target"],
}
variables.update(state_dict["model_state"])
model_cls = nest_net.create_model(imagenet_config.model_name, imagenet_config)
model = functools.partial(model_cls, num_classes=1000)


import PIL

#img = PIL.Image.open('dog.jpg')
img = PIL.Image.open("spoon.jpg")
#img = PIL.Image.open('13-0014.jpg')
img
#%%
def predict(image):
  logits = model(train=False).apply(variables, image, mutable=False)
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
    seed = jax.random.PRNGKey(0)
    model = flax.linen.Conv(features=32, kernel_size=(3,3), padding="SAME", name="CONV1")
    params = model.init(seed, inputs)
    x = model.apply(params,inputs)
    return x

input = _preprocess(img)
x=_try_conv(input)
cls, prob = predict(input)
print(f'ImageNet class id: {cls[0]}, prob: {prob[0]}')