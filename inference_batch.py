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
from configs import dog_breeds
from jax import grad  # for gradCAT

from models import nest_modules
from libml import self_attention
from models import basic_nest_defs
import pandas as pd
import trials
from models import nest_for_grad
from helpers_for_specific_data import dog_head_shapes_helpers as helpers
# Hide any GPUs form TensorFlow. Otherwise TF might reserve memory and make
# it unavailable to JAX.
tf.config.experimental.set_visible_devices([], "GPU")
logging.set_verbosity(logging.INFO)

print("JAX devices:\n" + "\n".join([repr(d) for d in jax.devices()]))
print('Current folder content', os.listdir())


#checkpoint_dir = "./nested-transformer/checkpoints/"
#checkpoint_dir = "./checkpoints/"
##checkpoint_dir = "./checkpoints/nest_dogs/checkpoints-0/"
##remote_checkpoint_dir = "gs://gresearch/nest-checkpoints/nest-b_imagenet"
##print('List checkpoints: ')

# Use checkpoint of host 0.
#imagenet_config = imagenet_nest.get_config()
imagenet_config = dog_breeds.get_config()
#state_dict = train.checkpoint.load_state_dict(
#    os.path.join(checkpoint_dir, os.path.basename(remote_checkpoint_dir)))
##state_dict = train.checkpoint.load_state_dict(checkpoint_dir)

##variables = {
##    "params": state_dict["optimizer"]["target"],
##}
##variables.update(state_dict["model_state"])
##model_cls = nest_net.create_model(imagenet_config.model_name, imagenet_config)
#model = functools.partial(model_cls, num_classes=1000)

##model = functools.partial(model_cls, num_classes=2)
import PIL


root = '/home/tali/Images'
dogs_dirs =['n02088094-Afghan_hound']
clss =[0]
#dogs_dirs=['n02086079-Pekinese', 'n02086240-Shih-Tzu', 'n02108089-boxer', 'n02108422-bull_mastiff',
#           'n02108915-French_bulldog', 'n02110627-affenpinscher', 'n02110958-pug','n02112137-chow',
#           'n02088094-Afghan_hound','n02090622-borzoi','n02091032-Italian_greyhound','n02091831-Saluki',
#           'n02092002-Scottish_deerhound','n02100735-English_setter','n02100877-Irish_setter',
#           'n02101006-Gordon_setter','n02106030-collie','n02106166-Border_collie']
allowed_labels = [3, 4, 91, 92, 94, 100, 102, 108,9, 18, 20, 25, 26, 61, 62, 63, 80, 81]
#clss =[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,0, 0, 0, 0, 0]


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


def inference_sample(img_path: str):
    img = PIL.Image.open(img_path)
    input = _preprocess(img)
    cls, prob = predict(input)
    print(f'class id: {cls[0]}, prob: {prob[0]}')
    return cls, prob

def create_img(img_path: str, out_path: str):
    img = PIL.Image.open(img_path)
    inputs = _preprocess(img)
    grads3, grads2, grads1, state3, state2, state1, agg_state3, agg_state2, agg_state1 = (
        nest_for_grad.calc_maps_and_grads__(inputs))
    hh = nest_for_grad.grad_cat_level3(state3['intermediates']['features_maps'][0], grads3,
                        grid_size=(1, 1), patch_size=(14, 14),win_part=2)
    hhh = nest_for_grad.grad_cat_level3(state2['intermediates']['features_maps'][0], grads2,
                        grid_size=(2, 2), patch_size=(14, 14),win_part=4)
    trials.plot_grid(img_path, out_path, False, hh, hhh)


def inference_dir(root : str, dir_name:str, dir_cls : int):
    print(f'correct_cls: {dir_cls}')
    imgs_path = os.path.join(root, dir_name)
    dirs = os.listdir(imgs_path)
    for item in dirs:
        img_path = os.path.join(imgs_path ,item)
        if os.path.isfile(img_path):
            cls, prob = inference_sample(img_path)
            if cls == dir_cls:
                correct = True
            else:
                correct = False

def create_imgs_dir(root : str, dir_name:str, dir_cls : int, out_root: str):
    print(f'correct_cls: {dir_cls}')
    imgs_path = os.path.join(root, dir_name)
    out_imgs_path = os.path.join(out_root, dir_name)
    os.makedirs(out_imgs_path,exist_ok = True)
    dirs = os.listdir(imgs_path)
    for item in dirs:
        img_path = os.path.join(imgs_path, item)
        fname = os.path.splitext(item)[0]
        out_path = os.path.join(out_imgs_path,fname+'.png')
        if os.path.isfile(img_path):
            create_img(img_path, out_path)

def run_on_all(root: str, dirs_names : list[str], dirs_cls: list[int]):
    cnt = 0
    for d in dirs_names:
        inference_dir(root, d, dirs_cls[cnt])
        cnt = cnt+1

def create_all_images(root: str, dirs_names : list[str], dirs_cls: list[int], out_root: str):
    cnt = 0
    for d in dirs_names:
        create_imgs_dir(root, d, dirs_cls[cnt], out_root)
        cnt = cnt+1

def run_on_db(config, data_rng):
    ds_info, train_ds, eval_ds = input_pipeline.create_datasets(config, data_rng)
    train_ds = train_ds.filter(helpers.predicate)
    train_ds = train_ds.map(helpers.update_label)
    train_ds = train_ds.map(helpers.update_image_to_bb)

    # df = tfds.as_dataframe(eval_ds1.take(3), ds_info)
    eval_sz = eval_ds.reduce(0, lambda x, _: x + 1)
    for step, batch in enumerate(eval_ds):
        batch = jax.tree_map(np.asarray, batch)
        output_path = "/home/tali/trials/gg.png"
        create_img(batch["image"],output_path)


#run_on_all(root, dogs_dirs, clss)
create_all_images(root, dogs_dirs, clss, '/home/tali/trials/')
