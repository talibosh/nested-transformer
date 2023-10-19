import functools
import math
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
from libml import attn_utils
from libml import self_attention
import numpy as np
import train
from models import nest_net
import os
from typing import List


default_kernel_init = attn_utils.trunc_normal(stddev=0.02)
default_bias_init = jax.nn.initializers.zeros

# for each level
# class nesTBrik (ftr map)=>class nest dense (logits)=>grad=>grad*ftrmap
# find max value (as in pseudo alg) and continue in correct direction

# GradCat
class GradCat:
    def RunGradCat(self, features_maps: List[list], grads: List[list]):
        traversal_path = [0]
        grade_values = [999]
        assert features_maps.__len__() == grads.__len__()
        for idx in range(len(features_maps)):
            curr_level_ftrs_maps = features_maps(idx)  # list of feature maps of curr level
            curr_grads = grads(idx)  # list of grads of curr level
            assert len(curr_level_ftrs_maps) == len(curr_grads)
            for count in range(len(curr_grads)):
                h1 = curr_level_ftrs_maps(count)*curr_grads(count)
                h1_ = nn.avg_pool(h1,
                window_shape=(curr_level_ftrs_maps(count).shape(0)//4, curr_level_ftrs_maps(count).shape(1)//2)
                            )
                n_star = h1_.argmax()
                grade = h1_.max()
                traversal_path.append(n_star)
                grade_values.append(grade)
        return traversal_path, grade_values



