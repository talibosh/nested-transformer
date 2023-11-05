import jax
from typing import Any, Callable, Sequence
from jax import random, numpy as jnp
import flax
from flax import linen as nn

# We create one dense layer instance (taking 'features' parameter as input)
model = nn.Dense(features=5)
key1, key2 = random.split(random.PRNGKey(42))
x = random.normal(key1, (10,)) # Dummy input data
params = model.init(key2, x) # Initialization call
jax.tree_util.tree_map(lambda x: x.shape, params) # Checking output shapes
model.apply(params, x)

y=4