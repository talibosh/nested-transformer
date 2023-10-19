from models import nest_bricks
import jax
import train
import os
from configs import imagenet_nest
import functools
class GradNesT:
    def __init__(self, model_name, config, checkpoint_dir:os.path, num_classes:int):
        self.config = config
        self.num_layers_per_block = config.num_layers_per_block
        self.num_blocks = len(self.num_layers_per_block)
        self.state_dict = train.checkpoint.load_state_dict(checkpoint_dir)
        self.variables = {
            "params": self.state_dict["optimizer"]["target"],
        }
        self.variables.update(self.state_dict["model_state"])
        dense_layer = nest_bricks.NesTDenseBlock.create_model(model_name, config)
        self.nestDense = functools.partial(dense_layer, num_classes=num_classes)
        aggregation_layer = nest_bricks.NesTAggregateBrick(model_name, config)
        self.nestAggregate = functools.partial(aggregation_layer, num_classes=num_classes)
        transformer_layer = nest_bricks.NesTTransformerBrick(model_name, config)
        self.nestTransformers = functools.partial(transformer_layer, num_classes=num_classes)


    def do_first_layer(self, inputs):
        x = self.nestTransformers(train=False).apply(self.variables, inputs, 0, mutable=False)
        return x

    def aggregate_and_transform(self, x, level):
        x = self.nestAggregate(train=False).apply(self.variables, x, level, mutable=False)
        x = self.nestTransformers(train=False).apply(self.variables, x, level, mutable=False)
        return x

    def do_dense_layer(self, x):
        logits = self.nestDense.apply(x)
        return logits
    def do_bef_grad_level_transformers(self, grad_level, inputs):
        x = self.do_first_layer(inputs)
        for level in range(1, grad_level):
            x = self.aggregate_and_transform(x, level)
        return x # these are the features maps

    def do_post_grad_level_transformers(self, grad_level, x):
        for level in range(grad_level+1, len(self.config.num_layers_per_block)):
            x = self.nestAggregate(train=False).apply(self.variables, x, level, mutable=False)
            x = self.nestTransformers(train=False).apply(self.variables, x, level, mutable=False)
        return x

    def do_post_grad_level(self, grad_level, x):
        x = self.do_post_grad_level_transformers(grad_level, x)
        logits = self.do_dense_layer(x)
        return logits

    def do_grad(self, grad_level, x):
        res = jax.grad(self.do_post_grad_level(self, grad_level, x))
        return res
    def calc_maps_and_grads(self, inputs):
        features_maps = []
        grads = []
        for grad_level in range(1, self.num_blocks):
            x = self.do_bef_grad_level_transformers(grad_level, inputs)
            grad_res = self.do_grad(grad_level, x)
            features_maps.append(x)
            grads.append(grad_res)
        return features_maps, grads







