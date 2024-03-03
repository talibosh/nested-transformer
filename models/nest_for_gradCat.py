import functools
from models import basic_nest_defs
import jax
import flax
from libml import attn_utils
import math
from models import nest_net
import flax.linen as nn
import ml_collections
import train
import numpy as np
class NestForGradCAT():
    def __init__(self, checkpoint_dir: str, config:ml_collections.ConfigDict, num_classes:int):
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        state_dict = train.checkpoint.load_state_dict(self.checkpoint_dir)

        variables = {
            "params": state_dict["optimizer"]["target"],
        }
        variables.update(state_dict["model_state"])
        self.variables = variables
        #full model
        model_cls = nest_net.create_model(self.config.model_name, self.config)
        self.full_model = functools.partial(model_cls, num_classes=num_classes)
        #create the model bricks of the model
        #Patch embedding
        self.config.classname = 'nest_modules.PatchEmbaddingBlock'
        model_cls_patch_embed = basic_nest_defs.create_model(self.config.model_name, self.config)
        self.model_patch_embed = functools.partial(model_cls_patch_embed, num_classes=num_classes)
        #Block (devide the image into sub boxes
        self.config.classname = 'nest_modules.BlockImages'
        model_cls_block_images = basic_nest_defs.create_model(self.config.model_name, self.config)
        self.model_block_images = functools.partial(model_cls_block_images, num_classes=num_classes)
        #positional embedding and encoder
        self.config.classname = 'nest_modules.PosEmbedAndEncodeBlock'
        model_cls_posembed_encode = basic_nest_defs.create_model(self.config.model_name, self.config)
        self.model_posembed_encode = functools.partial(model_cls_posembed_encode, num_classes=num_classes)
        #aggregate results for next hyrarchy
        self.config.classname = 'nest_modules.AggregateBlock'
        model_cls_aggregate = basic_nest_defs.create_model(self.config.model_name, self.config)
        self.model_aggregate = functools.partial(model_cls_aggregate, num_classes=num_classes)
        #dense block
        self.config.classname = 'nest_modules.DenseBlock'
        model_cls_dense = basic_nest_defs.create_model(self.config.model_name, self.config)
        self.model_dense = functools.partial(model_cls_dense, num_classes=num_classes)

        MAX_LEVEL = 3


    def predict(self,image): #image after preprocess
        logits, state = self.full_model(train=False).apply(self.variables, image, mutable=['intermediates'])
        # Return predicted class and confidence.
        return logits.argmax(axis=-1), nn.softmax(logits, axis=-1).max(axis=-1)


    def do_bef_grad_level_transformers_3(self, inputs):
        x, state = self.model_posembed_encode(train=False, level=0).apply(self.variables, inputs, mutable='intermediates')
        grid_size = int(math.sqrt(x.shape[1]))
        x = attn_utils.unblock_images(
            x, grid_size=(grid_size, grid_size), patch_size=(14, 14))

        x, agg_state = self.model_aggregate(train=False, level=0).apply(self.variables, x, mutable='intermediates')
        x, state = self.model_posembed_encode(train=False, level=1).apply(self.variables, x, mutable='intermediates')
        grid_size = int(math.sqrt(x.shape[1]))
        x = attn_utils.unblock_images(
            x, grid_size=(grid_size, grid_size), patch_size=(14, 14))

        x, agg_state = self.model_aggregate(train=False, level=1).apply(self.variables, x, mutable='intermediates')
        x, state = self.model_posembed_encode(train=False, level=2).apply(self.variables, x, mutable='intermediates')

        return x, state, agg_state

    def do_post_grad_level_3(self, inputs):
        x = inputs
        prob = self.model_dense(train=False, level=2).apply(self.variables, x, mutable=False)
        return prob

    def do_bef_grad_level_transformers_2(self, inputs):
        x, state = self.model_posembed_encode(train=False, level=0).apply(self.variables, inputs, mutable='intermediates')
        grid_size = int(math.sqrt(x.shape[1]))
        x = attn_utils.unblock_images(
            x, grid_size=(grid_size, grid_size), patch_size=(14, 14))

        x, agg_state = self.model_aggregate(train=False, level=0).apply(self.variables, x, mutable='intermediates')

        x, state = self.model_posembed_encode(train=False, level=1).apply(self.variables, x, mutable='intermediates')
        grid_size = int(math.sqrt(x.shape[1]))
        x = attn_utils.unblock_images(
            x, grid_size=(grid_size, grid_size), patch_size=(14, 14))

        return x, state, agg_state

    def do_post_grad_level_2(self, inputs):
        x, agg_state = self.model_aggregate(train=False, level=1).apply(self.variables, inputs, mutable='intermediates')
        x, state = self.model_posembed_encode(train=False, level=2).apply(self.variables, x, mutable='intermediates')
        prob = self.model_dense(train=False, level=2).apply(self.variables, x, mutable=False)
        return prob

    def do_bef_grad_level_transformers_1(self, inputs):
        x, state = self.model_posembed_encode(train=False, level=0).apply(self.variables, inputs, mutable='intermediates')
        grid_size = int(math.sqrt(x.shape[1]))
        x = attn_utils.unblock_images(
            x, grid_size=(grid_size, grid_size), patch_size=(14, 14))

        x, agg_state = self.model_aggregate(train=False, level=0).apply(self.variables, x, mutable='intermediates')

        x, state = self.model_posembed_encode(train=False, level=1).apply(self.variables, x, mutable='intermediates')
        grid_size = int(math.sqrt(x.shape[1]))
        x = attn_utils.unblock_images(
            x, grid_size=(grid_size, grid_size), patch_size=(14, 14))

        return x, state, agg_state

    def do_post_grad_level_1(self, inputs):
        x, agg_state = self.model_aggregate(train=False, level=1).apply(self.variables, inputs, mutable='intermediates')
        x, state = self.model_posembed_encode(train=False, level=2).apply(self.variables, x, mutable='intermediates')
        prob = self.model_dense(train=False, level=2).apply(self.variables, x, mutable=False)
        return prob

    def calc_ftr_maps_and_grads_part(self, inputs):
        x, state3, agg_state3 = self.do_bef_grad_level_transformers_3(inputs)
        grad_func3 = jax.grad(self.do_post_grad_level_3)
        grads3 = grad_func3(x)

        x, state2, agg_state2 = self.do_bef_grad_level_transformers_2(inputs)
        grad_func2 = jax.grad(self.do_post_grad_level_2)
        grads2 = grad_func2(x)
        x, state1, agg_state1 = self.do_bef_grad_level_transformers_1(inputs)
        grad_func1 = jax.grad(self.do_post_grad_level_1)
        grads1 = grad_func1(x)
        return grads3, grads2, grads1, state3, state2, state1, agg_state3, agg_state2, agg_state1

    def calc_ftr_maps_and_grads(self, inputs):
        x = self.model_patch_embed(train=False).apply(self.variables, inputs, mutable=False)
        x = self.model_block_images(train=False).apply(self.variables, x, mutable=False)
        grads3, grads2, grads1, state3, state2, state1, agg_state3, agg_state2, agg_state1 = \
            (self.calc_ftr_maps_and_grads_part(x))
        return grads3, grads2, grads1, state3, state2, state1, agg_state3, agg_state2, agg_state1

    def do_grad_cat_level(self, feature_map, grads, grid_size, patch_size, win_part):
        """
        create heatmaps
        the level we work on is defined by grid size ((1,1)- divide to 4 squares, (2,2)-16 squares
        """
        ftrs_shaped = attn_utils.unblock_images(
            feature_map, grid_size=grid_size, patch_size=patch_size)
        # now x is 1,14,14,512
        if grid_size[0] is 1:
            grads_shaped = attn_utils.unblock_images(
                grads, grid_size=grid_size, patch_size=patch_size)
        else:
            grads_shaped = grads

        pooled_grads = grads_shaped.squeeze().mean((0, 1))
        #pooled_grads = grads_shaped.squeeze()
        conv_output = ftrs_shaped.squeeze()

        for i in range(len(pooled_grads)):
            conv_output = conv_output.at[:, :, i].set(conv_output[:, :, i] * pooled_grads[i])

        #conv_output = np.multiply(pooled_grads, conv_output)
        heatmap = conv_output.mean(axis=-1)

        heatmap1 = flax.linen.relu(heatmap)# / heatmap.max()
        #heatmap1 = heatmap / heatmap.max()
        if np.max(heatmap1) == 0:
            heatmap1 = np.zeros(heatmap1.shape)
        else:
            heatmap1 = (heatmap1 - heatmap1.min()) / (heatmap1.max() - heatmap1.min())

        #heatmap3 = heatmap3.reshape(1, heatmap3.shape[0], heatmap3.shape[1], 1)
        #h11_ = flax.linen.avg_pool(heatmap3,
        #                           window_shape=(heatmap3.shape[1] // win_part, heatmap3.shape[2] // win_part),
        #                           strides=(heatmap3.shape[1] // win_part, heatmap3.shape[2] // win_part))
        heatmap1 = heatmap1.reshape(1, heatmap1.shape[0], heatmap1.shape[1], 1)
        heatmap1_squares_avg = flax.linen.avg_pool(heatmap1,
                                   window_shape=(heatmap1.shape[1] // win_part, heatmap1.shape[2] // win_part),
                                   strides=(heatmap1.shape[1] // win_part, heatmap1.shape[2] // win_part))

        hm1 = np.array(heatmap1)
        hm1 = hm1.squeeze()

        return hm1, heatmap1_squares_avg


    def create_heatmaps_and_avg_heatmaps(self, inputs):
        grads3, grads2, grads1, state3, state2, state1, agg_state3, agg_state2, agg_state1 = self.calc_ftr_maps_and_grads(inputs)
        heatmap3, avg_heatmap3 = self.do_grad_cat_level(state3['intermediates']['features_maps'][0], grads3,
                                           grid_size=(1, 1), patch_size=(14, 14), win_part=2)
        heatmap2, avg_heatmap2 = self.do_grad_cat_level(state2['intermediates']['features_maps'][0], grads2,
                                            grid_size=(2, 2), patch_size=(14, 14), win_part=4)
        return heatmap3, avg_heatmap3, heatmap2, avg_heatmap2

