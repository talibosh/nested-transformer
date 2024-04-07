import pandas as pd
import tensorflow as tf
import numpy as np
import os
import cv2
from utils import merge_images
from utils import seg_utils
from libml import preprocess

allowed_labels_long = tf.constant([9, 18, 20, 25, 26, 61, 62, 63, 80, 81])
allowed_labels_flat = tf.constant([3, 4, 91, 92, 94, 100, 102, 108])
allowed_labels = tf.constant([3, 4, 91, 92, 94, 100, 102, 108,9, 18, 20, 25, 26, 61, 62, 63, 80, 81],  dtype=tf.int32)
values = tf.constant([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,0, 0, 0, 0, 0],  dtype=tf.int32)

table = tf.lookup.StaticHashTable(
    initializer=tf.lookup.KeyValueTensorInitializer(
        keys=allowed_labels,
        values=values
    ),
    default_value= tf.constant(2,  dtype=tf.int32)
)

def predicate(x, allowed_labels = allowed_labels):
    label = x['label']
    isallowed = tf.equal(allowed_labels, tf.cast(label, allowed_labels.dtype))
    reduced = tf.reduce_sum(tf.cast(isallowed, tf.float32))
    return tf.greater(reduced, tf.constant(0.))

def relabel_map(example):
    new_label = table.lookup(example['label'])
    example['label'] = np.full((1,1), new_label, int)
    return example

def update_label(example, allowed_long=allowed_labels_long, allowed_flat=allowed_labels_flat):
    # label = x['label']
    is_long = tf.equal(allowed_long, tf.cast(example['label'], allowed_long.dtype))
    is_flat = tf.equal(allowed_flat, tf.cast(example['label'], allowed_flat.dtype))
    reduced_long = tf.reduce_sum(tf.cast(is_long, tf.float32))
    reduced_flat = tf.reduce_sum(tf.cast(is_flat, tf.float32))
    if tf.greater(reduced_long, tf.constant(0.)):
        example['label']= np.full((1, 1), 0, dtype=int)
    if tf.greater(reduced_flat, tf.constant(0.)):
        example['label']= np.full((1, 1), 1, dtype=int)
    #else:
    #    example['label']= np.full((1, 1), 2, dtype=int)
    return example

def update_image_to_bb(example):
    image = example["image"]
    #shape = tf.io.extract_jpeg_shape(image)  # get image shape (height, width)
    bndbox = example["objects"]["bbox"]
    #xmin = int(bndbox[0, 0] * float(shape[0]))  # min row
    #ymin = int(bndbox[0, 1] * float(shape[1]))  # min col
    #xmax = int(bndbox[0, 2] * float(shape[0]))  # max row
    #ymax = int(bndbox[0, 3] * float(shape[1]))  # max col
    #image = preprocess.decode_crop_bounding_box_and_resize(image, 224,
    #                                            xmin, ymin, xmax - xmin + 1, ymax - ymin + 1)
    #mean = np.array(preprocess.IMAGENET_DEFAULT_MEAN).reshape(1, 1, 3)
    #std = np.array(preprocess.IMAGENET_DEFAULT_STD).reshape(1, 1, 3)
    #image = (image - mean) / std
    example["image"] = image



def merge_cats_images(imgs_root:str, msks_root:str, df:pd.DataFrame, out_path:str):
    segs_names = ['face', 'ears', 'eyes', 'mouth']
    face_masks_root = os.path.join(msks_root, 'face_images', 'masks')
    ears_masks_root = os.path.join(msks_root, 'ears_images')
    eyes_masks_root = os.path.join(msks_root, 'eyes_images')
    mouth_masks_root = os.path.join(msks_root, 'mouth_images')

    imgs_list = df["FullPath"].tolist()
    for img_full_path in imgs_list:
        ears_msk_path = img_full_path.replace(imgs_root, ears_masks_root)
        face_msk_path = img_full_path.replace(imgs_root, face_masks_root)

        eyes_msk_path = img_full_path.replace(imgs_root, eyes_masks_root)
        mouth_msk_path = img_full_path.replace(imgs_root, mouth_masks_root)
        if os.path.isfile(face_msk_path) is False or os.path.isfile(ears_msk_path) is False or os.path.isfile(eyes_msk_path) is False or os.path.isfile(mouth_msk_path) is False:
            continue
        tmp_sz = (224, 224)
        seg_sz = (224, 224)
        face_seg = seg_utils.get_seg_image(face_msk_path, img_full_path, tmp_sz, seg_sz)
        ears_seg = seg_utils.get_seg_image(ears_msk_path, img_full_path, tmp_sz, seg_sz)
        eyes_seg = seg_utils.get_seg_image(eyes_msk_path, img_full_path, tmp_sz, seg_sz)
        mouth_seg = seg_utils.get_seg_image(mouth_msk_path, img_full_path, tmp_sz, seg_sz)
        seg1 = np.concatenate([face_seg, ears_seg],1)
        seg2 = np.concatenate([eyes_seg, mouth_seg], 1)
        new_data = np.concatenate([seg1, seg2], 0)
        new_data = cv2.resize(new_data, tmp_sz, cv2.INTER_LINEAR)
        save_path = img_full_path.replace(imgs_root, out_path)
        cv2.imwrite(save_path, new_data)

def get_heats_plots_out_path(img_path: str, heats_plot_out_folder:str):
    head, tail = os.path.split(img_path)
    f_splits = tail.split('_')
    id = f_splits[1]
    valence_name = 'pain'
    if img_path.find('no pain') >= 0 or img_path.find('no_pain') >= 0:
        valence_name = 'no pain'

    heat_maps_loc = os.path.join(heats_plot_out_folder, id, valence_name, 'heats_plots')
    # heat_maps_loc = head.replace(self.imgs_root_folder, self.heats_folder)
    #ff = heat_maps_loc.replace('.jpg', '_' + heatmap_name + '.npy')
    return heat_maps_loc

def get_heatmap_for_img(img_path: str, heats_root:str, heatmap_name: str):
    head, tail = os.path.split(img_path)
    f_splits = tail.split('_')
    id = f_splits[1]
    valence_name = 'pain'
    if img_path.find('no pain') >= 0 or img_path.find('no_pain') >= 0:
        valence_name = 'no pain'

    heat_maps_loc = os.path.join(heats_root, id, valence_name, 'heats', tail)
    # heat_maps_loc = head.replace(self.imgs_root_folder, self.heats_folder)
    ff = heat_maps_loc.replace('.jpg', '_' + heatmap_name + '.npy')
    return ff

def create_cats_heats_images(imgs_root:str, msks_root:str, heats_root:str, df:pd.DataFrame, out_path:str):

    imgs_list = df["FullPath"].tolist()
    for img_full_path in imgs_list:

        face_msk_path = img_full_path.replace(imgs_root, msks_root)
        img_full_path = img_full_path.replace('/home/tali/cats_pain_proj/face_images/masked_images', '/home/tali/cats_pain_proj/face_images/masked_images')
        if os.path.isfile(face_msk_path) is False:
            continue
        out_sz = (224, 224)
        am3_path=get_heatmap_for_img(img_full_path,heats_root, '3')
        am2_path=get_heatmap_for_img(img_full_path, heats_root, '2')
        if os.path.isfile(am3_path) is False:
            continue
        out_folder=get_heats_plots_out_path(img_full_path, out_path)
        alpha = 0.7
        seg_utils.create_heatmaps(face_msk_path, img_full_path,am3_path, am2_path, out_sz,alpha, out_folder)


if __name__ == "__main__":

    #df = pd.read_csv("/home/tali/cats_pain_proj/face_images/cats.csv")
    #imgs_root = '/home/tali/cats_pain_proj/face_images'
    #msks_root = '/home/tali/cats_pain_proj'
    #merge_cats_images(imgs_root, msks_root, df, '/home/tali/cats_pain_proj/seg_images_224')

    df = pd.read_csv("/home/tali/cats_pain_proj/face_images/masked_images/cats_masked.csv")
    imgs_root = '/home/tali/cats_pain_proj/face_images/masked_images'
    msks_root = '/home/tali/cats_pain_proj/face_images/masked_images/masks'
    heats_root = '/home/tali/trials/cats_finetune_mask_seg_test50_1/'
    out_path = heats_root
    create_cats_heats_images(imgs_root, msks_root, heats_root, df, out_path)




