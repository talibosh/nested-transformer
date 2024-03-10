import cv2
import numpy as np
from matplotlib import pyplot as plt


def combine_2_images(img_arr_1: np.array, img_arr_2: np.array, axis: int):
    return np.concatenate([img_arr_1, img_arr_2], axis)

def combine_images(img_paths: list[str], desired_one_image_shape:tuple[int, int],axis: int):
    first_img = cv2.imread(img_paths[0])
    first_img = cv2.resize(first_img, desired_one_image_shape, cv2.INTER_CUBIC)
    for img_path in img_paths[1:len(img_paths)]:
        img = cv2.imread(img_path)
        img = cv2.resize(img, desired_one_image_shape, cv2.INTER_CUBIC)
        first_img = combine_2_images(first_img, img, axis)
    return first_img

def merge_both_axis_images(horizontal_img_batches: list[list[str]], desired_one_image_shape:tuple[int, int], out_img_sz:tuple[int, int]):
    first_image = combine_images(horizontal_img_batches[0], desired_one_image_shape, 1)
    for list_ in horizontal_img_batches[1: len(horizontal_img_batches)]:
        hor_image1 = combine_images(list_, desired_one_image_shape, 1)
        first_image = combine_2_images(first_image, hor_image1, 0)
    first_image = cv2.resize(first_image, out_img_sz, cv2.INTER_CUBIC)
    return first_image

#img_paths = ['/home/tali/cats_pain_proj/face_images/pain/cat_1_video_3.2.jpg', '/home/tali/cats_pain_proj/ears_images/pain/cat_1_video_3.2.jpg']
#combine_images(img_paths, (224, 224),1)