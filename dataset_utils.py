
from PIL import Image
import os, sys
import numpy as np
from matplotlib import pyplot as plt
import keras
import matplotlib as mpl
import pandas as pd

def calc_data_set_mean_and_std(imgs_path:list[str]):
    rgb_values = np.concatenate(
        [Image.open(img).getdata() for img in imgs_path],
        axis=0
    ) / 255.

    # rgb_values.shape == (n, 3),
    # where n is the total number of pixels in all images,
    # and 3 are the 3 channels: R, G, B.

    # Each value is in the interval [0; 1]

    mu_rgb = np.mean(rgb_values, axis=0)  # mu_rgb.shape == (3,)
    std_rgb = np.std(rgb_values, axis=0)  # std_rgb.shape == (3,)

    return mu_rgb, std_rgb


if __name__ == "__main__":
    in_csv_path = '/home/tali/cats_pain_proj/face_images/cats.csv'#'/home/tali/cropped_cats_pain/cats.csv'
    df = pd.read_csv(in_csv_path)
    imgs_list = df["FullPath"].tolist()
    mean, std = calc_data_set_mean_and_std(imgs_list)
    print("mean=" + {mean} + " std="+{std})