import cv2
from PIL import Image
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


path = '/home/tali/mappingPjt/nst12/imgs/'
dirs = os.listdir( path )

def resize():
  for item in dirs:
    if os.path.isfile(path +item):
      im = Image.open(path +item)
      f, e = os.path.splitext(path +item)
      imResize = im.resize((224 ,224))
      imResize.save(f + ' resized.jpg', 'JPEG', quality=90)

def parse_image(source, square_size):
  src = Image.open(source)
  width, height = src.size
  image_results = []
  for x in range(0, width, square_size):  #
    for y in range(0, height, square_size):
      top_left = (x, y)  # left top of the rect
      bottom_right = (x + square_size, y + square_size)  # right bottom of the rect
      # the current format is used, because it's the cheapest
      # way to explain a rectange, lookup Rects
      test = src.crop((top_left[1], top_left[0], bottom_right[1], bottom_right[0]))
      image_results.append(test)

  return image_results

def run_rec_on_dir(square_size):
  for item in dirs:
    if os.path.isfile(path + item):
      parse_image(path + item, square_size)
#resize()
#run_rec_on_dir(56)

from matplotlib import image
from matplotlib import pyplot as plt
import keras
import matplotlib as mpl

def calc_grade_on_segment(msk_path:str, heatmaps:list[np.array], outsz=(224,224)):
  im = Image.open(msk_path)
  msk = im.resize((outsz[1], outsz[0]), resample  = Image.NEAREST)
  thresh = np.median(np.unique(msk))
  msk1 = np.array(msk)
  msk1[msk1<thresh] = 0
  msk1[msk1>=thresh] = 1
  seg_grades =[]
  rest_of_img_grades = []
  for map in heatmaps:
    rszd_map = np.resize(map, outsz)
    relevant_heat = rszd_map*msk1
    grade_sum = np.sum(relevant_heat)
    grade_normalized = grade_sum*np.sum(msk1)/(outsz[0]*outsz[1])
    seg_grades.append(grade_normalized)
    rest_img_grade = (np.sum(rszd_map)-grade_sum)*(outsz[0]*outsz[1]-np.sum(msk1))/(outsz[0]*outsz[1])
    rest_of_img_grades.append(rest_img_grade)
  return seg_grades, rest_of_img_grades


def plot_heatmap(fname:str, out_path:str = [], show: bool = False, heatmap_level3 = [], heatmap_level2 = [], outsz=(224,224)):
  alpha=0.3
  im = cv2.imread(fname) #Image.open(fname)
  img = cv2.resize(im,outsz, cv2.INTER_CUBIC) #im.resize(outsz)

  fname_ = os.path.basename(fname)

  fname3 = "3" + fname_
  fname2 = "2" + fname_
  fname_both = "b" + fname_

  if heatmap_level3 is not []:
    heatmap_level3=np.array(heatmap_level3)
    heatmap_level3=heatmap_level3.squeeze()
    # Apply a colormap

    hm3 = cv2.resize(heatmap_level3, (224, 224), cv2.INTER_CUBIC)
    hm3_ = cm.jet(hm3)
    superimposed_img3 = hm3_ * alpha + img
    superimposed_img3 = keras.utils.array_to_img(superimposed_img3)

    # Save the superimposed image
    superimposed_img3.save(os.path.join(out_path, fname3))

    #heatmap_level3 = np.uint8(255 * heatmap_level3)
  if heatmap_level2 is not []:
    heatmap_level2 = np.array(heatmap_level2)
    heatmap_level2 = heatmap_level2.squeeze()
   # heatmap_level2 = np.uint8(255 * heatmap_level2)

    hm2 = cv2.resize(heatmap_level2, (224, 224), cv2.INTER_CUBIC)
    hm2_ = cm.jet(heatmap_level2)
    superimposed_img2 = hm2_ * alpha + img
    superimposed_img2 = keras.utils.array_to_img(superimposed_img2)
    superimposed_img2.save(os.path.join(out_path, fname2))

  # Use jet colormap to colorize heatmap
  jet = mpl.colormaps["jet"]


  both = hm2_*0.3 + hm3_*0.7
  superimposed_both = both*alpha + img
  superimposed_both = keras.utils.array_to_img(superimposed_both)
  superimposed_both.save(os.path.join(out_path, fname_both))



def plot_grid(fname: str, out_path: str = [], show: bool = False, grades_level3=[], grades_level2=[]):
  im = Image.open(fname)
  data = im.resize((224,224))

  #data = image.imread(fname)

  # to draw a line from (200,300) to (500,100)
  width, height = data.size[0], data.size[1]

  step_w = int(width/4)
  step_h = int(height/4)

  #draw vertical lines
  h_pos=[0, height-1]
  for c in range(step_w, width, step_w):
    w_pos=[c-1, c-1]
    if c == width/2:
      color = 'magenta'
    else:
      color = 'blue'
    plt.plot(w_pos, h_pos, color=color, linewidth=3)


  #draw horiz lines
  w_pos=[0, width-1]
  for r in range(step_h, height, step_h):
    h_pos=[r-1, r-1]
    if r == height/2:
      color = 'magenta'
    else:
      color = 'blue'
    plt.plot(w_pos, h_pos, color=color, linewidth=3)

  idx=0
  g3 = np.reshape(grades_level3,(1,4))
  max_idx = np.argmax(g3)
  if grades_level3 is not []:
    for h in range(1,4,2):
      for w in range(1,4,2):
        my_grade = f'{g3[0][idx]:.3f}'
        plt.text(w*step_w-1, h*step_h-1, my_grade, fontsize='large', weight="bold")
        idx = idx+1

  idx1 = 0
  if grades_level2 is not []:
    g2 = np.reshape(grades_level2,(4,4))
    for h in range(0, 4):
      if h < 2:
        ver_half = 0
      else:
        ver_half = 2
      for w in range(0, 4):
        if w < 2:
          hor_half = 0
        else:
          hor_half = 1
        qtr = ver_half+hor_half
        my_grade= f'{g2[h][w]:.3f}'
        color='black'
        fontsize='medium'
        if qtr == max_idx:
          color = 'red'
          fontsize = 'xx-large'
        plt.text(int(step_w/2) + w * step_w - 1, int(step_h/2)+h * step_h - 1,my_grade,
                 fontsize=fontsize, color=color)
        idx1 = idx1 + 1
  plt.imshow(data)
  if out_path is not []:
    plt.savefig(out_path)

  if show:
    plt.show()

  plt.close()

def plot_heatmap_on_image(fname: str, out_path: str = [], show: bool = False):
  im = Image.open(fname)
  data = im.resize((224,224))
  plt.imshow(data)
  if out_path is not []:
    plt.savefig(out_path)

  if show:
    plt.show()

  plt.close()



#fname = '/home/tali/mappingPjt/nst12/imgs/n02100877_7560 resized.jpg'
#plot_grid(fname, [10,20,30,40], [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])