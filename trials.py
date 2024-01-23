
from PIL import Image
import os, sys
import numpy as np


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