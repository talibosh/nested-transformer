
from PIL import Image
import os, sys


path = '/home/tali/mappingPjt/nst/imgs/'
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

def plot_grid(fname):
  data = image.imread(fname)

  # to draw a line from (200,300) to (500,100)

  x2 = [0, 223]
  y2 = [55, 55]
  plt.plot(x2, y2, color="blue", linewidth=3)
  x3 = [55, 55]
  y3 = [0, 223]
  plt.plot(x3, y3, color="blue", linewidth=3)
  x4 = [0, 223]
  y4 = [167, 167]
  plt.plot(x4, y4, color="blue", linewidth=3)
  x5 = [167, 167]
  y5 = [0, 223]
  plt.plot(x5, y5, color="blue", linewidth=3)
  x0 = [111, 111]
  y0 = [0, 223]
  plt.plot(x0, y0, color="magenta", linewidth=3)
  x1 = [0, 223]
  y1 = [111, 111]
  plt.plot(x1, y1, color="magenta", linewidth=3)
  plt.imshow(data)
  plt.show()

fname = '/home/tali/mappingPjt/nst/imgs/n07711569_mashed_potato resized.jpg'
plot_grid(fname)