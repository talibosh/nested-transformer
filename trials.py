
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


resize()