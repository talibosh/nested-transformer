import pandas as pd
import os
import glob

from PIL import Image
from pathlib import Path

def create_df(root_path, out_path):
    f_list=[]
    v_list=[]
    fnames_list=[]
    cat_names_list=[]
    path_pain = os.path.join(root_path,'pain')
    path_no_pain = os.path.join(root_path, 'no_pain')
    for paths,dirs,files in os.walk(root_path):
        for d in dirs:
            if d == 'pain':
                valence = 1
            if d == 'no_pain':
                valence = 0
            for f in glob.glob(os.path.join(paths,d,"*.jpg")):
                f_list.append(f)
                v_list.append(valence)
                fname = os.path.split(f)[1]
                fnames_list.append(fname)
                splitted = fname.split('_')
                cat_names_list.append(splitted[1])
                # open image in png format
                #img_png = Image.open(f)

                # The image object is used to save the image in jpg format
                #new_filename = Path(fname).stem + ".jpg"
                #img_png.convert('RGB').save(os.path.join('/home/tali/cats_pain_proj/face_images/jpg/',d,new_filename))
    df = pd.DataFrame({'CatId': cat_names_list,'Filename': fnames_list,'FullPath': f_list, 'Valence':v_list})
    df.to_csv(out_path)

root_path = '/home/tali/cats_pain_proj/seg_images_224/'
out_path = os.path.join(root_path,'cats.csv')
create_df(root_path, out_path)

