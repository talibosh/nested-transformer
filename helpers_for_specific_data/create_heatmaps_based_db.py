import os
import numpy as np
import pandas as pd
import cv2


def create_merged_img(img, heatmap):
    #assert img.shape == heatmap.shape
    img=cv2.resize(img, (heatmap.shape[0],heatmap.shape[1]))
    hm_3_channel = np.stack((heatmap,) * 3, axis=-1)
    new_img = np.multiply(img, hm_3_channel)
    float_image_normalized = (new_img - new_img.min()) / (new_img.max() - new_img.min())

    # Scale the normalized float image to the range [0, 255]
    scaled_image = (float_image_normalized * 255).astype(np.uint8)
    return scaled_image

def get_heat_map_path(heatmaps_root:str, proj:str, heatmap_type:str, img_full_path:str, id:str, video_name:str, valence:str ):
    filename = os.path.basename(img_full_path)
    if proj == 'dogs':
        valence_pytorch=valence
        if valence == "P":
            valence_nest = 'pos'
        if valence == "N":
            valence_nest = 'neg'
    if proj == 'cats':
        valence_pytorch = valence
        if valence == 'Yes':
            valence_nest = 'pain'
        if valence == 'No':
            valence_nest = 'no_pain'
    if proj == 'horses':
        valence_pytorch = valence
        valence_nest = valence

    heat_fname = filename.replace('.jpg', '_' + heatmap_type + '.npy')
    heatmap_path_nest = os.path.join(heatmaps_root, id, video_name, valence_nest, "heats", heat_fname)

    heat_fname = filename.replace('.jpg', '.npy')
    heatmap_path_pytorch = os.path.join(heatmaps_root, heatmap_type, id, valence_pytorch, video_name, heat_fname)
    if os.path.isfile(heatmap_path_nest):
        return heatmap_path_nest
    if os.path.isfile(heatmap_path_pytorch):
        return heatmap_path_pytorch
    return ''

def get_heatmap(heatmap_path, np_face_msk:np.array, out_shape=(224,224)):
    if heatmap_path=='':
        return np.ones(out_shape)
    hm = np.load(heatmap_path)
    heatmap = cv2.resize(hm, out_shape)
    heatmap[heatmap < 0] = 0
    face_heat = heatmap * np_face_msk
    relevant_heat = heatmap / np.sum(face_heat)
    return relevant_heat

def create_face_msk_from_img(img:np.array, out_shape:tuple):

    face_msk = np.all(img != [0, 0, 0], axis=-1)
    face_msk = cv2.resize(face_msk.astype(np.uint8),out_shape)
    return face_msk

def create_heatmap_based_db(df:pd.DataFrame,orig_ds_root:str, heatmaps_root:str, proj:str, net_type:str,heatmap_type:str, out_root:str):
    new_full_names = []
    for idx, row in df.iterrows():
        img_full_path=row['fullpath']

        id = str(row['id'])
        video_name=''
        if proj=='dogs':
            video_name =str(row['video'])
        valence=row['label']
        hm_path = get_heat_map_path(heatmaps_root, proj, heatmap_type, img_full_path, id, video_name, valence)
        img = cv2.imread(img_full_path)
        out_shape = (224, 224)
        face_msk=create_face_msk_from_img(img, out_shape)
        hm=get_heatmap(hm_path, face_msk, out_shape = out_shape)
        merged_img=create_merged_img(img, hm)
        new_path= img_full_path.replace(orig_ds_root, out_root)
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        cv2.imwrite(new_path,merged_img)
        new_full_names.append(new_path)
    outdf=df
    outdf['fullpath']=new_full_names
    outdf.to_csv(os.path.join(out_root,'dataset.csv'))

if __name__ == "__main__":
    proj='dogs'
    net_type='dino'
    net_type = 'NesT'
    #dogs
    match proj:
        case 'dogs':
            ds_df = '/home/tali/dogs_annika_proj/data_set/dataset_masked.csv'
            if net_type.__contains__('dino'):
                orig_ds_root = '/home/tali/dogs_annika_proj/data_set'
                heatmaps_root ='/home/tali/dogs_annika_proj/pytorch_dino/maps/'
                heatmap_type = 'cam'
                out_root = '/home/tali/dogs_annika_proj/pytorch_dino/cam_based/data_set/'
                df = pd.read_csv(ds_df)
                df = df[df['id'].isin([25,26,27,28,29,30])]
            if net_type.__contains__('NesT'):
                orig_ds_root = '/home/tali/dogs_annika_proj/data_set'
                heatmaps_root ="/home/tali/dogs_annika_proj/res_25_mini_masked_all_maps/"
                heatmap_type = 'grad_cam'
                out_root = '/home/tali/dogs_annika_proj/nest/grad_cam_based/data_set/'
                df = pd.read_csv(ds_df)
                df = df[df['id'].isin([25,26,27,28,29,30])]

    create_heatmap_based_db(df,orig_ds_root, heatmaps_root, proj, net_type, heatmap_type, out_root)

