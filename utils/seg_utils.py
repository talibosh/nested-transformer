import os
import cv2
import pandas as pd
import numpy as np


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
def get_seg_image(msk_path:str, img_path:str, tmp_sz:tuple[int,int] = (224,224), out_sz:tuple[int, int] = (56, 56)):
    if os.path.isfile(msk_path) == False:
        return np.zeros(1,1)
    mask_img = cv2.imread(msk_path)
    img = cv2.imread(img_path)
    #assert(mask_img.shape == img.shape)
    rszd_msk = cv2.resize(mask_img, out_sz, cv2.INTER_NEAREST)
    rszd_image = cv2.resize(img, out_sz, cv2.INTER_NEAREST)
    thresh = np.median(np.unique(rszd_msk))

    rszd_msk[rszd_msk < thresh] = 0
    rszd_msk[rszd_msk >= thresh] = 1

    seg_image = np.multiply(rszd_msk, rszd_image)
    x1, y1, w, h = cv2.boundingRect(rszd_msk[:,:,0])
    x2 = x1 + w
    y2 = y1 + h
    start = (x1, y1)
    end = (x2, y2)
    colour = (1, 0, 0)
    thickness = -1
    #self.bb_msk = self.np_msk.copy()
    #ff = cv2.rectangle(seg_image, start, end, colour, thickness)
    cut_seg = seg_image[y1:y2, x1:x2]
    #rszd_seg_np_image = cv2.resize(seg_image, out_sz, cv2.INTER_CUBIC)
    cut_seg = cv2.resize(cut_seg, out_sz, cv2.INTER_NEAREST)
    return cut_seg

def create_masked_active_map(im: np.array, msk: np.array,  activation_map:np.array, out_sz: tuple[int, int]):
    img = cv2.resize(im, out_sz, cv2.INTER_NEAREST)
    mask = cv2.resize(msk, out_sz, cv2.INTER_NEAREST)
    thresh = np.median(np.unique(mask))
    np_msk = np.array(mask)
    np_msk[np_msk < thresh] = 0
    np_msk[np_msk >= thresh] = 1

    relu_map = activation_map
    relu_map[relu_map < 0] = 0
    if np.abs(np.sum(relu_map - activation_map )):
        y =44
    #grade_map = cv2.resize(activation_map, out_sz, cv2.INTER_NEAREST)
    factor = int(out_sz[0]/activation_map.shape[0])
    #grade_map = np.kron(activation_map, np.ones((factor,factor), dtype = activation_map.dtype))
    grade_map = cv2.resize(relu_map, out_sz, cv2.INTER_LINEAR)
    relu_map = grade_map
    relu_map[relu_map < 0] = 0
    relu_map_masked = np.multiply(relu_map,np_msk[:,:,0])

    return relu_map_masked


def create_masked_activation_map(msk_path: str, img_path: str, activation_map_path, out_sz: tuple[int, int], alpha:float, out_folder:str):
    im = cv2.imread(img_path)
    msk = cv2.imread(msk_path)
    activation_map = np.load(activation_map_path)
    masked_heatmap = create_masked_active_map(im, msk, activation_map, out_sz)
    return masked_heatmap

def create_heat_map(img:np.array, activation_map:np.array):
    norm_map = (activation_map-np.min(activation_map))/(np.max(activation_map)-np.min(activation_map))
    single_map = 255*norm_map
    single_map = single_map.astype(np.uint8)
    jet_map = cv2.applyColorMap(single_map, cv2.COLORMAP_JET)
    super_imposed_map = img*0.8 + 0.3*jet_map
    super_imposed_map = cv2.resize(super_imposed_map, (224,224), cv2.INTER_LINEAR)
    return norm_map, jet_map, super_imposed_map

def create_combined_maps(activation_map1:np.array, activation_map2:np.array, alpha:float):
    comb1 = activation_map1*alpha + activation_map2*(1-alpha)
    comb2 = activation_map1*activation_map2
    return comb1, comb2

def save_heat_map(jet_map:np.array, out_path: str):
    cv2.imwrite(jet_map, out_path)


def create_heatmaps(msk_path: str, img_path: str, activation3_map_path, activation2_map_path, out_sz: tuple[int, int], alpha:float, out_folder:str):
    im = cv2.imread(img_path)
    if msk_path == "":
        msk = np.ones(im.shape)
    else:
        msk = cv2.imread(msk_path)
    #a = np.load(activation3_map_path)
    activation_map3 = np.load(activation3_map_path)
    #activation_map2 = np.load(activation2_map_path)
    am3 = create_masked_active_map(im, msk, activation_map3, out_sz)
    #am2 = create_masked_active_map(im, msk, activation_map2, out_sz)
    #both, dup =create_combined_maps(am3, am2, alpha)
    img = cv2.resize(im, out_sz, cv2.INTER_NEAREST)
    #norm_mapb, jet_mapb, super_imposed_mapb = create_heat_map(img, both)
    #norm_mapd, jet_mapd, super_imposed_mapd = create_heat_map(img, dup)
    norm_map3, jet_map3, super_imposed_map3 = create_heat_map(img, am3)
    #norm_map2, jet_map2, super_imposed_map2 = create_heat_map(img, am2)
    fname_ = os.path.basename(img_path)

    fname3 = "3_" + fname_
    #fname2 = "2_" + fname_
    #fname_both = "b_" + fname_
    #fname_dup = "d_" + fname_
    #cv2.imwrite( os.path.join(out_folder, fname_both),super_imposed_mapb)
    #cv2.imwrite( os.path.join(out_folder, fname_dup),super_imposed_mapd)
    #cv2.imwrite( os.path.join(out_folder, fname2),super_imposed_map2)
    cv2.imwrite( os.path.join(out_folder, fname3),super_imposed_map3)

if __name__ == "__main__":
    msk_path=""
    csv_path='/home/tali/dogs_annika_proj/cropped_face/total_25_mini_masked.csv'
    res_folder = "/home/tali/dogs_annika_proj/res_25_gc_mini_masked/"
    df=pd.read_csv(csv_path)
    for index, row in df.iterrows():
        id = row["id"]
        #if id!=3:
        #    continue
        im_path = row["fullpath"]
        label = row['label']
        infered = row['Infered_Class']
        if label=="P" and infered==1 or label=="N" and infered==0:
            if label == "P":
                ff = "pos"
            else:
                ff = "neg"
            folder = os.path.join(res_folder, str(row["id"]), str(row["video"]), ff)
            heats_folder = os.path.join(folder, "heats")
            a, file_name = os.path.split(im_path)
            name, extension = os.path.splitext(file_name)
            am3_path = os.path.join(heats_folder,name + "_3.npy")
            am2_path = os.path.join(heats_folder, name + "_2.npy")
            out_folder = os.path.join(folder, "heats_plots")
            msk_path = im_path.replace("masked_images", "masks/face")
            if os.path.exists(msk_path) == False:
                msk_path = ""
            create_heatmaps(msk_path, im_path, am3_path, am2_path, (224, 224), 0.7, out_folder)


#msk_path="/home/tali/cats_pain_proj/face_images/masked_images/masks/no_pain/cat_1_video_1.4.jpg"
#im_path="/home/tali/cats_pain_proj/face_images/masked_images/no_pain/cat_1_video_1.4.jpg"
#activation3_map_path = "/home/tali/trials/cats_finetune_mask_seg_test50_1/1/no pain/heats/cat_1_video_1.4_3.npy"
#activation2_map_path = "/home/tali/trials/cats_finetune_mask_seg_test50_1/1/no pain/heats/cat_1_video_1.4_2.npy"
#create_heatmaps(msk_path, im_path, activation3_map_path, activation2_map_path, (224,224), 0.7, '/home/tali/trials/cats_finetune_mask_seg_test50_1/1/no pain/heats_plots')
