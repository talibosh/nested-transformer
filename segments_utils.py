import pandas
from PIL import Image
import os, sys
import numpy as np
from matplotlib import pyplot as plt
import keras
import matplotlib as mpl
import pandas as pd
from typing import List, Dict
import cv2


class OneSegOneHeatmapCalc:
    def __init__(self, msk_path: str, num_of_shows: int,face_msk_path:str, outSz: tuple[int, int],manip_type:str):
        if os.path.isfile(msk_path) == False or os.path.isfile(face_msk_path) == False:
            self.orig_msk = []
            return
        self.num_of_shows = num_of_shows
        self.orig_msk = Image.open(msk_path)
        self.orig_face_msk = Image.open(face_msk_path)
        self.outSz = outSz
        self.rszd_msk, self.np_msk = self.rszNconvet2NP(self.orig_msk, outSz)
        self.rszd_face_msk, self.np_face_msk = self.rszNconvet2NP(self.orig_face_msk, outSz)
        self.manipulation_type = manip_type
        # find bb
        x1, y1, w, h = cv2.boundingRect(self.np_msk)
        x2 = x1 + w
        y2 = y1 + h
        start = (x1, y1)
        end = (x2, y2)
        colour = (1, 0, 0)
        thickness = -1
        self.bb_msk = self.np_msk.copy()
        cv2.rectangle(self.bb_msk, start, end, colour, thickness)

        # Draw bounding rectangle
        # start = (x1, y1)
        # end = (x2, y2)
        # colour = (3, 0, 0)
        # thickness = 1
        # rectangle_img = cv2.rectangle(self.np_msk, start, end, colour, thickness)
        # print("x1:", x1, "x2:", x2, "y1:", y1, "y2:", y2)
        # plt.imshow(rectangle_img, cmap="gray")
        # plt.show()

    def rszNconvet2NP(self, orig_msk: Image, out_sz: tuple[int, int]):
        rszd_msk = orig_msk.resize((out_sz[1], out_sz[0]), resample=Image.NEAREST)
        thresh = np.median(np.unique(rszd_msk))
        np_msk = np.array(rszd_msk)
        np_msk[np_msk < thresh] = 0
        np_msk[np_msk >= thresh] = 1
        return rszd_msk, np_msk

    def calc_relevant_heat(self, heatmap: np.array):
        #rszd_heat = cv2.resize(heatmap, dsize=self.outSz, interpolation=cv2.INTER_LINEAR)
        # normalize heatmap - values are [0-1] and sums up to 1
        #rszd_heat = (rszd_heat - rszd_heat.min()) / (rszd_heat.max() - rszd_heat.min())
        heatmap[heatmap<0]=0
        face_heat=heatmap *self.np_face_msk

        if self.manipulation_type == 'power':
            face_heat = np.power(face_heat,2)
        if self.manipulation_type == 'exp':
            face_heat = np.exp(face_heat*2)

        relevant_heat = face_heat * self.np_msk

        relevant_heat = relevant_heat / (np.sum(face_heat)+1e-15)
        relevant_bb_heat = heatmap * self.bb_msk*self.np_face_msk
        relevant_bb_heat = relevant_bb_heat / (np.sum(face_heat)+1e-15)
        return relevant_heat, relevant_bb_heat, heatmap

    def calc_grade_by_seg(self, relevant_heat: np.array, rszd_heat: np.array, msk: np.array):
        prob_grade = np.sum(relevant_heat)
        mean_relevant_heat = np.mean(relevant_heat)
        var_relevant_heat = np.var(relevant_heat)
        tmp=rszd_heat*self.np_face_msk
        rest_img_map = tmp/(np.sum(tmp)+10e-7) - relevant_heat
        mean_rest_of_img = np.mean(rest_img_map)
        var_rest_of_img = np.var(rest_img_map)
        cnr = np.abs(mean_relevant_heat - mean_rest_of_img) / np.sqrt(var_relevant_heat + var_rest_of_img)
        #area = np.sum(msk) / (msk.shape[0] * msk.shape[1])
        #area = np.sum(msk) / np.sum(self.np_face_msk)
        area = np.sum(np.multiply(msk, self.np_face_msk))/ np.sum(self.np_face_msk)
        area_pixels = np.sum(np.multiply(msk, self.np_face_msk))
        return prob_grade, cnr, area, area_pixels

    def calc_grade_sums_by_seg(self, relevant_heat: np.array, rszd_heat: np.array):
        grade_sum = np.sum(relevant_heat)
        grade_ratio = np.sum(self.np_msk) / (self.outSz[0] * self.outSz[1])
        grade_normalized = grade_sum / grade_ratio
        rest_img_grade = (np.sum(rszd_heat) - grade_sum) / (1 - grade_ratio)
        mean_relevant_heat = np.mean(relevant_heat)
        var_relevant_heat = np.var(relevant_heat)
        rest_img_map = rszd_heat - relevant_heat
        mean_rest_of_img = np.mean(rest_img_map)
        var_rest_of_img = np.var(rest_img_map)
        cnr = np.abs(mean_relevant_heat - mean_rest_of_img) / np.sqrt(var_relevant_heat + var_rest_of_img)
        grade_normalized = mean_relevant_heat * (self.outSz[0] * self.outSz[1])
        rest_img_grade = mean_rest_of_img * (self.outSz[0] * self.outSz[1])
        return grade_normalized, rest_img_grade, grade_normalized - rest_img_grade, cnr


class OneImgOneSeg:

    def __init__(self, alpha: float, msk_path: str,face_msk_path:str, img_path: str, heatmap_paths: list[str], max_segs_num: int,
                 out_sz: tuple[int, int], manip_type:str):
        self.msk_path = msk_path
        self.face_msk_path =face_msk_path
        self.img_path = img_path
        self.heatmap_paths = heatmap_paths
        self.out_sz = out_sz
        self.max_segs_num = max_segs_num
        self.alpha = alpha
        self.manip_type = manip_type
        self.save_heats_images_root = heatmap_paths

    def create_both_heatmap(self, hm3: np.array, hm2: np.array, alpha: float):
        assert (hm3.shape == hm2.shape)
        hmb = hm3 * alpha + hm2 * (1 - alpha)
        #hmb = hm3 + hm2 * (1 - alpha)
        #hmb = (hmb - hmb.min()) / (hmb.max() - hmb.min())
        hmb = hmb / np.sum(hmb)
        return hmb

    def create_both_dup_heatmap(self, hm3, hm2):
        assert (hm3.shape == hm2.shape)
        hmb = hm3 * hm2
        #hmb = (hmb - hmb.min()) / (hmb.max() - hmb.min())
        hmb = hmb / np.sum(hmb)
        return hmb

    def overlay_heatmap_img(self, img:np.array, heatmap:np.array):
        norm_map = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap)+1e-15)
        norm_map=cv2.resize(norm_map,[img.shape[1],img.shape[0]],cv2.INTER_CUBIC)
        single_map = 255 * norm_map
        single_map = single_map.astype(np.uint8)
        jet_map = cv2.applyColorMap(single_map, cv2.COLORMAP_JET)
        jet_np_heatmap = np.float32(jet_map) / 255
        #jet_np_heatmap=cv2.resize(jet_np_heatmap,img.shape())
        fimg = np.float32(img)
        fimg=fimg/255
        if np.max(fimg) > 1:
            raise Exception(
                "The input image should np.float32 in the range [0, 1]")

        cam = 0.4 * jet_np_heatmap + 0.7 * fimg
        super_imposed_map = cam / np.max(cam)
        return np.uint8(255 * super_imposed_map)
        #super_imposed_map = img * 0.7 + 0.3 * jet_map
        #super_imposed_map = cv2.resize(super_imposed_map, (224, 224), cv2.INTER_LINEAR)
        #return super_imposed_map

    def get_msk_for_img(self):
        # msk_path = img_path.replace(self.imgs_root_folder, self.msks_folder)
        osh = OneSegOneHeatmapCalc(self.msk_path, self.max_segs_num,self.face_msk_path, self.out_sz, self.manip_type)
        if osh.orig_msk == []:
            return []
        else:
            return osh

    def get_one_heatmap_for_img(self, heatmap_path: str):
        heatmap = np.load(heatmap_path)
        rszd_heatmap = cv2.resize(heatmap, dsize=self.out_sz, interpolation=cv2.INTER_LINEAR)
        # rszd_heatmap = np.resize(heatmap, self.out_sz)
        # normalize resized image
        rszd_heatmap[rszd_heatmap < 0] = 0
        #rszd_heatmap = (rszd_heatmap - rszd_heatmap.min()) / (rszd_heatmap.max() - rszd_heatmap.min())
        #rszd_heatmap = rszd_heatmap / np.sum(rszd_heatmap)
        return rszd_heatmap

    def analyze_img(self):
        osh = self.get_msk_for_img()
        probs = []
        cnrs = []
        areas = []
        areas_pixels = []

        probs_bb = []
        cnrs_bb = []
        areas_bb = []
        areas_pixels_bb = []
        for hmp in self.heatmap_paths:
            if osh == []:
                probs.append(0)
                cnrs.append(0)
                areas.append(0)
                areas_pixels.append(0)
                probs_bb.append(0)
                cnrs_bb.append(0)
                areas_bb.append(0)
                areas_pixels_bb.append(0)

            else:
                hm = self.get_one_heatmap_for_img(hmp)
                relevant_heat, relevant_bb_heat, rszd_heat = osh.calc_relevant_heat(hm)
                if self.msk_path == self.face_msk_path: #face segment
                    img = cv2.imread(self.img_path)
                    img = cv2.resize(img, [224,224])
                    superimposed = self.overlay_heatmap_img( img, relevant_heat)
                    outf = hmp.replace('.npy', '_'+self.manip_type+'.jpg')
                    #superimposed=cv2.cvtColor(superimposed, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(outf, superimposed)

                prob_grade, cnr, area, area_pixels = osh.calc_grade_by_seg(relevant_heat, rszd_heat, osh.np_msk)
                prob_grade_bb, cnr_bb, area_bb, area_pixels_bb = osh.calc_grade_by_seg(relevant_bb_heat, rszd_heat, osh.bb_msk)
                probs.append(prob_grade)
                cnrs.append(cnr)
                areas.append(area)
                areas_pixels.append(area_pixels)
                probs_bb.append(prob_grade_bb)
                cnrs_bb.append(cnr_bb)
                areas_bb.append(area_bb)
                areas_pixels_bb.append(area_pixels_bb)

        return probs, cnrs, areas, areas_pixels, probs_bb, cnrs_bb, areas_bb, areas_pixels_bb

class OneImgAllSegs:
    def __init__(self, alpha: float, img_path: str,
                 segs_data: list[{'seg_name': str, 'instances_num': int, 'msk_path': str, 'heats_list': list[str]},
                            'outSz':tuple[int, int]], manip_type:str):
        self.alpha = alpha
        self.img_path = img_path
        self.segs_data = segs_data
        self.manip_type = manip_type

    def analyze_img(self):
        i = 0
        outs = {}
        for seg_data in self.segs_data:
            outs[i] = {}
            msk_path = seg_data['msk_path']
            face_msk_path = self.segs_data[0]['msk_path']
            heats_list = seg_data['heats_list']
            out_sz = seg_data['outSz']
            max_segs_num = seg_data['instances_num']
            oios = OneImgOneSeg(self.alpha, msk_path, face_msk_path, self.img_path, heats_list, max_segs_num, out_sz, self.manip_type)
            prob_grades, cnrs, areas, areas_pixels, prob_grades_bb, cnrs_bb, areas_bb, areas_pixels_bb = oios.analyze_img()
            outs[i]['full_path'] = self.img_path
            outs[i]['msk_path'] = seg_data['msk_path']
            outs[i]['outSz'] = seg_data['outSz']
            outs[i]['seg_name'] = seg_data['seg_name']
            outs[i]['prob_grades'] = prob_grades
            outs[i]['cnrs'] = cnrs
            outs[i]['areas'] = areas
            outs[i]['areas_pixels'] = areas_pixels
            outs[i]['prob_grades_bb'] = prob_grades_bb
            outs[i]['cnrs_bb'] = cnrs_bb
            outs[i]['areas_bb'] = areas_bb
            outs[i]['areas_pixels_bb'] = areas_pixels_bb
            i = i + 1
        return outs

    def overlay_heatmap_img(self, img:np.array, heatmap:np.array):
        norm_map = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
        single_map = 255 * norm_map
        single_map = single_map.astype(np.uint8)
        jet_map = cv2.applyColorMap(single_map, cv2.COLORMAP_JET)
        super_imposed_map = img * 0.7 + 0.4 * jet_map
        super_imposed_map = cv2.resize(super_imposed_map, (224, 224), cv2.INTER_LINEAR)
        return super_imposed_map


class CatsSegs:
    def __init__(self, alpha: float, df: pd.DataFrame, out_sz: tuple[int, int], res_folder: str, imgs_root: str,
                 msks_root: str, heats_root: str, manip_type):
        self.alpha = alpha
        self.out_sz = out_sz
        self.res_folder = res_folder
        self.df = df
        self.imgs_root = imgs_root
        self.msks_root = msks_root
        self.heats_root = heats_root
        self.segs_names = ['face', 'ears', 'eyes', 'mouth']
        self.face_masks_root = os.path.join(self.msks_root, 'face_images', 'masks')
        self.ears_masks_root = os.path.join(self.msks_root, 'ears_images')
        self.eyes_masks_root = os.path.join(self.msks_root, 'eyes_images')
        self.mouth_masks_root = os.path.join(self.msks_root, 'mouth_images')
        self.manip_type = manip_type

    def get_heatmap_for_img(self, img_path: str, heatmap_name: str):
        head, tail = os.path.split(img_path)
        f_splits = tail.split('_')
        id = f_splits[1]
        valence_name = 'pain'
        if img_path.find('no pain') >= 0 or img_path.find('no_pain') >= 0:
            valence_name = 'no pain'

        heat_maps_loc = os.path.join(self.heats_root, id, valence_name, 'heats', tail)
        # heat_maps_loc = head.replace(self.imgs_root_folder, self.heats_folder)
        ff = heat_maps_loc.replace('.jpg', '_' + heatmap_name + '.npy')
        return ff

    def analyze_one_img(self, img_full_path):
        face_msk_path = img_full_path.replace(self.imgs_root, self.face_masks_root)
        ears_msk_path = img_full_path.replace(self.imgs_root, self.ears_masks_root)
        eyes_msk_path = img_full_path.replace(self.imgs_root, self.eyes_masks_root)
        mouth_msk_path = img_full_path.replace(self.imgs_root, self.mouth_masks_root)
        heat3 = self.get_heatmap_for_img(img_full_path, '3')
        heat2 = self.get_heatmap_for_img(img_full_path, '2')
        heats_list = [heat3, heat2]
        seg_face = {'seg_name': 'face', 'instances_num': 1, 'msk_path': face_msk_path, 'heats_list': heats_list,
                    'outSz': (224, 224)}
        seg_ears = {'seg_name': 'ears', 'instances_num': 2, 'msk_path': ears_msk_path, 'heats_list': heats_list,
                    'outSz': (224, 224)}
        seg_eyes = {'seg_name': 'eyes', 'instances_num': 2, 'msk_path': eyes_msk_path, 'heats_list': heats_list,
                    'outSz': (224, 224)}
        seg_mouth = {'seg_name': 'mouth', 'instances_num': 1, 'msk_path': mouth_msk_path, 'heats_list': heats_list,
                     'outSz': (224, 224)}
        segs_data = [seg_face, seg_ears, seg_eyes, seg_mouth]
        oias = OneImgAllSegs(self.alpha, img_full_path, segs_data, self.manip_type)
        outs = oias.analyze_img()
        return outs

    def analyze_img_lists(self, imgs_paths: list[str]):
        all_outs = {}
        i = 0
        for img_full_path in imgs_paths:
            outs = self.analyze_one_img(img_full_path)
            all_outs[i] = outs
            i = i + 1
        return all_outs

    def analyze_all(self):
        cls = 1  # pain
        #eval = self.df[self.df["Valence"] == cls]
        #eval = eval[eval["Infered_Class"] == cls]
        eval = self.df[self.df["Valence"] == self.df["Infered_Class"]]
        #eval = eval[eval["Valence"] == cls]
        eval = eval[eval["Prob"] > 0.5]
        imgs_paths = eval['FullPath'].tolist()
        all_outs = self.analyze_img_lists(imgs_paths)
        return all_outs

    def create_res_df(self, all_outs, out_csv):
        id = []
        img_name = []
        full_path = []
        valence = []
        face_area = []
        face_area_bb = []
        face_prob3 = []
        face_cnr3 = []
        face_ng3 = []
        face_prob2 = []
        face_cnr2 = []
        face_ng2 = []
        face_probb = []
        face_cnrb = []
        face_ngb = []
        face_probd = []
        face_cnrd = []
        face_ngd = []
        face_prob3_bb = []
        face_cnr3_bb = []
        face_ng3_bb = []
        face_prob2_bb = []
        face_cnr2_bb = []
        face_ng2_bb = []
        face_probb_bb = []
        face_cnrb_bb = []
        face_ngb_bb = []
        face_probd_bb = []
        face_cnrd_bb = []
        face_ngd_bb = []
        ears_area = []
        ears_area_bb = []
        ears_prob3 = []
        ears_cnr3 = []
        ears_ng3 = []
        ears_prob2 = []
        ears_cnr2 = []
        ears_ng2 = []
        ears_probb = []
        ears_cnrb = []
        ears_ngb = []
        ears_probd = []
        ears_cnrd = []
        ears_ngd = []
        ears_prob3_bb = []
        ears_cnr3_bb = []
        ears_ng3_bb = []
        ears_prob2_bb = []
        ears_cnr2_bb = []
        ears_ng2_bb = []
        ears_probb_bb = []
        ears_cnrb_bb = []
        ears_ngb_bb = []
        ears_probd_bb = []
        ears_cnrd_bb = []
        ears_ngd_bb = []
        eyes_area = []
        eyes_area_bb = []
        eyes_prob3 = []
        eyes_cnr3 = []
        eyes_ng3 = []
        eyes_prob2 = []
        eyes_cnr2 = []
        eyes_ng2 = []
        eyes_probb = []
        eyes_cnrb = []
        eyes_ngb = []
        eyes_probd = []
        eyes_cnrd = []
        eyes_ngd = []
        eyes_prob3_bb = []
        eyes_cnr3_bb = []
        eyes_ng3_bb = []
        eyes_prob2_bb = []
        eyes_cnr2_bb = []
        eyes_ng2_bb = []
        eyes_probb_bb = []
        eyes_cnrb_bb = []
        eyes_ngb_bb = []
        eyes_probd_bb = []
        eyes_cnrd_bb = []
        eyes_ngd_bb = []
        mouth_area = []
        mouth_area_bb = []
        mouth_prob3 = []
        mouth_cnr3 = []
        mouth_ng3 = []
        mouth_prob2 = []
        mouth_cnr2 = []
        mouth_ng2 = []
        mouth_probb = []
        mouth_cnrb = []
        mouth_ngb = []
        mouth_probd = []
        mouth_cnrd = []
        mouth_ngd = []
        mouth_prob3_bb = []
        mouth_cnr3_bb = []
        mouth_ng3_bb = []
        mouth_prob2_bb = []
        mouth_cnr2_bb = []
        mouth_ng2_bb = []
        mouth_probb_bb = []
        mouth_cnrb_bb = []
        mouth_ngb_bb = []
        mouth_probd_bb = []
        mouth_cnrd_bb = []
        mouth_ngd_bb = []
        # id, img name, img full path, valence, face_area, face_prob, face_cnr, ears_area, ears_prob, ears_cnr, eyes_area, eyes_prob, eyes_cnr, mouth_area, mouth_prob, mouth_cnr
        for i in range(all_outs.__len__()):
            img_data = all_outs[i]
            face_data = img_data[0]
            ears_data = img_data[1]
            eyes_data = img_data[2]
            mouth_data = img_data[3]


            head, tail = os.path.split(face_data['full_path'])
            f_splits = tail.split('_')
            id_str = f_splits[1]
            valence_name = 1 #'pain'
            if face_data['full_path'].find('no pain') >= 0 or face_data['full_path'].find('no_pain') >= 0:
                valence_name = 0 #no pain


            if face_data['areas'][0] == 0 or ears_data['areas'][0]==0 or eyes_data['areas'][0]==0 or mouth_data['areas'][0]==0:
                continue
            full_path.append(face_data['full_path'])
            img_name.append(tail)
            id.append(int(id_str))
            valence.append(valence_name)


            face_area.append(face_data['areas'][0])
            face_prob3.append(face_data['prob_grades'][0])
            face_cnr3.append(face_data['cnrs'][0])
            face_ng3.append(face_data['prob_grades'][0] / face_data['areas'][0])
            face_prob2.append(face_data['prob_grades'][1])
            face_cnr2.append(face_data['cnrs'][1])
            face_ng2.append(face_data['prob_grades'][1] / face_data['areas'][0])
            face_probb.append(face_data['prob_grades'][2])
            face_cnrb.append(face_data['cnrs'][2])
            face_ngb.append(face_data['prob_grades'][2] / face_data['areas'][0])
            face_probd.append(face_data['prob_grades'][3])
            face_cnrd.append(face_data['cnrs'][3])
            face_ngd.append(face_data['prob_grades'][3] / face_data['areas'][0])
            face_area_bb.append(face_data['areas_bb'][0])
            face_prob3_bb.append(face_data['prob_grades_bb'][0])
            face_cnr3_bb.append(face_data['cnrs_bb'][0])
            face_ng3_bb.append(face_data['prob_grades_bb'][0]/face_data['areas'][0])
            face_prob2_bb.append(face_data['prob_grades_bb'][1])
            face_cnr2_bb.append(face_data['cnrs_bb'][1])
            face_ng2_bb.append(face_data['prob_grades_bb'][1] / face_data['areas'][0])
            face_probb_bb.append(face_data['prob_grades_bb'][2])
            face_cnrb_bb.append(face_data['cnrs_bb'][2])
            face_ngb_bb.append(face_data['prob_grades_bb'][2] / face_data['areas'][0])
            face_probd_bb.append(face_data['prob_grades_bb'][3])
            face_cnrd_bb.append(face_data['cnrs'][3])
            face_ngd_bb.append(face_data['prob_grades_bb'][3] // face_data['areas'][0])

            ears_area.append(ears_data['areas'][0])
            ears_prob3.append(ears_data['prob_grades'][0])
            ears_cnr3.append(ears_data['cnrs'][0])
            ears_ng3.append(ears_data['prob_grades'][0]/ears_data['areas'][0])
            ears_prob2.append(ears_data['prob_grades'][1])
            ears_cnr2.append(ears_data['cnrs'][1])
            ears_ng2.append(ears_data['prob_grades'][1] / ears_data['areas'][0])
            ears_probb.append(ears_data['prob_grades'][2])
            ears_cnrb.append(ears_data['cnrs'][2])
            ears_ngb.append(ears_data['prob_grades'][2] / ears_data['areas'][0])
            ears_probd.append(ears_data['prob_grades'][3])
            ears_cnrd.append(ears_data['cnrs'][3])
            ears_ngd.append(ears_data['prob_grades'][3] / ears_data['areas'][0])
            ears_area_bb.append(ears_data['areas_bb'][0])
            ears_prob3_bb.append(ears_data['prob_grades_bb'][0])
            ears_cnr3_bb.append(ears_data['cnrs_bb'][0])
            ears_ng3_bb.append(ears_data['prob_grades_bb'][0] / ears_data['areas_bb'][0])
            ears_prob2_bb.append(ears_data['prob_grades_bb'][1])
            ears_cnr2_bb.append(ears_data['cnrs_bb'][1])
            ears_ng2_bb.append(ears_data['prob_grades_bb'][1] / ears_data['areas_bb'][0])
            ears_probb_bb.append(ears_data['prob_grades_bb'][2])
            ears_cnrb_bb.append(ears_data['cnrs_bb'][2])
            ears_ngb_bb.append(ears_data['prob_grades_bb'][2] / ears_data['areas_bb'][0])
            ears_probd_bb.append(ears_data['prob_grades_bb'][3])
            ears_cnrd_bb.append(ears_data['cnrs_bb'][3])
            ears_ngd_bb.append(ears_data['prob_grades_bb'][3] / ears_data['areas_bb'][0])

            eyes_area.append(eyes_data['areas'][0])
            eyes_prob3.append(eyes_data['prob_grades'][0])
            eyes_cnr3.append(eyes_data['cnrs'][0])
            eyes_ng3.append(eyes_data['prob_grades'][0]/eyes_data['areas'][0])
            eyes_prob2.append(eyes_data['prob_grades'][1])
            eyes_cnr2.append(eyes_data['cnrs'][1])
            eyes_ng2.append(eyes_data['prob_grades'][1] / eyes_data['areas'][0])
            eyes_probb.append(eyes_data['prob_grades'][2])
            eyes_cnrb.append(eyes_data['cnrs'][2])
            eyes_ngb.append(eyes_data['prob_grades'][2] / eyes_data['areas'][0])
            eyes_probd.append(eyes_data['prob_grades'][3])
            eyes_cnrd.append(eyes_data['cnrs'][3])
            eyes_ngd.append(eyes_data['prob_grades'][3] / eyes_data['areas'][0])

            eyes_area_bb.append(eyes_data['areas_bb'][0])
            eyes_prob3_bb.append(eyes_data['prob_grades_bb'][0])
            eyes_cnr3_bb.append(eyes_data['cnrs_bb'][0])
            eyes_ng3_bb.append(eyes_data['prob_grades_bb'][0] / eyes_data['areas_bb'][0])
            eyes_prob2_bb.append(eyes_data['prob_grades_bb'][1])
            eyes_cnr2_bb.append(eyes_data['cnrs_bb'][1])
            eyes_ng2_bb.append(eyes_data['prob_grades_bb'][1] / eyes_data['areas_bb'][0])
            eyes_probb_bb.append(eyes_data['prob_grades_bb'][2])
            eyes_cnrb_bb.append(eyes_data['cnrs_bb'][2])
            eyes_ngb_bb.append(eyes_data['prob_grades_bb'][2] / eyes_data['areas_bb'][0])
            eyes_probd_bb.append(eyes_data['prob_grades_bb'][3])
            eyes_cnrd_bb.append(eyes_data['cnrs_bb'][3])
            eyes_ngd_bb.append(eyes_data['prob_grades_bb'][3] / eyes_data['areas_bb'][0])

            mouth_area.append(mouth_data['areas'][0])
            mouth_prob3.append(mouth_data['prob_grades'][0])
            mouth_cnr3.append(mouth_data['cnrs'][0])
            if mouth_data['areas'][0] == 0:
                mouth_ng3.append(0)
            else:
                mouth_ng3.append(mouth_data['prob_grades'][0] / mouth_data['areas'][0])
            mouth_prob2.append(mouth_data['prob_grades'][1])
            mouth_cnr2.append(mouth_data['cnrs'][1])
            if mouth_data['areas'][0] == 0:
                mouth_ng2.append(0)
            else:
                mouth_ng2.append(mouth_data['prob_grades'][1] / mouth_data['areas'][0])
            mouth_probb.append(mouth_data['prob_grades'][2])
            mouth_cnrb.append(mouth_data['cnrs'][2])
            if mouth_data['areas'][0] == 0:
                mouth_ngb.append(0)
            else:
                mouth_ngb.append(mouth_data['prob_grades'][2] / mouth_data['areas'][0])
            mouth_probd.append(mouth_data['prob_grades'][3])
            mouth_cnrd.append(mouth_data['cnrs'][3])
            if mouth_data['areas'][0] == 0:
                mouth_ngd.append(0)
            else:
                mouth_ngd.append(mouth_data['prob_grades'][3] / mouth_data['areas'][0])

            mouth_area_bb.append(mouth_data['areas_bb'][0])
            mouth_prob3_bb.append(mouth_data['prob_grades_bb'][0])
            mouth_cnr3_bb.append(mouth_data['cnrs_bb'][0])
            if mouth_data['areas_bb'][0] == 0:
                mouth_ng3_bb.append(0)
            else:
                mouth_ng3_bb.append(mouth_data['prob_grades_bb'][0] / mouth_data['areas_bb'][0])
            mouth_prob2_bb.append(mouth_data['prob_grades_bb'][1])
            mouth_cnr2_bb.append(mouth_data['cnrs_bb'][1])
            if mouth_data['areas_bb'][0] == 0:
                mouth_ng2_bb.append(0)
            else:
                mouth_ng2_bb.append(mouth_data['prob_grades_bb'][1] / mouth_data['areas_bb'][0])
            mouth_probb_bb.append(mouth_data['prob_grades_bb'][2])
            mouth_cnrb_bb.append(mouth_data['cnrs_bb'][2])
            if mouth_data['areas_bb'][0] == 0:
                mouth_ngb_bb.append(0)
            else:
                mouth_ngb_bb.append(mouth_data['prob_grades_bb'][2] / mouth_data['areas_bb'][0])
            mouth_probd_bb.append(mouth_data['prob_grades_bb'][3])
            mouth_cnrd_bb.append(mouth_data['cnrs_bb'][3])
            if mouth_data['areas_bb'][0] == 0:
                mouth_ngd_bb.append(0)
            else:
                mouth_ngd_bb.append(mouth_data['prob_grades_bb'][3] / mouth_data['areas_bb'][0])

        df = pd.DataFrame({'Id': id, 'Filename': img_name,
                           'FullPath': full_path, 'Valence': valence,
                           'face_area': face_area,
                           'face_prob3': face_prob3, 'face_prob2': face_prob2,
                           'face_probb': face_probb, 'face_probd': face_probd,
                           'face_cnr3': face_cnr3, 'face_cnr2': face_cnr2,
                           'face_cnrb': face_cnrb, 'face_cnrd': face_cnrd,
                           'face_ng3': face_ng3, 'face_ng2': face_ng2,
                           'face_ngb': face_ngb, 'face_ngd': face_ngd,

                           'face_area_bb': face_area_bb,
                           'face_prob3_bb': face_prob3_bb, 'face_prob2_bb': face_prob2_bb,
                           'face_probb_bb': face_probb_bb, 'face_probd_bb': face_probd_bb,
                           'face_cnr3_bb': face_cnr3_bb, 'face_cnr2_bb': face_cnr2_bb,
                           'face_cnrb_bb': face_cnrb_bb, 'face_cnrd_bb': face_cnrd_bb,
                           'face_ng3_bb': face_ng3_bb, 'face_ng2_bb': face_ng2_bb,
                           'face_ngb_bb': face_ngb_bb, 'face_ngd_bb': face_ngd_bb,

                           'ears_area': ears_area,
                           'ears_prob3': ears_prob3, 'ears_prob2': ears_prob2,
                           'ears_probb': ears_probb, 'ears_probd': ears_probd,
                           'ears_cnr3': ears_cnr3, 'ears_cnr2': ears_cnr2,
                           'ears_cnrb': ears_cnrb, 'ears_cnrd': ears_cnrd,
                           'ears_ng3': ears_ng3, 'ears_ng2': ears_ng2,
                           'ears_ngb': ears_ngb, 'ears_ngd': ears_ngd,

                           'ears_area_bb': ears_area_bb,
                           'ears_prob3_bb': ears_prob3_bb, 'ears_prob2_bb': ears_prob2_bb,
                           'ears_probb_bb': ears_probb_bb, 'ears_probd_bb': ears_probd_bb,
                           'ears_cnr3_bb': ears_cnr3_bb, 'ears_cnr2_bb': ears_cnr2_bb,
                           'ears_cnrb_bb': ears_cnrb_bb, 'ears_cnrd_bb': ears_cnrd_bb,
                           'ears_ng3_bb': ears_ng3_bb, 'ears_ng2_bb': ears_ng2_bb,
                           'ears_ngb_bb': ears_ngb_bb, 'ears_ngd_bb': ears_ngd_bb,

                           'eyes_area': eyes_area,
                           'eyes_prob3': eyes_prob3, 'eyes_prob2': eyes_prob2,
                           'eyes_probb': eyes_probb, 'eyes_probd': eyes_probd,
                           'eyes_cnr3': eyes_cnr3, 'eyes_cnr2': eyes_cnr2,
                           'eyes_cnrb': eyes_cnrb, 'eyes_cnrd': eyes_cnrd,
                           'eyes_ng3': eyes_ng3, 'eyes_ng2': eyes_ng2,
                           'eyes_ngb': eyes_ngb, 'eyes_ngd': eyes_ngd,

                           'eyes_area_bb': eyes_area_bb,
                           'eyes_prob3_bb': eyes_prob3_bb, 'eyes_prob2_bb': eyes_prob2_bb,
                           'eyes_probb_bb': eyes_probb_bb, 'eyes_probd_bb': eyes_probd_bb,
                           'eyes_cnr3_bb': eyes_cnr3_bb, 'eyes_cnr2_bb': eyes_cnr2_bb,
                           'eyes_cnrb_bb': eyes_cnrb_bb, 'eyes_cnrd_bb': eyes_cnrd_bb,
                           'eyes_ng3_bb': eyes_ng3_bb, 'eyes_ng2_bb': eyes_ng2_bb,
                           'eyes_ngb_bb': eyes_ngb_bb, 'eyes_ngd_bb': eyes_ngd_bb,

                           'mouth_area': mouth_area,
                           'mouth_prob3': mouth_prob3, 'mouth_prob2': mouth_prob2,
                           'mouth_probb': mouth_probb, 'mouth_probd': mouth_probd,
                           'mouth_cnr3': mouth_cnr3, 'mouth_cnr2': mouth_cnr2,
                           'mouth_cnrb': mouth_cnrb, 'mouth_cnrd': mouth_cnrd,
                           'mouth_ng3': mouth_ng3, 'mouth_ng2': mouth_ng2,
                           'mouth_ngb': mouth_ngb, 'mouth_ngd': mouth_ngd,

                           'mouth_area_bb': mouth_area_bb,
                           'mouth_prob3_bb': mouth_prob3_bb, 'mouth_prob2_bb': mouth_prob2_bb,
                           'mouth_probb_bb': mouth_probb_bb, 'mouth_probd_bb': mouth_probd_bb,
                           'mouth_cnr3_bb': mouth_cnr3_bb, 'mouth_cnr2_bb': mouth_cnr2_bb,
                           'mouth_cnrb_bb': mouth_cnrb_bb, 'mouth_cnrd_bb': mouth_cnrd_bb,
                           'mouth_ng3_bb': mouth_ng3_bb, 'mouth_ng2_bb': mouth_ng2_bb,
                           'mouth_ngb_bb': mouth_ngb_bb, 'mouth_ngd_bb': mouth_ngd_bb
                           })
        df.to_csv(out_csv)

    def calc_mean_std_relevant(self, data: np.array, relevant_locs: list[int]):
        mean_ = np.mean(data[relevant_locs])
        std_ = np.std(data[relevant_locs])
        return mean_, std_

    def calc_normalized_grade(self, orig_grades: np.array, orig_areas: np.array, relevant_locs: list[int]):
        res = np.divide(orig_grades[relevant_locs], orig_areas[relevant_locs])
        return res

    def analyze_of_seg(self, relevant_locs: list[int], areas: np.array, data_lists: list[np.array]):
        means = []
        stds = []
        means_norm_grades = []
        norm_grades_stds = []
        for dt in data_lists:
            mean_dt, std_dt = self.calc_mean_std_relevant(dt, relevant_locs)
            means.append(mean_dt)
            stds.append(std_dt)
            norm_grades_dt = self.calc_normalized_grade(dt, areas, relevant_locs)
            mean_norm_grades, std_norm_grades = self.calc_mean_std_relevant(norm_grades_dt,
                                                                            np.nonzero(norm_grades_dt)[0].tolist())
            means_norm_grades.append(mean_norm_grades)
            norm_grades_stds.append(std_norm_grades)

        return means, stds, means_norm_grades, norm_grades_stds

    def create_lists(self,df:pd.DataFrame, seg_name:str, addition:str = '' ):
        seg_area = np.array(df[seg_name+'_area'+addition].tolist())
        seg_locs = np.nonzero(seg_area)[0].tolist()
        seg_list = []
        seg_list.append(np.array(df[seg_name+'_prob3'+addition].tolist()))  # face_prob3
        seg_list.append(np.array(df[seg_name+'_prob2'+addition].tolist()))
        seg_list.append(np.array(df[seg_name+'_probb'+addition].tolist()))
        seg_list.append(np.array(df[seg_name+'_probd'+addition].tolist()))
        seg_list.append(np.array(df[seg_name+'_cnr3'+addition].tolist()))
        seg_list.append(np.array(df[seg_name+'_cnr2'+addition].tolist()))
        seg_list.append(np.array(df[seg_name+'_cnrb'+addition].tolist()))
        seg_list.append(np.array(df[seg_name+'_cnrd'+addition].tolist()))
        seg_means, seg_stds, seg_norm_grades, seg_norm_grades_stds = self.analyze_of_seg(seg_locs, seg_area, seg_list)
        seg_dict = {seg_name+addition+'_p3':[seg_means[0]], seg_name+addition+'_p2':[seg_means[1]],
                    seg_name+addition+'_pb':[seg_means[2]],seg_name+addition+'_pd':[seg_means[3]],
                    seg_name + addition + '_cnr3': [seg_means[4]], seg_name + addition + '_cnr2': [seg_means[5]],
                    seg_name + addition + '_cnrb': [seg_means[6]], seg_name + addition + '_cnrd': [seg_means[7]],
                    seg_name+addition+'_ng3':[seg_norm_grades[0]], seg_name+addition+'_ng2':[seg_norm_grades[1]],
                    seg_name+addition+'_ngb':[seg_norm_grades[2]], seg_name+addition+'_ngd':[seg_norm_grades[3]]
        }
        return pd.DataFrame.from_dict(seg_dict)
    def analyze_df(self, df: pd.DataFrame):

      new_df = pd.DataFrame()
      img_name = np.array(df['Filename'].tolist())
      full_path = np.array(df['FullPath'].tolist())

      face_dict = self.create_lists(df,'face','' )
      face_dict_bb = self.create_lists(df, 'face', '_bb')
      ears_dict = self.create_lists(df, 'ears', '')
      ears_dict_bb = self.create_lists(df, 'ears', '_bb')
      eyes_dict = self.create_lists(df, 'eyes', '')
      eyes_dict_bb = self.create_lists(df, 'eyes', '_bb')
      mouth_dict = self.create_lists(df, 'mouth', '')
      mouth_dict_bb = self.create_lists(df, 'mouth', '_bb')
      new_df = pd.concat([new_df, face_dict, face_dict_bb, ears_dict, ears_dict_bb, eyes_dict,eyes_dict_bb,mouth_dict,mouth_dict_bb], axis=1)
      #new_df = pd.concat([new_df,  ears_dict, ears_dict_bb, eyes_dict, eyes_dict_bb, mouth_dict,mouth_dict_bb], axis=1)

      return new_df

    def statistics_all(self, df: pd.DataFrame):
        new_df = pd.DataFrame()
        eval_df = df
        ret_df = self.analyze_df(eval_df)
        ret_df.insert(loc=0, column='Id', value=id)
        new_df = pd.concat([new_df, ret_df], axis=0)
        return new_df

    def statistics_by_id(self, df: pd.DataFrame):
        ids = df['Id'].tolist()
        unique_ids = np.unique(ids)
        new_df = pd.DataFrame()
        for id in unique_ids:
            print('***************start ' + str(id) + ' *************************\n')
            eval_df = df[df["Id"] == id]
            ret_df = self.analyze_df(eval_df)
            ret_df.insert(loc=0, column='Id', value=id)
            new_df = pd.concat([new_df, ret_df], axis=0)
        return new_df


def plot_msk_on_img(img_pth, msk_pth):
    im = cv2.imread(img_pth)
    msk = cv2.imread(msk_pth)
    # assert(im.shape == msk.shape)
    im1 = cv2.resize(im, dsize=(32, 32), interpolation=cv2.INTER_LINEAR)
    msk1 = cv2.resize(msk, dsize=(32, 32), interpolation=cv2.INTER_NEAREST)
    msked_img = im1 * msk1
    plt.matshow(msked_img)
    plt.show()

def ststistics_for_type(arr : np.array, thresh:float):
    cnts = arr[arr > thresh]
    avg = np.mean(cnts)
    avg_all = np.mean(arr)
    return len(cnts) / len(arr), avg, avg_all
def statistics_thresh_importance(df: pd.DataFrame):
    #names = ['ears_ng3', 'ears_ng3_bb', 'ears_ngb', 'ears_ngb_bb', 'ears_ngd', 'ears_ngd_bb',
    #         'eyes_ng3', 'eyes_ng3_bb', 'eyes_ngb', 'eyes_ngb_bb', 'eyes_ngd', 'eyes_ngd_bb',
    #         'mouth_ng3', 'mouth_ng3_bb', 'mouth_ngb', 'mouth_ngb_bb', 'mouth_ngd', 'mouth_ngd_bb']

    names = ['ears_ng3',  'ears_ngb', 'ears_ngd',
             'eyes_ng3',  'eyes_ngb', 'eyes_ngd',
             'mouth_ng3', 'mouth_ngb', 'mouth_ngd']
    res = []
    avgs = []
    avgs_all = []
    for n in names:
        arr = np.array(df[n].tolist())
        r, a, aa =ststistics_for_type(arr, 1.0)
        res.append(r)
        avgs.append(a)
        avgs_all.append(aa)
    return res, avgs, avgs_all

def ananlyze_importance_per_image_per_map_type(ears_norm_grade:float, eyes_norm_grade:float, mouth_norm_grade:float):
    grades_arr = np.array([ears_norm_grade, eyes_norm_grade, mouth_norm_grade])
    sorted_ids = np.argsort(grades_arr)  # ascending
    is_important = np.zeros(3, dtype = bool) # ears, eyes, mouth
    relevance = np.zeros(3) # ears, eyes, mouth - 2 most relevant 0 least relevant

    if ears_norm_grade > 1.:
        is_important[0] = True
    if eyes_norm_grade > 1.:
        is_important[1] = True
    if mouth_norm_grade > 1.:
        is_important[2] = True

    loc = np.where(sorted_ids == 0)  # find ears grade loc
    relevance[0] = loc[0]

    loc = np.where(sorted_ids == 1)  # find eyes grade loc
    relevance[1] = loc[0]

    loc = np.where(sorted_ids == 2)  # find mouth grade loc
    relevance[2] = loc[0]
    return relevance, is_important

#statistics of how many image where supported by ears, eyes, mouth and combinations
def analyze_importance_of_segments(df: pd.DataFrame, map_type: str):
    names = ['ears_ng3', 'ears_ngb', 'ears_ngd',
             'eyes_ng3', 'eyes_ngb', 'eyes_ngd',
             'mouth_ng3', 'mouth_ngb', 'mouth_ngd']

    ears_ng = np.array(df['ears_ng'+ map_type].tolist())
    eyes_ng = np.array(df['eyes_ng'+ map_type].tolist())
    mouth_ng = np.array(df['mouth_ng'+ map_type].tolist())

    ears = 0
    eyes = 0
    mouth = 0
    ears_eyes = 0
    ears_mouth = 0
    eyes_mouth = 0
    ears_eyes_mouth = 0
    none_important = 0
    ears_avg = [0, 0, 0, 0] # ears, ears_eyes, ears_mouth, ears_eyes_mouth
    eyes_avg = [0, 0, 0, 0] # eyes, ears_eyes, eyes_mouth, ears_eyes_mouth
    mouth_avg = [0, 0, 0, 0] # mouth, ears_mouth, eyes_mouth, ears_eyes_mouth

    for i in range(ears_ng.__len__()):
        relevance, is_important = ananlyze_importance_per_image_per_map_type(ears_ng[i], eyes_ng[i], mouth_ng[i])
        if np.sum(is_important) == 3: # all grades > 1
            ears_eyes_mouth = ears_eyes_mouth + 1
            ears_avg[3] = ears_avg[3] + ears_ng[i]
            eyes_avg[3] = eyes_avg[3] + eyes_ng[i]
            mouth_avg[3] = mouth_avg[3] + mouth_ng[i]
        if np.sum(is_important) == 0:
            none_important = none_important + 1
        if np.sum(is_important) == 1: # single segment grade > 1
            if is_important[0] == 1:
                ears = ears + 1
                ears_avg[0] = ears_avg[0] + ears_ng[i]
            if is_important[1] == 1:
                eyes = eyes + 1
                eyes_avg[0] = eyes_avg[0] + eyes_ng[i]
            if is_important[2] == 1:
                mouth = mouth + 1
                mouth_avg[0] = mouth_avg[0] + mouth_ng[i]
        if np.sum(is_important) == 2: # 2  grades > 1
            if is_important[0] == 1 and is_important[1] == 1: # ears and eyes
                ears_eyes = ears_eyes + 1
                ears_avg[1] = ears_avg[1] + ears_ng[i]
                eyes_avg[1] = eyes_avg[1] + eyes_ng[i]
            if is_important[0] == 1 and is_important[2] == 1: # ears and mouth
                ears_mouth = ears_mouth + 1
                ears_avg[2] = ears_avg[2] + ears_ng[i]
                mouth_avg[1] = mouth_avg[1] + mouth_ng[i]
            if is_important[1] == 1 and is_important[2] == 1: # eyes and mouth
                eyes_mouth = eyes_mouth + 1
                eyes_avg[2] = eyes_avg[2] + eyes_ng[i]
                mouth_avg[2] = mouth_avg[2] + mouth_ng[i]

    ears_avg[0] = ears_avg[0] / ears
    eyes_avg[0] = eyes_avg[0] / eyes
    mouth_avg[0] = mouth_avg[0] / mouth

    ears_avg[1] = ears_avg[1] / ears_eyes
    eyes_avg[1] = eyes_avg[1] / ears_eyes
    mouth_avg[1] = mouth_avg[1] / ears_mouth

    ears_avg[2] = ears_avg[2] / ears_mouth
    eyes_avg[2] = eyes_avg[2] / eyes_mouth
    mouth_avg[2] = mouth_avg[2] / eyes_mouth

    ears_avg[3] = ears_avg[3] / ears_eyes_mouth
    eyes_avg[3] = eyes_avg[3] / ears_eyes_mouth
    mouth_avg[3] = mouth_avg[3] / ears_eyes_mouth
    none_important=none_important/ears_ng.__len__()
    ears=ears/ears_ng.__len__()
    eyes=eyes/ears_ng.__len__()
    mouth=mouth /ears_ng.__len__()
    ears_eyes=ears_eyes /ears_ng.__len__()
    ears_mouth=ears_mouth /ears_ng.__len__()
    eyes_mouth=eyes_mouth /ears_ng.__len__()
    ears_eyes_mouth=ears_eyes_mouth /ears_ng.__len__()

    return none_important, ears, eyes, mouth, ears_eyes, ears_mouth, eyes_mouth, ears_eyes_mouth, ears_avg, eyes_avg, mouth_avg

def calc_map_type_quality(df:pandas.DataFrame, map_type:str):
    ears_prob = np.array(df['ears_prob' + map_type].tolist())
    eyes_prob = np.array(df['eyes_prob' + map_type].tolist())
    mouth_prob = np.array(df['mouth_prob' + map_type].tolist())
    ears_area = np.array(df['ears_area'].tolist())
    eyes_area = np.array(df['eyes_area'].tolist())
    mouth_area = np.array(df['mouth_area'].tolist())
    total_prob = ears_prob+eyes_prob+mouth_prob
    outer_prob = np.ones(total_prob.shape)-total_prob
    total_area = ears_area+eyes_area+mouth_area
    outer_area = np.ones(total_area.shape) - total_area
    ng_ears = np.divide(ears_prob, ears_area)
    ng_eyes = np.divide(eyes_prob, eyes_area)
    ng_mouth = np.divide(mouth_prob, mouth_area)
    ng = np.divide(total_prob, total_area)
    ng_out = np.divide(outer_prob, outer_area)
    out_mean = np.mean(ng_out)
    out_median = np.median(ng_out)
    ng_mean = np.mean(ng)
    ng_median = np.median(ng)
    ears_mean = np.mean(ng_ears)
    eyes_mean = np.mean(ng_eyes)
    mouth_mean = np.mean(ng_mouth)
    ears_median = np.median(ng_ears)
    eyes_median = np.median(ng_eyes)
    mouth_median = np.median(ng_mouth)
    res_mean =ears_mean+eyes_mean+mouth_mean-3*out_mean
    res_median =ears_median+eyes_median+mouth_median-3*out_median
    return res_mean, res_median



if __name__ == "__main__":
    # img_path = '/home/tali/cats_pain_proj/face_images/pain/cat_10_video_1.1.jpg'
    # msk_path = '/home/tali/cats_pain_proj/eyes_images/pain/cat_10_video_1.1.jpg'
    # plot_msk_on_img(img_path, msk_path)


    #dogs
    df = pd.read_csv("/home/tali/dogs_annika_proj/cropped_face/total_10.csv")
    catsSegs = CatsSegs(alpha=0.8, df=df, out_sz=(224, 224), res_folder='/home/tali',
                        imgs_root='/home/tali/dogs_annika_proj/data_set/',
                        msks_root='/home/tali/dogs_annika_proj/data_set/',
                        heats_root='/home/tali/dogs_annika_proj/res_10_gc/')
    all_outs = catsSegs.analyze_all()
    out_df_path = '/home/tali/dogs_annika_proj/res_10_gc/analyze.csv'
    catsSegs.create_res_df(all_outs, out_df_path)

    df1 = pd.read_csv('/home/tali/trials/try_finetune_mask_224_all_cam.csv')
    evalP = df1[df1["Valence"] == 1]
    resP , avgsP, avgsAllP= statistics_thresh_importance(evalP)
    evalNP = df1[df1["Valence"] == 0]
    resNP , avgsNP, avgsAllNP= statistics_thresh_importance(evalNP)
    mean_3, med_3 = calc_map_type_quality(df1,'3')
    mean_b, med_b = calc_map_type_quality(df1, 'b')
    mean_d, med_d = calc_map_type_quality(df1, 'd')
    #none_important, ears, eyes, mouth, ears_eyes, ears_mouth, eyes_mouth, ears_eyes_mouth,ears_avg, eyes_avg, mouth_avg = analyze_importance_of_segments(evalP, 'd')

    df = pd.read_csv("/home/tali/cats_pain_proj/face_images/masked_images/cats_finetune_mask_infered50.csv")
    catsSegs = CatsSegs(alpha=0.8, df=df, out_sz=(224, 224), res_folder='/home/tali',
                        imgs_root='/home/tali/cats_pain_proj/face_images/masked_images',
                        msks_root='/home/tali/cats_pain_proj',
                        heats_root='/home/tali/trials/cats_finetune_mask_seg_test50_cam/')
    all_outs = catsSegs.analyze_all()
    out_df_path = '/home/tali/trials/try_finetune_mask_224_high_new_b5.csv'
    catsSegs.create_res_df(all_outs, out_df_path)

    df1 = pd.read_csv('/home/tali/trials/try_finetune_mask_high_new_b.csv')
    evalP = df1[df1["Valence"] == 1]
    resP, avgsP = statistics_thresh_importance(df1)
    df1 = pd.read_csv('/home/tali/trials/try_finetune_mask_224_no_pain_high.csv')

    ret_df1=catsSegs.analyze_df(df1)
    ret_df = catsSegs.statistics_by_id(df1)
    ret_df1 = catsSegs.statistics_all(df1)
    ret_df1.to_csv('/home/tali/trials/analysis_min_mask_ft_all_nobad_pain.csv')
