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
    def __init__(self, msk_path: str, num_of_shows: int, outSz: tuple[int, int]):
        if os.path.isfile(msk_path) == False:
            self.orig_msk = []
            return
        self.num_of_shows = num_of_shows
        self.orig_msk = Image.open(msk_path)
        self.outSz = outSz
        self.rszd_msk, self.np_msk = self.rszNconvet2NP(self.orig_msk, outSz)
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
        rszd_heat = cv2.resize(heatmap, dsize=self.outSz, interpolation=cv2.INTER_LINEAR)
        # normalize heatmap - values are [0-1] and sums up to 1
        rszd_heat = (rszd_heat - rszd_heat.min()) / (rszd_heat.max() - rszd_heat.min())
        rszd_heat = rszd_heat / np.sum(rszd_heat)
        relevant_heat = rszd_heat * self.np_msk
        relevant_bb_heat = rszd_heat * self.bb_msk
        return relevant_heat, relevant_bb_heat, rszd_heat

    def calc_grade_by_seg(self, relevant_heat: np.array, rszd_heat: np.array, msk: np.array):
        prob_grade = np.sum(relevant_heat)
        mean_relevant_heat = np.mean(relevant_heat)
        var_relevant_heat = np.var(relevant_heat)
        rest_img_map = rszd_heat - relevant_heat
        mean_rest_of_img = np.mean(rest_img_map)
        var_rest_of_img = np.var(rest_img_map)
        cnr = np.abs(mean_relevant_heat - mean_rest_of_img) / np.sqrt(var_relevant_heat + var_rest_of_img)
        area = np.sum(msk) / (msk.shape[0] * msk.shape[1])
        return prob_grade, cnr, area

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

    def __init__(self, alpha: float, msk_path: str, img_path: str, heatmap_paths: list[str], max_segs_num: int,
                 out_sz: tuple[int, int]):
        self.msk_path = msk_path
        self.img_path = img_path
        self.heatmap_paths = heatmap_paths
        self.out_sz = out_sz
        self.max_segs_num = max_segs_num
        self.alpha = alpha

    def create_both_heatmap(self, hm3: np.array, hm2: np.array, alpha: float):
        assert (hm3.shape == hm2.shape)
        hmb = hm3 * alpha + hm2 * (1 - alpha)
        hmb = (hmb - hmb.min()) / (hmb.max() - hmb.min())
        hmb = hmb / np.sum(hmb)
        return hmb

    def create_both_dup_heatmap(self, hm3, hm2):
        assert (hm3.shape == hm2.shape)
        hmb = hm3 * hm2
        hmb = (hmb - hmb.min()) / (hmb.max() - hmb.min())
        hmb = hmb / np.sum(hmb)
        return hmb

    def get_msk_for_img(self):
        # msk_path = img_path.replace(self.imgs_root_folder, self.msks_folder)
        osh = OneSegOneHeatmapCalc(self.msk_path, self.max_segs_num, self.out_sz)
        if osh.orig_msk == []:
            return []
        else:
            return osh

    def get_one_heatmap_for_img(self, heatmap_path: str):
        heatmap = np.load(heatmap_path)
        rszd_heatmap = cv2.resize(heatmap, dsize=self.out_sz, interpolation=cv2.INTER_LINEAR)
        # rszd_heatmap = np.resize(heatmap, self.out_sz)
        # normalize resized image
        rszd_heatmap = (rszd_heatmap - rszd_heatmap.min()) / (rszd_heatmap.max() - rszd_heatmap.min())
        rszd_heatmap = rszd_heatmap / np.sum(rszd_heatmap)
        return rszd_heatmap

    def analyze_img(self):
        osh = self.get_msk_for_img()
        if osh == []:
            return [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]
        hm3 = self.get_one_heatmap_for_img(self.heatmap_paths[0])
        hm2 = self.get_one_heatmap_for_img(self.heatmap_paths[1])
        hm_both = self.create_both_heatmap(hm3, hm2, self.alpha)
        hm_bothd = self.create_both_dup_heatmap(hm3, hm2)
        relevant_heat3, relevant_bb_heat3, rszd_heat3 = osh.calc_relevant_heat(hm3)
        prob_grade3, cnr3, area3 = osh.calc_grade_by_seg(relevant_heat3, rszd_heat3, osh.np_msk)
        prob_grade3_bb, cnr3_bb, area3_bb = osh.calc_grade_by_seg(relevant_bb_heat3, rszd_heat3, osh.bb_msk)
        relevant_heat2, relevant_bb_heat2, rszd_heat2 = osh.calc_relevant_heat(hm2)
        prob_grade2, cnr2, area2 = osh.calc_grade_by_seg(relevant_heat2, rszd_heat2, osh.np_msk)
        prob_grade2_bb, cnr2_bb, area2_bb = osh.calc_grade_by_seg(relevant_bb_heat2, rszd_heat2, osh.bb_msk)
        relevant_heatb, relevant_bb_heatb, rszd_heatb = osh.calc_relevant_heat(hm_both)
        prob_gradeb, cnrb, areab = osh.calc_grade_by_seg(relevant_heatb, rszd_heatb, osh.np_msk)
        prob_gradeb_bb, cnrb_bb, areab_bb = osh.calc_grade_by_seg(relevant_bb_heatb, rszd_heatb, osh.bb_msk)
        relevant_heatd, relevant_bb_heatd, rszd_heatd = osh.calc_relevant_heat(hm_bothd)
        prob_graded, cnrd, aread = osh.calc_grade_by_seg(relevant_heatd, rszd_heatd, osh.np_msk)
        prob_graded_bb, cnrd_bb, aread_bb = osh.calc_grade_by_seg(relevant_bb_heatd, rszd_heatd, osh.bb_msk)
        return [prob_grade3, prob_grade2, prob_gradeb, prob_graded], [cnr3, cnr2, cnrb, cnrd], [area3, area2, areab,
                                                                                                aread], \
            [prob_grade3_bb, prob_grade2_bb, prob_gradeb_bb, prob_graded_bb], [cnr3_bb, cnr2_bb, cnrb_bb, cnrd_bb], \
            [area3_bb, area2_bb, areab_bb, aread_bb]


class OneImgAllSegs:
    def __init__(self, alpha: float, img_path: str,
                 segs_data: list[{'seg_name': str, 'instances_num': int, 'msk_path': str, 'heats_list': list[str]},
                            'outSz':tuple[int, int]]):
        self.alpha = alpha
        self.img_path = img_path
        self.segs_data = segs_data

    def analyze_img(self):
        i = 0
        outs = {}
        for seg_data in self.segs_data:
            outs[i] = {}
            msk_path = seg_data['msk_path']
            heats_list = seg_data['heats_list']
            out_sz = seg_data['outSz']
            max_segs_num = seg_data['instances_num']
            oios = OneImgOneSeg(self.alpha, msk_path, self.img_path, heats_list, max_segs_num, out_sz)
            prob_grades, cnrs, areas, prob_grades_bb, cnrs_bb, areas_bb = oios.analyze_img()
            outs[i]['full_path'] = self.img_path
            outs[i]['msk_path'] = seg_data['msk_path']
            outs[i]['outSz'] = seg_data['outSz']
            outs[i]['seg_name'] = seg_data['seg_name']
            outs[i]['prob_grades'] = prob_grades
            outs[i]['cnrs'] = cnrs
            outs[i]['areas'] = areas
            outs[i]['prob_grades_bb'] = prob_grades_bb
            outs[i]['cnrs_bb'] = cnrs_bb
            outs[i]['areas_bb'] = areas_bb
            i = i + 1
        return outs


class CatsSegs:
    def __init__(self, alpha: float, df: pd.DataFrame, out_sz: tuple[int, int], res_folder: str, imgs_root: str,
                 msks_root: str, heats_root: str):
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
        oias = OneImgAllSegs(self.alpha, img_full_path, segs_data)
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
        eval = self.df[self.df["Valence"] == cls]
        eval = eval[eval["Infered_Class"] == cls]
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
        face_prob2 = []
        face_cnr2 = []
        face_probb = []
        face_cnrb = []
        face_probd = []
        face_cnrd = []
        face_prob3_bb = []
        face_cnr3_bb = []
        face_prob2_bb = []
        face_cnr2_bb = []
        face_probb_bb = []
        face_cnrb_bb = []
        face_probd_bb = []
        face_cnrd_bb = []
        ears_area = []
        ears_area_bb = []
        ears_prob3 = []
        ears_cnr3 = []
        ears_prob2 = []
        ears_cnr2 = []
        ears_probb = []
        ears_cnrb = []
        ears_probd = []
        ears_cnrd = []
        ears_prob3_bb = []
        ears_cnr3_bb = []
        ears_prob2_bb = []
        ears_cnr2_bb = []
        ears_probb_bb = []
        ears_cnrb_bb = []
        ears_probd_bb = []
        ears_cnrd_bb = []
        eyes_area = []
        eyes_area_bb = []
        eyes_prob3 = []
        eyes_cnr3 = []
        eyes_prob2 = []
        eyes_cnr2 = []
        eyes_probb = []
        eyes_cnrb = []
        eyes_probd = []
        eyes_cnrd = []
        eyes_prob3_bb = []
        eyes_cnr3_bb = []
        eyes_prob2_bb = []
        eyes_cnr2_bb = []
        eyes_probb_bb = []
        eyes_cnrb_bb = []
        eyes_probd_bb = []
        eyes_cnrd_bb = []
        mouth_area = []
        mouth_area_bb = []
        mouth_prob3 = []
        mouth_cnr3 = []
        mouth_prob2 = []
        mouth_cnr2 = []
        mouth_probb = []
        mouth_cnrb = []
        mouth_probd = []
        mouth_cnrd = []
        mouth_prob3_bb = []
        mouth_cnr3_bb = []
        mouth_prob2_bb = []
        mouth_cnr2_bb = []
        mouth_probb_bb = []
        mouth_cnrb_bb = []
        mouth_probd_bb = []
        mouth_cnrd_bb = []
        # id, img name, img full path, valence, face_area, face_prob, face_cnr, ears_area, ears_prob, ears_cnr, eyes_area, eyes_prob, eyes_cnr, mouth_area, mouth_prob, mouth_cnr
        for i in range(all_outs.__len__()):
            img_data = all_outs[i]
            face_data = img_data[0]
            ears_data = img_data[1]
            eyes_data = img_data[2]
            mouth_data = img_data[3]

            full_path.append(face_data['full_path'])
            head, tail = os.path.split(face_data['full_path'])
            f_splits = tail.split('_')
            id_str = f_splits[1]
            img_name.append(tail)
            id.append(int(id_str))
            valence.append(1)
            face_area.append(face_data['areas'][0])
            face_prob3.append(face_data['prob_grades'][0])
            face_cnr3.append(face_data['cnrs'][0])
            face_prob2.append(face_data['prob_grades'][1])
            face_cnr2.append(face_data['cnrs'][1])
            face_probb.append(face_data['prob_grades'][2])
            face_cnrb.append(face_data['cnrs'][2])
            face_probd.append(face_data['prob_grades'][3])
            face_cnrd.append(face_data['cnrs'][3])
            face_area_bb.append(face_data['areas_bb'][0])
            face_prob3_bb.append(face_data['prob_grades_bb'][0])
            face_cnr3_bb.append(face_data['cnrs_bb'][0])
            face_prob2_bb.append(face_data['prob_grades_bb'][1])
            face_cnr2_bb.append(face_data['cnrs_bb'][1])
            face_probb_bb.append(face_data['prob_grades_bb'][2])
            face_cnrb_bb.append(face_data['cnrs_bb'][2])
            face_probd_bb.append(face_data['prob_grades_bb'][3])
            face_cnrd_bb.append(face_data['cnrs'][3])
            ears_area.append(ears_data['areas'][0])
            ears_prob3.append(ears_data['prob_grades'][0])
            ears_cnr3.append(ears_data['cnrs'][0])
            ears_prob2.append(ears_data['prob_grades'][1])
            ears_cnr2.append(ears_data['cnrs'][1])
            ears_probb.append(ears_data['prob_grades'][2])
            ears_cnrb.append(ears_data['cnrs'][2])
            ears_probd.append(ears_data['prob_grades'][3])
            ears_cnrd.append(ears_data['cnrs'][3])
            ears_area_bb.append(ears_data['areas_bb'][0])
            ears_prob3_bb.append(ears_data['prob_grades_bb'][0])
            ears_cnr3_bb.append(ears_data['cnrs_bb'][0])
            ears_prob2_bb.append(ears_data['prob_grades_bb'][1])
            ears_cnr2_bb.append(ears_data['cnrs_bb'][1])
            ears_probb_bb.append(ears_data['prob_grades_bb'][2])
            ears_cnrb_bb.append(ears_data['cnrs_bb'][2])
            ears_probd_bb.append(ears_data['prob_grades_bb'][3])
            ears_cnrd_bb.append(ears_data['cnrs_bb'][3])

            eyes_area.append(eyes_data['areas'][0])
            eyes_prob3.append(eyes_data['prob_grades'][0])
            eyes_cnr3.append(eyes_data['cnrs'][0])
            eyes_prob2.append(eyes_data['prob_grades'][1])
            eyes_cnr2.append(eyes_data['cnrs'][1])
            eyes_probb.append(eyes_data['prob_grades'][2])
            eyes_cnrb.append(eyes_data['cnrs'][2])
            eyes_probd.append(eyes_data['prob_grades'][3])
            eyes_cnrd.append(eyes_data['cnrs'][3])

            eyes_area_bb.append(eyes_data['areas_bb'][0])
            eyes_prob3_bb.append(eyes_data['prob_grades_bb'][0])
            eyes_cnr3_bb.append(eyes_data['cnrs_bb'][0])
            eyes_prob2_bb.append(eyes_data['prob_grades_bb'][1])
            eyes_cnr2_bb.append(eyes_data['cnrs_bb'][1])
            eyes_probb_bb.append(eyes_data['prob_grades_bb'][2])
            eyes_cnrb_bb.append(eyes_data['cnrs_bb'][2])
            eyes_probd_bb.append(eyes_data['prob_grades_bb'][3])
            eyes_cnrd_bb.append(eyes_data['cnrs_bb'][3])

            mouth_area.append(mouth_data['areas'][0])
            mouth_prob3.append(mouth_data['prob_grades'][0])
            mouth_cnr3.append(mouth_data['cnrs'][0])
            mouth_prob2.append(mouth_data['prob_grades'][1])
            mouth_cnr2.append(mouth_data['cnrs'][1])
            mouth_probb.append(mouth_data['prob_grades'][2])
            mouth_cnrb.append(mouth_data['cnrs'][2])
            mouth_probd.append(mouth_data['prob_grades'][3])
            mouth_cnrd.append(mouth_data['cnrs'][3])

            mouth_area_bb.append(mouth_data['areas_bb'][0])
            mouth_prob3_bb.append(mouth_data['prob_grades_bb'][0])
            mouth_cnr3_bb.append(mouth_data['cnrs_bb'][0])
            mouth_prob2_bb.append(mouth_data['prob_grades_bb'][1])
            mouth_cnr2_bb.append(mouth_data['cnrs_bb'][1])
            mouth_probb_bb.append(mouth_data['prob_grades_bb'][2])
            mouth_cnrb_bb.append(mouth_data['cnrs_bb'][2])
            mouth_probd_bb.append(mouth_data['prob_grades_bb'][3])
            mouth_cnrd_bb.append(mouth_data['cnrs_bb'][3])

        df = pd.DataFrame({'Id': id, 'Filename': img_name,
                           'FullPath': full_path, 'Valence': valence,
                           'face_area': face_area,
                           'face_prob3': face_prob3, 'face_prob2': face_prob2,
                           'face_probb': face_probb, 'face_probd': face_probd,
                           'face_cnr3': face_cnr3, 'face_cnr2': face_cnr2,
                           'face_cnrb': face_cnrb, 'face_cnrd': face_cnrd,

                           'face_area_bb': face_area_bb,
                           'face_prob3_bb': face_prob3_bb, 'face_prob2_bb': face_prob2_bb,
                           'face_probb_bb': face_probb_bb, 'face_probd_bb': face_probd_bb,
                           'face_cnr3_bb': face_cnr3_bb, 'face_cnr2_bb': face_cnr2_bb,
                           'face_cnrb_bb': face_cnrb_bb, 'face_cnrd_bb': face_cnrd_bb,

                           'ears_area': ears_area,
                           'ears_prob3': ears_prob3, 'ears_prob2': ears_prob2,
                           'ears_probb': ears_probb, 'ears_probd': ears_probd,
                           'ears_cnr3': ears_cnr3, 'ears_cnr2': ears_cnr2,
                           'ears_cnrb': ears_cnrb, 'ears_cnrd': ears_cnrd,

                           'ears_area_bb': ears_area_bb,
                           'ears_prob3_bb': ears_prob3_bb, 'ears_prob2_bb': ears_prob2_bb,
                           'ears_probb_bb': ears_probb_bb, 'ears_probd_bb': ears_probd_bb,
                           'ears_cnr3_bb': ears_cnr3_bb, 'ears_cnr2_bb': ears_cnr2_bb,
                           'ears_cnrb_bb': ears_cnrb_bb, 'ears_cnrd_bb': ears_cnrd_bb,

                           'eyes_area': eyes_area,
                           'eyes_prob3': eyes_prob3, 'eyes_prob2': eyes_prob2,
                           'eyes_probb': eyes_probb, 'eyes_probd': eyes_probd,
                           'eyes_cnr3': eyes_cnr3, 'eyes_cnr2': eyes_cnr2,
                           'eyes_cnrb': eyes_cnrb, 'eyes_cnrd': eyes_cnrd,

                           'eyes_area_bb': eyes_area_bb,
                           'eyes_prob3_bb': eyes_prob3_bb, 'eyes_prob2_bb': eyes_prob2_bb,
                           'eyes_probb_bb': eyes_probb_bb, 'eyes_probd_bb': eyes_probd_bb,
                           'eyes_cnr3_bb': eyes_cnr3_bb, 'eyes_cnr2_bb': eyes_cnr2_bb,
                           'eyes_cnrb_bb': eyes_cnrb_bb, 'eyes_cnrd_bb': eyes_cnrd_bb,

                           'mouth_area': mouth_area,
                           'mouth_prob3': mouth_prob3, 'mouth_prob2': mouth_prob2,
                           'mouth_probb': mouth_probb, 'mouth_probd': mouth_probd,
                           'mouth_cnr3': mouth_cnr3, 'mouth_cnr2': mouth_cnr2,
                           'mouth_cnrb': mouth_cnrb, 'mouth_cnrd': mouth_cnrd,

                           'mouth_area_bb': mouth_area_bb,
                           'mouth_prob3_bb': mouth_prob3_bb, 'mouth_prob2_bb': mouth_prob2_bb,
                           'mouth_probb_bb': mouth_probb_bb, 'mouth_probd_bb': mouth_probd_bb,
                           'mouth_cnr3_bb': mouth_cnr3_bb, 'mouth_cnr2_bb': mouth_cnr2_bb,
                           'mouth_cnrb_bb': mouth_cnrb_bb, 'mouth_cnrd_bb': mouth_cnrd_bb
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
        seg_dict = {#seg_name+addition+'_p3':[seg_means[0]], seg_name+addition+'_p2':[seg_means[1]],
                    #seg_name+addition+'_pb':[seg_means[2]],seg_name+addition+'_pd':[seg_means[3]],
                    seg_name + addition + '_cnr3': [seg_means[4]], #seg_name + addition + '_cnr2': [seg_means[5]],
                    #seg_name + addition + '_cnrb': [seg_means[6]], seg_name + addition + '_cnrd': [seg_means[7]],
                    seg_name+addition+'_ng3':[seg_norm_grades[0]]}#, seg_name+addition+'_ng2':[seg_norm_grades[1]],
                    #seg_name+addition+'_ngb':[seg_norm_grades[2]], seg_name+addition+'_ngd':[seg_norm_grades[3]]
        #}
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
      #new_df = pd.concat([new_df, face_dict, face_dict_bb, ears_dict, ears_dict_bb, eyes_dict,eyes_dict_bb,mouth_dict,mouth_dict_bb], axis=1)
      new_df = pd.concat([new_df,  ears_dict, ears_dict_bb, eyes_dict, eyes_dict_bb, mouth_dict,mouth_dict_bb], axis=1)

      return new_df
    def analyze_df1(self, df: pd.DataFrame):
        ids = np.array(df['Id'].tolist())
        img_name = np.array(df['Filename'].tolist())
        full_path = np.array(df['FullPath'].tolist())
        face_area = np.array(df['face_area'].tolist())
        face_prob3 = np.array(df['face_prob3'].tolist())
        face_prob2 = np.array(df['face_prob2'].tolist())
        face_probb = np.array(df['face_probb'].tolist())
        face_probd = np.array(df['face_probd'].tolist())

        face_cnr3 = np.array(df['face_cnr3'].tolist())
        face_cnr2 = np.array(df['face_cnr2'].tolist())
        face_cnrb = np.array(df['face_cnrb'].tolist())
        face_cnrd = np.array(df['face_cnrd'].tolist())

        face_area_bb = np.array(df['face_area_bb'].tolist())
        face_prob3_bb = np.array(df['face_prob3_bb'].tolist())
        face_prob2_bb = np.array(df['face_prob2_bb'].tolist())
        face_probb_bb = np.array(df['face_probb_bb'].tolist())
        face_probd_bb = np.array(df['face_probd_bb'].tolist())
        face_cnr3_bb = np.array(df['face_cnr3_bb'].tolist())
        face_cnr2_bb = np.array(df['face_cnr2_bb'].tolist())
        face_cnrb_bb = np.array(df['face_cnrb_bb'].tolist())
        face_cnrd_bb = np.array(df['face_cnrd_bb'].tolist())

        face_locs = np.nonzero(face_area)[0].tolist()
        mean_face_area, std_face_area = self.calc_mean_std_relevant(face_area, face_locs)
        mean_fp3, std_fp3 = self.calc_mean_std_relevant(face_prob3, face_locs)
        mean_fp2, std_fp2 = self.calc_mean_std_relevant(face_prob2, face_locs)
        mean_fpb, std_fpb = self.calc_mean_std_relevant(face_probb, face_locs)
        mean_fpbd, std_fpbd = self.calc_mean_std_relevant(face_probd, face_locs)
        mean_cnr3, std_cnr3 = self.calc_mean_std_relevant(face_cnr3, face_locs)
        mean_cnr2, std_cnr2 = self.calc_mean_std_relevant(face_cnr2, face_locs)
        mean_cnrb, std_cnrb = self.calc_mean_std_relevant(face_cnrb, face_locs)
        mean_cnrbd, std_cnrd = self.calc_mean_std_relevant(face_cnrd, face_locs)
        norm_grades_fp3 = self.calc_normalized_grade(face_prob3, face_area, face_locs)
        norm_grades_fp2 = self.calc_normalized_grade(face_prob2, face_area, face_locs)
        norm_grades_fpb = self.calc_normalized_grade(face_probb, face_area, face_locs)
        norm_grades_fpd = self.calc_normalized_grade(face_probd, face_area, face_locs)
        mean_face_area_bb, std_face_area_bb = self.calc_mean_std_relevant(face_area_bb, face_locs)
        mean_fp3_bb, std_fp3_bb = self.calc_mean_std_relevant(face_prob3_bb, face_locs)
        mean_fp2_bb, std_fp2_bb = self.calc_mean_std_relevant(face_prob2_bb, face_locs)
        mean_fpb_bb, std_fpb_bb = self.calc_mean_std_relevant(face_probb_bb, face_locs)
        mean_fpbd_bb, std_fpbd_bb = self.calc_mean_std_relevant(face_probd_bb, face_locs)
        mean_face_cnr_bb3, std_face_cnr3_bb = self.calc_mean_std_relevant(face_cnr3_bb, face_locs)
        mean_face_cnr2_bb, std_face_cnr2_bb = self.calc_mean_std_relevant(face_cnr2_bb, face_locs)
        mean_face_cnrb_bb, std_face_cnrb_bb = self.calc_mean_std_relevant(face_cnrb_bb, face_locs)
        mean_face_cnrbd_bb, std_face_cnrd_bb = self.calc_mean_std_relevant(face_cnrd_bb, face_locs)
        norm_grades_fp3_bb = self.calc_normalized_grade(face_prob3_bb, face_area_bb, face_locs)
        norm_grades_fp2_bb = self.calc_normalized_grade(face_prob2_bb, face_area_bb, face_locs)
        norm_grades_fpb_bb = self.calc_normalized_grade(face_probb_bb, face_area_bb, face_locs)
        norm_grades_fpd_bb = self.calc_normalized_grade(face_probd_bb, face_area_bb, face_locs)
        mean_norm_fp3, std_norm_fp3 = self.calc_mean_std_relevant(norm_grades_fp3,
                                                                  np.nonzero(norm_grades_fp3)[0].tolist())
        mean_norm_fp2, std_norm_fp2 = self.calc_mean_std_relevant(norm_grades_fp2,
                                                                  np.nonzero(norm_grades_fp2)[0].tolist())
        mean_norm_fpb, std_norm_fpb = self.calc_mean_std_relevant(norm_grades_fpb,
                                                                  np.nonzero(norm_grades_fpb)[0].tolist())
        mean_norm_fpd, std_norm_fpd = self.calc_mean_std_relevant(norm_grades_fpd,
                                                                  np.nonzero(norm_grades_fpd)[0].tolist())
        mean_norm_fp3_bb, std_norm_fp3_bb = self.calc_mean_std_relevant(norm_grades_fp3_bb,
                                                                        np.nonzero(norm_grades_fp3_bb)[0].tolist())
        mean_norm_fp2_bb, std_norm_fp2_bb = self.calc_mean_std_relevant(norm_grades_fp2_bb,
                                                                        np.nonzero(norm_grades_fp2_bb)[0].tolist())
        mean_norm_fpb_bb, std_norm_fpb_bb = self.calc_mean_std_relevant(norm_grades_fpb_bb,
                                                                        np.nonzero(norm_grades_fpb_bb)[0].tolist())
        mean_norm_fpd_bb, std_norm_fpd_bb = self.calc_mean_std_relevant(norm_grades_fpd_bb,
                                                                        np.nonzero(norm_grades_fpd_bb)[0].tolist())

        ears_area = np.array(df['ears_area'].tolist())
        ears_prob3 = np.array(df['ears_prob3'].tolist())
        ears_prob2 = np.array(df['ears_prob2'].tolist())
        ears_probb = np.array(df['ears_probb'].tolist())
        ears_probd = np.array(df['ears_probd'].tolist())
        ears_cnr3 = np.array(df['ears_cnr3'].tolist())
        ears_cnr2 = np.array(df['ears_cnr2'].tolist())
        ears_cnrb = np.array(df['ears_cnrb'].tolist())
        ears_cnrd = np.array(df['ears_cnrd'].tolist())

        ears_area_bb = np.array(df['ears_area_bb'].tolist())
        ears_prob3_bb = np.array(df['ears_prob3_bb'].tolist())
        ears_prob2_bb = np.array(df['ears_prob2_bb'].tolist())
        ears_probb_bb = np.array(df['ears_probb_bb'].tolist())
        ears_probd_bb = np.array(df['ears_probd_bb'].tolist())
        ears_cnr3_bb = np.array(df['ears_cnr3_bb'].tolist())
        ears_cnr2_bb = np.array(df['ears_cnr2_bb'].tolist())
        ears_cnrb_bb = np.array(df['ears_cnrb_bb'].tolist())
        ears_cnrd_bb = np.array(df['ears_cnrd_bb'].tolist())

        ears_locs = np.nonzero(ears_area)[0].tolist()
        mean_ears_area, std_ears_area = self.calc_mean_std_relevant(ears_area, ears_locs)
        mean_ears3, std_ears3 = self.calc_mean_std_relevant(ears_prob3, ears_locs)
        mean_ears2, std_ears2 = self.calc_mean_std_relevant(ears_prob2, ears_locs)
        mean_earsb, std_earsb = self.calc_mean_std_relevant(ears_probb, ears_locs)
        mean_earsbd, std_earsbd = self.calc_mean_std_relevant(ears_probd, ears_locs)
        mean_ears_cnr3, std_cnr3 = self.calc_mean_std_relevant(ears_cnr3, ears_locs)
        mean_ears_cnr2, std_cnr2 = self.calc_mean_std_relevant(ears_cnr2, ears_locs)
        mean_ears_cnrb, std_cnrb = self.calc_mean_std_relevant(ears_cnrb, ears_locs)
        mean_ears_cnrbd, std_ears_cnrd = self.calc_mean_std_relevant(ears_cnrd, ears_locs)
        norm_grades_earsp3 = self.calc_normalized_grade(ears_prob3, ears_area, ears_locs)
        norm_grades_earsp2 = self.calc_normalized_grade(ears_prob2, ears_area, ears_locs)
        norm_grades_earspb = self.calc_normalized_grade(ears_probb, ears_area, ears_locs)
        norm_grades_earspd = self.calc_normalized_grade(ears_probd, ears_area, ears_locs)
        mean_ears_area_bb, std_ears_area_bb = self.calc_mean_std_relevant(ears_area_bb, ears_locs)
        mean_earsp3_bb, std_earsp3_bb = self.calc_mean_std_relevant(ears_prob3_bb, ears_locs)
        mean_earsp2_bb, std_earsp2_bb = self.calc_mean_std_relevant(ears_prob2_bb, ears_locs)
        mean_earspb_bb, std_earspb_bb = self.calc_mean_std_relevant(ears_probb_bb, ears_locs)
        mean_earspbd_bb, std_earspbd_bb = self.calc_mean_std_relevant(ears_probd_bb, ears_locs)
        mean_ears_cnr_bb3, std_ears_cnr3_bb = self.calc_mean_std_relevant(ears_cnr3_bb, ears_locs)
        mean_ears_cnr2_bb, std_ears_cnr2_bb = self.calc_mean_std_relevant(ears_cnr2_bb, ears_locs)
        mean_ears_cnrb_bb, std_ears_cnrb_bb = self.calc_mean_std_relevant(ears_cnrb_bb, ears_locs)
        mean_ears_cnrbd_bb, std_ears_cnrd_bb = self.calc_mean_std_relevant(ears_cnrd_bb, ears_locs)
        norm_grades_earsp3_bb = self.calc_normalized_grade(ears_prob3_bb, ears_area_bb, ears_locs)
        norm_grades_earsp2_bb = self.calc_normalized_grade(ears_prob2_bb, ears_area_bb, ears_locs)
        norm_grades_earspb_bb = self.calc_normalized_grade(ears_probb_bb, ears_area_bb, ears_locs)
        norm_grades_earspd_bb = self.calc_normalized_grade(ears_probd_bb, ears_area_bb, ears_locs)
        mean_norm_ears3, std_norm_ears3 = self.calc_mean_std_relevant(norm_grades_earsp3,
                                                                      np.nonzero(norm_grades_earsp3)[0].tolist())
        mean_norm_ears2, std_norm_ears2 = self.calc_mean_std_relevant(norm_grades_earsp2,
                                                                      np.nonzero(norm_grades_earsp2)[0].tolist())
        mean_norm_earsb, std_norm_earsb = self.calc_mean_std_relevant(norm_grades_earspb,
                                                                      np.nonzero(norm_grades_earspb)[0].tolist())
        mean_norm_earsd, std_norm_earsd = self.calc_mean_std_relevant(norm_grades_earspd,
                                                                      np.nonzero(norm_grades_earspd)[0].tolist())
        mean_norm_ears3_bb, std_norm_ears3_bb = self.calc_mean_std_relevant(norm_grades_earsp3_bb,
                                                                            np.nonzero(norm_grades_earsp3_bb)[
                                                                                0].tolist())
        mean_norm_ears2_bb, std_norm_ears2_bb = self.calc_mean_std_relevant(norm_grades_earsp2_bb,
                                                                            np.nonzero(norm_grades_earsp2_bb)[
                                                                                0].tolist())
        mean_norm_earsb_bb, std_norm_earsb_bb = self.calc_mean_std_relevant(norm_grades_earspb_bb,
                                                                            np.nonzero(norm_grades_earspb_bb)[
                                                                                0].tolist())
        mean_norm_earsd_bb, std_norm_earsd_bb = self.calc_mean_std_relevant(norm_grades_earspd_bb,
                                                                            np.nonzero(norm_grades_earspd_bb)[
                                                                                0].tolist())

        eyes_area = np.array(df['eyes_area3'].tolist())
        eyes_prob3 = np.array(df['eyes_prob3'].tolist())
        eyes_prob2 = np.array(df['eyes_prob2'].tolist())
        eyes_probb = np.array(df['eyes_probb'].tolist())
        eyes_probd = np.array(df['eyes_probd'].tolist())
        eyes_cnr3 = np.array(df['eyes_cnr3'].tolist())
        eyes_cnr2 = np.array(df['eyes_cnr2'].tolist())
        eyes_cnrb = np.array(df['eyes_cnrb'].tolist())
        eyes_cnrd = np.array(df['eyes_cnrd'].tolist())

        eyes_area_bb = np.array(df['eyes_area3_bb'].tolist())
        eyes_prob3_bb = np.array(df['eyes_prob3_bb'].tolist())
        eyes_prob2_bb = np.array(df['eyes_prob2_bb'].tolist())
        eyes_probb_bb = np.array(df['eyes_probb_bb'].tolist())
        eyes_probd_bb = np.array(df['eyes_probd_bb'].tolist())
        eyes_cnr3_bb = np.array(df['eyes_cnr3_bb'].tolist())
        eyes_cnr2_bb = np.array(df['eyes_cnr2_bb'].tolist())
        eyes_cnrb_bb = np.array(df['eyes_cnrb_bb'].tolist())
        eyes_cnrd_bb = np.array(df['eyes_cnrd_bb'].tolist())

        eyes_locs = np.nonzero(eyes_area)[0].tolist()
        mean_eyes_area, std_eyes_area = self.calc_mean_std_relevant(eyes_area, eyes_locs)
        mean_eyes3, std_eyes3 = self.calc_mean_std_relevant(eyes_prob3, eyes_locs)
        mean_eyes2, std_eyes2 = self.calc_mean_std_relevant(eyes_prob2, eyes_locs)
        mean_eyesb, std_eyesb = self.calc_mean_std_relevant(eyes_probb, eyes_locs)
        mean_eyesbd, std_eyesbd = self.calc_mean_std_relevant(eyes_probd, eyes_locs)
        mean_eyes_cnr3, std_cnr3 = self.calc_mean_std_relevant(eyes_cnr3, eyes_locs)
        mean_eyes_cnr2, std_cnr2 = self.calc_mean_std_relevant(eyes_cnr2, eyes_locs)
        mean_eyes_cnrb, std_cnrb = self.calc_mean_std_relevant(eyes_cnrb, eyes_locs)
        mean_eyes_cnrbd, std_eyes_cnrd = self.calc_mean_std_relevant(eyes_cnrd, eyes_locs)
        norm_grades_eyesp3 = self.calc_normalized_grade(eyes_prob3, eyes_area, eyes_locs)
        norm_grades_eyesp2 = self.calc_normalized_grade(eyes_prob2, eyes_area, eyes_locs)
        norm_grades_eyespb = self.calc_normalized_grade(eyes_probb, eyes_area, eyes_locs)
        norm_grades_eyespd = self.calc_normalized_grade(eyes_probd, eyes_area, eyes_locs)
        mean_eyes_area_bb, std_eyes_area_bb = self.calc_mean_std_relevant(eyes_area_bb, eyes_locs)
        mean_eyesp3_bb, std_eyesp3_bb = self.calc_mean_std_relevant(eyes_prob3_bb, eyes_locs)
        mean_eyesp2_bb, std_eyesp2_bb = self.calc_mean_std_relevant(eyes_prob2_bb, eyes_locs)
        mean_eyespb_bb, std_eyespb_bb = self.calc_mean_std_relevant(eyes_probb_bb, eyes_locs)
        mean_eyespbd_bb, std_eyespbd_bb = self.calc_mean_std_relevant(eyes_probd_bb, eyes_locs)
        mean_eyes_cnr_bb3, std_eyes_cnr3_bb = self.calc_mean_std_relevant(eyes_cnr3_bb, eyes_locs)
        mean_eyes_cnr2_bb, std_eyes_cnr2_bb = self.calc_mean_std_relevant(eyes_cnr2_bb, eyes_locs)
        mean_eyes_cnrb_bb, std_eyes_cnrb_bb = self.calc_mean_std_relevant(eyes_cnrb_bb, eyes_locs)
        mean_eyes_cnrbd_bb, std_eyes_cnrd_bb = self.calc_mean_std_relevant(eyes_cnrd_bb, eyes_locs)
        norm_grades_eyesp3_bb = self.calc_normalized_grade(eyes_prob3_bb, eyes_area_bb, eyes_locs)
        norm_grades_eyesp2_bb = self.calc_normalized_grade(eyes_prob2_bb, eyes_area_bb, eyes_locs)
        norm_grades_eyespb_bb = self.calc_normalized_grade(eyes_probb_bb, eyes_area_bb, eyes_locs)
        norm_grades_eyespd_bb = self.calc_normalized_grade(eyes_probd_bb, eyes_area_bb, eyes_locs)
        mean_norm_eyes3, std_norm_eyes3 = self.calc_mean_std_relevant(norm_grades_eyesp3,
                                                                      np.nonzero(norm_grades_eyesp3)[0].tolist())
        mean_norm_eyes2, std_norm_eyes2 = self.calc_mean_std_relevant(norm_grades_eyesp2,
                                                                      np.nonzero(norm_grades_eyesp2)[0].tolist())
        mean_norm_eyesb, std_norm_eyesb = self.calc_mean_std_relevant(norm_grades_eyespb,
                                                                      np.nonzero(norm_grades_eyespb)[0].tolist())
        mean_norm_eyesd, std_norm_eyesd = self.calc_mean_std_relevant(norm_grades_eyespd,
                                                                      np.nonzero(norm_grades_eyespd)[0].tolist())
        mean_norm_eyes3_bb, std_norm_eyes3_bb = self.calc_mean_std_relevant(norm_grades_eyesp3_bb,
                                                                            np.nonzero(norm_grades_eyesp3_bb)[
                                                                                0].tolist())
        mean_norm_eyes2_bb, std_norm_eyes2_bb = self.calc_mean_std_relevant(norm_grades_eyesp2_bb,
                                                                            np.nonzero(norm_grades_eyesp2_bb)[
                                                                                0].tolist())
        mean_norm_eyesb_bb, std_norm_eyesb_bb = self.calc_mean_std_relevant(norm_grades_eyespb_bb,
                                                                            np.nonzero(norm_grades_eyespb_bb)[
                                                                                0].tolist())
        mean_norm_eyesd_bb, std_norm_eyesd_bb = self.calc_mean_std_relevant(norm_grades_eyespd_bb,
                                                                            np.nonzero(norm_grades_eyespd_bb)[
                                                                                0].tolist())

        mouth_area = np.array(df['mouth_area3'].tolist())
        mouth_prob3 = np.array(df['mouth_prob3'].tolist())
        mouth_prob2 = np.array(df['mouth_prob2'].tolist())
        mouth_probb = np.array(df['mouth_probb'].tolist())
        mouth_probd = np.array(df['mouth_probd'].tolist())
        mouth_cnr3 = np.array(df['mouth_cnr3'].tolist())
        mouth_cnr2 = np.array(df['mouth_cnr2'].tolist())
        mouth_cnrb = np.array(df['mouth_cnrb'].tolist())
        mouth_cnrd = np.array(df['mouth_cnrd'].tolist())

        mouth_area_bb = np.array(df['mouth_area3_bb'].tolist())
        mouth_prob3_bb = np.array(df['mouth_prob3_bb'].tolist())
        mouth_prob2_bb = np.array(df['mouth_prob2_bb'].tolist())
        mouth_probb_bb = np.array(df['mouth_probb_bb'].tolist())
        mouth_probd_bb = np.array(df['mouth_probd_bb'].tolist())
        mouth_cnr3_bb = np.array(df['mouth_cnr3_bb'].tolist())
        mouth_cnr2_bb = np.array(df['mouth_cnr2_bb'].tolist())
        mouth_cnrb_bb = np.array(df['mouth_cnrb_bb'].tolist())
        mouth_cnrd_bb = np.array(df['mouth_cnrd_bb'].tolist())

        mouth_locs = np.nonzero(mouth_area)[0].tolist()
        mean_mouth_area, std_mouth_area = self.calc_mean_std_relevant(mouth_area, mouth_locs)
        mean_mouth3, std_mouth3 = self.calc_mean_std_relevant(mouth_prob3, mouth_locs)
        mean_mouth2, std_mouth2 = self.calc_mean_std_relevant(mouth_prob2, mouth_locs)
        mean_mouthb, std_mouthb = self.calc_mean_std_relevant(mouth_probb, mouth_locs)
        mean_mouthbd, std_mouthbd = self.calc_mean_std_relevant(mouth_probd, mouth_locs)
        mean_mouth_cnr3, std_cnr3 = self.calc_mean_std_relevant(mouth_cnr3, mouth_locs)
        mean_mouth_cnr2, std_cnr2 = self.calc_mean_std_relevant(mouth_cnr2, mouth_locs)
        mean_mouth_cnrb, std_cnrb = self.calc_mean_std_relevant(mouth_cnrb, mouth_locs)
        mean_mouth_cnrbd, std_mouth_cnrd = self.calc_mean_std_relevant(mouth_cnrd, mouth_locs)
        norm_grades_mouthp3 = self.calc_normalized_grade(mouth_prob3, mouth_area, mouth_locs)
        norm_grades_mouthp2 = self.calc_normalized_grade(mouth_prob2, mouth_area, mouth_locs)
        norm_grades_mouthpb = self.calc_normalized_grade(mouth_probb, mouth_area, mouth_locs)
        norm_grades_mouthpd = self.calc_normalized_grade(mouth_probd, mouth_area, mouth_locs)
        mean_mouth_area_bb, std_mouth_area_bb = self.calc_mean_std_relevant(mouth_area_bb, mouth_locs)
        mean_mouthp3_bb, std_mouthp3_bb = self.calc_mean_std_relevant(mouth_prob3_bb, mouth_locs)
        mean_mouthp2_bb, std_mouthp2_bb = self.calc_mean_std_relevant(mouth_prob2_bb, mouth_locs)
        mean_mouthpb_bb, std_mouthpb_bb = self.calc_mean_std_relevant(mouth_probb_bb, mouth_locs)
        mean_mouthpbd_bb, std_mouthpbd_bb = self.calc_mean_std_relevant(mouth_probd_bb, mouth_locs)
        mean_mouth_cnr_bb3, std_mouth_cnr3_bb = self.calc_mean_std_relevant(mouth_cnr3_bb, mouth_locs)
        mean_mouth_cnr2_bb, std_mouth_cnr2_bb = self.calc_mean_std_relevant(mouth_cnr2_bb, mouth_locs)
        mean_mouth_cnrb_bb, std_mouth_cnrb_bb = self.calc_mean_std_relevant(mouth_cnrb_bb, mouth_locs)
        mean_mouth_cnrbd_bb, std_mouth_cnrd_bb = self.calc_mean_std_relevant(mouth_cnrd_bb, mouth_locs)
        norm_grades_mouthp3_bb = self.calc_normalized_grade(mouth_prob3_bb, mouth_area_bb, mouth_locs)
        norm_grades_mouthp2_bb = self.calc_normalized_grade(mouth_prob2_bb, mouth_area_bb, mouth_locs)
        norm_grades_mouthpb_bb = self.calc_normalized_grade(mouth_probb_bb, mouth_area_bb, mouth_locs)
        norm_grades_mouthpd_bb = self.calc_normalized_grade(mouth_probd_bb, mouth_area_bb, mouth_locs)
        mean_norm_mouth3, std_norm_mouth3 = self.calc_mean_std_relevant(norm_grades_mouthp3,
                                                                        np.nonzero(norm_grades_mouthp3)[0].tolist())
        mean_norm_mouth2, std_norm_mouth2 = self.calc_mean_std_relevant(norm_grades_mouthp2,
                                                                        np.nonzero(norm_grades_mouthp2)[0].tolist())
        mean_norm_mouthb, std_norm_mouthb = self.calc_mean_std_relevant(norm_grades_mouthpb,
                                                                        np.nonzero(norm_grades_mouthpb)[0].tolist())
        mean_norm_mouthd, std_norm_mouthd = self.calc_mean_std_relevant(norm_grades_mouthpd,
                                                                        np.nonzero(norm_grades_mouthpd)[0].tolist())
        mean_norm_mouth3_bb, std_norm_mouth3_bb = self.calc_mean_std_relevant(norm_grades_mouthp3_bb,
                                                                              np.nonzero(norm_grades_mouthp3_bb)[
                                                                                  0].tolist())
        mean_norm_mouth2_bb, std_norm_mouth2_bb = self.calc_mean_std_relevant(norm_grades_mouthp2_bb,
                                                                              np.nonzero(norm_grades_mouthp2_bb)[
                                                                                  0].tolist())
        mean_norm_mouthb_bb, std_norm_mouthb_bb = self.calc_mean_std_relevant(norm_grades_mouthpb_bb,
                                                                              np.nonzero(norm_grades_mouthpb_bb)[
                                                                                  0].tolist())
        mean_norm_mouthd_bb, std_norm_mouthd_bb = self.calc_mean_std_relevant(norm_grades_mouthpd_bb,
                                                                              np.nonzero(norm_grades_mouthpd_bb)[
                                                                                  0].tolist())

    def analyze_by_id(self, df: pd.DataFrame):
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


if __name__ == "__main__":
    # img_path = '/home/tali/cats_pain_proj/face_images/pain/cat_10_video_1.1.jpg'
    # msk_path = '/home/tali/cats_pain_proj/eyes_images/pain/cat_10_video_1.1.jpg'
    # plot_msk_on_img(img_path, msk_path)

    df = pd.read_csv("/home/tali/cats_pain_proj/face_images/masked_images/cats_finetune_mask_infered50.csv")
    catsSegs = CatsSegs(alpha=0.7, df=df, out_sz=(224, 224), res_folder='/home/tali',
                        imgs_root='/home/tali/cats_pain_proj/face_images/masked_images',
                        msks_root='/home/tali/cats_pain_proj',
                        heats_root='/home/tali/trials/cats_finetune_mask_relu_res_test50/')
    all_outs = catsSegs.analyze_all()
    out_df_path = '/home/tali/trials/try_finetune_mask_224.csv'
    catsSegs.create_res_df(all_outs, out_df_path)

    df1 = pd.read_csv('/home/tali/trials/try_finetune_mask_224.csv')
    #catsSegs.analyze_df(df1)
    ret_df = catsSegs.analyze_by_id(df1)
    ret_df.to_csv('/home/tali/trials/analysis_min_mask_ft.csv')
