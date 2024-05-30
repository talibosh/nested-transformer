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
import segments_utils
from abc import ABC, abstractmethod
class AnimalSegs:
    def __init__(self, alpha: float, df: pd.DataFrame, out_sz: tuple[int, int], res_folder: str, imgs_root: str,
                 msks_root: str, heats_root: str, segs_names: list[str],segs_max_det:list[int],heatmaps_names: list[str]):
        self.alpha = alpha
        self.out_sz = out_sz
        self.res_folder = res_folder
        self.df = df
        self.imgs_root = imgs_root
        self.msks_root = msks_root
        self.heats_root = heats_root
        self.segs_names = segs_names
        self.segs_max_det = segs_max_det
        self.heatmaps_names = heatmaps_names


    def analyze_one_img(self, img_path:str, heatmaps_paths:list[str], masks_dictionary:list[dict]):
        required_fields = {
            "seg_name": str,
            "mask_path": str,
        }

        for d in masks_dictionary:
            for field, field_type in required_fields.items():
                if field not in d:
                    raise ValueError(f"Missing required field: {field}")
                if not isinstance(d[field], field_type):
                    raise TypeError(f"Incorrect type for field {field}. Expected {field_type.__name__}.")

        segs_data = []
        for seg_name, max_det in zip( self.segs_names, self.segs_max_det):
            mask_path=""
            for d in masks_dictionary:
                if "seg_name" in d and d["seg_name"] == seg_name:
                    mask_path = d["mask_path"]
                    break

            seg = {'seg_name': seg_name, 'instances_num': max_det, 'msk_path': mask_path, 'heats_list': heatmaps_paths,
                    'outSz': self.out_sz}
            segs_data.append(seg)

        oias = segments_utils.OneImgAllSegs(self.alpha, img_path, segs_data)
        outs = oias.analyze_img()
        return outs


    def analyze_all(self):
        eval = self.df[self.df["label"] == self.df["Infered_Class"]]
        eval = eval[eval["Prob"] > 0.5]
        imgs_paths = eval['full path'].tolist()
        all_outs = self.analyze_img_lists(imgs_paths)
        return all_outs

    def create_res_df(self, all_outs):

        id = []
        img_name = []
        full_path = []
        valence = []
        video = []
        infered_class=[]

        # Dictionary to hold the new lists
        analyze_res_lists = {}

        # Create new lists and add them to the dictionary
        for seg_name in self.segs_names:
            list_name_area = seg_name + "_area"
            list_name_area_bb = seg_name + "_area_bb"
            for heat_name in self.heatmaps_names:
                list_name_prob = seg_name + "_prob_" + heat_name
                list_name_cnr = seg_name + "_cnr_" + heat_name
                list_name_ng = seg_name + "_ng_" + heat_name

                list_name_prob_bb = seg_name + "_prob_" + heat_name + "_bb"
                list_name_cnr_bb = seg_name + "_cnr_" + heat_name + "_bb"
                list_name_ng_bb = seg_name + "_ng_" + heat_name + "_bb"

                analyze_res_lists[list_name_prob] = []
                analyze_res_lists[list_name_cnr] = []
                analyze_res_lists[list_name_area] = []
                analyze_res_lists[list_name_ng] = []
                analyze_res_lists[list_name_prob_bb] = []
                analyze_res_lists[list_name_cnr_bb] = []
                analyze_res_lists[list_name_area_bb] = []
                analyze_res_lists[list_name_ng_bb] = []

        for i in range(all_outs.__len__()):  # go over images
            one_img_res = all_outs[i]
            # use 1st segment (usually all face) to detect id, full_path, ...and so on
            full_path.append(one_img_res[0]["full_path"])
            img_name.append(os.path.basename(one_img_res[0]["full_path"]))
            id.append(one_img_res[0]["id"])
            video.append(one_img_res[0]["video"])
            valence.append(one_img_res[0]["valence"])
            infered_class.append(one_img_res[0]["Infered_Class"])
            for seg_idx in range(one_img_res.__len__()):  # go over segments in image
                seg_res = one_img_res[seg_idx]
                seg_name = seg_res["seg_name"]
                area_name = seg_name + "_area"
                area_name_bb = area_name + "_bb"
                area = seg_res['areas'][0]
                area_bb = seg_res['areas_bb'][0]
                analyze_res_lists[area_name].append(area)
                analyze_res_lists[area_name_bb].append(area_bb)
                for heat_idx in range(self.heatmaps_names.__len__()):
                    heat_name = self.heatmaps_names[heat_idx]
                    prob = seg_res['prob_grades'][heat_idx]
                    prob_bb = seg_res['prob_grades_bb'][heat_idx]
                    list_name_prob = seg_name + "_prob_" + heat_name
                    analyze_res_lists[list_name_prob].append(prob)
                    list_name_prob_bb = list_name_prob + "_bb"
                    analyze_res_lists[list_name_prob_bb].append(prob_bb)
                    list_name_cnr = seg_name + "_cnr_" + heat_name
                    analyze_res_lists[list_name_cnr].append(seg_res['cnrs'][heat_idx])
                    list_name_cnr_bb = list_name_cnr + "_bb"
                    analyze_res_lists[list_name_cnr_bb].append(seg_res['cnrs_bb'][heat_idx])
                    list_name_ng = seg_name + "_ng_" + heat_name
                    analyze_res_lists[list_name_ng].append(prob /(area+1e-10)) #avoid division by 0 if seg was not found
                    list_name_ng_bb = list_name_ng + "_bb"
                    analyze_res_lists[list_name_ng_bb].append(prob_bb / (area_bb+1e-10))#avoid division by 0 if seg was not found

        analyze_res_lists["id"] = id
        analyze_res_lists["video"] = video
        analyze_res_lists["valence"] = valence
        analyze_res_lists["Infered_Class"] = infered_class
        analyze_res_lists["img_name"] = img_name
        analyze_res_lists["full_path"] = full_path
        df = pd.DataFrame(analyze_res_lists)

        return df

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
        seg_dict={}
        seg_list = []
        for hmn in self.heatmaps_names:
            seg_list.append(np.array(df[seg_name + '_prob_' + hmn + addition].tolist()))
            mean, std, norm_grades, norm_grades_stds = self.analyze_of_seg(seg_locs, seg_area,seg_list)
            #seg_list.append(np.array(df[seg_name + '_cnr_' + hmn + addition].tolist()))
            seg_dict[seg_name + '_prob_' + hmn + addition] = mean
            seg_dict[seg_name + '_ng_' + hmn + addition] = norm_grades
        return pd.DataFrame.from_dict(seg_dict)
    def analyze_df(self, df: pd.DataFrame):

        new_df = pd.DataFrame()
        for seg_name in self.segs_names:
            seg_dict = self.create_lists(df, seg_name, '')
            seg_dict_bb = self.create_lists(df, seg_name, '_bb')
            new_df = pd.concat([new_df, seg_dict, seg_dict_bb], axis=1)
        return new_df

    def calc_map_type_quality(self, df:pandas.DataFrame, segs_to_ignore:list[str], map_type:str):
        total_mean =0
        total_outer_mean = 0
        total_median = 0
        total_outer_median = 0
        for seg_name in self.segs_names:
            if seg_name in segs_to_ignore:
                continue
            seg_prob = np.array(df[seg_name+"_prob_" + map_type].tolist())
            seg_area = np.array(df[seg_name+"_area"].tolist())
            seg_relevant_locs = np.nonzero(seg_area)
            seg_ng = np.divide(seg_prob[seg_relevant_locs], seg_area[seg_relevant_locs])
            seg_mean = np.mean(seg_ng)
            seg_median = np.median(seg_ng)
            seg_outer_area = np.ones(seg_area[seg_relevant_locs].shape)-seg_area[seg_relevant_locs]
            seg_outer_prob = np.ones(seg_prob[seg_relevant_locs].shape)-seg_prob[seg_relevant_locs]
            seg_outer_ng = np.divide(seg_outer_prob, seg_outer_area)
            seg_outer_mean = np.mean(seg_outer_ng)
            seg_outer_median = np.median(seg_outer_ng)
            total_mean = total_mean + seg_mean
            total_outer_mean = total_outer_mean + seg_outer_mean
            total_median = total_median + seg_median
            total_outer_median = total_outer_median + seg_outer_median

        quality_mean = total_mean - total_outer_mean
        quality_median = total_median - total_outer_median
        return quality_mean, quality_median

if __name__ == "__main__":
    6
    # img_path = '/home/tali/cats_pain_proj/face_images/pain/cat_10_video_1.1.jpg'
    # msk_path = '/home/tali/cats_pain_proj/eyes_images/pain/cat_10_video_1.1.jpg'
    # plot_msk_on_img(img_path, msk_path)

