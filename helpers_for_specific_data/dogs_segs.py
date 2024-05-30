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
from animal_segs import AnimalSegs


class DogsSegs(AnimalSegs):
    def __init__(self, alpha: float, df: pd.DataFrame, out_sz: tuple[int, int], res_folder: str, imgs_root: str,
                 msks_root: str, heats_root: str, segs_names: list[str],segs_max_det:list[int],heatmaps_names: list[str]):
        super().__init__(alpha, df, out_sz, res_folder, imgs_root, msks_root, heats_root, segs_names, segs_max_det, heatmaps_names)
        self.df["Infered_Class"] = self.df['Infered_Class'].replace({1:'P', 0:'N'})



    def get_heatmap_for_img(self, img_path: str, heatmap_name: str):
        head, tail = os.path.split(img_path)
        f_splits = tail.split('_')
        id = f_splits[1]
        valence_name = 'neg'
        if img_path.find('P'):
            valence_name = 'pos'

        heat_maps_loc = os.path.join(self.heats_root, id, valence_name, 'heats', tail)
        # heat_maps_loc = head.replace(self.imgs_root_folder, self.heats_folder)
        ff = heat_maps_loc.replace('.jpg', '_' + heatmap_name + '.npy')
        return ff



    def analyze_all(self):
        eval = self.df[self.df["label"] == self.df["Infered_Class"]]
        eval = eval[eval["Prob"] > 0.5]
        all_outs = {}
        i = 0
        for idx, row in eval.iterrows():

            id = str(row["id"])
            valence = str(row["label"])
            if valence == "P":
                valenceh = 'pos'
            else:
                valenceh = 'neg'
            video_name = str(row["video"])
            filename = os.path.basename(row["file_name"])
            heats_paths = []
            for heat_name in self.heatmaps_names:
                heat_fname = filename.replace('.jpg', '_' + heat_name + '.npy')
                heatmap_path = os.path.join(self.heats_root, id, video_name, valenceh, "heats",heat_fname)
                heats_paths.append(heatmap_path)
            msks_dict_list=[]
            for seg_name in self.segs_names:
                msk_path = os.path.join(self.msks_root, id, valence, video_name,"masks", seg_name, filename)
                msk_dict={"seg_name":seg_name, "mask_path":msk_path}
                msks_dict_list.append(msk_dict)
            outs=self.analyze_one_img(row["full path"], heats_paths, msks_dict_list)
            for idx in range(outs.__len__()):
                outs[idx]["id"] = row["id"]
                outs[idx]["video"] = row["video"]
                outs[idx]["valence"] = row["label"]
                outs[idx]["Infered_Class"] =  row["Infered_Class"]

            all_outs[i] = outs
            i = i + 1
        return all_outs




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

if __name__ == "__main__":
    # img_path = '/home/tali/cats_pain_proj/face_images/pain/cat_10_video_1.1.jpg'
    # msk_path = '/home/tali/cats_pain_proj/eyes_images/pain/cat_10_video_1.1.jpg'
    # plot_msk_on_img(img_path, msk_path)


    #dogs
    df = pd.read_csv("/home/tali/dogs_annika_proj/cropped_face/total_10.csv")
    dogSegs = DogsSegs(alpha=0.8, df=df, out_sz=(28, 28), res_folder='/home/tali',
                       imgs_root='/home/tali/dogs_annika_proj/data_set/',
                       msks_root='/home/tali/dogs_annika_proj/data_set/',
                       heats_root='/home/tali/dogs_annika_proj/res_10_gc/',
                       segs_names = ["face", "ear", "eye"], segs_max_det=[1, 2, 2], heatmaps_names=["3"])

    all_outs = dogSegs.analyze_all()
    out_df_path = '/home/tali/dogs_annika_proj/res_10_gc/analyze.csv'
    res_df = dogSegs.create_res_df(all_outs)
    res_df.to_csv(out_df_path)
    analysis_df = dogSegs.analyze_df(res_df)
    df_analysis_path = '/home/tali/dogs_annika_proj/res_10_gc/res_analysis.csv'
    analysis_df.to_csv(df_analysis_path)
    dogSegs.calc_map_type_quality(res_df, ['face'],'3')

