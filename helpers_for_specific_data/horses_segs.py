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


class HorsesSegs(AnimalSegs):
    def __init__(self, alpha: float, df: pd.DataFrame, out_sz: tuple[int, int], res_folder: str, imgs_root: str,
                 msks_root: str, heats_root: str, segs_names: list[str],segs_max_det:list[int],heatmaps_names: list[str]):
        super().__init__(alpha, df, out_sz, res_folder, imgs_root, msks_root, heats_root, segs_names, segs_max_det, heatmaps_names)
        self.df["Infered_Class"] = self.df['Infered_Class'].replace({1:'Yes', 0:'No'})



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

    def create_eval_df(self, res_df):
        eval_df = pd.DataFrame()
        videos = res_df["video"].to_list()
        unique_videos = np.unique(np.array(videos))
        for video in unique_videos:
            df = res_df[res_df["video"] == video]
            valence = (df["label"].tolist())[0]
            prediction = df["Infered_Class"].tolist()
            correct = sum(1 for x in prediction if x == valence)
            wrong = prediction.__len__() - correct
            if correct/(correct+wrong) < 0.6:
                continue
            success = 0
            if correct > wrong:
                success = 1
            if success:
                correct_df = df[df["Infered_Class"] == valence]
                if valence == 'P':
                    eval_df = pd.concat([eval_df, correct_df], axis=0, ignore_index=True)
        return eval_df

    def analyze_all(self):
        #eval = self.df
        eval = self.df[self.df["label"] == self.df["Infered_Class"]]
        eval = eval[eval["Prob"] > 0.5]
        all_outs = {}
        i = 0
        for idx, row in eval.iterrows():

            id = str(row["id"])
            valence = str(row["label"])

            filename = os.path.basename(row["file_name"])
            heats_paths = []
            for heat_name in self.heatmaps_names:
                heat_fname = filename.replace('.jpg', '_' + heat_name + '.npy')
                heatmap_path = os.path.join(self.heats_root, id, valence, "heats",heat_fname)
                heats_paths.append(heatmap_path)
            msks_dict_list=[]
            for seg_name in self.segs_names:
                msk_path = os.path.join(self.msks_root, valence, "masks", seg_name, filename)
                msk_dict={"seg_name":seg_name, "mask_path":msk_path}
                msks_dict_list.append(msk_dict)
            outs=self.analyze_one_img(row["fullpath"], heats_paths, msks_dict_list)
            for idx in range(outs.__len__()):
                outs[idx]["id"] = row["id"]
                outs[idx]["valence"] = row["label"]
                outs[idx]["Infered_Class"] = row["Infered_Class"]

            all_outs[i] = outs
            i = i + 1
        return all_outs



if __name__ == "__main__":
    # img_path = '/home/tali/cats_pain_proj/face_images/pain/cat_10_video_1.1.jpg'
    # msk_path = '/home/tali/cats_pain_proj/eyes_images/pain/cat_10_video_1.1.jpg'
    # plot_msk_on_img(img_path, msk_path)


    #horses
    df = pd.read_csv('/home/tali/horses/results/res25/total_res_25.csv')
    horsesSegs = HorsesSegs(alpha=0.8, df=df, out_sz=(28, 28), res_folder='/home/tali',
                       imgs_root='/home/tali/horses/dataset/',
                       msks_root='/home/tali/horses/dataset/',
                       heats_root='/home/tali/horses/results/res25/',
                       segs_names = ["face", "top", "middle", "bottom"], segs_max_det=[1, 1, 1, 1], heatmaps_names=["cam", "grad_cam", "eigen_cam"])

    all_outs = horsesSegs.analyze_all()
    out_df_path = '/home/tali/horses/results/res25/analyze.csv'
    res_df = horsesSegs.create_res_df(all_outs)
    res_df.to_csv(out_df_path)
    #analysis_df = dogSegs.analyze_df(res_df)
    #df_analysis_path = '/home/tali/dogs_annika_proj/res_25_mini_masked_all_maps/res_analysis.csv'
    #analysis_df.to_csv(df_analysis_path)
    cam_quality_mean, cam_quality_median, cam_res=horsesSegs.calc_map_type_quality(res_df, ['face'],'cam')
    gcam_quality_mean, gcam_quality_median, gcam_res=horsesSegs.calc_map_type_quality(res_df, ['face'], 'grad_cam')
    ecam_quality_mean, ecam_quality_median, ecam_res=horsesSegs.calc_map_type_quality(res_df, ['face'], 'eigen_cam')

