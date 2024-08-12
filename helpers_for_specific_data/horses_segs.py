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
import glob

class HorsesSegs(AnimalSegs):
    def __init__(self, alpha: float, df: pd.DataFrame, out_sz: tuple[int, int], res_folder: str, imgs_root: str,
                 msks_root: str, heats_root: str, segs_names: list[str],segs_max_det:list[int],heatmaps_names: list[str],manip_type:str):
        return_after_super=False
        if df.empty:
            return_after_super=True
        super().__init__(alpha, df, out_sz, res_folder, imgs_root, msks_root, heats_root, segs_names, segs_max_det, heatmaps_names,manip_type)
        if return_after_super:
            return
        if not("Infered_Class_nums" in self.df.columns):
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
        filtered_df = res_df[res_df['label'] == res_df['Infered_Class']]
        return filtered_df

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
            valence = str(row["label"])

            filename = os.path.basename(row["file_name"])
            heats_paths = []
            for heat_name in self.heatmaps_names:
                heat_fname1 = filename.replace('.jpg', '_' + heat_name + '.npy')
                heatmap_path1 = os.path.join(self.heats_root, id,  valence, "heats", heat_fname1)
                heat_fname2 = filename.replace('.jpg', '.npy')
                heatmap_path2 = os.path.join(self.heats_root, heat_name, id, valence,  heat_fname2)
                a1 = os.path.isfile(heatmap_path1)
                a2 = os.path.isfile(heatmap_path2)
                if a1:
                    heats_paths.append(heatmap_path1)
                if a2:
                    heats_paths.append(heatmap_path2)

            #for heat_name in self.heatmaps_names:
            #    heat_fname = filename.replace('.jpg', '_' + heat_name + '.npy')
            #    heatmap_path = os.path.join(self.heats_root, id, valence, "heats",heat_fname)
            #    heats_paths.append(heatmap_path)
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

def calc_qualities(df:pd.DataFrame, heats_root:str, heats_names:list[str], out_df_path:str,manip_type:str):
    horsesSegs = HorsesSegs(alpha=0.8, df=df, out_sz=(28, 28), res_folder='/home/tali',
                            imgs_root='/home/tali/horses/dataset/',
                            msks_root='/home/tali/horses/dataset/',
                            heats_root=heats_root,
                            segs_names=["face", "top", "middle", "bottom"], segs_max_det=[1, 1, 1, 1],
                            heatmaps_names=heats_names,manip_type = manip_type)

    all_outs = horsesSegs.analyze_all()
    #out_df_path = '/home/tali/horses/results/res25/analyze.csv'
    res_df = horsesSegs.create_res_df(all_outs)
    res_df.to_csv(out_df_path)
    # analysis_df = dogSegs.analyze_df(res_df)
    # df_analysis_path = '/home/tali/dogs_annika_proj/res_25_mini_masked_all_maps/res_analysis.csv'
    # analysis_df.to_csv(df_analysis_path)
    summary_path = os.path.join(os.path.dirname(out_df_path), 'summary_' + manip_type + '.json')
    cuts_dict = {'all': 'all', 'Yes': 'valence', 'No': 'valence'}
    horsesSegs.summarize_results_and_calc_qualities(res_df, cuts_dict, summary_path)
    return summary_path

def run_horses():
    def create_pytorch_path(type:str,ft:str, root_path:str,add:str)->str:
        out_path = os.path.join(root_path, 'pytorch_'+type+ft,add)
        return out_path

    heats_names = ['grad_cam','xgrad_cam','grad_cam_plusplus','power_grad_cam']
    root_path = '/home/tali/horses'
    net_types = ['vit','dino','resnet50','nest-tiny']
    #net_types = ['dino', 'nest-tiny']
    run_type =['']
    manip_type=['']
    summaries={}
    for rt in run_type:
        for i,type in enumerate(net_types):
            for manipulation in manip_type:
                if type == 'nest-tiny':
                    inference_file =  '/home/tali/horses/results/res25/total_res_25.csv'
                    heats_root = '/home/tali/horses/results/res25/'
                    out_df_path = os.path.join(heats_root,'analysis_'+manipulation+'.csv')
                else:
                    inference_file = create_pytorch_path(type,rt,root_path,'inference.csv')
                    heats_root = create_pytorch_path(type,rt,root_path,'maps')
                    out_df_path = create_pytorch_path(type, rt, root_path, 'analysis_' + manipulation + '.csv')

                # Delete each .jpg file
                jpg_files = glob.glob(os.path.join(heats_root, "**", "*.jpg"), recursive=True)

                for file_path in jpg_files:
                    os.remove(file_path)
                summary_path=calc_qualities(pd.read_csv(inference_file), heats_root, heats_names,out_df_path, manipulation)
                summaries[type]=summary_path
    return  summaries
def plot_horses(net_jsons:dict):
    heats_names = ['grad_cam', 'xgrad_cam', 'grad_cam_plusplus', 'power_grad_cam']
    horsesSegs = HorsesSegs(alpha=0.8, df=pd.DataFrame(), out_sz=(28, 28), res_folder='/home/tali',
                            imgs_root='/home/tali/horses/dataset/',
                            msks_root='/home/tali/horses/dataset/',
                            heats_root='',
                            segs_names=["face", "top", "middle", "bottom"], segs_max_det=[1, 1, 1, 1],
                            heatmaps_names=heats_names, manip_type='')
    net_colors = {'resnet50': 'red', 'vit': 'green', 'dino': 'blue', 'nest-tiny': 'orange'}
    outdir = '/home/tali/horses/plots/'
    os.makedirs(outdir,exist_ok=True)
    horsesSegs.go_over_jsons_and_plot(net_colors, net_jsons,outdir)

if __name__ == "__main__":
    # img_path = '/home/tali/cats_pain_proj/face_images/pain/cat_10_video_1.1.jpg'
    # msk_path = '/home/tali/cats_pain_proj/eyes_images/pain/cat_10_video_1.1.jpg'
    # plot_msk_on_img(img_path, msk_path)
    summaries = run_horses()
    plot_horses(summaries)
