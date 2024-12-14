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

class CatsSegs(AnimalSegs):
    def __init__(self, alpha: float, df: pd.DataFrame, out_sz: tuple[int, int], res_folder: str, imgs_root: str,
                 msks_root: str, heats_root: str, segs_names: list[str],segs_max_det:list[int],
                 heatmaps_names: list[str],manip_type:str):
        return_after_super = False
        if df.empty:
            return_after_super=True
        super().__init__(alpha, df, out_sz, res_folder, imgs_root, msks_root, heats_root, segs_names, segs_max_det,
                         heatmaps_names,manip_type)
        if return_after_super:
            return
        if not("Infered_Class_nums" in self.df.columns):
            self.df["Infered_Class"] = self.df['Infered_Class'].replace({1:'Yes', 0:'No'})
        self.df["label"] = self.df['label'].replace({1: 'Yes', 0: 'No'})



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
        '''
        eval_df = pd.DataFrame()
        ids = res_df["id"].to_list()
        unique_ids = np.unique(np.array(ids))
        for id in unique_ids:
            df = res_df[res_df["id"] == id]
            valence = df["label"].tolist()
            prediction = df["Infered_Class"].tolist()
            comparison = [a == b for a, b in zip(valence, prediction)]
            correct = sum(1 for x in comparison if x == True)
            #correct = sum(1 for x in prediction if x == valence)
            wrong = prediction.__len__() - correct
            if correct/(correct+wrong) < 0.5:
                continue
            success = 0
            if correct > wrong:
                success = 1
            if success:
                correct_df = df[df["Infered_Class"] == valence]
                #if valence == 'P':
                eval_df = pd.concat([eval_df, correct_df], axis=0, ignore_index=True)
        return eval_df
        '''
    def analyze_all(self):
        #eval = self.df
        eval = self.create_eval_df(self.df)
        #eval = self.df[self.df["label"] == self.df["Infered_Class"]]
        #eval = eval[eval["Prob"] > 0.5]
        all_outs = {}
        i = 0
        for idx, row in eval.iterrows():

            id = str(row["id"])
            valence = str(row["label"])

            filename = os.path.basename(row["file_name"])
            heats_paths = []
            if valence == 'Yes':
                v='pain'
            if valence=='No':
                v='no_pain'

            filename = os.path.basename(row["file_name"])
            heats_paths = []
            for heat_name in self.heatmaps_names:
                heat_fname1 = filename.replace('.jpg', '_' + heat_name + '.npy')
                heatmap_path1 = os.path.join(self.heats_root, id,  v, "heats", heat_fname1)
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
            if len(heats_paths)==0:
                continue
            msks_dict_list=[]
            if valence=='Yes':
                v='pain'
            if valence=='No':
                v='no_pain'

            for seg_name in self.segs_names:
                subfolder=''
                if seg_name=='face':
                    subfolder='masks'
                msk_path = os.path.join(self.msks_root,seg_name+'_images',subfolder,v, filename)
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
    catsSegs = CatsSegs(alpha=0.8, df=df, out_sz=(28, 28), res_folder='/home/tali',
                            imgs_root='/home/tali/cats_pain_proj/face_images/masked_images/',
                            msks_root='/home/tali/cats_pain_proj/',
                            heats_root=heats_root,
                            segs_names=["face", "ears", "eyes", "mouth"], segs_max_det=[1, 2, 2, 1],
                            heatmaps_names=heats_names,
                            manip_type=manip_type)

    all_outs = catsSegs.analyze_all()
    if all_outs=={}:
        return
    #out_df_path = '/home/tali/horses/results/res25/analyze.csv'
    res_df = catsSegs.create_res_df(all_outs)
    res_df.to_csv(out_df_path)
    # analysis_df = dogSegs.analyze_df(res_df)
    # df_analysis_path = '/home/tali/dogs_annika_proj/res_25_mini_masked_all_maps/res_analysis.csv'
    # analysis_df.to_csv(df_analysis_path)
    summary_path = os.path.join(os.path.dirname(out_df_path), 'summary_' + manip_type + '.json')
    cuts_dict={'all':'all','Yes':'valence','No':'valence'}
    catsSegs.summarize_results_and_calc_qualities(res_df,cuts_dict, summary_path)
    if summary_path.endswith('power.json'):
        catsSegs.map_names_powered(summary_path, summary_path)
    return summary_path



#cam_quality_mean, cam_quality_median, cam_res = horsesSegs.calc_map_type_quality(res_df, ['face'], 'cam')
    #gcam_quality_mean, gcam_quality_median, gcam_res = horsesSegs.calc_map_type_quality(res_df, ['face'], 'grad_cam')
    #ecam_quality_mean, ecam_quality_median, ecam_res = horsesSegs.calc_map_type_quality(res_df, ['face'], 'eigen_cam')
def run_cats():
    def create_pytorch_path(type:str,ft:str, root_path:str,add:str)->str:
        out_path = os.path.join(root_path, 'pytorch_'+type+ft,add)
        return out_path

    heats_names = ['grad_cam','xgrad_cam','grad_cam_plusplus']
    root_path = '/home/tali/cats_pain_proj'
    net_types = ['vit','dino','resnet50','nest-tiny']
    #net_types = ['dino','nest-tiny']
    run_type =['']
    manip_type=['','power']
    summaries = {}
    for rt in run_type:
        for i,type in enumerate(net_types):
            for manipulation in manip_type:
                if type == 'nest-tiny':
                    inference_file = '/home/tali/cats_pain_proj/restiny25/cats_finetune_mask_25_high_lr.csv'
                    #inference_file = '/home/tali/cats_pain_proj/restiny25/cats_finetune_mask_25.csv'
                    heats_root = '/home/tali/cats_pain_proj/restiny25/'
                    out_df_path = os.path.join(heats_root,'analysis_'+manipulation+'.csv')
                else:
                    inference_file = create_pytorch_path(type,rt,root_path,'inference.csv')
                    heats_root = create_pytorch_path(type,rt,root_path,'maps')
                    out_df_path = create_pytorch_path(type, rt, root_path, 'analysis_' + manipulation + '.csv')

                jpg_files = glob.glob(os.path.join(heats_root, "**", "*.jpg"), recursive=True)

                # Delete each .jpg file
                for file_path in jpg_files:
                    os.remove(file_path)
                summary_path=calc_qualities(pd.read_csv(inference_file), heats_root,  heats_names,out_df_path, manipulation)
                if not(summary_path==None):
                    summaries[type + '_' + manipulation] = summary_path
    return summaries

def plot_cats(net_jsons:dict):
    heats_names = ['grad_cam', 'xgrad_cam', 'grad_cam_plusplus',
                   'grad_cam_power', 'xgrad_cam_power', 'grad_cam_plusplus_power']

    catsSegs = CatsSegs(alpha=0.8, df=pd.DataFrame(), out_sz=(28, 28), res_folder='/home/tali',
                            imgs_root='/home/tali/cats_pain_proj/face_images/masked_images/',
                            msks_root='/home/tali/cats_pain_proj/',
                            heats_root='',
                            segs_names=["face", "ears", "eyes", "mouth"], segs_max_det=[1, 1, 1, 1],
                            heatmaps_names=heats_names, manip_type='')
    net_colors = {'resnet50': 'red', 'vit': 'green', 'dino': 'blue', 'nest-tiny': 'orange'}
    #net_colors = {'dino': 'blue', 'nest-tiny': 'orange'}
    outdir = '/home/tali/cats_pain_proj/plots/'
    os.makedirs(outdir, exist_ok=True)
    #catsSegs.go_over_jsons_and_plot(net_colors, net_jsons, outdir, 'scaled')
    #catsSegs.go_over_jsons_and_plot(net_colors, net_jsons, outdir, 'quals')
    #net_colors = {'dino': 'blue', 'nest-tiny': 'orange'}
    catsSegs.go_over_jsons_and_plot(net_colors, net_jsons, outdir, 'seg_quals')
    #catsSegs.go_over_jsons_and_plot(net_colors, net_jsons, outdir, 'seg_scaled')

if __name__ == "__main__":
    # img_path = '/home/tali/cats_pain_proj/face_images/pain/cat_10_video_1.1.jpg'
    # msk_path = '/home/tali/cats_pain_proj/eyes_images/pain/cat_10_video_1.1.jpg'
    # plot_msk_on_img(img_path, msk_path)
    summaries = run_cats()
    plot_cats(summaries)
