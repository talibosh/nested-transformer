import pandas
from PIL import Image
import os, sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as patches
import keras
import matplotlib as mpl
import pandas as pd
from typing import List, Dict
import cv2
import segments_utils
from abc import ABC, abstractmethod
import json
from collections import OrderedDict
class AnimalSegs:
    def __init__(self, alpha: float, df: pd.DataFrame, out_sz: tuple[int, int], res_folder: str, imgs_root: str,
                 msks_root: str, heats_root: str, segs_names: list[str],segs_max_det:list[int],heatmaps_names: list[str],
                 manip_type:str):
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
        self.manip_type = manip_type

    def overlay_heatmap_img(self, img:np.array, heatmap:np.array):
        norm_map = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
        single_map = 255 * norm_map
        single_map = single_map.astype(np.uint8)
        jet_map = cv2.applyColorMap(single_map, cv2.COLORMAP_JET)
        super_imposed_map = img * 0.7 + 0.4 * jet_map
        super_imposed_map = cv2.resize(super_imposed_map, (224, 224), cv2.INTER_LINEAR)
        return super_imposed_map
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

        oias = segments_utils.OneImgAllSegs(self.alpha, img_path, segs_data, self.manip_type)
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
            #list_name_area_bb = seg_name + "_area_bb"
            for heat_name in self.heatmaps_names:
                list_name_prob = seg_name + "_prob_" + heat_name
                #list_name_cnr = seg_name + "_cnr_" + heat_name
                list_name_ng = seg_name + "_ng_" + heat_name

                #list_name_prob_bb = seg_name + "_prob_" + heat_name + "_bb"
                #list_name_cnr_bb = seg_name + "_cnr_" + heat_name + "_bb"
                #list_name_ng_bb = seg_name + "_ng_" + heat_name + "_bb"

                analyze_res_lists[list_name_prob] = []
                #analyze_res_lists[list_name_cnr] = []
                analyze_res_lists[list_name_area] = []
                analyze_res_lists[list_name_ng] = []
                #analyze_res_lists[list_name_prob_bb] = []
                #analyze_res_lists[list_name_cnr_bb] = []
                #analyze_res_lists[list_name_area_bb] = []
                #analyze_res_lists[list_name_ng_bb] = []

        for i in range(all_outs.__len__()):  # go over images
            one_img_res = all_outs[i]
            # use 1st segment (usually all face) to detect id, full_path, ...and so on
            full_path.append(one_img_res[0]["full_path"])
            img_name.append(os.path.basename(one_img_res[0]["full_path"]))
            id.append(one_img_res[0]["id"])
            if "video" in one_img_res[0]:
                video.append(one_img_res[0]["video"])
            valence.append(one_img_res[0]["valence"])
            infered_class.append(one_img_res[0]["Infered_Class"])

            for seg_idx in range(one_img_res.__len__()):  # go over segments in image
                seg_res = one_img_res[seg_idx]
                seg_name = seg_res["seg_name"]
                area_name = seg_name + "_area"
                #area_name_bb = area_name + "_bb"
                area = seg_res['areas'][0]
                #area_bb = seg_res['areas_bb'][0]
                analyze_res_lists[area_name].append(area)
                #analyze_res_lists[area_name_bb].append(area_bb)
                for heat_idx in range(self.heatmaps_names.__len__()):
                    heat_name = self.heatmaps_names[heat_idx]
                    prob = seg_res['prob_grades'][heat_idx]
                    #prob_bb = seg_res['prob_grades_bb'][heat_idx]
                    list_name_prob = seg_name + "_prob_" + heat_name
                    analyze_res_lists[list_name_prob].append(prob)
                    #list_name_prob_bb = list_name_prob + "_bb"
                    #analyze_res_lists[list_name_prob_bb].append(prob_bb)
                    #list_name_cnr = seg_name + "_cnr_" + heat_name
                    #analyze_res_lists[list_name_cnr].append(seg_res['cnrs'][heat_idx])
                    #list_name_cnr_bb = list_name_cnr + "_bb"
                    #analyze_res_lists[list_name_cnr_bb].append(seg_res['cnrs_bb'][heat_idx])
                    list_name_ng = seg_name + "_ng_" + heat_name
                    analyze_res_lists[list_name_ng].append(prob /(area+1e-10)) #avoid division by 0 if seg was not found
                    #list_name_ng_bb = list_name_ng + "_bb"
                    #analyze_res_lists[list_name_ng_bb].append(prob_bb / (area_bb+1e-10))#avoid division by 0 if seg was not found

        analyze_res_lists["id"] = id
        if video != []:
            analyze_res_lists["video"] = video
        analyze_res_lists["valence"] = valence
        analyze_res_lists["Infered_Class"] = infered_class
        analyze_res_lists["img_name"] = img_name
        analyze_res_lists["full_path"] = full_path
        df = pd.DataFrame(analyze_res_lists)

        return df

    def summarize_results_and_calc_qualities(self, res_df:pd.DataFrame,cuts_dict:dict, summary_path:str):
        #file = open(summary_path, "w")
        out_dict={}
        for key in cuts_dict:
            out_dict[key] = {}
            if key == 'all':
                res_to_chk = res_df
            else:
                res_to_chk = res_df[res_df[cuts_dict[key]] == key]
            print('**********Analyze '+ cuts_dict[key] +' = '+ key +'***************')
            #file.write('**********Analyze ' + cuts_dict[key] + ' = ' + key + '***************'+ '\n')
            for hn in self.heatmaps_names:
                res = self.calc_map_type_quality(res_to_chk, ['face'], hn)
                out_dict[key][hn] = {}
                out_dict[key][hn] = res
                #print(hn + " qual_mean:" + str(qual_mean) + " perc:" + str(perc) + " total_qual:" + str(total_qual))
                #file.write(
                #    hn + " qual_mean:" + str(qual_mean) + " perc:" + str(perc) + " total_qual:" + str(total_qual) + '\n')
                for key1, value in res.items():
                    print(f"{key1}: {value}")
                    #file.write(f"{key}: {value}\n")
        #file.close()

        file =   open(summary_path, 'w')
        json.dump(out_dict, file, indent=4)
        file.close()

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
            curr_list =np.array(df[seg_name + '_prob_' + hmn + addition].tolist())
            seg_list.append(curr_list)
            mean, std, norm_grades, norm_grades_stds = self.analyze_of_seg(seg_locs, seg_area,curr_list)
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
        total_mean_prob_of_segs=0
        total_mean_area_of_segs=0
        perc_good_grades = 0
        total_qual = 0
        total_qual_high =0
        num_of_used_segs = self.segs_names.__len__()
        res=dict()
        for seg_name in self.segs_names:
            if seg_name in segs_to_ignore:
                num_of_used_segs=num_of_used_segs-1
                continue
            seg_prob = np.array(df[seg_name+"_prob_" + map_type].tolist())
            seg_area = np.array(df[seg_name+"_area"].tolist())
            seg_relevant_locs = np.nonzero(seg_area)
            seg_ng = np.divide(seg_prob[seg_relevant_locs], seg_area[seg_relevant_locs])
            seg_prob_mean = np.mean(seg_prob[seg_relevant_locs])
            seg_area_mean = np.mean(seg_area[seg_relevant_locs])
            good_ng = seg_ng[seg_ng > 1]
            perc_good_ng = good_ng.__len__()/seg_ng.__len__()

            seg_mean_high_ng = np.mean(good_ng) if perc_good_ng>0 else 0
            seg_mean = np.mean(seg_ng)
            seg_median = np.median(seg_ng)
            res[seg_name+"_mean"]=seg_mean
            res[seg_name + "_mean_high_ng"] = seg_mean_high_ng
            #res[seg_name+"_median"] = seg_median
            res[seg_name+"_prob_mean"]=seg_prob_mean
            res[seg_name + "_area_mean"] = seg_area_mean
            res[seg_name + "_perc_good_ng"] = perc_good_ng
            res[seg_name + "_percent*ng"] = perc_good_ng*seg_mean
            res[seg_name + "_percent*high_ng"] = perc_good_ng * seg_mean_high_ng
            perc_good_grades = max(perc_good_ng,perc_good_grades)
            seg_outer_area = np.ones(seg_area[seg_relevant_locs].shape)-seg_area[seg_relevant_locs]
            seg_outer_prob = np.ones(seg_prob[seg_relevant_locs].shape)-seg_prob[seg_relevant_locs]
            seg_outer_ng = np.divide(seg_outer_prob, seg_outer_area)
            seg_outer_mean = np.mean(seg_outer_ng)
            seg_outer_median = np.median(seg_outer_ng)
            total_qual = total_qual + perc_good_ng*seg_mean
            total_qual_high = total_qual_high + perc_good_ng*seg_mean_high_ng
            total_mean = total_mean + seg_mean
            total_outer_mean = total_outer_mean + seg_outer_mean
            total_median = total_median + seg_median
            total_outer_median = total_outer_median + seg_outer_median
            total_mean_area_of_segs = total_mean_area_of_segs + seg_area_mean
            total_mean_prob_of_segs = total_mean_prob_of_segs + seg_prob_mean
        outer_mean_area = 1  - total_mean_area_of_segs
        outer_mean_prob =1 -total_mean_prob_of_segs
        outer_mean_ng = outer_mean_prob/outer_mean_area
        qual_mean = total_mean_prob_of_segs/total_mean_area_of_segs
        #quality_mean = total_mean -num_of_used_segs*outer_mean_ng #total_outer_mean
        quality_mean=0
        quality_median=0
        #quality_median = total_median - total_outer_median
        res["qual_mean"] = qual_mean
        res["max_perc_good_grades"]=perc_good_grades
        res["total_qual"]=total_qual
        res["total_qual_high"]=total_qual_high/num_of_used_segs
        ordered_data = OrderedDict()
        ordered_data["total_qual_high"] = res.pop("total_qual_high")
        ordered_data["total_qual"] = res.pop("total_qual")

        # Add the remaining items
        ordered_data.update(res)
        return ordered_data

    def create_data_from_summary_json_file(self,summary_path:str):
        with open(summary_path, 'r') as file:
            data = json.load(file) #data is dict
        #cut into list of dicts
        dicts = {}
        def change_key_names(my_dict:dict, old_key:str, new_key:str):
            try:
                my_dict[new_key] = my_dict.pop(old_key)
            except:
                print('no such key ' + old_key)
            return my_dict
        def rmv_keys(my_dict:dict, ):
            # Keys to keep
            value_to_exclude = 'face'
            # Create a set excluding the specific value
            keys_to_keep = {item for item in self.segs_names if item != value_to_exclude}
            #keys_to_keep.add('quality')
            keys_to_keep.add('quality_score')
            # Create a new dictionary with only the desired keys
            filtered_dict = {key: my_dict[key] for key in keys_to_keep if key in my_dict}
            return filtered_dict

        for idx, key in enumerate(data.keys()):
            curr_dict = data[key]
            new_dict = {}
            for hm in curr_dict.keys():
                hm_dict = curr_dict[hm]
                new_dict[hm]={}
                #hm_dict = change_key_names(hm_dict,'total_qual', 'quality')
                hm_dict = change_key_names(hm_dict, 'total_qual_high', 'quality_score')
                for seg in self.segs_names:
                    #hm_dict = change_key_names(hm_dict, seg+'_mean', seg)
                    hm_dict = change_key_names(hm_dict, seg + '_percent*high_ng', seg)

                hm_dict = rmv_keys(hm_dict)
                new_dict[hm] = hm_dict
            dicts[key] = new_dict

        return dicts

    def go_over_jsons_and_plot(self, net_colors:dict, net_jsons:dict):

        final_dicts = {}
        for key in net_jsons.keys():
            dicts = self.create_data_from_summary_json_file(net_jsons[key])
            for i,type_analyze in enumerate(dicts.keys()):
                if (type_analyze in final_dicts)==False:
                    final_dicts[type_analyze]={}
                final_dicts[type_analyze][key] =  dicts[type_analyze]

        segments_marks={ 'quality_score':'d'}
        marks_options= ['5','^','o','*']
        for i,seg in enumerate(self.segs_names):
            if seg == 'face':
                continue
            segments_marks[seg]=marks_options[i]

        for k in final_dicts.keys():
            self.plot_results(net_colors, segments_marks, final_dicts[k])


    def plot_results(self,net_colors:dict,segments_marks:dict,data:dict):
        # X-axis labels
        x_labels = self.heatmaps_names#['gc', 'xgc', 'gc++', 'pgc']

        #net_colors={'resnet50':'red', 'ViT':'green', 'ViT-dino':'blue', 'NesT':'orange'}
        #segments_marks={'eyes':'o','ears':'^','mouth':'*','quality':'2'}
        '''
        data = {'resnet50':{'grad_cam': {'ears':0.859,'eyes':0.42,'mouth':0.645,'quality':0.626},
                             'xgrad_cam':{'ears':0.859,'eyes':0.42,'mouth':0.645,'quality':0.626},
                                'grad_cam_plusplus':{'ears':0.98,'eyes':1.045,'mouth':0.928,'quality':1.382},
                                'power_grad_cam':   {'ears':1.162,'eyes':0.44,'mouth':0.566,'quality':0.798}},
                    'ViT': {'grad_cam': {'ears': 0.823, 'eyes': 1.55, 'mouth': 1.07, 'quality': 1.76},
                                 'xgrad_cam': {'ears': 1.04, 'eyes': 0.99, 'mouth': 0.924, 'quality': 1.2},
                                 'grad_cam_plusplus': {'ears': 0.98, 'eyes': 1.045, 'mouth': 0.928, 'quality': 2.07},
                                 'power_grad_cam': {'ears': 0.67, 'eyes': 0.5, 'mouth': 1.09, 'quality': 0.74}},
                    'ViT-dino': {'grad_cam': {'ears': 1.43, 'eyes': 0.95, 'mouth': 0.82, 'quality': 1.58},
                            'xgrad_cam': {'ears': 1.006, 'eyes': 1.02, 'mouth':1.129, 'quality': 1.67},
                            'grad_cam_plusplus': {'ears': 1.02, 'eyes': 2.079, 'mouth': 0.674, 'quality': 1.42},
                            'power_grad_cam': {'ears': 2.12, 'eyes': 6.35, 'mouth': 0.54, 'quality': 6.86}},
                    'NesT': {'grad_cam': {'ears': 1.161, 'eyes': 1.219, 'mouth': 1.059, 'quality': 3.157},
                                 'xgrad_cam': {'ears': 1.161, 'eyes': 1.219, 'mouth': 1.059, 'quality': 3.157},
                                 'grad_cam_plusplus': {'ears': 1.161, 'eyes': 1.219, 'mouth': 1.059, 'quality': 3.157},
                                 'power_grad_cam': {'ears': 1.161, 'eyes': 1.219, 'mouth': 1.059, 'quality': 3.157}}
                }

            '''
            # Colors and markers for the groups
            #colors = ['red', 'green', 'blue', 'orange'] #every net has a color
            #markers = ['o', '^', 's', '*']#every

            # Create figure and axes
        fig, ax = plt.subplots()

            # Set the limits of the axes
        ax.set_xlim(0, x_labels.__len__()+2)
        ax.set_ylim(0,10)

            # Set x-ticks and labels
        ax.set_xticks(range(1,1+len(x_labels)))
        ax.set_xticklabels(x_labels,rotation = 0)

        #ax.set_yticks(np.arange(0, 2, 0.1))

        for index, netType in enumerate(data.keys()):
            color = net_colors[netType]
            net_data = data[netType]
            for x_tick,heatmap in enumerate(net_data.keys()):
                hm_data = net_data[heatmap]
                for i, segment in enumerate(hm_data.keys()):
                    marker = segments_marks[segment]
                    ax.scatter(x_tick+1, hm_data[segment], edgecolors=color,facecolors='none', marker=marker, s=100)

            # Plot the data
            #for i in range(4):  # 4 groups
            #    for j in range(4):  # 4 numbers in each group
            #        ax.scatter(i, data[i, j], color=colors[i], marker=markers[j], s=100)

            # Adding legend
        for i, seg in enumerate(segments_marks.keys()):
            ax.scatter([], [], edgecolors='k',facecolors='none', marker=segments_marks[seg], s=100, label=seg)
        for i, net in enumerate(net_colors.keys()):
            ax.scatter([], [], color=net_colors[net], marker='s', s=100, label=net)

            #for i, marker in enumerate(markers):
            #    ax.scatter([], [], color='k', marker=marker, s=100, label=f'Shape {marker}')
            #for i, color in enumerate(colors):
            #    ax.scatter([], [], color=color, marker='o', s=100, label=f'Group {color}')

        ax.legend(loc='upper right')

            # Set labels
        ax.set_xlabel('heat maps types')
        ax.set_ylabel('mean normalyzed grades and qualities')

            # Display the plot
        plt.title('Summary')
        plt.grid(True)
        plt.show()



if __name__ == "__main__":
    6
    # img_path = '/home/tali/cats_pain_proj/face_images/pain/cat_10_video_1.1.jpg'
    # msk_path = '/home/tali/cats_pain_proj/eyes_images/pain/cat_10_video_1.1.jpg'
    # plot_msk_on_img(img_path, msk_path)

