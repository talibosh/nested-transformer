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
            list_name_area_pixels = seg_name + "_area_pixels"
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
                analyze_res_lists[list_name_area_pixels] = []
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
                area_pixels_name = seg_name + "_area_pixels"
                #area_name_bb = area_name + "_bb"
                area = seg_res['areas'][0]
                area_pixels = seg_res['areas_pixels'][0]
                #area_bb = seg_res['areas_bb'][0]
                analyze_res_lists[area_name].append(area)
                analyze_res_lists[area_pixels_name].append(area_pixels)
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

    def calc_measures(self, df:pandas.DataFrame, segs_to_ignore:list[str], map_type:str):
        res = dict()
        data_for_analysis = dict()
        for seg_name in self.segs_names:
            if seg_name in segs_to_ignore:
                num_of_used_segs = num_of_used_segs - 1
                continue
            seg_prob = np.array(df[seg_name + "_prob_" + map_type].tolist())
            seg_area = np.array(df[seg_name + "_area"].tolist())
            seg_relevant_locs = np.nonzero(seg_area)
            seg_ng = np.divide(seg_prob, seg_area+0.000001)
            seg_prob_mean = np.mean(seg_prob[seg_relevant_locs])
            seg_area_mean = np.mean(seg_area[seg_relevant_locs])
            #good_ng = seg_ng[seg_ng > 1]
            #perc_good_ng = good_ng.__len__() / seg_ng.__len__()
            data_for_analysis[seg_name+'_prob'] = seg_prob
            data_for_analysis[seg_name + '_area'] = seg_area
            data_for_analysis[seg_name + '_ng'] = seg_ng
            data_for_analysis[seg_name + '_relevant_locs'] = seg_relevant_locs
            data_for_analysis[seg_name + '_prob_mean'] = seg_prob_mean
            data_for_analysis[seg_name + '_area_mean'] = seg_area_mean
            seg_prob_with_mean = seg_area[seg_area == 0] = seg_prob_mean
            seg_area_with_mean = seg_area[seg_area == 0] = seg_area_mean
            seg_ng_with_mean = np.divide(seg_prob_with_mean, seg_area_with_mean)
            data_for_analysis[seg_name + '_prob_with_mean'] = seg_prob_with_mean
            data_for_analysis[seg_name + '_area_with_mean'] = seg_area_with_mean
            data_for_analysis[seg_name + '_ng_with_mean'] = seg_ng_with_mean

        # find out number of relevant images



    def calc_map_type_quality(self, df:pandas.DataFrame, segs_to_ignore:list[str], map_type:str):
        total_mean =0
        #total_outer_mean = 0
        total_median = 0
        #total_outer_median = 0
        total_mean_prob_of_segs=0
        total_mean_area_of_segs=0
        perc_good_grades = 0
        total_qual = 0
        total_qual_high =0
        total_scaled = 0
        total_scaled_high = 0
        num_of_used_segs = self.segs_names.__len__()
        res=dict()
        means=[]
        #find mean of areas
        for seg_name in self.segs_names:
            if seg_name in segs_to_ignore:
                num_of_used_segs=num_of_used_segs-1
                continue
            areas = np.array(df[seg_name + "_area"].tolist())
            areas = areas[areas>0]
            means.append(np.mean(areas))
        min_value = min(means)
        max_value = max(means)
        restrain_factor = 1#1-(min_value/max_value)


        for seg_name in self.segs_names:
            if seg_name in segs_to_ignore:
                num_of_used_segs=num_of_used_segs-1
                continue
            seg_prob = np.array(df[seg_name+"_prob_" + map_type].tolist())
            seg_area = np.array(df[seg_name+"_area"].tolist())
            seg_area_pixels = np.array(df[seg_name + "_area_pixels"].tolist())
            seg_relevant_locs = np.nonzero(seg_area)
            seg_ng = np.divide(seg_prob[seg_relevant_locs], seg_area[seg_relevant_locs])
            good_ng = seg_ng[seg_ng > 1]
            perc_good_ng = good_ng.__len__() / seg_ng.__len__()
            seg_ng0 = np.divide(seg_prob[seg_relevant_locs], np.power(seg_area[seg_relevant_locs],restrain_factor))
            seg_prob_density = np.divide(seg_prob[seg_relevant_locs], np.power(seg_area_pixels[seg_relevant_locs],restrain_factor))
            seg_scaled_grade = seg_prob_density*100
            seg_prob_mean = np.mean(seg_prob[seg_relevant_locs])
            seg_area_mean = np.mean(seg_area[seg_relevant_locs])
            seg_area_pixels_mean = np.mean(seg_area_pixels[seg_relevant_locs])
            seg_scaled_grade_mean = np.mean(seg_scaled_grade)

            #seg_mean_high_ng = np.mean(good_ng) if perc_good_ng>0 else 0
            seg_mean_high_ng = np.mean(seg_ng0[seg_ng > 1]) if perc_good_ng>0 else 0
            seg_mean_high_scaled_grade = np.mean(seg_scaled_grade[seg_ng > 1]) if perc_good_ng>0 else 0
            seg_mean = np.mean(seg_ng0)
            seg_mean_scaled = np.mean(seg_scaled_grade)
            seg_median = np.median(seg_ng)
            res[seg_name+"_mean"]=seg_mean
            res[seg_name + "_mean_high_ng"] = seg_mean_high_ng
            res[seg_name + "_mean_scaled"] = seg_mean_scaled
            res[seg_name + "_mean_high_scaled"] = seg_mean_high_scaled_grade
            #res[seg_name+"_median"] = seg_median
            res[seg_name+"_prob_mean"]=seg_prob_mean
            res[seg_name + "_area_mean"] = seg_area_mean
            res[seg_name + "_area_pixels_mean"] = seg_area_pixels_mean
            res[seg_name + "_perc_good_ng"] = perc_good_ng
            res[seg_name + "_percent*ng"] = perc_good_ng*seg_mean
            res[seg_name + "_percent*high_ng"] = perc_good_ng * seg_mean_high_ng
            res[seg_name + "_percent*high_scaled_grade"] = perc_good_ng * seg_mean_high_scaled_grade
            perc_good_grades = max(perc_good_ng,perc_good_grades)
            #seg_outer_area = np.ones(seg_area[seg_relevant_locs].shape)-seg_area[seg_relevant_locs]
            #seg_outer_prob = np.ones(seg_prob[seg_relevant_locs].shape)-seg_prob[seg_relevant_locs]
            #seg_outer_ng = np.divide(seg_outer_prob, seg_outer_area)
            #seg_outer_mean = np.mean(seg_outer_ng)
            #seg_outer_median = np.median(seg_outer_ng)
            total_qual = total_qual + perc_good_ng*seg_mean
            total_scaled = total_scaled + perc_good_ng * seg_scaled_grade_mean
            total_qual_high = total_qual_high + perc_good_ng*seg_mean_high_ng
            total_scaled_high = total_scaled_high + perc_good_ng * seg_mean_high_scaled_grade
            total_mean = total_mean + seg_mean
            #total_outer_mean = total_outer_mean + seg_outer_mean
            total_median = total_median + seg_median
            #total_outer_median = total_outer_median + seg_outer_median
            total_mean_area_of_segs = total_mean_area_of_segs + seg_area_mean
            total_mean_prob_of_segs = total_mean_prob_of_segs + seg_prob_mean
        #outer_mean_area = 1  - total_mean_area_of_segs
        #outer_mean_prob =1 -total_mean_prob_of_segs
        #outer_mean_ng = outer_mean_prob/outer_mean_area
        qual_mean = total_mean_prob_of_segs/total_mean_area_of_segs
        #quality_mean = total_mean -num_of_used_segs*outer_mean_ng #total_outer_mean
        quality_mean=0
        quality_median=0
        #quality_median = total_median - total_outer_median
        res["qual_mean"] = qual_mean
        res["max_perc_good_grades"]=perc_good_grades
        res["total_qual"]=total_qual
        res["total_qual_high"]=total_qual_high/num_of_used_segs
        res["total_scaled"] = total_scaled
        res["total_scaled_high"] = total_scaled_high / num_of_used_segs
        ordered_data = OrderedDict()
        ordered_data["total_qual_high"] = res.pop("total_qual_high")
        ordered_data["total_qual"] = res.pop("total_qual")
        ordered_data["total_scaled"] = res.pop("total_scaled")
        ordered_data["total_scaled_high"] = res.pop("total_scaled_high")

        # Add the remaining items
        ordered_data.update(res)
        return ordered_data

    def rmv_keys(self, my_dict: dict, keys_to_keep):
        # Keys to keep
        #value_to_exclude = 'face'

        # Create a new dictionary with only the desired keys
        filtered_dict = {key: my_dict[key] for key in keys_to_keep if key in my_dict}
        return filtered_dict

    def create_data_from_summary_json_file(self,summary_path:str, plot_type:str):
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

        value_to_exclude = 'face'
        if plot_type == 'quals':
            keys_to_keep={'quality_score'}
        elif plot_type == 'scaled':
            keys_to_keep = {'scaled_score'}
        else:
            keys_to_keep = {item for item in self.segs_names if item != value_to_exclude}
        # keys_to_keep.add('quality')

        for idx, key in enumerate(data.keys()):
            curr_dict = data[key]
            new_dict = {}
            for hm in curr_dict.keys():
                hm_dict = curr_dict[hm]
                new_dict[hm]={}
                #hm_dict = change_key_names(hm_dict,'total_qual', 'quality')
                hm_dict = change_key_names(hm_dict, 'total_qual_high', 'quality_score')
                hm_dict = change_key_names(hm_dict, 'total_scaled_high', 'scaled_score')
                for seg in self.segs_names:
                    #hm_dict = change_key_names(hm_dict, seg+'_mean', seg)
                    if plot_type=='seg_quals':
                        hm_dict = change_key_names(hm_dict, seg + '_percent*high_ng', seg)
                    if plot_type == 'seg_scaled':
                        hm_dict = change_key_names(hm_dict, seg + '_percent*high_scaled_grade', seg)
                # Create a set excluding the specific value

                hm_dict = self.rmv_keys(hm_dict,keys_to_keep)
                new_dict[hm] = hm_dict
            dicts[key] = new_dict

        return dicts

    def go_over_jsons_and_plot(self, net_colors:dict, net_jsons:dict,outdir:str,plot_type:str):

        final_dicts = {}
        keys2chk = list(net_jsons.keys())
        #if plot_type == 'seg_quals' or plot_type == 'seg_scaled':
        #    keys2chk = [key for key in keys2chk if key not in ['vit_', 'vit_power', 'resnet50_', 'resnet50_power']]

        for key in keys2chk:
            dicts = self.create_data_from_summary_json_file(net_jsons[key],plot_type)
            for i,type_analyze in enumerate(dicts.keys()):
                if (type_analyze in final_dicts)==False:
                    final_dicts[type_analyze]={}
                final_dicts[type_analyze][key] =  dicts[type_analyze]

        segments_marks = {}
        if plot_type == 'quals':
            segments_marks={ 'quality_score':'s'}#, 'scaled_score': 's'}
        elif plot_type == 'scaled':
            segments_marks={ 'scaled_score':'s'}#, 'scaled_score': 's'}
        else:
            marks_options= ['5','^','o','*']
            for i,seg in enumerate(self.segs_names):
                if seg == 'face':
                    continue
                segments_marks[seg]=marks_options[i]

        for k in final_dicts.keys():
            plot_name = plot_type + '_' + k
            self.plot_results(net_colors, segments_marks, final_dicts[k],outdir,plot_name)


    def plot_results(self,net_colors:dict,segments_marks:dict,data:dict,outpath:str,plot_type:str):
        # X-axis labels
        x_labels = self.heatmaps_names#['gc', 'xgc', 'gc++', 'pgc']

        #net_colors={'resnet50':'red', 'ViT':'green', 'ViT-dino':'blue', 'NesT':'orange'}
        #segments_marks={'eyes':'o','ears':'^','mouth':'*','quality':'2'}

            # Colors and markers for the groups
            #colors = ['red', 'green', 'blue', 'orange'] #every net has a color
            #markers = ['o', '^', 's', '*']#every

            # Create figure and axes
        fig, ax = plt.subplots(figsize=(10, 6))

            # Set the limits of the axes
        ax.set_xlim(0, x_labels.__len__()+2)
        if plot_type.startswith('quals'):
            ax.set_ylim(0,2.5)
        if plot_type.startswith('scaled'):
            ax.set_ylim(0, 2.5)
        if plot_type.startswith('seg_quals'):
            ax.set_ylim(0, 5)
        if plot_type.startswith('seg_scaled'):
            ax.set_ylim(0, 2)

            # Set x-ticks and labels
        ax.set_xticks(range(1,1+len(x_labels)))
        ax.tick_params(axis='x', labelsize=7)
        ax.set_xticklabels(x_labels,rotation = 7)

        #ax.set_yticks(np.arange(0, 2, 0.1))
        def remove_last_underscore(input_string):
            # Find the position of the last underscore
            last_underscore_index = input_string.rfind('_')

            # If an underscore is found, slice the string up to that point
            if last_underscore_index != -1:
                return input_string[:last_underscore_index]
            else:
                # If no underscore is found, return the original string
                return input_string

        for index, netType in enumerate(data.keys()):
            curr_net = remove_last_underscore(netType)
            color = net_colors[curr_net]
            net_data = data[netType]
            for x_tick,heatmap in enumerate(net_data.keys()):
                hm_data = net_data[heatmap]
                for i, segment in enumerate(hm_data.keys()):
                    marker = segments_marks[segment]
                    ax.scatter(self.heatmaps_names.index(heatmap)+1, hm_data[segment], edgecolors=color,facecolors='none', marker=marker, s=100)

            # Plot the data
            #for i in range(4):  # 4 groups
            #    for j in range(4):  # 4 numbers in each group
            #        ax.scatter(i, data[i, j], color=colors[i], marker=markers[j], s=100)

            # Adding legend
        segs_name = {'top': 'Ears', 'middle': 'Eyes', 'bottom': 'Muzzles',
                     'ear': 'Ears','eye':'Eyes', 'ears':'Ears', 'eyes':'Eyes','mouth':'Mouth'}
        for i, seg in enumerate(segments_marks.keys()):
            if not(seg=='quality_score') and not(seg=='scaled_score') :
                ax.scatter([], [], edgecolors='k',facecolors='none', marker=segments_marks[seg], s=100, label=segs_name[seg])

            #if not(seg=='scaled_score'):
            #    ax.scatter([], [], edgecolors='k',facecolors='none', marker=segments_marks[seg], s=100, label=segs_name[seg])

        net_name={'resnet50':'ResNet50', 'vit':'ViT','dino':'ViT-dino','nest-tiny':'NesT'}
        for i, net in enumerate(net_colors.keys()):
            ax.scatter([], [], color=net_colors[net], marker='s', s=100, label=net_name[net])

        #for i, marker in enumerate(markers):
         #   ax.scatter([], [], color='k', marker=marker, s=100, label=f'Shape {marker}')
            #for i, color in enumerate(colors):
            #    ax.scatter([], [], color=color, marker='o', s=100, label=f'Group {color}')

        ax.legend(loc='upper right')

            # Set labels
        ax.set_xlabel('heat map type')
        ax.set_ylabel('segs quality score')

            # Display the plot
        #plt.title('Summary')
        plt.grid(True)
        plt.savefig(os.path.join(outpath, plot_type+'.jpg'))
        plt.show()
    def map_names_powered(self,summary1:str, summary2:str):
        heats_mapping = {'grad_cam': 'grad_cam_power', 'xgrad_cam': 'xgrad_cam_power',
                         'grad_cam_plusplus': 'grad_cam_plusplus_power', 'power_grad_cam': 'power_grad_cam_power'}
        with open(summary1, 'r') as file:
            data1 = json.load(file)

        def rename_fields(data, field_mapping):
            if isinstance(data, list):
                return [rename_fields(item, field_mapping) for item in data]
            elif isinstance(data, dict):
                return {field_mapping.get(key, key): rename_fields(value, field_mapping) for key, value in data.items()}
            else:
                return data

        updated_data = rename_fields(data1, heats_mapping)
        with open(summary2, 'w') as file:
            json.dump(updated_data, file, indent=4)


if __name__ == "__main__":
    6
    # img_path = '/home/tali/cats_pain_proj/face_images/pain/cat_10_video_1.1.jpg'
    # msk_path = '/home/tali/cats_pain_proj/eyes_images/pain/cat_10_video_1.1.jpg'
    # plot_msk_on_img(img_path, msk_path)

