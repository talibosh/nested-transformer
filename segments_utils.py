
from PIL import Image
import os, sys
import numpy as np
from matplotlib import pyplot as plt
import keras
import matplotlib as mpl
import pandas as pd

class one_segment_one_heatmap_calc:
  def __init__(self, msk_path:str, outSz = (224,224)):
    if os.path.isfile(msk_path) == False:
      self.orig_msk = []
      return
    self.orig_msk = Image.open(msk_path)
    self.outSz = outSz
    self.rszd_msk, self.np_msk = self.rszNconvet2NP(self.orig_msk, outSz)

  def rszNconvet2NP(self, orig_msk:Image, out_sz = (224,224)):
    rszd_msk = orig_msk.resize((out_sz[1], out_sz[0]), resample=Image.NEAREST)
    thresh = np.median(np.unique(rszd_msk))
    np_msk = np.array(rszd_msk)
    np_msk[np_msk < thresh] = 0
    np_msk[np_msk >= thresh] = 1
    return rszd_msk, np_msk

  def calc_relevant_heat(self, heatmap: np.array):
    rszd_heat = np.resize(heatmap, self.outSz)
    relevant_heat = rszd_heat * self.np_msk
    return relevant_heat, rszd_heat

  def calc_grade_sums_by_seg(self, relevant_heat: np.array, rszd_heat: np.array):
    grade_sum = np.sum(relevant_heat)
    grade_ratio = np.sum(self.np_msk) / (self.outSz[0] * self.outSz[1])
    grade_normalized = grade_sum / grade_ratio
    rest_img_grade = (np.sum(rszd_heat) - grade_sum) / (1-grade_ratio)
    mean_relevant_heat = np.mean(relevant_heat)
    var_relevant_heat = np.var(relevant_heat)
    rest_img_map = rszd_heat-relevant_heat
    mean_rest_of_img = np.mean(rest_img_map)
    var_rest_of_img = np.var(rest_img_map)
    cnr = np.abs(mean_relevant_heat-mean_rest_of_img)/np.sqrt(var_relevant_heat+var_rest_of_img)

    return grade_normalized, rest_img_grade, cnr




class WorkWithSegs:
  def __init__(self, alpha:float, imgs_root_folder:str,  msks_folder: str, heats_folder: str, res_folder:str, df: pd.DataFrame, out_sz=(224,224)):
    self.df = df
    self.alpha = alpha
    self.msks_folder = msks_folder
    self.out_sz = out_sz
    self.res_folder = res_folder
    self.heats_folder = heats_folder
    self.imgs_root_folder = imgs_root_folder
    ids = df['CatId'].tolist()
    self.unique_ids = np.unique(ids)
    classes = df['Valence'].tolist()
    self.unique_classes = np.unique(classes)
    self.num_classes = self.unique_classes.__len__()
    self.gn3 = []
    self.gn2 = []
    self.gnb = []
    self.rig3 = []
    self.rig2 = []
    self.rigb = []
    self.cnr3 = []
    self.cnr2 = []
    self.cnrb = []
    self.full_pths=[]
    self.valence=[]
    self.id=[]
    self.img_name=[]

  def create_both_heatmap(self, hm3, hm2):
    hmb = hm3*self.alpha + hm2*(1-self.alpha)
    return hmb
  def get_df_for_analyze(self,  id: int, cls: int):
    eval_df = self.df[self.df["CatId"] == id]
    eval = eval_df[eval_df["Valence"] == cls]
    eval = eval[eval["Infered_Class"] == cls]
    return eval

  def get_msk_for_img(self, img_path: str):
    msk_path = img_path.replace(self.imgs_root_folder, self.msks_folder)
    osh = one_segment_one_heatmap_calc(msk_path, self.out_sz)
    if osh.orig_msk == []:
      return []
    else:
      return osh

  def get_heatmap_for_img(self, img_path: str, heatmap_name: str, id: int, valence: int):
    head, tail = os.path.split(img_path)
    valence_name = 'pain'
    if valence == 0:
      valence_name = 'no pain'
    heat_maps_loc = os.path.join(self.heats_folder,str(id), valence_name, 'heats', tail)
    #heat_maps_loc = head.replace(self.imgs_root_folder, self.heats_folder)
    ff = heat_maps_loc.replace('.jpg','_'+heatmap_name+'.npy')
    heatmap = np.load(ff)
    rszd_heatmap = np.resize(heatmap, self.out_sz)
    return rszd_heatmap

  def analyze_img(self, img_path: str, osh:one_segment_one_heatmap_calc, id:int, valence: int):
    hm3 = self.get_heatmap_for_img(img_path, '3',  id= id, valence =valence)
    hm2 = self.get_heatmap_for_img(img_path, '2',  id= id, valence =valence)
    hm_both = self.create_both_heatmap(hm3, hm2)
    relevant_heat3, rszd_heat3 = osh.calc_relevant_heat(hm3)
    grade_normalized3, rest_img_grade3, cnr3 = osh.calc_grade_sums_by_seg(relevant_heat3, rszd_heat3)
    relevant_heat2, rszd_heat2 = osh.calc_relevant_heat(hm2)
    grade_normalized2, rest_img_grade2, cnr2 = osh.calc_grade_sums_by_seg(relevant_heat2, rszd_heat2)
    relevant_heatb, rszd_heatb = osh.calc_relevant_heat(hm_both)
    grade_normalized_b, rest_img_grade_b, cnrb = osh.calc_grade_sums_by_seg(relevant_heatb, rszd_heatb)
    return [grade_normalized3, grade_normalized2, grade_normalized_b], [rest_img_grade3, rest_img_grade2, rest_img_grade_b], [cnr3, cnr2, cnrb]

  def analyze_imgs_list(self, imgs_list, valence: int, id: int):
    for img_pth in imgs_list:
      osh = self.get_msk_for_img(img_pth)
      if osh == []:
        continue
      grade_normalized, rest_img_grades, cnrs = self.analyze_img(img_pth, osh, id=id, valence=valence)
      self.rig3.append(rest_img_grades[0])
      self.rig2.append(rest_img_grades[1])
      self.rigb.append(rest_img_grades[2])
      self.gn3.append(grade_normalized[0])
      self.gn2.append(grade_normalized[1])
      self.gnb.append(grade_normalized[2])
      self.cnr3.append(cnrs[0])
      self.cnr2.append(cnrs[1])
      self.cnrb.append(cnrs[2])
      self.full_pths.append(img_pth)
      self.valence.append(valence)
      self.id.append(id)
      self.img_name.append(os.path.basename(img_pth))

  def analyze_eval_df(self, eval_df: pd.DataFrame):
    imgs_pths = eval_df["FullPath"].tolist()
    if imgs_pths == []:
      return
    self.analyze_imgs_list(imgs_pths, eval_df["Valence"].tolist()[0], eval_df["CatId"].tolist()[0] )

  def create_res_df(self, out_path: str):
    df = pd.DataFrame({'Id': self.id, 'Filename': self.img_name,
                       'FullPath': self.full_pths,'Valence': self.valence,
                       'normGrade3': self.gn3, 'normGrade2': self.gn2,'normGradeBoth': self.gnb,
                       'restOfImgGrade3': self.rig3, 'restOfImgGrade2': self.rig2, 'restOfImgGradeBoth': self.rigb,
                       'cnr3': self.cnr3, 'cnr2':self.cnr2, 'cnrb':self.cnrb})
    df.to_csv(out_path)

  def analyze_all(self):
    for id in self.unique_ids:
      print('*******' + str(id) + '*******')
      for cls in self.unique_classes:
        eval_df = self.get_df_for_analyze(id, cls)
        self.analyze_eval_df(eval_df)

def grade_maps_metrices(df: pd.DataFrame):
  normGrade3 = np.array(df['normGrade3'].tolist())
  normGrade2 = np.array(df['normGrade2'].tolist())
  normGradeBoth = np.array(df['normGradeBoth'].tolist())
  restOfImgGrade3 = np.array(df['restOfImgGrade3'].tolist())
  restOfImgGrade2 = np.array(df['restOfImgGrade2'].tolist())
  restOfImgGradeBoth = np.array(df['restOfImgGradeBoth'].tolist())
  res3 = normGrade3-restOfImgGrade3
  res2 = normGrade2-restOfImgGrade2
  resBoth = normGradeBoth-restOfImgGradeBoth
  r3 = (res3 > 0)
  badloc3 = np.where(r3 == False)[0]
  r2 = (res2 > 0)
  badloc2 = np.where(r2 == False)[0]
  rb = (resBoth > 0)
  badlocb = np.where(rb == False)[0]
  return r3, r2, rb


if __name__ == "__main__":
  #df = pd.read_csv("/home/tali/cats_pain_proj/face_images/cats_norm1_infered60.csv")
  #wws = WorkWithSegs(alpha = 0.7, imgs_root_folder = '/home/tali/cats_pain_proj/face_images/',
  #                     msks_folder = '/home/tali/cats_pain_proj/face_images/masks/' ,
  #                     heats_folder = '/home/tali/trials/cats_bb_res_test60/', res_folder='',
  #                     df = df, out_sz=(32, 32))
  #wws.analyze_all()
  #wws.create_res_df('/home/tali/cats_pain_proj/face_images/grade_map_analyze.csv')
  df = pd.read_csv('/home/tali/cats_pain_proj/face_images/grade_map_analyze.csv')
  r3, r2, rb = grade_maps_metrices(df)

















