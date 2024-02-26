
from PIL import Image
import os, sys
import numpy as np
from matplotlib import pyplot as plt
import keras
import matplotlib as mpl
import pandas as pd
from typing import List, Dict

class OneSegOneHeatmapCalc:
  def __init__(self, msk_path : str, num_of_shows : int = 1, outSz = (224,224)):
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
    #normalize heatmap - values are [0-1] and sums up to 1
    rszd_heat = (rszd_heat - rszd_heat.min())/(rszd_heat.max() - rszd_heat.min())
    rszd_heat = rszd_heat/np.sum(rszd_heat)
    relevant_heat = rszd_heat * self.np_msk
    return relevant_heat, rszd_heat

  def calc_grade_by_seg(selfself, relevant_heat: np.array, rszd_heat: np.array):
    prob_grade = np.sum(relevant_heat)
    mean_relevant_heat = np.mean(relevant_heat)
    var_relevant_heat = np.var(relevant_heat)
    rest_img_map = rszd_heat - relevant_heat
    mean_rest_of_img = np.mean(rest_img_map)
    var_rest_of_img = np.var(rest_img_map)
    cnr = np.abs(mean_relevant_heat - mean_rest_of_img) / np.sqrt(var_relevant_heat + var_rest_of_img)
    return prob_grade, cnr

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
    grade_normalized = mean_relevant_heat * (self.outSz[0] * self.outSz[1])
    rest_img_grade = mean_rest_of_img * (self.outSz[0] * self.outSz[1])
    return grade_normalized, rest_img_grade, grade_normalized-rest_img_grade, cnr


class OneImgOneSeg:

  def __init__(self, msk_path: str, img_path: str, heatmap_paths: list[str], max_segs_num: int, out_sz = (224,224)):
    self.msk_path = msk_path
    self.img_path = img_path
    self.heatmap_paths = heatmap_paths
    self.outsz = out_sz
    self.max_segs_num = max_segs_num
  #def __init__(self, msks_folder: str, imgs_root_folder: str,  heats_folder: str, res_folder: str, in_df: pd.DataFrame, outsz =(224, 224)):
  #  self.in_df = in_df
  #  self.msks_folder = msks_folder
  #  self.res_folder = res_folder
  #  self.heats_folder = heats_folder
  #  self.imgs_root_folder = imgs_root_folder
  #  ids = df['CatId'].tolist()
  #  self.unique_ids = np.unique(ids)
  #  classes = df['Valence'].tolist()
  #  self.unique_classes = np.unique(classes)
  #  self.num_classes = self.unique_classes.__len__()
  #  self.out_sz = outsz

  def create_both_heatmap(self, hm3: np.array, hm2: np.array, alpha: float):
    assert(hm3.shape == hm2.shape)
    hmb = hm3*alpha + hm2*(1-alpha)
    hmb = (hmb-hmb.min()) / (hmb.max()-hmb.min())
    return hmb

  def create_both_dup_heatmap(self, hm3, hm2):
    assert (hm3.shape == hm2.shape)
    hmb = hm3 * hm2
    hmb = (hmb - hmb.min()) / (hmb.max() - hmb.min())
    return hmb

  #def get_df_for_analyze(self,  id: int, cls: int):
  #  eval_df = self.in_df[self.in_df["CatId"] == id]
  #  eval = eval_df[eval_df["Valence"] == cls]
  #  eval = eval[eval["Infered_Class"] == cls]
  #  return eval

  def get_msk_for_img(self):
    #msk_path = img_path.replace(self.imgs_root_folder, self.msks_folder)
    osh = OneSegOneHeatmapCalc(self.msk_path, self.max_segs_num, self.outsz)
    if osh.orig_msk == []:
      return []
    else:
      return osh

  def get_one_heatmap_for_img(self, heatmap_path: str):
    heatmap = np.load(heatmap_path)
    rszd_heatmap = np.resize(heatmap, self.out_sz)
    # normalize resized image
    rszd_heatmap = (rszd_heatmap - rszd_heatmap.min()) / (rszd_heatmap.max() - rszd_heatmap.min())
    rszd_heatmap = rszd_heatmap/np.sum(rszd_heatmap)
    return rszd_heatmap


  #def get_heatmap_for_img(self, img_path: str, heatmap_name: str, id: int, valence: int):
  #  head, tail = os.path.split(img_path)
  #  valence_name = 'pain'
  #  if valence == 0:
  #    valence_name = 'no pain'
  #  heat_maps_loc = os.path.join(self.heats_folder, str(id), valence_name, 'heats', tail)
  #  # heat_maps_loc = head.replace(self.imgs_root_folder, self.heats_folder)
  #  ff = heat_maps_loc.replace('.jpg', '_' + heatmap_name + '.npy')
  #  heatmap = np.load(ff)
  #  rszd_heatmap = np.resize(heatmap, self.out_sz)
  #  # normalize resized image
  #  rszd_heatmap = (rszd_heatmap - rszd_heatmap.min()) / (rszd_heatmap.max() - rszd_heatmap.min())
  #  rszd_heatmap = rszd_heatmap/np.sum(rszd_heatmap)
  #  return rszd_heatmap

  def analyze_img(self):
    osh = self.get_msk_for_img()
    if osh == []:
      return [], []
    hm3 = self.get_heatmap_for_img(self.heatmap_paths[0])
    hm2 = self.get_heatmap_for_img(self.heatmap_paths[1])
    hm_both = self.create_both_heatmap(hm3, hm2)
    hm_bothd = self.create_both_dup_heatmap(hm3, hm2)
    relevant_heat3, rszd_heat3 = osh.calc_relevant_heat(hm3)
    prob_grade3, cnr3 = osh.calc_grade_by_seg(relevant_heat3, rszd_heat3)
    relevant_heat2, rszd_heat2 = osh.calc_relevant_heat(hm2)
    prob_grade2, cnr2 = osh.calc_grade_by_seg(relevant_heat2, rszd_heat2)
    relevant_heatb, rszd_heatb = osh.calc_relevant_heat(hm_both)
    prob_gradeb, cnrb = osh.calc_grade_by_seg(relevant_heatb, rszd_heatb)
    relevant_heatd, rszd_heatd = osh.calc_relevant_heat(hm_bothd)
    prob_graded, cnrd = osh.calc_grade_by_seg(relevant_heatd, rszd_heatd)
    return [prob_grade3, prob_grade2, prob_gradeb, prob_graded], [cnr3, cnr2, cnrb, cnrd]

class OneImgAllSegs:
  def __init__(self, alpha:float, img_path: str, segs_data:list[{'seg_name':str, 'instances_num':int, 'msk_path':str, 'heats_list':list[str]}, 'outSz':tuple[int,int]]):
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
      oios = OneImgOneSeg(msk_path, self.img_path, heats_list,max_segs_num, out_sz)
      prob_grades, cnrs = oios.analyze_img()
      outs[i]['seg_data']=seg_data
      outs[i]['prob_grades'] = prob_grades
      outs[i]['cnrs'] = cnrs
      i = i+1
    return outs


class CatsSegs:
  def __init__(self, alpha: float, df: pd.DataFrame, out_sz: tuple[int, int], res_folder: str, imgs_root: str, msks_root: str, heats_root: str):
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
    if img_path.find('no pain') or img_path.find('no_pain'):
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
    heats_list = [ heat3, heat2]
    seg_face={'seg_name':'face','instances_num':1,'msk_path':face_msk_path,'heats_list':heats_list,'outSz':(224,224)}
    seg_ears={'seg_name':'ears','instances_num':2,'msk_path':ears_msk_path,'heats_list':heats_list,'outSz': (224, 224)}
    seg_eyes={'seg_name':'eyes','instances_num':2,'msk_path':eyes_msk_path,'heats_list': heats_list,'outSz': (224, 224)}
    seg_mouth={'seg_name':'mouth','instances_num':1,'msk_path':mouth_msk_path,'heats_list': heats_list,'outSz': (224, 224)}
    segs_data=[seg_face, seg_ears, seg_eyes, seg_mouth]
    oias = OneImgAllSegs(self.alpha, img_full_path, segs_data)
    outs = oias.analyze_img()
    return outs

  def analyze_img_lists(self, imgs_paths: list[str]):
    for img_full_path in imgs_paths:
      self.analyze_one_img(img_full_path)

  def analyze_all(self):
    imgs_paths = self.df['FullPath'].tolist()
    self.analyze_img_lists(imgs_paths)


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
    self.gnbd = []
    self.rig3 = []
    self.rig2 = []
    self.rigb = []
    self.rigbd = []
    self.cnr3 = []
    self.cnr2 = []
    self.cnrb = []
    self.cnrbd = []
    self.full_pths=[]
    self.valence=[]
    self.id=[]
    self.img_name=[]

  def create_both_heatmap(self, hm3, hm2):
    hmb = hm3*self.alpha + hm2*(1-self.alpha)
    hmb = (hmb-hmb.min()) / (hmb.max()-hmb.min())
    return hmb

  def create_both_dup_heatmap(self, hm3, hm2):
    hmb = hm3 * hm2
    hmb = (hmb - hmb.min()) / (hmb.max() - hmb.min())
    return hmb

  def get_df_for_analyze(self,  id: int, cls: int):
    eval_df = self.df[self.df["CatId"] == id]
    eval = eval_df[eval_df["Valence"] == cls]
    eval = eval[eval["Infered_Class"] == cls]
    return eval

  def get_msk_for_img(self, img_path: str):
    msk_path = img_path.replace(self.imgs_root_folder, self.msks_folder)
    osh = OneSegOneHeatmapCalc(msk_path, self.out_sz)
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
    #normalize resized image
    rszd_heatmap = (rszd_heatmap - rszd_heatmap.min()) / (rszd_heatmap.max() - rszd_heatmap.min())
    return rszd_heatmap

  def analyze_img(self, img_path: str, osh:OneSegOneHeatmapCalc, id:int, valence: int):
    hm3 = self.get_heatmap_for_img(img_path, '3',  id= id, valence =valence)
    hm2 = self.get_heatmap_for_img(img_path, '2',  id= id, valence =valence)
    hm_both = self.create_both_heatmap(hm3, hm2)
    hm_bothd = self.create_bothd_heatmap(hm3, hm2)
    relevant_heat3, rszd_heat3 = osh.calc_relevant_heat(hm3)
    grade_normalized3, rest_img_grade3, sub3, cnr3 = osh.calc_grade_sums_by_seg(relevant_heat3, rszd_heat3)
    relevant_heat2, rszd_heat2 = osh.calc_relevant_heat(hm2)
    grade_normalized2, rest_img_grade2, sub2, cnr2 = osh.calc_grade_sums_by_seg(relevant_heat2, rszd_heat2)
    relevant_heatb, rszd_heatb = osh.calc_relevant_heat(hm_both)
    grade_normalized_b, rest_img_grade_b, subb, cnrb = osh.calc_grade_sums_by_seg(relevant_heatb, rszd_heatb)
    relevant_heatbd, rszd_heatbd = osh.calc_relevant_heat(hm_bothd)
    grade_normalized_bd, rest_img_grade_bd, subbd, cnrbd = osh.calc_grade_sums_by_seg(relevant_heatbd, rszd_heatbd)

    return [grade_normalized3, grade_normalized2, grade_normalized_b, grade_normalized_bd], [rest_img_grade3, rest_img_grade2, rest_img_grade_b, rest_img_grade_bd], [cnr3, cnr2, cnrb, cnrbd]

  def analyze_imgs_list(self, imgs_list, valence: int, id: int):
    for img_pth in imgs_list:
      osh = self.get_msk_for_img(img_pth)
      if osh == []:
        continue
      grade_normalized, rest_img_grades, cnrs = self.analyze_img(img_pth, osh, id=id, valence=valence)
      self.rig3.append(rest_img_grades[0])
      self.rig2.append(rest_img_grades[1])
      self.rigb.append(rest_img_grades[2])
      self.rigbd.append(rest_img_grades[3])
      self.gn3.append(grade_normalized[0])
      self.gn2.append(grade_normalized[1])
      self.gnb.append(grade_normalized[2])
      self.gnbd.append(grade_normalized[3])
      self.cnr3.append(cnrs[0])
      self.cnr2.append(cnrs[1])
      self.cnrb.append(cnrs[2])
      self.cnrbd.append(cnrs[3])
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
                       'normGrade3': self.gn3, 'restOfImgGrade3': self.rig3,
                       'normGrade2': self.gn2, 'restOfImgGrade2': self.rig2,
                       'normGradeBoth': self.gnb,'restOfImgGradeBoth': self.rigb,
                       'normGradeBothd': self.gnbd, 'restOfImgGradeBothd': self.rigbd,
                       'cnr3': self.cnr3, 'cnr2':self.cnr2, 'cnrb':self.cnrb,'cnrbd':self.cnrbd})
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
  df = pd.read_csv("/home/tali/cats_pain_proj/face_images/cats_norm1_infered50.csv")
  catsSegs = CatsSegs(alpha=0.7, df =df, out_sz=(32, 32), res_folder = '/home/tali',
                      imgs_root='/home/tali/cats_pain_proj/face_images/',
                      msks_root = '/home/tali/cats_pain_proj',
                      heats_root = '/home/tali/trials/cats_bb_res_test50_minus/')
  catsSegs.analyze_all()

  wws = WorkWithSegs(alpha = 0.7, imgs_root_folder = '/home/tali/cats_pain_proj/face_images/',
                       msks_folder = '/home/tali/cats_pain_proj/face_images/masks/',
                       heats_folder = '/home/tali/trials/cats_bb_res_test50_minus/', res_folder='',
                       df = df, out_sz=(32, 32))
  wws.analyze_all()
  wws.create_res_df('/home/tali/cats_pain_proj/face_images/grade_map_analyze50.csv')
  df = pd.read_csv('/home/tali/cats_pain_proj/face_images/grade_map_analyze50.csv')
  r3, r2, rb = grade_maps_metrices(df)

















