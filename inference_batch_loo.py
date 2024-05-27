import tensorflow as tf
tf.config.experimental.set_visible_devices([], "GPU")

import pandas

from libml import preprocess
import numpy as np
from models import nest_for_gradCat as nstGradCat
import ml_collections
import PIL
import trials
import os


class inference_and_gradCAT_one_id():

    def __init__(self, dataset_mean, dataset_std):
        self.mean = dataset_mean
        self.std = dataset_std
    def __preprocess(self,image):
        image = np.array(image.resize((224, 224))).astype(np.float32) / 255
        mean = np.array(self.mean).reshape(1, 1, 3) #np.array(preprocess.IMAGENET_DEFAULT_MEAN).reshape(1, 1, 3)
        std = np.array(self.std).reshape(1, 1, 3) #np.array(preprocess.IMAGENET_DEFAULT_STD).reshape(1, 1, 3)
        image = (image - mean) / std
        return image[np.newaxis, ...]
    def inference_one_id_one_class(self,  images_list: list[str], model:nstGradCat.NestForGradCAT):
        predicted_class = []
        probs = []
        for img in images_list:
            image = PIL.Image.open(img)
            image = self.__preprocess(image)
            cls, prob = model.predict(image)
            #print(f'ImageNet class id: {cls[0]}, prob: {prob[0]}')
            predicted_class.append(cls[0])
            probs.append(prob[0])
        return predicted_class, probs

    def gradCAT_one_id_one_class(self, class_id: int, images_list: list[str], model: nstGradCat.NestForGradCAT, out_dir: str = "", show: bool = False):
        out_dir_heats_plots = os.path.join(out_dir, "heats_plots")
        out_dir_heats = os.path.join(out_dir, "heats")

        os.makedirs(out_dir_heats_plots, exist_ok=True)
        os.makedirs(out_dir_heats, exist_ok=True)

        for img_path in images_list:
            image = PIL.Image.open(img_path)
            image = self.__preprocess(image)
            heatmap3, avg_heatmap3, heatmap2, avg_heatmap2 = model.create_heatmaps_and_avg_heatmaps(image)
            fname = os.path.basename(img_path)
            out_path1 = os.path.join(out_dir, fname)
            os.makedirs(out_dir,exist_ok=True)
            #trials.plot_grid(img_path, out_path1, show, avg_heatmap3, avg_heatmap2)
            #trials.plot_heatmap(img_path, out_dir_heats_plots, show, heatmap3 , heatmap2 ,(224,224))
            ff = out_path1.replace(out_dir, out_dir_heats)
            f3 = ff.replace('.jpg','_3.npy')
            np.save(f3, heatmap3)
            #f2 = ff.replace('.jpg', '_2.npy')
            #np.save(f2, heatmap2)
            #head, tail = os.path.split(img_path)
            #if class_id is 0:
            #    h='no_pain'
            #if class_id is 1:
            #    h='pain'
            #msk_path = os.path.join(head, '..','masks', h, tail)
            #seg_grades, rest_of_img_grades = trials.calc_grade_on_segment(msk_path, heatmaps)





class inference_and_gradCAT_loo():

    def __init__(self, chk_points_root: str, config: ml_collections.ConfigDict):
        self.chk_points_root = chk_points_root
        self.config = config
    def inference_loo(self, df:pandas.DataFrame, out_path:str):
        idName=""
        labelName=""
        if config.dataset == "cats_pain":
            idName = "CatId"
            labelName = "Valence"
            fullPathName="FullPath"
        elif config.dataset == "dogs_anika":
            idName = "id"
            labelName = "label"
            fullPathName = "full path"
        ids = df[idName].tolist()
        unique_ids = np.unique(ids)
        classes = df[labelName].tolist()
        unique_classes = np.unique(classes)
        num_classes = unique_classes.__len__()
        new_df = pandas.DataFrame()
        for id in unique_ids:
            #if id<15:
            #    continue
            id_df = pandas.DataFrame()
            print('***************start ' + str(id) + ' *************************\n')
            model = nstGradCat.NestForGradCAT(os.path.join(self.chk_points_root, str(id),'checkpoints-0'), self.config, num_classes)
            eval_df = df[df[idName] == id]
            eval_cls_1 = eval_df[eval_df[labelName] == unique_classes[1]]
            eval_cls_0 = eval_df[eval_df[labelName] == unique_classes[0]]
            infer = inference_and_gradCAT_one_id(self.config.mean, self.config.std)
            preds0, probs0 = infer.inference_one_id_one_class( eval_cls_0[fullPathName].tolist(), model)
            preds1, probs1 = infer.inference_one_id_one_class( eval_cls_1[fullPathName].tolist(), model)
            #df_preds_probs_no_pain = pandas.DataFrame({'Infered_Class': preds0,'Prob': probs0})
            #df_preds_probs_pain = pandas.DataFrame({'Infered_Class': preds1, 'Prob': probs1})
            updated_eval_cls1=eval_cls_1
            updated_eval_cls1['Infered_Class'] = preds1
            updated_eval_cls1['Prob'] = probs1
            updated_eval_cls0 = eval_cls_0
            updated_eval_cls0['Infered_Class'] = preds0
            updated_eval_cls0['Prob'] = probs0
            id_df = pandas.concat([id_df, updated_eval_cls1, updated_eval_cls0], axis=0)
            csv_name = 'dog_'+str(id)+'.csv'
            id_df.to_csv(os.path.join(out_path,csv_name))
            new_df = pandas.concat([new_df, updated_eval_cls1, updated_eval_cls0], axis=0)
        return new_df

    def create_infer_csv_loo(self, in_df_path:str, out_path:str):
        df=pandas.read_csv(in_df_path)
        infered_df = self.inference_loo(df, os.path.split(out_path)[0])
        infered_df.to_csv(out_path)

    def gradCAT_loo(self, df:pandas.DataFrame, out_root:str):
        idName = ""
        labelName = ""
        infered_cls_name = ""
        has_video = False
        if config.dataset == "cats_pain":
            idName = "CatId"
            labelName = "Valence"
            fullPathName = "FullPath"
            infered_cls_name = "Infered_Class"
            dir_name_1 = 'no_pain'
            dir_name_0 = 'pain'
        elif config.dataset == "dogs_anika":
            idName = "id"
            labelName = "label"
            fullPathName = "full path"
            infered_cls_name = 'Infered_Class'
            dir_name_1 = 'pos'
            dir_name_0 = 'neg'
            has_video = True
        ids = df[idName].tolist()
        unique_ids = np.unique(ids)
        classes = df[labelName].tolist()
        infered_classes = df[infered_cls_name].tolist()
        unique_classes = np.unique(classes)
        num_classes = unique_classes.__len__()
        new_df = pandas.DataFrame()
        for id in unique_ids:
            if id<28:
                continue
            model = nstGradCat.NestForGradCAT(os.path.join(self.chk_points_root, str(id),'checkpoints-0'), self.config, num_classes)
            eval_df = df[df[idName] == id]
            if has_video is True:
                videos_list = np.unique(eval_df['video'].to_list())
            else:
                videos_list = [""]
            for v in videos_list:
                eval_df_v = eval_df[eval_df["video"] == v]
                eval_1 = eval_df_v[eval_df_v[labelName] == 'P']
                eval_1 = eval_1[eval_1[infered_cls_name] == 1]

                eval_0 = eval_df_v[eval_df_v[labelName] == 'N']
                eval_0 = eval_0[eval_0[infered_cls_name] == 0]
                grad_cat = inference_and_gradCAT_one_id(self.config.mean, self.config.std)
                dir0 = os.path.join(out_root, str(id),str(v), dir_name_0)
                grad_cat.gradCAT_one_id_one_class(0, eval_0[fullPathName].tolist(), model, dir0)
                dir1 = os.path.join(out_root, str(id),str(v), dir_name_1)
                grad_cat.gradCAT_one_id_one_class(1, eval_1[fullPathName].tolist(), model, dir1)


    def run_grad_CAT_loo(self, in_df_path : str, out_root : str):
        df = pandas.read_csv(in_df_path)
        self.gradCAT_loo(df, out_root)



if __name__ == "__main__":
    #from configs import imagenet_nest
    #config = imagenet_nest.get_config()
    #model = nstGradCat.NestForGradCAT('/home/tali/mappingPjt/nst12/checkpoints/nest-b_imagenet/', config, 1000)
    #img_path = 'n02086079_499.jpg'
    #igi = inference_and_gradCAT_one_id()
    #igi.inference_one_id_one_class(135, [img_path], model)
    #igi.gradCAT_one_id_one_class(135, [img_path], model, '/home/tali/test_imgnet', False)

    from configs import cats_pain, anika_nest
    in_csv_path = '/home/tali/dogs_annika_proj/cropped_face/dogs_cropped_frames.csv'
    out_csv_path = "/home/tali/dogs_annika_proj/cropped_face/total_10.csv"
    chkpoints_root = '/home/tali/mappingPjt/nst12/checkpoints/anika_dogs/'
    config = anika_nest.get_config()
    loo_oper = inference_and_gradCAT_loo(chkpoints_root, config)
    #loo_oper.create_infer_csv_loo(in_csv_path, out_csv_path)
    loo_oper.run_grad_CAT_loo(out_csv_path, '/home/tali/dogs_annika_proj/res_10_gc/')


    #in_csv_path = "/home/tali/cats_pain_proj/face_images/masked_images/cats_masked.csv"#'/home/tali/cropped_cats_pain/cats.csv'
    #out_csv_path = "/home/tali/cats_pain_proj/face_images/masked_images/cats_finetune_mask_85.csv"
    #csv_path ="/home/tali/cats_pain_proj/face_images/masked_images/cats_finetune_mask_infered50.csv" #'/home/tali/cropped_cats_pain/cats_norm1_infered.csv'
    #chkpoints_root = '/home/tali/mappingPjt/nst12/checkpoints/cats_pain/'
    #config = cats_pain.get_config()
    #loo_oper = inference_and_gradCAT_loo(chkpoints_root, config)
    #loo_oper.create_infer_csv_loo(in_csv_path, out_csv_path)
    #loo_oper.run_grad_CAT_loo(csv_path, '/home/tali/trials/cats_finetune_mask_seg_test50_cam')





