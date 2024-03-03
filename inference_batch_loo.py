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
    def inference_one_id_one_class(self, class_id: int, images_list: list[str], model:nstGradCat.NestForGradCAT):
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
            trials.plot_grid(img_path, out_path1, show, avg_heatmap3, avg_heatmap2)
            trials.plot_heatmap(img_path, out_dir_heats_plots, show, heatmap3 , heatmap2 ,(224,224))
            ff = out_path1.replace(out_dir, out_dir_heats)
            f3 = ff.replace('.jpg','_3.npy')
            np.save(f3, heatmap3)
            f2 = ff.replace('.jpg', '_2.npy')
            np.save(f2, heatmap2)
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
    def inference_loo(self, df:pandas.DataFrame):
        ids = df['CatId'].tolist()
        unique_ids = np.unique(ids)
        classes = df['Valence'].tolist()
        unique_classes = np.unique(classes)
        num_classes = unique_classes.__len__()
        new_df = pandas.DataFrame()
        for id in unique_ids:
            print('***************start ' + str(id) + ' *************************\n')
            model = nstGradCat.NestForGradCAT(os.path.join(self.chk_points_root, str(id),'checkpoints-0'), self.config, num_classes)
            eval_df = df[df["CatId"] == id]
            eval_pain = eval_df[eval_df["Valence"] == 1]
            eval_no_pain = eval_df[eval_df["Valence"] == 0]
            infer = inference_and_gradCAT_one_id(self.config.mean, self.config.std)
            preds0, probs0 = infer.inference_one_id_one_class(0, eval_no_pain["FullPath"].tolist(), model)
            preds1, probs1 = infer.inference_one_id_one_class(1, eval_pain["FullPath"].tolist(), model)
            #df_preds_probs_no_pain = pandas.DataFrame({'Infered_Class': preds0,'Prob': probs0})
            #df_preds_probs_pain = pandas.DataFrame({'Infered_Class': preds1, 'Prob': probs1})
            updated_eval_no_pain=eval_no_pain
            updated_eval_no_pain['Infered_Class'] = preds0
            updated_eval_no_pain['Prob'] = probs0
            updated_eval_pain = eval_pain
            updated_eval_pain['Infered_Class'] = preds1
            updated_eval_pain['Prob'] = probs1
            new_df = pandas.concat([new_df, updated_eval_no_pain, updated_eval_pain], axis=0)
        return new_df

    def create_infer_csv_loo(self, in_df_path:str, out_path:str):
        df=pandas.read_csv(in_df_path)
        infered_df = self.inference_loo(df)
        infered_df.to_csv(out_path)

    def gradCAT_loo(self, df:pandas.DataFrame, out_root:str):
        ids = df['CatId'].tolist()
        unique_ids = np.unique(ids)
        classes = df['Valence'].tolist()
        infered_classes = df['Infered_Class'].tolist()
        unique_classes = np.unique(classes)
        num_classes = unique_classes.__len__()
        new_df = pandas.DataFrame()
        for id in unique_ids:
            if id<1:
                continue
            model = nstGradCat.NestForGradCAT(os.path.join(self.chk_points_root, str(id),'checkpoints-0'), self.config, num_classes)
            eval_df = df[df["CatId"] == id]
            eval_pain = eval_df[eval_df["Valence"] == 1]
            eval_pain = eval_pain[eval_pain["Infered_Class"] == 1]

            eval_no_pain = eval_df[eval_df["Valence"] == 0]
            eval_no_pain = eval_no_pain[eval_no_pain["Infered_Class"] == 0]
            grad_cat = inference_and_gradCAT_one_id(self.config.mean, self.config.std)
            no_pain_dir = os.path.join(out_root, str(id), 'no pain')
            grad_cat.gradCAT_one_id_one_class(0, eval_no_pain["FullPath"].tolist(), model, no_pain_dir)
            pain_dir = os.path.join(out_root, str(id), 'pain')
            grad_cat.gradCAT_one_id_one_class(1, eval_pain["FullPath"].tolist(), model, pain_dir)


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

    from configs import cats_pain
    in_csv_path = '/home/tali/cats_pain_proj/face_images/masked_images/cats_masked.csv'#'/home/tali/cropped_cats_pain/cats.csv'
    out_csv_path = '/home/tali/cats_pain_proj/face_images/masked_images/cats_finetune_mask_infered50.csv'
    csv_path ='/home/tali/cats_pain_proj/face_images/masked_images/cats_finetune_mask_infered50.csv' #'/home/tali/cropped_cats_pain/cats_norm1_infered.csv'
    chkpoints_root = '/home/tali/mappingPjt/nst12/checkpoints/nest_cats/'
    config = cats_pain.get_config()
    loo_oper = inference_and_gradCAT_loo(chkpoints_root, config)
    #loo_oper.create_infer_csv_loo(in_csv_path, out_csv_path)
    loo_oper.run_grad_CAT_loo(csv_path, '/home/tali/trials/cats_finetune_mask_relu_res_test50')





