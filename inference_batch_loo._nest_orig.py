import tensorflow as tf
tf.config.experimental.set_visible_devices([], "GPU")

import pandas

from libml import preprocess
import numpy as np
#from models import nest_for_gradCat as nstGradCat
from models import nest_net
import ml_collections
import PIL
import trials
import os
import train
import functools
import flax.linen as nn

class inference_and_gradCAT_one_id():
    def __preprocess(self,image):
        image = np.array(image.resize((224, 224))).astype(np.float32) / 255
        mean = np.array(preprocess.IMAGENET_DEFAULT_MEAN).reshape(1, 1, 3)
        std = np.array(preprocess.IMAGENET_DEFAULT_STD).reshape(1, 1, 3)
        image = (image - mean) / std
        return image[np.newaxis, ...]
    def inference_one_id_one_class(self, class_id: int, images_list: list[str], model, variables):
        predicted_class = []
        probs = []
        for img in images_list:
            image = PIL.Image.open(img)
            image = self.__preprocess(image)
            #cls, prob = model.predict(image)
            logits, state = model(train=False).apply(variables, image, mutable=['intermediates'])

            # Return predicted class and confidence.
            cls, prob = logits.argmax(axis=-1), nn.softmax(logits, axis=-1).max(axis=-1)

            #print(f'ImageNet class id: {cls[0]}, prob: {prob[0]}')
            predicted_class.append(cls[0])
            probs.append(prob[0])
        return predicted_class, probs


class inference_loo():

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
            checkpoint_dir = os.path.join(self.chk_points_root,str(id),'checkpoints-0')
            state_dict = train.checkpoint.load_state_dict(checkpoint_dir)

            variables = {
                "params": state_dict["optimizer"]["target"],
            }
            variables.update(state_dict["model_state"])
            model_cls = nest_net.create_model(self.config.model_name, self.config)
            # model = functools.partial(model_cls, num_classes=1000)

            model = functools.partial(model_cls, num_classes=2)

            #model = nstGradCat.NestForGradCAT(os.path.join(self.chk_points_root, str(id),'checkpoints-0'), self.config, num_classes)
            eval_df = df[df["CatId"] == id]
            eval_pain = eval_df[eval_df["Valence"] == 1]
            eval_no_pain = eval_df[eval_df["Valence"] == 0]
            infer = inference_and_gradCAT_one_id()
            preds0, probs0 = infer.inference_one_id_one_class(0, eval_no_pain["FullPath"].tolist(), model, variables)
            preds1, probs1 = infer.inference_one_id_one_class(1, eval_pain["FullPath"].tolist(), model, variables)
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




if __name__ == "__main__":
    from configs import cats_pain
    in_csv_path = '/home/tali/cats_pain_proj/face_images/cats.csv'#'/home/tali/cropped_cats_pain/cats.csv'
    out_csv_path = '/home/tali/cats_pain_proj/face_images/cats_norm1_infered60_orig.csv'
    csv_path ='/home/tali/cats_pain_proj/face_images/cats_norm1_infered60_orig.csv' #'/home/tali/cropped_cats_pain/cats_norm1_infered.csv'
    chkpoints_root = '/home/tali/mappingPjt/nst12/checkpoints/nest_cats_norm1_60/'
    config = cats_pain.get_config()
    loo_oper = inference_loo(chkpoints_root, config)
    loo_oper.create_infer_csv_loo(in_csv_path, out_csv_path)
    #loo_oper.run_grad_CAT_loo(csv_path, '/home/tali/trials/cats_bb_res_test60')



