import pandas as pd
import os
import numpy as np

def calc_metrices(valence_list: list[int], pred_valence_list: list[int]):
    if valence_list.__len__()!=pred_valence_list.__len__():
        return -1 #wrong len
    TN=0
    TP=0
    FP=0
    FN=0
    for i in range(valence_list.__len__()):
        if valence_list[i] == 0 and pred_valence_list[i] == 0:
            TN = TN + 1
        if valence_list[i]==1 and pred_valence_list[i]==1:
            TP=TP+1
        if valence_list[i] == 0 and pred_valence_list[i] == 1:
            FP = FP + 1
        if valence_list[i]==1 and pred_valence_list[i]==0:
            FN=FN+1
    accuracy = (TP+TN)/(TP+TN+FP+FN+ 1e-12)
    precision =TP/(TP+FP+ 1e-12)
    recall = TP/(TP+FN+ 1e-12)
    f1_score = 2/(1/(precision+ 1e-12) + 1/(recall+ 1e-12))
    return accuracy, precision, recall ,f1_score

def calc_metrices_from_df(df: pd.DataFrame):
    valence = df["Valence"].tolist()
    prediction = df["Infered_Class"].tolist()
    accuracy, precision, recall, f1_score=calc_metrices(valence, prediction)
    return accuracy, precision, recall, f1_score


def calc_metrices_by_id(df: pd.DataFrame):
    ids = df['CatId'].tolist()
    unique_ids = np.unique(ids)
    accuracy=[]
    precision=[]
    recall=[]
    f1_score=[]
    for id in unique_ids:
        print('***************start ' + str(id) + ' *************************\n')
        eval_df = df[df["CatId"] == id]
        accuracy_, precision_, recall_, f1_score_ = calc_metrices_from_df(eval_df)
        accuracy.append(accuracy_)
        precision.append(precision_)
        recall.append(recall_)
        f1_score.append(f1_score_)
    res_df={'id':unique_ids.tolist(), 'accuracy':accuracy, 'precision':precision, 'recall':recall, 'f1_score':f1_score}
    return pd.DataFrame.from_dict(res_df)

df_path= '/home/tali/cats_pain_proj/face_images/masked_images/cats_finetune_mask_infered50.csv' #'/home/tali/cropped_cats_pain/cats_norm1_infered.csv'
accuracy, precision, recall, f1_score = calc_metrices_from_df(pd.read_csv(df_path))
#print("accuracy: "+ str(accuracy) + " precision " + str(precision + " recall "+ str(recall) + " f1 "+str(f1_score)))
new_dfb= calc_metrices_by_id(pd.read_csv(df_path))
new_dfb.to_csv('/home/tali/cats_pain_proj/face_images/masked_images/cats_finetune_mask_infered50_by_id.csv')





