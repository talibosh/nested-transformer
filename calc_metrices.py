import pandas as pd
import os
import numpy as np


def combine_csvs(dir_path:str, out_csv_full_path: str):
    # Initialize an empty list to store DataFrames
    dfs = []

    # Iterate over each file in the directory
    for filename in os.listdir(dir_path):
        if filename.endswith('.csv') and not(filename.startswith('dogs')):
            # Read the CSV file and append it to the list of DataFrames
            filepath = os.path.join(dir_path, filename)
            df = pd.read_csv(filepath)
            dfs.append(df)

    # Combine all DataFrames into one
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.to_csv(out_csv_full_path)
    return combined_df
def calc_metrices_maj_call(valence_list: list[int], pred_valence_list: list[int], video_names_list: list[int]):
    y=1
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

def calc_metrices_from_df_dogs(df: pd.DataFrame):
    valence = df["label"].tolist()
    converted_list = [1 if x == 'P' else 0 for x in valence]
    prediction = df["Infered_Class"].tolist()
    accuracy, precision, recall, f1_score=calc_metrices(converted_list, prediction)
    return accuracy, precision, recall, f1_score

def calc_metrices_from_df_dogs_maj_call(res_df: pd.DataFrame):
    TN_expec = 0
    TP_expec = 0
    FP_expec = 0
    FN_expec = 0

    TN_dis = 0
    TP_dis = 0
    FP_dis = 0
    FN_dis = 0

    videos = res_df["video"].to_list()
    unique_videos = np.unique(np.array(videos))
    for video in unique_videos:
        df = res_df[res_df["video"] == video]
        valence = (df["label"].tolist())[0]
        prediction = df["Infered_Class"].tolist()
        converted_list = ['P' if x == 1 else 'N' for x in prediction]
        converted_list=np.array(converted_list)
        correct = (np.where(converted_list == valence))[0].__len__()
        wrong = converted_list.__len__()-correct
        success = 0
        if correct > wrong:
            success = 1
        if valence == "P":
            if success:
                TP_expec=TP_expec+1
                TN_dis= TN_dis+1
            else:
                FN_expec=FN_expec+1
                FP_dis = FP_dis+1
        else: #"N"
            if success:
                TN_expec=TN_expec+1
                TP_dis=TP_dis+1
            else:
                FP_expec=FP_expec+1
                FN_dis=FN_dis+1
    accuracy_expec = (TP_expec + TN_expec) / (TP_expec + TN_expec + FP_expec + FN_expec+ 1e-12)
    precision_expec = TP_expec / (TP_expec + FP_expec + 1e-12)
    recall_expec = TP_expec / (TP_expec + FN_expec + 1e-12)
    f1_score_expec = 2 / (1 / (precision_expec + 1e-12) + 1 / (recall_expec + 1e-12))

    accuracy_dis = (TP_dis + TN_dis) / (TP_dis + TN_dis + FP_dis + FN_dis + 1e-12)
    precision_dis = TP_dis / (TP_dis + FP_dis + 1e-12)
    recall_dis = TP_dis / (TP_dis + FN_dis + 1e-12)
    f1_score_dis = 2 / (1 / (precision_dis + 1e-12) + 1 / (recall_dis + 1e-12))

    return accuracy_expec, precision_expec, recall_expec, f1_score_expec,accuracy_dis, precision_dis, recall_dis, f1_score_dis,


def calc_metrices_from_df_cats(df: pd.DataFrame):
    valence = df["label"].tolist()
    prediction = df["Infered_Class"].tolist()
    accuracy, precision, recall, f1_score=calc_metrices(valence, prediction)
    return accuracy, precision, recall, f1_score
def calc_metrices_from_df_horses(df: pd.DataFrame):
    valence = df["label"].tolist()
    converted_list = [1 if x == 'Yes' else 0 for x in valence]
    prediction = df["Infered_Class"].tolist()
    accuracy, precision, recall, f1_score=calc_metrices(converted_list, prediction)
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



#cats
df_path = '/home/tali/cats_pain_proj/restiny25/cats_finetune_mask_25_high_lr.csv'
df = pd.read_csv(df_path)
calc_metrices_from_df_cats(df)
#horses
#df_path = '/home/tali/horses/results/res25/total_res_25.csv'
#df = pd.read_csv(df_path)
#calc_metrices_from_df_horses(df)
#combine_csvs('/home/tali/dogs_annika_proj/cropped_face/', '/home/tali/dogs_annika_proj/cropped_face/total_10.csv')
#df_path= "/home/tali/cats_pain_proj/face_images/masked_images/cats_finetune_mask_85.csv" #'/home/tali/cropped_cats_pain/cats_norm1_infered.csv'

df_path = "/home/tali/dogs_annika_proj/cropped_face/total_25_mini_masked.csv"
df = pd.read_csv(df_path)
df= df[df["id"].isin([25,26,27,28,29,30])]
calc_metrices_from_df_dogs_maj_call(df)

#print("accuracy: "+ str(accuracy) + " precision " + str(precision + " recall "+ str(recall) + " f1 "+str(f1_score)))
new_dfb= calc_metrices_by_id(pd.read_csv(df_path))
new_dfb.to_csv('/home/tali/cats_pain_proj/face_images/masked_images/cats_finetune_mask_infered50_by_id.csv')





