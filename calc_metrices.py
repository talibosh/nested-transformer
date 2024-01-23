import pandas as pd
import os

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
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    precision =TP/(TP+FP)
    recall = TP/(TP+FN)
    f1_score = 2/(1/precision + 1/recall)
    return accuracy, precision, recall ,f1_score

def calc_metrices_from_df(df: pd.DataFrame):
    valence = df["Valence"].tolist()
    prediction = df["Infered_Class"].tolist()
    accuracy, precision, recall, f1_score=calc_metrices(valence, prediction)
    return accuracy, precision, recall, f1_score

df_path= '/home/tali/cat_pain1/cats1_infered.csv'
accuracy, precision, recall, f1_score = calc_metrices_from_df(pd.read_csv(df_path))
print("accuracy: "+ str(accuracy) + " precision " + str(precision + " recall "+ str(recall) + " f1 "+str(f1_score)))





