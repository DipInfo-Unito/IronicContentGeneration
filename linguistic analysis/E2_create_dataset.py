import pandas as pd
from utils.prepare_initial_df import *


#Prepare the dataframe
dir_young = "./aggregated_outputs/Young_True_mask_no_user.csv"
dir_old= "./aggregated_outputs/Old_True_mask_no_user.csv"

id_original_toremove = [
"1579723813620944896",
"1580035289006817280",
"1579757446801002496",
"fjc7rcq",
"1579742011846184960",
"1579753297002758146",
"1579739548212670465",
"fjcj3me",
"fj5preb",
"fj6ylhi",
"1579571193338040320",
"1579742664349876224",
"1572501686031360000",
"1572499496688570370",
"fjam4g8",
"1580015843953766401",
"1579904821766672384",
"1579754796760039424"
]

df_YoungOld, df_original_y_iro, df_original_o_iro = Experiment_2(dir_young, dir_old, id_original_toremove)


young = pd.read_csv("./qualtrics annotation/young_annotations.csv").drop([0,1])
old = pd.read_csv("./qualtrics annotation/old_annotations.csv").drop([0,1])


attention_cols = [c for c in young.columns if c.startswith("attention")]
young = young.drop(attention_cols, axis=1)


ann_young = clean_df(young, df_YoungOld)
ann_old = clean_df(old, df_YoungOld)

ann_young["annotator_group"] = "young"
ann_old["annotator_group"] = "old"

df_annotations = pd.concat([ann_young,ann_old],axis=0)
df_annotations = df_annotations.merge(df_YoungOld[["id_original", "parent_text"]], on="id_original")

print(ann_young.shape, ann_old.shape, df_annotations.shape)

df_annotations =df_annotations.sort_values(by="id_original")
df_annotations = df_annotations.rename(columns={"parent_text": "Post", "Young": "Y-model", "Old":"O-model", "PROLIFIC_PID":"Participant id"})

df_annotations = df_annotations[['id_original', 'Participant id', 'Duration (in seconds)', 'Finished',
       'RecordedDate', 'ResponseId', 'model_id', 'label_id', 'Post', 'O-model',
       'Y-model', 'label', 'annotator_group']]


demographics = demographics = pd.read_csv("/home/marem/VscProjects/theGIRLS/dem_data/demographics_E2.csv")
df_annotations = df_annotations.merge(demographics, on="Participant id")


print("total anotations: ", df_annotations.shape)
print("total annotators: ", len(df_annotations["Participant id"].drop_duplicates().tolist()))
df_annotations.to_csv("./final_datasets/Experiment_2_dataset.csv")