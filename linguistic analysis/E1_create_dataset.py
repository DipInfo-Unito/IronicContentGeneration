import pandas as pd

from utils.prepare_initial_df import Experiment_1

#Prepare the dataframe
dir_IRO = "./aggregated_outputs/aggregated_True_mask_no_user.csv"
dir_NIRO = "./aggregated_outputs/aggregated_False_mask_no_user.csv"

id_original_toremove = [
"1579300909359980544",
"1579737375923924992",
"1579750140017278976",
"1579748088373477377",
"fj5vjya",
"1579758197686284289",
"1579716607030751233",
"1569996671358238720",
"1580039891336851456",
"1579837055194267649",
"1578129800728023040",
"1579748811144310785",
"1538755893692604417"]

model_iro, model_niro, original_iro, original_not = Experiment_1(dir_IRO, dir_NIRO, id_original_toremove)


def get_aggregated_text(row):
    if row["model"] == "ironic":
        return model_iro.loc[model_iro["id_original"] == row["id_original"], "aggregated"].values[0]
    elif row["model"] == "notironic":
        return model_niro.loc[model_niro["id_original"] == row["id_original"], "aggregated"].values[0]
    else:
        return None
    

def convert_label(label):
    if label == -2:
        return "Strongly Disagree"
    elif label == -1:
        return "Disagree"
    elif label == 0:
        return "Neither Agree nor Disagree"
    elif label == 1:
        return "Agree"
    elif label == 2:
        return "Strongly Agree"
    else:
        return "error"





df_annotations = pd.read_csv("./qualtrics annotation/E1_clean.csv", sep=",")

attention_cols = [c for c in df_annotations.columns if c.startswith("attention")]
df_annotations = df_annotations.drop(attention_cols, axis=1)


for i in df_annotations["PROLIFIC_PID"]: 
    if i == "663c1a1ed18ce928e9b08f9a":
        print ("upsy")


col_names = df_annotations.columns.tolist()

col_id = [col for col in col_names if "ironic" in col or "notironic" in col]


# Usiamo melt per ristrutturare il dataframe
df_melted = df_annotations.melt(id_vars=["StartDate", "EndDate", "Status", "IPAddress", "Progress", "Duration (in seconds)",
                  "Finished", "RecordedDate", "ResponseId", "RecipientLastName", "all_attentions",
                  'RecipientFirstName', 'RecipientEmail', 'ExternalReference',
                  'LocationLatitude', 'LocationLongitude', 'DistributionChannel',
                  'UserLanguage', 'Q801', 'Q1108', "PROLIFIC_PID"], value_vars=col_id,
                    var_name="model_id", value_name="label")


# Separiamo la colonna "model_id" in due colonne "model" e "id"
df_melted[["model", "id_original"]] = df_melted["model_id"].str.split("_", expand=True)


# Riorganizziamo le colonne e ordiniamo per "date" per avere l'output desiderato
df_final = df_melted[["id_original", "PROLIFIC_PID", "Duration (in seconds)",
                  "Finished", "RecordedDate", "ResponseId", "all_attentions",
                    "model", "label"]].sort_values(by=["id_original","PROLIFIC_PID"])

list_ann_pre = (df_final["PROLIFIC_PID"].drop_duplicates().tolist())

df_final["id_original"] = df_final["id_original"].str.replace(']', '', regex=False)
df_final = df_final.dropna(subset="label")
df_final.reset_index(drop=True, inplace=True)

list_ann_post = (df_final["PROLIFIC_PID"].drop_duplicates().tolist())
for i in list_ann_pre:
    if i not in list_ann_post:
        print(i)


df_final = df_final.merge(model_iro[["parent_text", "id_original"]], on="id_original")
df_final["aggregated"] = df_final.apply(get_aggregated_text, axis=1)
df_final["label"] = df_final["label"].apply(convert_label)

df_final = df_final[['id_original', 'PROLIFIC_PID', 'Duration (in seconds)', 'Finished',
       'RecordedDate', 'ResponseId', 'all_attentions', 'parent_text',
       'aggregated','model', 'label']]

df_final = df_final.rename(columns={"parent_text": "Post", "aggregated": "Reply", "PROLIFIC_PID":"Participant id"}).replace({"ironic":"IRO", "notironic":"NIRO"})
demographics = pd.read_csv("/home/marem/VscProjects/theGIRLS/dem_data/demographics_E1.csv")
df_final = df_final.merge(demographics, on="Participant id")

df_final.to_csv("./final_datasets/E1_dataset_merge.csv")
print(df_final.shape)