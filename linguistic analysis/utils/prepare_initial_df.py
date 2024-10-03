import pandas as pd 

#------------------------------------------ For the FIRST EXPERIMENT ------------------------------------------
def Experiment_1 (directory_IRO, directory_NIRO, id_original_toremove):
    model_iro = pd.read_csv(directory_IRO, sep=",")
    model_niro = pd.read_csv(directory_NIRO, sep=",")

    model_iro = model_iro[~model_iro.id_original.isin(id_original_toremove)].reset_index(drop=True)
    model_niro = model_niro[~model_niro.id_original.isin(id_original_toremove)].reset_index(drop=True)

    original_iro = model_iro[model_iro["label"].isin(["ironic"])]
    original_not = model_iro[model_niro["label"].isin(["serious"])]

    print ("Dataframe shapes:")
    print(model_iro.shape, model_niro.shape, original_iro.shape, original_not.shape)


    model_iro = model_iro[["id_original", "parent_text", "text", "aggregated", "label"]]
    model_niro = model_niro[["id_original", "parent_text", "text", "aggregated", "label"]]
    model_iro["parent_text"] = model_iro["parent_text"].astype(str)

    return model_iro, model_niro, original_iro, original_not

    

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


#------------------------------------------ For the SECOND EXPERIMENT ------------------------------------------


#Experiment 2
def Experiment_2 (directory_young, directory_old, id_original_toremove):
    df_young = pd.read_csv(directory_young, sep=",")
    df_old = pd.read_csv(directory_old, sep=",")

    print(df_young.shape, df_old.shape)

    df_young = df_young[["id_original", "text", "parent_text", "label", "source", "origin", "Young"]]
    df_old = df_old[["Old"]]


    df = pd.concat([df_old, df_young], axis=1)
    df_ = df [["id_original","parent_text", "text", "Old", "Young", "label", "source"]]
    df_ = df.drop_duplicates(subset=["id_original"])
    print(df_.shape)
    df_clean = df_[~df_.id_original.isin(id_original_toremove)].reset_index(drop=True)
    print(df_clean.shape)
    df_original_y = df[df["origin"].isin(["young"])]
    df_original_y = df_original_y.drop(columns="Old")
    df_original_y_iro = df_original_y[df_original_y["label"].isin(["ironic"])]
    df_original_y_not = df_original_y[df_original_y["label"].isin(["serious"])]


    df_original_o = df[df["origin"].isin(["old"])]
    df_original_o = df_original_o.drop(columns="Young")
    df_original_o_iro = df_original_o[df_original_o["label"].isin(["ironic"])]
    df_original_o_not = df_original_o[df_original_o["label"].isin(["serious"])]

    return df_clean, df_original_y_iro, df_original_o_iro


def get_corresponding_value(row):
    col1_values = row['model_id'].split('#')
    col2_number = row['label_id'].split('.')[0]

    for value in col1_values:
        number, word = value.split('.')
        if number == col2_number:
            return word
        
        
def clean_df(df, df_clean):
    col_names = df.columns.tolist()
    col_id = [col for col in col_names if "1.Old#2.Young" in col or "1.Young#2.Old" in col]

    df_melted = df.melt(id_vars=["StartDate", "EndDate", "Status", "IPAddress", "Progress", "Duration (in seconds)",
                  "Finished", "RecordedDate", "ResponseId", "RecipientLastName",
                  'RecipientFirstName', 'RecipientEmail', 'ExternalReference',
                  'LocationLatitude', 'LocationLongitude', 'DistributionChannel',
                  'UserLanguage', "PROLIFIC_PID"], value_vars=col_id,
                    var_name="model_id", value_name="label_id")

    # Separiamo la colonna "model_id" in due colonne "model" e "id"
    df_melted[["id_original", "model_id"]] = df_melted["model_id"].str.split("_", expand=True)

    # Riorganizziamo le colonne e ordiniamo per "date" per avere l'output desiderato
    df_final = df_melted[["id_original", "PROLIFIC_PID", "Duration (in seconds)",
                  "Finished", "RecordedDate", "ResponseId",
                    "model_id", "label_id"]].sort_values(by=["id_original","PROLIFIC_PID"])
    
    df_final = df_final.dropna(subset="label_id")
    df_final.reset_index(drop=True, inplace=True)

    df_final = df_final.merge(df_clean[["Young", "Old", "id_original"]], on="id_original")

    df_final['label'] = df_final.apply(get_corresponding_value, axis=1)
    df_final['label'] = df_final['label'].fillna("Neither").str.replace(']','', regex=False)

    return df_final