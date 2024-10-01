import pandas as pd 

#Experiment 1 
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