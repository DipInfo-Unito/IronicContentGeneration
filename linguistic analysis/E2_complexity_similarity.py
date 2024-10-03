import pandas as pd 
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np

from utils.Complexity_Similarity import cosine_distance_wordembedding_method, calculate_tree_heights, human_similarity
from utils.prepare_initial_df import Experiment_2

#Prepare the dataset
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



#syntactical complexity
df_complexity_generated = df_YoungOld[["id_original","Young", "Old"]]
df_complexity_human_young = df_original_y_iro[["id_original", "text"]]
df_complexity_human_old = df_original_o_iro[["id_original", "text"]]

df_complexity_generated = calculate_tree_heights(df_complexity_generated, 'Young', "generated reply Young")
df_complexity_generated = calculate_tree_heights(df_complexity_generated, 'Old', "generated reply Old")
df_complexity_human_young = calculate_tree_heights(df_complexity_human_young, 'text', "human reply Young ironic")
df_complexity_human_old = calculate_tree_heights(df_complexity_human_old, 'text', "human reply Old ironic")




#text similarity with the post
list_parent = df_YoungOld["parent_text"].astype(str).tolist()
list_text = df_YoungOld["text"].astype(str).tolist()
list_generated_young = df_YoungOld["Young"].astype(str).tolist()
list_generated_old = df_YoungOld["Old"].astype(str).tolist()
list_id = df_YoungOld["id_original"].tolist()


dict_similarity = {}

for i in range(len(list_parent)):
    parent_txt = list_parent[i]
    txt = list_text[i]
    generated_txt_young= list_generated_young[i]
    generated_txt_old= list_generated_old[i]
    id = list_id[i]
    computed_similarity_txt = cosine_distance_wordembedding_method(parent_txt,txt)
    computed_similarity_generated_young = cosine_distance_wordembedding_method(parent_txt,generated_txt_young)
    computed_similarity_generated_old = cosine_distance_wordembedding_method(parent_txt,generated_txt_old)
    dict_similarity[id]=[computed_similarity_txt,computed_similarity_generated_young,computed_similarity_generated_old]

similarity_text = pd.DataFrame.from_dict(dict_similarity, orient="index", columns=["sim_text", "sim_generated_young", "sim_generated_old"])

similarity_text.reset_index(inplace=True)
similarity_text.rename(columns={"index": "id_original"}, inplace=True)




list_sim_post_human_young = human_similarity(df_original_y_iro)
list_sim_post_human_old = human_similarity(df_original_o_iro)


print("mean similarity parent-human young ironic-->", round(np.average(list_sim_post_human_young),3))
print("mean similarity parent-human old ironic-->", round(np.average(list_sim_post_human_young),3))
print("mean similarities parent-model Young -->", round(np.average(similarity_text["sim_generated_young"].tolist()),3))
print("mean similarities parent-model Old -->", round(np.average(similarity_text["sim_generated_old"].tolist()),3))

print()
print("std similarity parent-human young ironic-->", round(np.std(list_sim_post_human_young),3))
print("std similarity parent-human old ironic-->", round(np.std(list_sim_post_human_young),3))
print("std similarities parent-model Young -->", round(np.std(similarity_text["sim_generated_young"].tolist()),3))
print("std similarities parent-model Old -->", round(np.std(similarity_text["sim_generated_old"].tolist()),3))