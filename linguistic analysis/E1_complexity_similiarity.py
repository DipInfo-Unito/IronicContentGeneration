import pandas as pd 
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np

from utils.Complexity_Similarity import cosine_distance_wordembedding_method, calculate_tree_heights, human_similarity
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



#syntactical complexity
df_complexity_iro = model_iro[["id_original","aggregated"]].rename(columns={"aggregated": "aggregated_iro"})
df_complexity_not = model_niro[["id_original","aggregated"]].rename(columns={"aggregated": "aggregated_not"})
df_complexity_human = model_iro[["id_original","text"]]

df_complexity_iro = calculate_tree_heights(df_complexity_iro, 'aggregated_iro', "IRO model")
df_complexity_not = calculate_tree_heights(df_complexity_not, 'aggregated_not', "NIRO model")
df_complexity_human = calculate_tree_heights(df_complexity_human, 'text', "Human reply")

df_complexity = pd.merge(df_complexity_iro, df_complexity_not, on="id_original")
df_complexity = pd.merge(df_complexity, df_complexity_human, on="id_original")

# l = df_complexity["IRO model"].tolist()
# text = df_complexity["aggregated_iro"].tolist()
# count = 0
# for i in range(len(l)):
#     if l[i] > 20:
#         print(text[i])

df_complexity_clean = df_complexity.drop(df_complexity[df_complexity["IRO model"] == 128].index)
print(df_complexity.shape, df_complexity_clean.shape)

fig = sns.displot(data=df_complexity_clean[["IRO model","NIRO model"]], kde=True, height=8, aspect=1)
fig.set_axis_labels('Tree high', 'Count', fontsize=10)
# ax.set_ylabel('Percentage', fontsize=10)
plt.setp(fig._legend.get_texts(), fontsize='20')
fig.savefig("./linguistic analysis/plots/plot_syntactic_complexity_iro-niro.png")



#text similarity with the post
list_parent = model_iro["parent_text"].astype(str).tolist()
list_text = model_iro["text"].astype(str).tolist()
list_generated_iro = model_iro["aggregated"].astype(str).tolist()
list_generated_not = model_niro["aggregated"].astype(str).tolist()
list_id = model_iro["id_original"].tolist()


dict_similarity = {}

for i in range(len(list_parent)):
    parent_txt = list_parent[i]
    txt = list_text[i]
    generated_txt_iro= list_generated_iro[i]
    generated_txt_not= list_generated_not[i]
    id = list_id[i]
    computed_similarity_txt = cosine_distance_wordembedding_method(parent_txt,txt)
    computed_similarity_generated_iro = cosine_distance_wordembedding_method(parent_txt,generated_txt_iro)
    computed_similarity_generated_not = cosine_distance_wordembedding_method(parent_txt,generated_txt_not)
    dict_similarity[id]=[computed_similarity_txt,computed_similarity_generated_iro,computed_similarity_generated_not]

similarity_text = pd.DataFrame.from_dict(dict_similarity, orient="index", columns=["Human reply", "IRO model", "NIRO model"])

similarity_text.reset_index(inplace=True)
similarity_text.rename(columns={"index": "id_original"}, inplace=True)

fig = sns.displot(similarity_text[["Human reply", "IRO model", "NIRO model"]],kde=True)
fig.set_axis_labels("Similarity", "Number of texts")
plt.setp(fig._legend.get_texts(), fontsize='13')
fig.savefig("./linguistic analysis/plots/similarity_parent_exp1.png")


list_sim_human_iro = human_similarity(original_iro)
list_sim_human_not = human_similarity(original_not)


print("mean similarity parent-human reply ironic-->", round(np.average(list_sim_human_iro),3))
print("mean similarity parent-human reply not ironic-->", round(np.average(list_sim_human_not),3))
print("mean similarities parent-model IRO -->", round(np.average(similarity_text["IRO model"].tolist()),3))
print("mean similarities parent-model NIRO -->", round(np.average(similarity_text["NIRO model"].tolist()),3))

print()
print("std similarity parent-human reply ironic-->", round(np.std(list_sim_human_iro),3))
print("std similarity parent-human reply not ironic-->", round(np.std(list_sim_human_not),3))
print("std similarity parent-human reply -->", round(np.std(similarity_text["Human reply"].tolist()),3))
print("std similarities parent-model IRO -->", round(np.std(similarity_text["IRO model"].tolist()),3))
print("std similarities parent-model NIRO -->", round(np.std(similarity_text["NIRO model"].tolist()),3))