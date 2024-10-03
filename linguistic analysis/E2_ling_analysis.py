import pandas as pd
from utils.LinguisticAnalysis import *
from utils.prepare_initial_df import Experiment_2


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




print("\n\n", "----------------------------------------Extract number of tokens----------------------------------------")
print("HUMAN TEXT")
h_post = len_token(df_YoungOld, "parent_text", "post")
h_reply = len_token(df_YoungOld, "text", "reply")
h_reply_iro = len_token (df_original_y_iro, "text", "human Young ironic")
h_reply_not = len_token (df_YoungOld, "Young", "modelYoung")
print("GENERATED TEXT")
g_IRO = len_token(df_original_o_iro, "text", "human Old ironic")
g_NIRO = len_token (df_YoungOld, "Old", "Old")


print("\n\n", "----------------------------------------Extract mean of interjections----------------------------------------")
print("HUMAN TEXT")
print('human post: ', round(np.average(len(get_interjection(df_YoungOld['parent_text'].tolist())))))
print('human reply: ', round(np.average(len(get_interjection(df_YoungOld["text"].tolist())))))
print("human reply Young labeled as ironic: ",round(np.average(len(get_interjection(df_original_y_iro['text'].tolist())))))
print("human reply Old labeled as not ironic: ",round(np.average(len(get_interjection(df_original_o_iro['text'].tolist())))))
print("GENERATED TEXT")
print('generated Young reply: ', round(np.average(len(get_interjection(df_YoungOld['Young'].tolist())))))
print('generated Old reply: ', round(np.average(len(get_interjection(df_YoungOld['Old'].tolist())))))


print("\n\n", "----------------------------------------Extract mean of negations----------------------------------------")
print("HUMAN TEXT")
print('human post: ', round(np.average(len(get_negation(df_YoungOld['parent_text'].astype(str).tolist())))))
print('human reply: ', round(np.average(len(get_negation(df_YoungOld['text'].astype(str).tolist())))))
print("human reply Young labeled as ironic: ",round(np.average(len(get_negation(df_original_y_iro['text'].astype(str).tolist())))))
print("human reply Old labeled as not ironic: ",round(np.average(len(get_negation(df_original_o_iro['text'].astype(str).tolist())))))
print("GENERATED TEXT")
print('generated IRO reply: ', round(np.average(len(get_negation(df_YoungOld['Young'].astype(str).tolist())))))
print('generated NIRO reply): ', round(np.average(len(get_negation(df_YoungOld['Old'].astype(str).tolist())))))




print("\n\n", "----------------------------------------Extract type token ratio----------------------------------------")
print("HUMAN TEXT")
print("human post", round(ttr(df_YoungOld['parent_text'].tolist()),3))
print('human reply: ', round(ttr(df_YoungOld['text'].tolist()),3))
print('human reply Young labeled as ironic: ',  round(ttr(df_original_y_iro['text'].tolist()),3))
print('human reply Old labeled as not ironic: ',  round(ttr(df_original_o_iro['text'].tolist()),3))
print("GENERATED TEXT")
print('generated Young reply: ', round(ttr(df_YoungOld['Young'].tolist()),3))
print('ggenerated Old reply: ', round(ttr(df_YoungOld['Old'].tolist()),3))


print("\n\n", "----------------------------------------Extract named entities----------------------------------------")
h_ent, dict_h_ents = list_entities(df_YoungOld,'text', 'id_original')
parent_ent, dict_parent_ents = list_entities(df_YoungOld,"parent_text", "id_original")
h_ent_y, dict_h_ents_y = list_entities(df_original_y_iro,'text', 'id_original')
h_ent_o, dict_h_ents_o = list_entities(df_original_o_iro,'text', 'id_original')
generated_y_ent, dict_g_y_ents = list_entities(df_YoungOld, 'Young', 'id_original')
generated_o_ent, dict_g_o_ents = list_entities(df_YoungOld, 'Old', 'id_original')


print('post: ', parent_ent)
print('human reply: ', h_ent)
print("human young ironic replies: ", h_ent_y)
print("human old ironic replies ", h_ent_o)
print('generated replies Young: ', generated_y_ent)
print('generated replies Old: ', generated_o_ent)

print("\n","MATCH POST-IRO named entities")
count_post, count_generated_young, count_y_match, list_ner_y = matching_entities(dict_parent_ents, dict_g_y_ents)

print("\n","MATCH POST-NIRO named entities")
count_post, count_generated_old, count_o_match, list_ner_o = matching_entities(dict_parent_ents, dict_g_o_ents)



print("\n\n", "----------------------------------------Nominal utterances----------------------------------------")
df_nominal = df_YoungOld[['id_original']]
df_nominal["nominal_young_aggregated"] = df_YoungOld["Young"].apply(nominal_utterance) #.fillna(True)
df_nominal["nominal_old_aggregated"] = df_YoungOld["Old"].apply(nominal_utterance) #.fillna(True)

print(df_nominal["nominal_young_aggregated"].value_counts())
print()
print(df_nominal["nominal_old_aggregated"].value_counts())