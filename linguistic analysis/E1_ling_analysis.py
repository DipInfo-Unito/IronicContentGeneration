import pandas as pd
from utils.LinguisticAnalysis import *
from utils.prepare_initial_df import Experiment_1


#Prepare the dataframe

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




print("\n\n", "----------------------------------------Extract number of tokens----------------------------------------")
print("HUMAN TEXT")
h_post = len_token(model_iro, "parent_text", "human post")
h_reply = len_token (model_iro, "text", "human reply")
h_reply_iro = len_token (original_iro, "text", "human reply labeled as ironic")
h_reply_not = len_token (original_not, "text", "human reply labeled as not ironic")
print("GENERATED TEXT")
g_IRO = len_token (model_iro, "aggregated", "generated IRO reply")
g_NIRO =len_token (model_niro, "aggregated", "generated NIRO reply")


print("\n\n", "----------------------------------------Extract mean of interjections----------------------------------------")
print("HUMAN TEXT")
print('human post: ', round(np.average(len(get_interjection(model_iro['parent_text'].tolist())))))
print('human reply: ', round(np.average(len(get_interjection(model_iro["text"].tolist())))))
print("human reply labeled as ironic: ",round(np.average(len(get_interjection(original_iro['text'].tolist())))))
print("human reply labeled as not ironic: ",round(np.average(len(get_interjection(original_not['text'].tolist())))))
print("GENERATED TEXT")
print('generated IRO reply: ', round(np.average(len(get_interjection(model_iro['aggregated'].tolist())))))
print('generated NIRO reply: ', round(np.average(len(get_interjection(model_niro['aggregated'].tolist())))))


print("\n\n", "----------------------------------------Extract mean of negations----------------------------------------")
print("HUMAN TEXT")
print('human post: ', round(np.average(len(get_negation(model_iro['parent_text'].astype(str).tolist())))))
print('human reply: ', round(np.average(len(get_negation(model_iro['text'].astype(str).tolist())))))
print("human reply labeled as ironic: ",round(np.average(len(get_negation(original_iro['text'].astype(str).tolist())))))
print("human reply labeled as not ironic: ",round(np.average(len(get_negation(original_not['text'].astype(str).tolist())))))
print("GENERATED TEXT")
print('generated IRO reply: ', round(np.average(len(get_negation(model_iro['aggregated'].astype(str).tolist())))))
print('generated NIRO reply): ', round(np.average(len(get_negation(model_niro['aggregated'].astype(str).tolist())))))




print("\n\n", "----------------------------------------Extract type token ratio----------------------------------------")
print("HUMAN TEXT")
print("human post", round(ttr(model_iro['parent_text'].tolist()),3))
print('human reply: ', round(ttr(model_iro['text'].tolist()),3))
print('human reply labeled as ironic: ',  round(ttr(original_iro['text'].tolist()),3))
print('human reply labeled as not ironic: ',  round(ttr(original_not['text'].tolist()),3))
print("GENERATED TEXT")
print('generated IRO reply: ', round(ttr(model_iro['aggregated'].tolist()),3))
print('ggenerated NIRO reply: ', round(ttr(model_niro['aggregated'].tolist()),3))


print("\n\n", "----------------------------------------Extract named entities----------------------------------------")
h_ent, dict_h_ents = list_entities(model_iro,'text', 'id_original')
generated_iro_ent, dict_g_iro_ents = list_entities(model_iro, 'aggregated', 'id_original')
generated_niro_ent, dict_g_niro_ents = list_entities(model_niro, 'aggregated', 'id_original')
parent_ent, dict_parent_ents = list_entities(model_iro,"parent_text", "id_original")

print('human post:')
print('\t', parent_ent)
print('human reply:')
print('\t', h_ent)
print('generated reply:')
print('\t',generated_iro_ent)

print("\n","MATCH POST-IRO named entities")
count_post, count_generated_iro, count_i_match = matching_entities(dict_parent_ents, dict_g_iro_ents)
print("full values in post: ", count_post)
print("full values in generated IRO reply: ", count_generated_iro)
print("matching cases post-IRO: ", count_i_match)

print("\n","MATCH POST-NIRO named entities")
count_post, count_generated_niro, count_n_match = matching_entities(dict_parent_ents, dict_g_niro_ents)
print("full values in post: ", count_post)
print("full values in generated NIRO reply: ", count_generated_niro)
print("matching cases post-IRO: ", count_n_match)



print("\n\n", "----------------------------------------Nominal utterances----------------------------------------")
df_nominal = model_iro[['id_original']]
df_nominal["nominal_iro_aggregated"] = model_iro["aggregated"].apply(nominal_utterance) #.fillna(True)
df_nominal["nominal_not_aggregated"] = model_niro["aggregated"].apply(nominal_utterance) #.fillna(True)

print(df_nominal["nominal_iro_aggregated"].value_counts())
print()
print(df_nominal["nominal_not_aggregated"].value_counts())