import random

import pandas as pd

random.seed(42)

young_df = pd.concat([pd.read_csv("GenerationAggregated_old_split/Young_train_set.csv"), pd.read_csv("GenerationAggregated_old_split/Young_validation_set.csv"), pd.read_csv("GenerationAggregated_old_split/Young_test_set.csv")])
old_df = pd.concat([pd.read_csv("GenerationAggregated_old_split/Old_train_set.csv"), pd.read_csv("GenerationAggregated_old_split/Old_validation_set.csv"), pd.read_csv("GenerationAggregated_old_split/Old_test_set.csv")])

young_df["origin"] = "young"
old_df["origin"] = "old"
df_old_young_union = pd.concat([young_df, old_df])

all_ids = list(set(df_old_young_union["id_original"]))
train_ids = random.sample(sorted(all_ids), int(len(all_ids)*0.8))

df_young_train = young_df[young_df["id_original"].isin(train_ids)]
df_old_train = old_df[old_df["id_original"].isin(train_ids)]

df_old_young_union_test = df_old_young_union[~df_old_young_union["id_original"].isin(train_ids)]

print(len(df_young_train), len(df_old_train), len(df_old_young_union_test), len(set(df_old_young_union_test["id_original"])))
print(df_young_train["label"].value_counts())
print(df_old_train["label"].value_counts())

#df_young_train.to_csv("GenerationAggregated/Young_train_set.csv")
#df_old_train.to_csv("GenerationAggregated/Old_train_set.csv")
#df_old_young_union_test.to_csv("GenerationAggregated/Old_Young_union_test_set.csv")
