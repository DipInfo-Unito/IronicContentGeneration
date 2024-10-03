import pandas as pd
import numpy as np 

import seaborn as sns
import matplotlib.pyplot as plt

import spacy
import spacy_udpipe

#uncomment if needed
# spacy.cli.download("en_core_web_lg")
# spacy_udpipe.download("en") 
nlp = spacy.load("en_core_web_lg")


from utils.Preprocessing import cleaning, tokenizer, tokenizer_spacy, lemmatize_spacy

#Number of tokens
def len_token (df, column, case_name):
  len_token = []
  for ans in df[column]:
    tokens = tokenizer_spacy(ans)
    len_token.append(len(tokens))
  print(f"mean token in {case_name} replies: ", round(np.average(len_token)))

  return len_token


#Interjection
def get_interjection(list_txt):
    nlp = spacy_udpipe.load('en')
    interjections=[]
    for i in list_txt:
        txt = nlp(i)
        for t in txt:
            if t.pos_ == 'INTJ':
                # print(t.pos_)
                interjections.append(t.lemma_)
    return interjections


#Negation
def get_negation (list_txt):
    negations = []
    for i in list_txt:
        txt = nlp(i)
        for token in txt:
            if token.dep_ == "neg":
                negations.append(token.text)
    
    return negations

#Type-token ratio
def ttr(lista):
    tokens=[]
    for i in lista:
        l = tokenizer(i)
        tokens.extend(l)
    ttr = len(list(set(tokens)))/len(tokens)

    return ttr


#Named entities
def list_entities(df, column_text, column_id):
    # list_txt = df[column_text].tolist()
    labels= ['WORK_OF_ART', 'ORG', 'PERSON', 'GPE', 'LOC', 'EVENT', 'NORP', 'PRODUCT', 'DATE', 'LANGUAGE', 'LAW']    
    nlp = spacy.load("en_core_web_lg")
    entities = []
    id_entities = {id: [] for id in df[column_id]}

    texts = df[column_text].tolist()
    ids = df[column_id].tolist()

    for txt, id in zip(texts, ids):
        doc = nlp(str(txt))
        for t in doc.ents:
             if t.label_ in labels:
                entities.append(t.text)
                id_entities[id].append(t.text)
        

    return len(entities), id_entities


#matching named entities
def matching_entities (dict_1, dict_2):
    count_1 = 0
    count_2 =0
    count_match =0
    list_matches = []
    list_ner_in = []

    for key,value in dict_1.items():
        if len(value) != 0:
            count_1 +=1
        for k,v in dict_2.items():
            if key == k:
                if len(value) != 0 and len(v) != 0:
                        count_match +=1
                        list_matches.append(key)

    for k,v in dict_2.items():
        if len(v) != 0:
            count_2 +=1
            list_ner_in.append(k)

    print("full values in parent: ", count_1)
    print("full values in generated: ", count_2)
    print("matching cases: ", count_match)

    return count_1, count_2, count_match, list_ner_in


def ner_df (dict_parent, dict_h, dict_g, list_ner_in, model, DIR, naming):

    df_ner_parent = pd.DataFrame(dict_parent.items(), columns=["id_original", "ner_parent"])
    df_ner_h = pd.DataFrame(dict_h.items(), columns=["id_original", "ner_original"])
    df_ner_g = pd.DataFrame(dict_g.items(), columns=["id_original", "ner_aggregated"])

    df_ner = pd.merge(df_ner_parent, df_ner_g, on="id_original")
    df_ner = pd.merge(df_ner, model[["id_original", "parent_text","aggregated"]], on="id_original")
    df_ner_match = df_ner[df_ner.id_original.isin(list_ner_in)].reset_index(drop=True)
    df_ner_match.to_csv(DIR + f"ner_{naming}.csv")

    return df_ner_match



#Nominal utterances 
def nominal_utterance(sent):
    doc = nlp(sent)

    for token in doc: 
        if token.pos_ == "NOUN" or token.pos_ == "PROPN" or token.pos_ == "PRON":
            if len([child for child in token.children]) >0:
                for child in token.children:
                    if child.pos_=='NOUN' or child.pos_=='ADJ' or child.pos_=='ADV' or child.pos_=='PROPN' or child.pos_=='NUM':
                         return True

        elif (token.pos_=='VERB' or token.pos_=='ADJ' or token.pos_=='ADV') and (token.dep_=='nsubj' or token.dep_=='obl' or token.dep_=='obj' or token.dep_=='iobj'):
            if len([child for child in token.children]) >0:
                for child in token.children:
                    if child.pos_=='NOUN' or child.pos_=='ADJ' or child.pos_=='ADV' or child.pos_=='PROPN' or child.pos_=='NUM':
                        return True

        else:
            return False
        

