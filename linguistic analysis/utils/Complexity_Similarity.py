import numpy as np
import gensim.downloader as api
from scipy.spatial.distance import cosine
import spacy 

model = api.load('word2vec-google-news-300')

#uncomment if needed
# spacy.cli.download("en_core_web_lg")
# spacy_udpipe.download("en") 
nlp = spacy.load("en_core_web_lg")


#Tree height
def get_tree_height(token):
    if not any(token.children):
        return 1  # If the token has no children, the height is 1.
    else:
        return 1 + max(get_tree_height(child) for child in token.children)  # The height is 1 plus the maximum height of its children's subtrees.

def get_sentence_tree_height(doc):
    heights = []
    for sent in doc.sents: 
        roots = [token for token in sent if token.head == token]  # Find the head
        if not roots:
            heights.append(0)
        else:
            heights.append(max(get_tree_height(root) for root in roots))
    return heights

def calculate_tree_heights(df, column_name,new_column_name):
    # Use nlp.pipe to process the sentences in batches
    sentences = df[column_name].tolist()
    docs = nlp.pipe(sentences)
    
    tree_heights = []
    for doc in docs:
        heights = get_sentence_tree_height(doc)
        if heights:
            tree_heights.append(heights[0])  # Assuming one sentence per row
        else:
            tree_heights.append(0)
    
    df[new_column_name] = tree_heights
    return df



#Text similarity
def get_average_word2vec(tokens_list, vector, generate_missing=False, k=300):
    if len(tokens_list)<1:
        return np.zeros(k)
    if generate_missing:
        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]
    else:
        vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged

def cosine_distance_wordembedding_method(s1, s2):
    vector_1 = get_average_word2vec(s1.split(), model)
    vector_2 = get_average_word2vec(s2.split(), model)
    if np.all(vector_1 == 0) or np.all(vector_2 == 0): #if one of the vectors is zero they are considered maximally dissimilar
        return 1.0
    cosine_distance = cosine(vector_1, vector_2)
    return 1-cosine_distance


def human_similarity (df):
    list_parent = df["parent_text"].astype(str).tolist()
    list_text = df["text"].astype(str).tolist()

    list_sim_human = []

    for i in range(len(list_parent)):
        parent_txt = list_parent[i]
        txt = list_text[i]
        computed_similarity = cosine_distance_wordembedding_method(parent_txt,txt)
        list_sim_human.append(computed_similarity)

    return list_sim_human