import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk.stem import WordNetLemmatizer

nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('sentiwordnet')

lemmatizer = WordNetLemmatizer()


def penn_to_wn(tag):
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None

# Returns list of pos-neg and objective score. But returns empty list if not present in senti wordnet.
def get_sentiment(word,tag):
    wn_tag = penn_to_wn(tag)
    
    if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
        return []

    #Lemmatization
    lemma = lemmatizer.lemmatize(word, pos=wn_tag)
    if not lemma:
        return []

    #Synset is a special kind of a simple interface that is present in NLTK to look up words in WordNet. 
    #Synset instances are the groupings of synonymous words that express the same concept. 
    #Some of the words have only one Synset and some have several.
    synsets = wn.synsets(word, pos=wn_tag)
    if not synsets:
        return []

    # Take the first sense, the most common
    synset = synsets[0]
    swn_synset = swn.senti_synset(synset.name())

    return [synset.name(), swn_synset.pos_score(),swn_synset.neg_score(),swn_synset.obj_score()]

    pos=neg=obj=count=0


def extract_sentiment (df):
    pos=neg=obj=count=0
    senti_score = []

    for pos_val in df['pos_tags']:
        senti_val = [get_sentiment(x,y) for (x,y) in pos_val]
        for score in senti_val:
            try:
                pos = pos + score[1]  #positive score is stored at 2nd position
                neg = neg + score[2]  #negative score is stored at 3rd position
            except:
                continue
        senti_score.append(pos - neg)
        pos=neg=0    
    
    print(senti_score)
    df['senti_score'] = senti_score
    # print(df['senti_score'])

    return df


def overall_sentiment (df):
    overall=[]
    for i in range(len(df)):
        if df['senti_score'][i]>= 0.05:
            overall.append('Positive')
        elif df['senti_score'][i]<= -0.05:
            overall.append('Negative')
        else:
            overall.append('Neutral')
    df['overall_sentiment']=overall

    return df