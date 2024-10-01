import re 
import spacy

nlp = spacy.load("en_core_web_lg")
stopwords = nlp.Defaults.stop_words

def cleaning(sent):
  string = sent.lower()
  string = re.sub(r"((http|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?)", " ", string, flags=re.U)
  # string = re.sub(r'[@#]', ' ', string, flags=re.U)
  string = re.sub(r"([\.\,\!\?\;\:\-\_\|“”\"\'\\\/\%\&\$\£\€\@\#\[\]\)\(])", r" \1 ", string)
  string = re.sub(r"([0-9]+(\.[0-9]+)?)",r" \1 ", string)
  # string = re.sub(r"[^a-zA-Z0-9àèéìùòç“”\.\,\!\?\;\:\-\_\|\"\'\\\/\%\&\$\£\€\@\#\[\]\)\(]",r" ", string)
  string = re.sub(r"[^a-zA-Z0-9àèéìùòç]",r" ", string)
  string = re.sub(r"\s+",r" ", string)

  return string.strip()


def tokenizer(sent):
  string = cleaning(sent)
  list_tokens= string.split(' ')

  return list_tokens


def tokenizer_spacy(sent):
  string = cleaning(sent)
  tokenized_sent = []
  for i in nlp(string):
    tokenized_sent.append(i)
  
  return tokenized_sent


def lemmatize_spacy(sent):
  lemmatized_sent=[]
  sent_cleaned= cleaning(sent)
    
  for i in nlp(sent_cleaned):
    if i.text not in stopwords:
        lemmatized_sent.append(i.lemma_)
                         
  return ' '.join(lemmatized_sent)