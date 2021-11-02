from IPython import get_ipython

### TEXT PREPROCESSING

#!py -3.8 -m pip install -r requirements.txt

import time
import nltk
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from ast import literal_eval
import re, string
import pandas as pd

def data():
    global language, dataset_path
    
    keyb = input("Choose language: type English, Spanish or Greek.")
    if(keyb.casefold() == "english"):
        language = 'english'
    elif(keyb.casefold() == "spanish"):
        language = 'spanish'
    elif(keyb.casefold() == "greek"):
        language = 'greek'
    else:
        print("Invalid input. Please type English, Spanish or Greek.")
        data()

    dataset_path = "dataset_"+ language +".txt"

    return language, dataset_path

data()

df =  pd.read_csv(dataset_path)
tweets = df["tweet"].tolist()

def preprocess_word(w):
    # Removes punctuation
    
    translator = str.maketrans('', '', string.punctuation)
    punctuation = w.translate(translator)

    return punctuation

def preprocessing(x):
    # Returns a nested list of the processed sentences
    
    # Removes mentions, numbers and links
    mentions = [re.sub(r'@\w+',"", sent) for sent in tweets]
    numbers = [re.sub('[0-9]+', "", sent) for sent in mentions]
    links = [re.sub(r'http\S+', "", sent) for sent in numbers]
    emoji = [re.sub("[\U0001F600-\U0001F64F]+", "", sent) for sent in links]
    symbols = [re.sub("[\U0001F300-\U0001F5FF]+", "", sent) for sent in emoji]

    # Removes stopwords
    stop_words = set(stopwords.words(language))
    filtered_sentence = [w for w in symbols if not w.lower() in stop_words]
    
    # Removes lower text, word tokenization
    lower = [[sent.lower()] for sent in filtered_sentence]
    in_list = [word for sent in lower for word in sent]
    word_tokenized = [word_tokenize(sent) for sent in in_list]
    word_tokenized = [sent for sent in word_tokenized if sent]
    
    for _id, sent in enumerate(word_tokenized):
        word_tokenized[_id] =  [preprocess_word(w) for w in sent]
    
    # Removes empty elements, sentences and retweets
    words = [[word for word in sent if word != '' and word != 'rt' and len(word)>1] for sent in word_tokenized]
    sentences = [sent for sent in words if sent]

    return sentences

text = preprocessing(tweets)


### BAG OF WORDS 

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from ast import literal_eval

try: 
    assert(literal_eval(str(text)) == text.copy())
except AssertionError:
    print('failed to convert')
    
final_str = ([" ".join(x) for x in text])

count_vect = CountVectorizer()
bow = count_vect.fit_transform(final_str).toarray()

vocab = count_vect.get_feature_names()


### EMBEDDINGS 

from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import numpy as np

model = Word2Vec(sentences=text, window=5, min_count=1, workers=4)
model.save("word2vec.model")

embeddings = [model.wv[word] for word in text]

# Calculate the word vector average for every sentence:
v_average = []
for i in text:
    av = np.mean(model.wv[i], axis=0)
    v_average.append(av)


### SYNTAX

def flatten_list(x):
#Takes a nested list and converts it into a list of elements
#where every sublist is a new element

    new_list = [] 
    
    for sent in x:
        sentences = " ".join(sent)
        new_list.append(sentences)
    
    return new_list

new = flatten_list(text)

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import spacy
import string
import pprint
get_ipython().run_line_magic('matplotlib', 'inline')

#Append every word to a wordset
wordset = set()
for sentence in text:
    for word in sentence:
        wordset.add(word)

#Add every word of the dataset as a node
base_graph = nx.Graph()
base_graph.add_nodes_from(wordset)

rep = {}
processed_sentences = []

if language == 'english':
    nlp = spacy.load("en_core_web_sm")
elif language == 'spanish':
    nlp = spacy.load("es_core_news_sm")
elif language == 'greek':
    nlp = spacy.load("el_core_news_sm")

timestamps1 = []

start_time1 = time.time()
for sent_id, sent in enumerate(new[:40]):
    sentence_graph = base_graph.copy()
    processed_sentences.append(nlp(sent))
    if sent_id % 5 == 0:
        timestamps1.append(time.time() - start_time1)

index = []
for i in range(5, len(text)):
    if i % 5 == 0:
        index.append(i)
    else:
        continue

for x,y in zip(index, timestamps1):
    print("Creating graphs: ", x, " sentences in ", y, "seconds")

#Add edges between the nodes according to syntactic relations
start_time2 = time.time()
timestamps2 = []
for sent_id, sent in enumerate(processed_sentences[:40]):
    for token in sent:
        nodeA = token.text
        nodeB = token.head.text
        sentence_graph.add_edge(nodeA, nodeB)
        sentence_representation =  nx.adjacency_matrix(sentence_graph) #sparse matrix
        rep[sent_id] = sentence_representation.toarray()
    if sent_id % 5 == 0:
        timestamps2.append(time.time() - start_time2)

for x,y in zip(index, timestamps2):
    print("Adding edges: ", x, " sentences in ", y, "seconds")

# Flatten the sentence representation array
arr = []
key_order = sorted(rep.keys())

for sent_id in key_order:
    outer_list = rep[sent_id]
    vector = np.reshape(vector, [-1])
    arr.append(vector)


# options = {
#    "font_size": 20,
#    "node_size": 30,
#    "node_color": "white",
#    "edgecolors": 'blue',
#    "linewidths": 1,
#    "width": 1,
#
# plt.figure(3,figsize=(33,33))
# nx.draw(sentence_graph, with_labels=True, **options)


### CLASSIFICATION

#### Classification using Bag-of-Words:
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC

x = bow
y = df['class'].astype(int)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
logr = LogisticRegression()
logr.fit(x_train, y_train)
bow_predictions = logr.predict(x_test)

bow_report = classification_report(y_test, bow_predictions)
print(bow_report)

#### Classification using embeddings:
v_train, v_test, y_train, y_test = train_test_split(v_average, y, test_size=0.25, random_state=0)
logr.fit(v_train, y_train)
emb_predictions = logr.predict(v_test)

emb_report = classification_report(y_test, emb_predictions)
print(emb_report)

#### Classification using both Bag-of-Words and embeddings: 
conc = np.concatenate([bow, v_average], axis=1)

c_train, c_test, y_train, y_test = train_test_split(conc, y, test_size=0.25, random_state=0)
logr.fit(c_train, y_train)
bow_emb_predictions = logr.predict(c_test)

bow_emb_report = classification_report(y_test, bow_emb_predictions)
print(bow_emb_report)

#### Classification using syntax
g_train, g_test, y_train, y_test = train_test_split(arr, y, test_size=0.25, random_state=0)
logr.fit(g_train, y_train)
syntax_predictions = logr.predict(g_test)

syntax_report = classification_report(y_test, syntax_predictions)
print(syntax_report)



# if laptop dies, use some dimensionality reduction method (eg PCA)