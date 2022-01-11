import time
import nltk
import re
import string
import spacy
import pprint
import pandas as pd
import numpy as np
import pandas as pd
import statistics
import networkx as nx
import matplotlib.pyplot as plt
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from ast import literal_eval
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer


print("\n-------PREPROCESSING--------\n")

def read_english_file(filename):
    lst = []
    with open(filename, encoding="utf8") as file:
        for line in file:
            line_content = line.rstrip()
            line_sep = line_content.split(sep=",", maxsplit=6)
            lst.append(line_sep)

    return lst

def read_spanish_file(filename):
    lst = []
    with open(filename, encoding="utf8") as file:
        for line in file:
            line_content = line.rstrip()
            line_sep = line_content.split(sep=",", maxsplit=1)
            line_rsep = line_sep[1].rsplit(sep=",", maxsplit=1)
            line_rsep.append(line_sep[0])
            lst.append(line_rsep)

    return lst

def read_greek_file(filename):
    lst = []
    with open(filename, encoding="utf8") as file:
        for line in file:
            line_content = line.rstrip()
            line_sep = line_content.split(sep=",", maxsplit=2)
            lst.append(line_sep)

    return lst


# Configuration
###################

# language: set to None for interactive runs
language = 'english'
# limit the vocabulary size, or set to None for unrestricted vocab
max_vocabulary_size = 125
# limit the number of data, for testing purposes. set to None for no limiting
num_limit_data = 100

#################


def get_language(language=None):
    
    if language is None:
        available = ["english", "spanish", "greek"]
        keyb = input("Choose language: type English, Spanish or Greek.")
        language = keyb.strip().lower()
        while language not in available:
            print("Invalid input -- supported languages are", available)
            if language is not None:
                # if not interactive, exit
                exit(1)

    print("Running with language:", language)

    return language

language = get_language(language)
dataset_path = "dataset_"+ language +".txt"

if language == "english":
    data = read_english_file(dataset_path)
elif language == "spanish":
    data = read_spanish_file(dataset_path)
elif language == "greek":
    dirty_data = read_greek_file(dataset_path)
    data = [dirty_data[0]] + [line for line in dirty_data[1:] if line[0].startswith("1")]

df = pd.DataFrame(data)
df.columns = df.iloc[0]
df = df.iloc[1:, :]
df = df.dropna()
df[df.astype(str)['tweet'] != '[]']
df = df[df['class'] != '']

if num_limit_data is not None:
    # use shuffle-splitting to make sure the data limiting keeps
    # stuff from all classes, so that classifiers don't complain
    from sklearn.model_selection import StratifiedShuffleSplit
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=num_limit_data, random_state=42)
    idx_to_keep = list(splitter.split(df, df['class']))[0][1]
    df = df.iloc[idx_to_keep]
    print("Limited dataset to", len(df), "instances")

tweets = df["tweet"].tolist()

print("Loading data...")

def preprocess_word(w):
    # Removes punctuation
    
    translator = str.maketrans('', '', string.punctuation)
    punctuation = w.translate(translator)

    return punctuation

def preprocessing(x):
    # Returns a nested list of the processed sentences

    # Removes mentions, numbers and links
    mentions = [re.sub(r'@\w+',"", sent) for sent in x]
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
    sentences = [sent for sent in words]
    # mark only non-empty sentences to retain
    retain_index = [idx for (idx, sent) in enumerate(sentences) if len(sent) > 0]
    return sentences, retain_index

print("Preprocessing", len(tweets), "tweets")
text, retained_index = preprocessing(tweets)

# apply the retained index collection to the text
# and the dataframe (to keep track of the corresp. labels)
print("Fetching retained instances")
text = [text[i] for i in retained_index]
df = df.iloc[retained_index]

print("\n-------REPRESENTATION--------\n")

# BAG OF WORDS
try: 
    assert(literal_eval(str(text)) == text.copy())
except AssertionError:
    print('Failed to convert')
    
final_str = ([" ".join(x) for x in text])

print("Building Bag of Words")
count_vect = CountVectorizer(max_features=max_vocabulary_size)
bow = count_vect.fit_transform(final_str).toarray()
vocab = count_vect.get_feature_names()


# EMBEDDINGS
print("\nBuilding word2vec")
model = Word2Vec(sentences=text, window=5, min_count=1, workers=4)
model.save("word2vec.model")

embeddings = [model.wv[word] for word in text]

# Calculate the word vector average for every sentence:
print("Averaging text embeddings")
v_average = [np.mean(emb, axis=0) for emb in embeddings]


# SYNTAX
def flatten_list(x):
# Takes a nested list and converts it into a list of elements
# where every sublist is a new element

    new_list = []

    for sent in x:
        sentences = " ".join(sent)
        new_list.append(sentences)

    return new_list

print("\nFlattening text")
new = flatten_list(text)

# Append every word to a wordset
wordset = set()
for sentence in text:
    for word in sentence:
        wordset.add(word)

# Add every word of the dataset as a node
base_graph = nx.Graph()
base_graph.add_nodes_from(wordset)

rep = {}
processed_sentences = []

if language == "english":
    nlp = spacy.load("en_core_web_sm")
elif language == "spanish":
    nlp = spacy.load("es_core_news_sm")
elif language == "greek":
    nlp = spacy.load("el_core_news_sm")

timestamps1 = []

print("Extracting dependency graphs:")
start_time1 = time.time()
for sent_id, sent in enumerate(new):
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

print("\nBuilding syntactic graphs:")

# Add edges between the nodes according to syntactic relations
start_time2 = time.time()
timestamps2 = []

for sent_id, sent in enumerate(processed_sentences):
    sentence_graph = base_graph.copy()
    for token in sent:
        nodeA = token.text
        nodeB = token.head.text
        sentence_graph.add_edge(nodeA, nodeB)

    sentence_representation =  nx.adjacency_matrix(sentence_graph) #sparse matrix
    rep[sent_id] = sentence_representation
    if sent_id % 10 == 0:
        timestamps2.append(time.time() - start_time2)
        print("Calculated", len(timestamps2) * 5, "steps")

# for x,y in zip(index, timestamps2):
#     print("Adding edges: ", x, " sentences in ", y, "seconds")

# Flatten the sentence representation array
print("\nFlattening adjacency matrices")

# Select a fixed vocabulary size as a statistic on document vocabularies
key_order = sorted(rep.keys())

# Reshape - convert every vector to 1D
reshaped_list = []
for id in key_order:
    outer_list = rep[id]
    reshaped_vector = np.reshape(outer_list, (1, -1))
    reshaped_list.append(reshaped_vector)

# Calculate mean
mean = round(statistics.mean([vector.shape[1] for vector in reshaped_list]))
print("MEAN IS ", mean)

# Resize
# Truncate long vectors
# Add zeros to short vectors
arr = []
for vector in reshaped_list:
    csr_vector = vector.tocsr()
    resized_vector = csr_vector[0, :mean]
    # + ADD DUMMY COLUMNS TO SHORT VECTORS
    arr.append(resized_vector)

options = {
    "font_size": 20,
    "node_size": 30,
    "node_color": "white",
    "edgecolors": 'blue',
    "linewidths": 1,
     "width": 1,
 }

plt.figure(3,figsize=(33,33))
nx.draw(sentence_graph, with_labels=True, **options)

print("\n-------CLASSIFICATION-------\n")

# Classification using Bag-of-Words:
print("Classifying BOW - LR")

y = df['class'].astype(int)

x = bow
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
logr = LogisticRegression()
logr.fit(x_train, y_train)
bow_predictions = logr.predict(x_test)

bow_report = classification_report(y_test, bow_predictions)
print(bow_report)


# Classification using embeddings:
print("Classifying embeddings - LR")
v_train, v_test, y_train, y_test = train_test_split(v_average, y, test_size=0.25, random_state=0)
logr.fit(v_train, y_train)
emb_predictions = logr.predict(v_test)

emb_report = classification_report(y_test, emb_predictions)
print(emb_report)


# Classification using both Bag-of-Words and embeddings:
print("Classifying BOW + embeddings - LR")
conc = np.concatenate([bow, v_average], axis=1)

c_train, c_test, y_train, y_test = train_test_split(conc, y, test_size=0.25, random_state=0)
logr = LogisticRegression()
logr.fit(c_train, y_train)
bow_emb_predictions = logr.predict(c_test)

bow_emb_report = classification_report(y_test, bow_emb_predictions)
print(bow_emb_report)


# Classification using syntax
print("Classifying syntax - LR")

g_train, g_test, y_train, y_test = train_test_split(arr, y, test_size=0.25, random_state=0)

logr = LogisticRegression()
logr.fit(g_train, y_train)
syntax_predictions = logr.predict(g_test)

syntax_report = classification_report(y_test, syntax_predictions)
print(syntax_report)


# if laptop dies, use some dimensionality reduction method (eg PCA)
