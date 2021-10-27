#!/usr/bin/env python
# coding: utf-8

# ### TEXT PREPROCESSING

# In[1]:


import nltk
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from ast import literal_eval
import re, string
import pandas as pd

# In[ ]:


# In[2]:


df = pd.read_csv("labeled_data.csv")
tweets = df['tweet'].tolist()


# In[3]:


def preprocess_word(w):
    # Removes punctuation

    translator = str.maketrans('', '', string.punctuation)
    punctuation = w.translate(translator)

    return punctuation


# In[4]:


def preprocessing(x):
    # Returns a nested list of the processed sentences

    # Removes mentions, numbers and links
    mentions = [re.sub(r'@\w+', "", sent) for sent in tweets]
    numbers = [re.sub('[0-9]+', "", sent) for sent in mentions]
    links = [re.sub(r'http\S+', "", sent) for sent in numbers]

    # Removes stopwords
    stop_words = set(stopwords.words('english'))
    filtered_sentence = [w for w in links if not w.lower() in stop_words]

    # Removes lower text, word tokenization
    lower = [[sent.lower()] for sent in filtered_sentence]
    in_list = [word for sent in lower for word in sent]
    word_tokenized = [word_tokenize(sent) for sent in in_list]
    word_tokenized = [sent for sent in word_tokenized if sent]

    for _id, sent in enumerate(word_tokenized):
        word_tokenized[_id] = [preprocess_word(w) for w in sent]

    # Removes empty elements, sentences and retweets
    words = [[word for word in sent if word != '' and word != 'rt' and len(word) > 1] for sent in word_tokenized]
    sentences = [sent for sent in words if sent]

    return sentences


# In[5]:


text = preprocessing(tweets)
print(text[:50])

# ### BAG OF WORDS

# In[6]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from ast import literal_eval

try:
    assert (literal_eval(str(text)) == text.copy())
except AssertionError:
    print('failed to convert')

final_str = ([" ".join(x) for x in text])

count_vect = CountVectorizer()
bow = count_vect.fit_transform(final_str).toarray()
print(bow[:50])

# In[7]:


vocab = count_vect.get_feature_names()

# In[8]:


# Returns the frequency of every word in total
# sumindex = [sum(x) for x in zip(*bow)]


# ### TF-IDF

# In[9]:


# from sklearn.feature_extraction.text import TfidfVectorizer

# tfidf_vect = TfidfVectorizer()
# tfidf = count_vect.fit_transform(final_str).toarray()
# print(tfidf[:50])


# ### EMBEDDINGS 

# In[10]:


from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import numpy as np

model = Word2Vec(sentences=text, window=5, min_count=1, workers=4)
model.save("word2vec.model")

vector = model.wv['no']  # returns numpy vector of a word
sims = model.wv.most_similar('no', topn=10)  # returns similar words

embeddings = [model.wv[word] for word in text]

# Calculate the word vector average for every sentence:

# In[11]:


v_average = []
for i in text:
    av = np.mean(model.wv[i], axis=0)
    v_average.append(av)

# from sklearn.decomposition import IncrementalPCA
# from sklearn.manifold import TSNE                 
# import numpy as np                                  
# 
# 
# def reduce_dimensions(model):
#     num_dimensions = 2 
# 
# 
#     vectors = np.asarray(model.wv.vectors)
#     labels = np.asarray(model.wv.index_to_key) 
# 
#     tsne = TSNE(n_components=num_dimensions, random_state=0)
#     vectors = tsne.fit_transform(vectors)
# 
#     x_vals = [v[0] for v in vectors]
#     y_vals = [v[1] for v in vectors]
#     return x_vals, y_vals, labels
# 
# 
# x_vals, y_vals, labels = reduce_dimensions(model)
# 
# def plot_with_plotly(x_vals, y_vals, labels, plot_in_notebook=True):
#     from plotly.offline import init_notebook_mode, iplot, plot
#     import plotly.graph_objs as go
# 
#     trace = go.Scatter(x=x_vals, y=y_vals, mode='text', text=labels)
#     data = [trace]
# 
#     if plot_in_notebook:
#         init_notebook_mode(connected=True)
#         iplot(data, filename='word-embedding-plot')
#     else:
#         plot(data, filename='word-embedding-plot.html')
# 
# 
# def plot_with_matplotlib(x_vals, y_vals, labels):
#     import matplotlib.pyplot as plt
#     import random
# 
#     random.seed(0)
# 
#     plt.figure(autosize=True)
#     plt.scatter(x_vals, y_vals)
# 
#     indices = list(range(len(labels)))
#     selected_indices = random.sample(indices, 25)
#     for i in selected_indices:
#         plt.annotate(labels[i], (x_vals[i], y_vals[i]))
# 
# try:
#     get_ipython()
# except Exception:
#     plot_function = plot_with_matplotlib
# else:
#     plot_function = plot_with_plotly
# 
# plot_function(x_vals, y_vals, labels)

# ### SYNTAX

# In[12]:


import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Masking

padded = pad_sequences(text, padding="post", truncating="post", maxlen=10, dtype=object)


# In[13]:


# print(padded)


# In[14]:


# print(type(padded))
# print(type(padded[0]))
# print(type(padded[0][0]))


# In[15]:


def to_string(x):
    sequences = []

    for sent in x:
        sentences = sent.tolist()
        sequences.append([str(word) for word in sentences])

    return sequences


sequences = to_string(padded)
print(sequences[:50])


# In[16]:


def flatten_list(x):
    # Takes a nested list and converts it into a list of elements
    # where every sublist is a new element

    new_list = []

    for sent in x:
        sentences = " ".join(sent)
        new_list.append(sentences)

    return new_list


new = flatten_list(sequences)
# print(new)


# In[17]:


import spacy


def dependency_parsing(x):
    # Returns a nested list of syntactic label

    nlp = spacy.load("en_core_web_sm")
    dependencies = []
    for sent in x:
        doc = nlp(sent)
        new_list = [token.dep_ for token in doc]
        dependencies.append(new_list)

    return dependencies


dep = dependency_parsing(new)

# In[18]:


import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import string
import pprint

get_ipython().run_line_magic('matplotlib', 'inline')

# In[19]:


# Append every word to a wordset
wordset = set()
for sentence in text:
    for word in sentence:
        wordset.add(word)

# In[20]:


print(wordset)

# In[22]:


# Add every word of the dataset as a node
base_graph = nx.Graph()
base_graph.add_nodes_from(wordset)
# nx.draw(base_graph, with_labels=True)


# In[ ]:


rep = {}
processed_sentences = []
nlp = spacy.load("en_core_web_sm")

for sent_id, sent in enumerate(new):
    sentence_graph = base_graph.copy()
    processed_sentences.append(nlp(sent))

# In[ ]:


# Add edges between the nodes according to syntactic relations
for sent_id, sent in enumerate(processed_sentences):
    for token in sent:
        nodeA = token.text
        nodeB = token.head.text
        sentence_graph.add_edge(nodeA, nodeB)
        sentence_representation = nx.adjacency_matrix(sentence_graph)  # sparse matrix
        rep[sent_id] = sentence_representation.toarray()

# In[ ]:


# Flatten the sentence representation array
arr = []
for outer_list_id, outer_list in enumerate(rep.values()):
    arr.append([item for inner_list in outer_list for inner_list in outer_list])

# In[ ]:


# [item for inner_list in outer_list for inner_list tin outer_list]


# In[ ]:


# arr = {}
# for sent_id, outer_list in rep.items():
#  ...
#  arr[sent_id] = ....


# In[ ]:


print('List:     ', type(arr), len(arr))
print('Sentence: ', type(arr[0]), len(arr[0]))
print('Item:     ', type(arr[0][0]))

# In[ ]:


options = {
    "font_size": 20,
    "node_size": 30,
    "node_color": "white",
    "edgecolors": 'blue',
    "linewidths": 1,
    "width": 1,
}
plt.figure(3, figsize=(33, 33))
nx.draw(sentence_graph, with_labels=True, **options)

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.spy(sentence_representation)
plt.title("Sentence Representation Sparsity");

# ### CLASSIFICATION

# #### Classification using Bag-of-Words:

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC

# In[ ]:


x = bow
y = df['class'].astype(int)

# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
logr = LogisticRegression()
logr.fit(x_train, y_train)
bow_predictions = logr.predict(x_test)
print(bow_predictions)

# In[ ]:


bow_report = classification_report(y_test, bow_predictions)
print(bow_report)

# #### Classification using embeddings:

# In[ ]:


v_train, v_test, y_train, y_test = train_test_split(v_average, y, test_size=0.25, random_state=0)
logr.fit(v_train, y_train)
emb_predictions = logr.predict(v_test)
print(emb_predictions)

# In[ ]:


emb_report = classification_report(y_test, emb_predictions)
print(emb_report)

# #### Classification using both Bag-of-Words and embeddings:

# In[ ]:


conc = np.concatenate([bow, v_average], axis=1)

# In[ ]:


c_train, c_test, y_train, y_test = train_test_split(conc, y, test_size=0.25, random_state=0)
logr.fit(c_train, y_train)
bow_emb_predictions = logr.predict(c_test)
print(bow_emb_predictions)

# In[ ]:


bow_emb_report = classification_report(y_test, bow_emb_predictions)
print(bow_emb_report)

# #### Classification using syntax

# In[ ]:


g_train, g_test, y_train, y_test = train_test_split(arr, y, test_size=0.25, random_state=0)
logr.fit(g_train, y_train)
syntax_predictions = logr.predict(g_test)
print(syntax_predictions)

# In[ ]:


syntax_report = classification_report(y_test, syntax_predictions)
print(syntax_report)

# In[ ]:


# if laptop dies, use some dimensionality reduction method (eg PCA)
