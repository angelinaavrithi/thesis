#!/usr/bin/env python
# coding: utf-8

# ### TEXT PREPROCESSING

# In[1]:


import nltk
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import sent_tokenize, word_tokenize
from ast import literal_eval
import re, string
import pandas as pd


df =  pd.read_csv("dataset.txt")
tweets = df['tweet'].tolist()

def preprocess_word(w):
    # Removes punctuation
    
    translator = str.maketrans('', '', string.punctuation)
    punctuation = w.translate(translator)

    return punctuation


def preprocessing(x):
    # Returns a nested list of the processed sentences
    
    mentions = [re.sub(r'@\w+',"", sent) for sent in tweets] #removes mentions
    numbers = [re.sub('[0-9]+', "", sent) for sent in mentions] #removes numbers
    links = [re.sub(r'http\S+', "", sent) for sent in numbers] #removes links
    
    lower = [[sent.lower()] for sent in links] #lower text
    in_list = [word for sent in lower for word in sent]
    word_tokenized = [word_tokenize(sent) for sent in in_list]
    word_tokenized = [sent for sent in word_tokenized if sent] #word tokenization
    
    for _id, sent in enumerate(word_tokenized):
        word_tokenized[_id] =  [preprocess_word(w) for w in sent]

    words = [[word for word in sent if word != '' and word != 'rt' and len(word)>1] for sent in word_tokenized] #removes useless words
    sentences = [sent for sent in words if sent] #removes empty sentences

    return sentences

text = preprocessing(tweets)


# ### BAG OF WORDS 

# In[2]:


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
print(bow)


# In[3]:


vocab = count_vect.get_feature_names()


# In[4]:


#Returns the frequency of every word in total
sumindex = [sum(x) for x in zip(*bow)]


# ### TF-IDF

# In[5]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vect = TfidfVectorizer()
tfidf = count_vect.fit_transform(final_str).toarray()
print(tfidf)


# ### EMBEDDINGS 

# In[6]:


from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import numpy as np

model = Word2Vec(sentences=text, window=5, min_count=1, workers=4)
model.save("word2vec.model")

vector = model.wv['no'] #returns numpy vector of a word
sims = model.wv.most_similar('no', topn=10) #returns similar words

embeddings = [model.wv[word] for word in text]


# Calculate the word vector average for every sentence:

# In[7]:


v_average = []
for i in text:
    av = np.mean(model.wv[i], axis=0)
    v_average.append(av)


# In[8]:


from sklearn.decomposition import IncrementalPCA  
from sklearn.manifold import TSNE                 
import numpy as np                                  


def reduce_dimensions(model):
    num_dimensions = 2 


    vectors = np.asarray(model.wv.vectors)
    labels = np.asarray(model.wv.index_to_key) 

    tsne = TSNE(n_components=num_dimensions, random_state=0)
    vectors = tsne.fit_transform(vectors)

    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
    return x_vals, y_vals, labels


x_vals, y_vals, labels = reduce_dimensions(model)

def plot_with_plotly(x_vals, y_vals, labels, plot_in_notebook=True):
    from plotly.offline import init_notebook_mode, iplot, plot
    import plotly.graph_objs as go

    trace = go.Scatter(x=x_vals, y=y_vals, mode='text', text=labels)
    data = [trace]

    if plot_in_notebook:
        init_notebook_mode(connected=True)
        iplot(data, filename='word-embedding-plot')
    else:
        plot(data, filename='word-embedding-plot.html')


def plot_with_matplotlib(x_vals, y_vals, labels):
    import matplotlib.pyplot as plt
    import random

    random.seed(0)

    plt.figure(autosize=True)
    plt.scatter(x_vals, y_vals)

    indices = list(range(len(labels)))
    selected_indices = random.sample(indices, 25)
    for i in selected_indices:
        plt.annotate(labels[i], (x_vals[i], y_vals[i]))

try:
    get_ipython()
except Exception:
    plot_function = plot_with_matplotlib
else:
    plot_function = plot_with_plotly

plot_function(x_vals, y_vals, labels)


# ### SYNTAX

# In[9]:


def flatten_list(x):
#Takes a nested list and converts it into a list of elements
#where every sublist is a new element

    new_list = [] 
    
    for sent in x:
        sentences = " ".join(sent)
        new_list.append(sentences)
    
    return new_list

new = flatten_list(text)
print(new)


# In[10]:


import spacy

def dependency_parsing(x):
# Returns a nested list of syntactic labels

    nlp = spacy.load("en_core_web_sm")
    dependencies = []
    for sent in x:
        doc = nlp(sent)
        new_list = [token.dep_ for token in doc]
        dependencies.append(new_list)
        
    return dependencies

dep = dependency_parsing(new)


# In[11]:


import networkx as nx
import matplotlib.pyplot as plt
import string
import pprint
get_ipython().run_line_magic('matplotlib', 'inline')


# In[12]:


#Append every word to a wordset
wordset = set()
for sentence in text:
    for word in sentence:
        wordset.add(word)


# In[13]:


#Add every word as a node
base_graph = nx.Graph()
base_graph.add_nodes_from(wordset)
nx.draw(base_graph, with_labels=True)


# In[14]:


repr = {}

nlp = spacy.load("en_core_web_sm")

for sentence_id, sentence_contents in enumerate(new):
    sentence_graph = base_graph.copy()
    processed_sentence =  nlp(' '.join(new))
    
print(processed_sentence)


# In[15]:


#Add edges between the nodes according to syntactic relations
for token in processed_sentence:
    nodeA = token.text
    nodeB = token.head.text
    #print('\tadding edge between', nodeA, 'and', nodeB)
    sentence_graph.add_edge(nodeA, nodeB)
    sentence_representation =  nx.adjacency_matrix(sentence_graph) #sparse matrix
    #print('\t sparse matrix has ',sentence_representation.count_nonzero(),'nonzero elements')
    repr[sentence_id] = sentence_representation


# In[16]:


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


# In[17]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.spy(sentence_representation)
plt.title("Sentence Representation Sparsity");


# In[18]:


print(sentence_representation)


# #### DataFrame

# In[19]:


A = pd.DataFrame(columns=['Tweets','Class','BoW','Embeddings'])
A['Tweets'] = [sent for sent in text]


# In[20]:


A['Class'] = df['class']
print(A)


# #### Frequency & Embeddings lists

# In[21]:


wfrequency_list = []
for sent in text:
    for word in sent:
        ind = vocab.index(word)
        #print('\n',word, '\nIndex:', ind)
        freq = sumindex[ind]
        #print("BOW frequency:", freq)
        wfrequency_list.append(freq)


# In[22]:


wembeddings_list = []
for l in embeddings:
    for subl in l:
        wembeddings_list.append(subl)


# ### CLASSIFICATION

# #### Classification using Bag-of-Words:

# In[23]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC


# In[24]:


x = bow
y = df['class'].astype(int)


# In[25]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
logr = LogisticRegression()
logr.fit(x_train, y_train)
bow_predictions = logr.predict(x_test)
print(bow_predictions)


# In[26]:


bow_report = classification_report(y_test, bow_predictions)
print(bow_report)


# #### Classification using embeddings:

# In[27]:


v_train, v_test, y_train, y_test = train_test_split(v_average, y, test_size=0.25, random_state=0)
logr.fit(v_train, y_train)
emb_predictions = logr.predict(v_test)
print(emb_predictions)


# In[28]:


emb_report = classification_report(y_test, emb_predictions)
print(emb_report)


# #### Classification using both Bag-of-Words and embeddings: 

# In[29]:


conc = np.concatenate([bow, v_average], axis=1)


# In[30]:


c_train, c_test, y_train, y_test = train_test_split(conc, y, test_size=0.25, random_state=0)
logr.fit(c_train, y_train)
bow_emb_predictions = logr.predict(c_test)
print(bow_emb_predictions)


# In[31]:


bow_emb_report = classification_report(y_test, bow_emb_predictions)
print(bow_emb_report)


# #### Classification using syntax

# In[33]:


g_train, g_test, y_train, y_test = train_test_split(repr, y, test_size=0.25, random_state=0)
logr.fit(g_train, y_train)
bow_predictions = logr.predict(g_test)
print(syntax_predictions)

