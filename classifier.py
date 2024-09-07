#!/usr/bin/env python
# coding: utf-8

import re
import string
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec
from keras.preprocessing.sequence import pad_sequences
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


### LOAD & PREPROCESS DATA

# Load data
def load_data(filepath):
    df = pd.read_csv(filepath)
    return df['tweet'].tolist(), df['class'].astype(int).values


# Text preprocessing functions
def preprocess_word(w):
    translator = str.maketrans('', '', string.punctuation)
    return w.translate(translator)


def preprocessing(tweets):
    mentions = [re.sub(r'@\w+', "", sent) for sent in tweets]
    numbers = [re.sub('[0-9]+', "", sent) for sent in mentions]
    links = [re.sub(r'http\S+', "", sent) for sent in numbers]
    stop_words = set(stopwords.words('english'))
    filtered_sentence = [w for w in links if not w.lower() in stop_words]
    lower = [[sent.lower()] for sent in filtered_sentence]
    in_list = [word for sent in lower for word in sent]
    word_tokenized = [word_tokenize(sent) for sent in in_list]
    word_tokenized = [sent for sent in word_tokenized if sent]
    for _id, sent in enumerate(word_tokenized):
        word_tokenized[_id] = [preprocess_word(w) for w in sent]
    words = [[word for word in sent if word != '' and word != 'rt' and len(word) > 1] for sent in word_tokenized]
    return [sent for sent in words if sent]


### TEXT REPRESENTATIONS

# Bag-of-Words
def create_bow(final_str):
    count_vect = CountVectorizer()
    bow = count_vect.fit_transform(final_str).toarray()
    vocab = count_vect.get_feature_names_out()
    return bow, vocab


# Embeddings
def create_embeddings(text):
    model = Word2Vec(sentences=text, window=5, min_count=1, workers=4)
    model.save("word2vec.model")
    embeddings = [model.wv[word] for word in text]
    v_average = [np.mean(model.wv[sent], axis=0) for sent in text]
    return model, v_average


# Syntax processing
def create_dependencies(text):
    # Pad the sequences
    padded = pad_sequences(text, padding="post", truncating="post", maxlen=10, dtype=object)

    # Convert padded sequences to strings
    sequences = [[str(word) for word in sent.tolist()] for sent in padded]
    sentences = [" ".join(sent) for sent in sequences]

    # Dependency parsing
    nlp = spacy.load("en_core_web_sm")
    parsed_dependencies = [[token.dep_ for token in nlp(sent)] for sent in sentences]

    return parsed_dependencies


def create_graphs(text):
    wordset = {word for sentence in text for word in sentence}
    base_graph = nx.Graph()
    base_graph.add_nodes_from(wordset)

    rep = {}
    processed_sentences = []
    nlp = spacy.load("en_core_web_sm")

    for sent in text:
        sentence_graph = base_graph.copy()
        processed_sentences.append(nlp(sent))

    for sent_id, sent in enumerate(processed_sentences):
        for token in sent:
            nodeA = token.text
            nodeB = token.head.text
            sentence_graph.add_edge(nodeA, nodeB)
        sentence_representation = nx.adjacency_matrix(sentence_graph)
        rep[sent_id] = sentence_representation.toarray()

    fixed_sentence_order = sorted(rep.keys())
    arr = [np.reshape(np.concatenate(rep[sent_id]), [-1]) for sent_id in fixed_sentence_order]
    return arr


def plot_graphs(sentence_graph, sentence_representation):
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
    plt.spy(sentence_representation)
    plt.title("Sentence Representation Sparsity")
    plt.show()


# CLASSIFICATION

def classify(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
    logr = LogisticRegression()
    logr.fit(x_train, y_train)
    predictions = logr.predict(x_test)
    report = classification_report(y_test, predictions)
    return predictions, report


def main(filepath):
    tweets, y = load_data(filepath)
    text = preprocessing(tweets)
    final_str = [" ".join(x) for x in text]

    # Bag-of-Words
    bow, vocab = create_bow(final_str)

    # Embeddings
    model, v_average = create_embeddings(text)

    # Dependency parsing
    dep = create_dependencies(text)

    # Create graphs
    arr = create_graphs(text)

    # Plotting (optional, uncomment if needed)
    # plot_graphs(sentence_graph, sentence_representation)

    # Classification
    bow_predictions, bow_report = classify(bow, y)
    print("Bag-of-Words Predictions:", bow_predictions)
    print(bow_report)

    v_train, v_test, y_train, y_test = train_test_split(v_average, y, test_size=0.25, random_state=0)
    logr = LogisticRegression()
    logr.fit(v_train, y_train)
    emb_predictions = logr.predict(v_test)
    emb_report = classification_report(y_test, emb_predictions)
    print("Embeddings Predictions:", emb_predictions)
    print(emb_report)

    conc = np.concatenate([bow, v_average], axis=1)
    bow_emb_predictions, bow_emb_report = classify(conc, y)
    print("Combined BoW and Embeddings Predictions:", bow_emb_predictions)
    print(bow_emb_report)

    syntax_predictions, syntax_report = classify(arr, y)
    print("Syntax Predictions:", syntax_predictions)
    print(syntax_report)


if __name__ == "__main__":
    main("labeled_data.csv")
