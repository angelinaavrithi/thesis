import time
import nltk
import re
import string
import scipy.sparse
import spacy
import pprint
import statistics
# noinspection PyPackageRequirements
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from ast import literal_eval

import unidecode as unidecode
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer


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


def preprocess_word(w):
    # Removes punctuation
    translator = str.maketrans('', '', string.punctuation)
    punctuation = w.translate(translator)
    return punctuation


def preprocessing(x, language):
    # Removes mentions, numbers, links, and other unwanted elements
    mentions = [re.sub(r'@\w+', "", sent) for sent in x]
    numbers = [re.sub('[0-9]+', "", sent) for sent in mentions]
    links = [re.sub(r'http\S+', "", sent) for sent in numbers]
    emoji = [re.sub("[\U0001F600-\U0001F64F]+", "", sent) for sent in links]
    symbols = [re.sub("[\U0001F300-\U0001F5FF]+", "", sent) for sent in emoji]
    spanish_punct = [sent.replace('¿', '').replace('¡', '') for sent in symbols]
    if language == 'spanish':
        decoded = [unidecode.unidecode(sent) for sent in spanish_punct]
    else:
        decoded = spanish_punct
    duplicates = [re.sub(r'(.)\1+$', r'\1', sent) for sent in decoded]

    # Removes stopwords
    stop_words = set(stopwords.words(language))
    filtered_sentence = [w for w in duplicates if not w.lower() in stop_words]

    # Lower text, word tokenization
    lower = [[sent.lower()] for sent in filtered_sentence]
    in_list = [word for sent in lower for word in sent]
    word_tokenized = [word_tokenize(sent) for sent in in_list]
    word_tokenized = [sent for sent in word_tokenized if sent]

    for _id, sent in enumerate(word_tokenized):
        word_tokenized[_id] = [preprocess_word(w) for w in sent]

    # Removes empty elements, sentences, and retweets
    words = [[word for word in sent if word != '' and word != 'rt' and len(word) > 1] for sent in word_tokenized]
    sentences = [sent for sent in words]
    retain_index = [idx for (idx, sent) in enumerate(sentences) if len(sent) > 0]
    return sentences, retain_index


def flatten_list(x):
    # Takes a nested list and converts it into a list of elements where every sublist is a new element
    new_list = [" ".join(sent) for sent in x]
    return new_list


def build_bow(text, max_vocabulary_size):
    final_str = ([" ".join(x) for x in text])
    print("Building Bag of Words")
    count_vect = CountVectorizer(max_features=max_vocabulary_size)
    bow = count_vect.fit_transform(final_str).toarray()
    vocab = count_vect.get_feature_names_out()
    return bow, vocab


def build_embeddings(text):
    print("\nBuilding word2vec")
    model = Word2Vec(sentences=text, window=5, min_count=1, workers=4)
    model.save("word2vec.model")
    embeddings = [model.wv[word] for word in text]
    print("Averaging text embeddings")
    v_average = [np.mean(emb, axis=0) for emb in embeddings]
    return v_average


def build_syntax_graphs(text, language):
    print("\nFlattening text")
    new = flatten_list(text)
    wordset = set(word for sentence in text for word in sentence)
    print(wordset)
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

    index = [i for i in range(5, len(text)) if i % 5 == 0]
    for x, y in zip(index, timestamps1):
        print("Creating graphs: ", x, " sentences in ", y, "seconds")

    print("\nBuilding syntactic graphs:")
    start_time2 = time.time()
    timestamps2 = []

    for sent_id, sent in enumerate(processed_sentences):
        sentence_graph = base_graph.copy()
        for token in sent:
            nodeA = token.text
            nodeB = token.head.text
            sentence_graph.add_edge(nodeA, nodeB)
        sentence_representation = nx.adjacency_matrix(sentence_graph)  # sparse matrix
        rep[sent_id] = sentence_representation
        if sent_id % 10 == 0:
            timestamps2.append(time.time() - start_time2)
            print("Calculated", len(timestamps2) * 5, "out of", 5 * round((len(processed_sentences) / 2) / 5), "steps")

    print("\nFlattening adjacency matrices")
    key_order = sorted(rep.keys())
    reshaped_list = [np.reshape(rep[id], (1, -1)) for id in key_order]

    min_len = min([vector.shape[1] for vector in reshaped_list])
    resized_list = [vector.tocsr()[:1, :min_len] for vector in reshaped_list]
    arr = scipy.sparse.vstack(resized_list)
    return arr

from sklearn.preprocessing import LabelEncoder
import numpy as np
import scipy.sparse

def build_syntax_labels(text, language):
    print("\nFlattening text")
    new = flatten_list(text) 
    processed_sentences = []

    if language == "english":
        nlp = spacy.load("en_core_web_sm")
    elif language == "spanish":
        nlp = spacy.load("es_core_news_sm")
    elif language == "greek":
        nlp = spacy.load("el_core_news_sm")

    print("\nExtracting dependency labels:")
    max_len = 0  # track the longest sentence

    for sent_id, sent in enumerate(new):
        doc = nlp(sent)  
        sentence_dependencies = [token.dep_ for token in doc]
        processed_sentences.append(sentence_dependencies)
        max_len = max(max_len, len(sentence_dependencies))

    # padding
    padded_sentences = []
    for sentence in processed_sentences:
        while len(sentence) < max_len:
            sentence.append("PAD") 
        padded_sentences.append(sentence)

    flattened_sentences = [label for sentence in padded_sentences for label in sentence]

    # convert labels into integers
    le = LabelEncoder()
    le.fit(flattened_sentences) 

    encoded_sentences = [le.transform(sentence) for sentence in padded_sentences]

    sparse_syntax_matrix = scipy.sparse.csr_matrix(encoded_sentences)
    print(sparse_syntax_matrix)

    return sparse_syntax_matrix  


def classify_and_report(x, y, method_name):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
    logr = LogisticRegression()
    logr.fit(x_train, y_train)
    predictions = logr.predict(x_test)
    report = classification_report(y_test, predictions)
    print(f"Classifying {method_name} || Logistic Regression")
    print(report)


def print_matrix_info(matrix, name="Matrix"):
    """Prints the shape and size of a matrix"""
    print(f"{name} shape: {matrix.shape}")
    print(f"{name} number of samples (rows): {matrix.shape[0]}")
    print(f"{name} number of features (columns): {matrix.shape[1]}")
    print(f"{name} memory usage: {matrix.data.nbytes / 1024**2:.2f} MB")


def main():
    print("\n-------PREPROCESSING--------\n")
    # Configuration
    language = 'english'  # Set to None for interactive runs
    max_vocabulary_size = None  # Limit the vocabulary size, or set to None for unrestricted vocab
    num_limit_data = None  # Limit the number of data, set to None for no limiting

    language = get_language(language)
    dataset_path = "dataset_" + language + ".txt"

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
        from sklearn.model_selection import StratifiedShuffleSplit
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=num_limit_data, random_state=42)
        idx_to_keep = list(splitter.split(df, df['class']))[0][1]
        df = df.iloc[idx_to_keep]
        print("Limited dataset to", len(df), "instances")

    tweets = df["tweet"].tolist()

    print("Loading data...")
    text, retained_index = preprocessing(tweets, language)

    print("Fetching retained instances")
    text = [text[i] for i in retained_index]
    df = df.iloc[retained_index]

    # Print preprocessed data for cross-checking
    preprocessed_text = flatten_list(text)
    df['preprocessed_tweet'] = preprocessed_text
    df.to_csv(f"preprocessed_data_{language}.csv", index=False, encoding='utf-8')
    print(f"\nPreprocessed data exported to preprocessed_data_{language}.csv")

    print("\n-------REPRESENTATION--------\n")

    # Bag-of-Words representation
    bow, vocab = build_bow(text, max_vocabulary_size)

    # Embeddings representation
    v_average = build_embeddings(text)

    # Syntax representation
    arr = build_syntax_labels(text, language)

    print("\n-------CLASSIFICATION-------\n")

    y = df['class'].astype(int)

    # Classification using Bag-of-Words
    classify_and_report(bow, y, "BOW")

    # Classification using embeddings
    classify_and_report(v_average, y, "Embeddings")

    # Classification using both Bag-of-Words and embeddings
    conc = np.concatenate([bow, v_average], axis=1)
    classify_and_report(conc, y, "BOW + Embeddings")

    # Classification using syntax
    classify_and_report(arr, y, "Syntax")

    # Combine all representations
    print("\nCombining BoW, Embeddings, and Syntax representations")
    arr_dense = arr.toarray()
    combined_features = np.concatenate([bow, v_average, arr_dense], axis=1)
    classify_and_report(combined_features, y, "Combined (BoW + Embeddings + Syntax)")


if __name__ == "__main__":
    main()
