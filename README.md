## Hate speech detection: repository for my BA thesis. 
The goal is to create an automatic tool that detects hateful text in english, spanish and greek data.

### Data: 
We use texts retrieved from Twitter. Following a multi-lingual approach, we use an english, a spanish and a greek dataset. The user is asked to choose among the three languages by typing "english", "spanish" or "greek".

Both the spanish and the greek dataset have two labels; "hateful" and "neutral". The english dataset has a third "offensive" label, which indicates a sentiment between hate and neutral.

### Feature selection:
We classify text according to four feature sets:
1. BoW (Bag-of-Words).
2. Word embeddings.
3. BoW + Embeddings.
4. Syntax. More specifically, we create a syntax graph for every sentence according to the spacy dependency parsing. Each graph is converted to its corresponding adjacency matrix.

### Classifier: 
We use a logistic regression classifier.
