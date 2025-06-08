import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import nltk
import gensim.downloader as api
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
import os
import requests
from zipfile import ZipFile


warnings.filterwarnings('ignore')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
data = pd.read_csv('train_essays_.csv')
print("Dataset Info:")
print(data.info())
print("\nLabel Distribution:")
print(data['generated'].value_counts())

minority_count = data['generated'].value_counts().min()
k_neighbors = min(2, minority_count - 1)

# Text preprocessing
def preprocess_text(text):
    if not isinstance(text, str):
        text = str(text)
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(t) for t in tokens]

data['tokens'] = data['text'].apply(preprocess_text)
data['processed_text'] = data['tokens'].apply(lambda tokens: ' '.join(tokens))
y_original = data['generated']

# Evaluation function
def train_and_evaluate(X, y, name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = SVC(kernel='linear', class_weight='balanced')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'confusion_matrix_{name.lower()}.png')
    plt.close()
    return acc, prec, rec, f1

results = {}

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(data['processed_text']).toarray()
X_tfidf, y_tfidf = SMOTE(random_state=42, k_neighbors=k_neighbors).fit_resample(X_tfidf, y_original)
results['TF-IDF'] = train_and_evaluate(X_tfidf, y_tfidf, 'TF-IDF')

# Word2Vec
w2v_model = api.load("word2vec-google-news-300")
def average_word_vectors(tokens, model, dim=300):
    vecs = [model[w] for w in tokens if w in model]
    return np.mean(vecs, axis=0) if vecs else np.zeros(dim)
X_w2v = np.array([average_word_vectors(t, w2v_model) for t in data['tokens']])
X_w2v, y_w2v = SMOTE(random_state=42, k_neighbors=k_neighbors).fit_resample(X_w2v, y_original)
results['Word2Vec'] = train_and_evaluate(X_w2v, y_w2v, 'Word2Vec')

# Doc2Vec
tagged = [TaggedDocument(words=tokens, tags=[i]) for i, tokens in enumerate(data['tokens'])]
d2v_model = Doc2Vec(tagged, vector_size=100, window=5, min_count=2, workers=4, epochs=20)
X_d2v = np.array([d2v_model.infer_vector(t) for t in data['tokens']])
X_d2v, y_d2v = SMOTE(random_state=42, k_neighbors=k_neighbors).fit_resample(X_d2v, y_original)
results['Doc2Vec'] = train_and_evaluate(X_d2v, y_d2v, 'Doc2Vec')

# GloVe
# Download if not found
if not os.path.exists("glove.6B.100d.txt"):
    print("Downloading GloVe embeddings...")
    url = "http://nlp.stanford.edu/data/glove.6B.zip"
    zip_path = "glove.6B.zip"
    with open(zip_path, "wb") as f:
        f.write(requests.get(url).content)
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall()
    os.remove(zip_path)

def load_glove(path):
    glove = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            glove[word] = np.asarray(values[1:], dtype='float32')
    return glove

glove_model = load_glove('glove.6B.100d.txt')
X_glove = np.array([average_word_vectors(t, glove_model, dim=100) for t in data['tokens']])
X_glove, y_glove = SMOTE(random_state=42, k_neighbors=k_neighbors).fit_resample(X_glove, y_original)
results['GloVe'] = train_and_evaluate(X_glove, y_glove, 'GloVe')

# BERT
bert_model = SentenceTransformer('all-MiniLM-L6-v2')
X_bert = bert_model.encode(data['processed_text'], convert_to_numpy=True)
X_bert, y_bert = SMOTE(random_state=42, k_neighbors=k_neighbors).fit_resample(X_bert, y_original)
results['BERT'] = train_and_evaluate(X_bert, y_bert, 'BERT')

# Results comparison plot
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.15
x = np.arange(len(results))
for i, metric in enumerate(metrics):
    scores = [results[m][i] for m in results]
    ax.bar(x + i * bar_width, scores, bar_width, label=metric)
ax.set_xticks(x + bar_width * 1.5)
ax.set_xticklabels(results.keys())
ax.set_title('Performance Comparison Across Embedding Techniques')
ax.legend()
plt.tight_layout()
plt.savefig('embedding_comparison.png')
plt.close()

# t-SNE clustering visualization for BERT
X_vis = TSNE(n_components=2, random_state=42).fit_transform(X_bert)
kmeans = KMeans(n_clusters=2, random_state=42).fit(X_bert)
clusters = kmeans.labels_
plt.figure(figsize=(8, 6))
plt.scatter(X_vis[:, 0], X_vis[:, 1], c=clusters, cmap='viridis', alpha=0.6, label='K-Means Clusters')
plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y_bert, cmap='coolwarm', marker='x', alpha=0.3, label='True Labels')
plt.legend()
plt.title('t-SNE Clustering on BERT Embeddings')
plt.savefig('tsne_kmeans_bert.png')
plt.close()

# Theoretical formalism summary
print("\nTheoretical Formalism Summary:")
print("TF-IDF: TF(t,d) * log(N / (1 + df(t)))")
print("Word2Vec: CBOW/Skip-Gram predict word embeddings.")
print("Doc2Vec: Paragraph vector techniques like PV-DM or PV-DBOW.")
print("GloVe: Matrix factorization on global word co-occurrence.")
print("BERT: Contextual embeddings using Transformer-based models.")
