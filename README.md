# IMDB NLP — Trigram Language Model & Skip-Gram Word Embeddings 🎬📝

An NLP pipeline on the IMDB review dataset covering trigram-based probabilistic 
language modeling for sentence recommendation, full text preprocessing, Skip-Gram 
word embeddings, cosine similarity analysis, and 2D PCA visualization of semantic 
space.

---

## 🧩 Problem Statement

### Part I — Sentence Recommendation using Trigram Language Model
Design a probabilistic trigram language model trained on the IMDB corpus to 
recommend which of two test sentences is most relevant to the training data.

### Part II — Text Preprocessing & Feature Extraction
Implement a full NLP preprocessing pipeline and extract word features using 
Skip-Gram (Word2Vec) embeddings for similarity analysis.

### Part III — Similarity Analysis & Visualization
Identify the top 2 most similar words using cosine similarity and visualize 
word embeddings in 2D semantic space using PCA.

---

## 📦 Dataset

- **File:** `IMDB.csv`
- **Columns:** `ID`, `review`
- **Content:** IMDB movie reviews used as training corpus

---

## 🔬 Methodology

### Part I — Trigram Language Model

**Preprocessing:**
- Remove HTML tags (`<br />` etc.)
- Remove non-alphabetic characters
- Lowercase all text

**Model:**
- Built using NLTK `trigrams` + `defaultdict(Counter)`
- Trigram probability: `P(w3 | w1, w2) = count(w1,w2,w3) / count(w1,w2)`
- Smoothing: unseen trigrams assigned probability `1e-6`

**Test Sentences:**
| Sentence | Probability |
|----------|------------|
| *"The first thing that struck me about Oz was its brutality..."* | 1.76 × 10⁻¹¹ |
| *"This was the most I'd laughed at one of Woody's comedies in years"* | 1.80 × 10⁻⁸ |

> ✅ **Test Sentence 2 is more relevant** — higher trigram probability indicates 
> better alignment with the IMDB training corpus.

---

### Part II — Text Preprocessing Pipeline

Applied sequentially on all IMDB reviews:

| Step | Method | Tool |
|------|--------|------|
| Tokenization | `word_tokenize` | NLTK |
| Lowercasing | `.lower()` | Python |
| Stop Words Removal | English stopwords | NLTK |
| Stemming | Porter Stemmer | NLTK |
| Lemmatization | WordNet Lemmatizer | NLTK |
| HTML/Special char removal | Regex | re |

---

### Part III — Feature Extraction & Similarity Analysis

**Word Embeddings — Skip-Gram (Word2Vec)**
| Parameter | Value |
|-----------|-------|
| vector_size | 100 |
| window | 5 |
| sg | 1 (Skip-Gram) |
| min_count | 1 |
| workers | 4 |

**Similarity Metric — Cosine Similarity**
```
similarity = (vec1 · vec2) / (||vec1|| × ||vec2||)
```
- Chosen because it measures the angle between vectors, not magnitude
- Ideal for word embeddings where direction encodes semantic meaning

**Visualization:**
- Words subset: `violence, brutality, unflinching, scenes, oz, struck, first, thing`
- PCA reduces 100-dimensional vectors → 2D for visualization
- Plotted in semantic space with annotated scatter plot

---

## 🛠️ Setup & Usage

### Prerequisites
- Python 3.8+
- Jupyter Notebook / Google Colab

### Install Dependencies
```bash
pip install pandas numpy matplotlib scikit-learn nltk gensim
```

### Download NLTK Resources
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

### Run
```bash
jupyter notebook IMDB_NLP.ipynb
```

---

## 📁 File Structure

```
IMDB-NLP-Trigram-SkipGram-Analysis/
├── IMDB_NLP.ipynb          # Main notebook
├── IMDB.csv                # IMDB reviews dataset
└── README.md
```

---

## 📝 License

MIT License — for academic use.
