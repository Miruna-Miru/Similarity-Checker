# utils.py
import pdfplumber
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Global stopwords list for better performance
stop_words = set(stopwords.words('english'))

# Function to extract text from PDFs or TXT files
def extract_text(file):
    if file.name.endswith(".pdf"):
        text = ''
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + '\n'
        return text
    elif file.name.endswith(".txt"):
        return file.read().decode('utf-8')
    else:
        return ""

# Function to preprocess the text (lowercase, remove stopwords, lemmatize)
def preprocess(text):
    # Lowercasing and cleaning text
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Tokenization and removing stopwords
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    
    # Lemmatization for better word normalization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    
    return tokens

# Function to compute the similarity score between two texts
def compute_similarity(text1, text2):
    # Using TF-IDF Vectorizer for semantic similarity (cosine similarity)
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([text1, text2])
    sim_score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return sim_score

# Function to highlight similar words between two lines
def highlight_words(line1, line2):
    matcher = SequenceMatcher(None, line1, line2)
    highlighted1 = ""
    highlighted2 = ""

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            highlighted1 += f"<span style='background-color: #d4edda'>{line1[i1:i2]}</span>"
            highlighted2 += f"<span style='background-color: #d4edda'>{line2[j1:j2]}</span>"
        else:
            highlighted1 += line1[i1:i2]
            highlighted2 += line2[j1:j2]

    return highlighted1, highlighted2

# Function to find similar lines based on a threshold score
def find_similar_lines(text1, text2, threshold=0.):
    # Tokenize text into sentences (improves line-by-line matching)
    lines1 = [l.strip() for l in sent_tokenize(text1) if l.strip()]
    lines2 = [l.strip() for l in sent_tokenize(text2) if l.strip()]
    similar = []

    for line1 in lines1:
        for line2 in lines2:
            score = SequenceMatcher(None, line1, line2).ratio()
            if score >= threshold:
                hl1, hl2 = highlight_words(line1, line2)
                similar.append((hl1, hl2, score))

    return similar
