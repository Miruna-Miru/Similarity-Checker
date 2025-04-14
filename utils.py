# utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdfplumber
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# 1. Text extraction
def extract_text(file):
    if file.name.endswith(".pdf"):
        text = ''
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                if page.extract_text():
                    text += page.extract_text() + '\n'
        return text
    elif file.name.endswith(".txt"):
        return file.read().decode('utf-8')
    return ""

# 2. Preprocessing
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    return tokens

# 3. Vocabulary (basic dummy)
def build_vocab(tokens_list):
    vocab = {}
    idx = 1
    for tokens in tokens_list:
        for token in tokens:
            if token not in vocab:
                vocab[token] = idx
                idx += 1
    return vocab

# 4. Encoding
def encode(tokens, vocab, max_len=64):
    encoded = [vocab.get(t, 0) for t in tokens]
    if len(encoded) > max_len:
        return torch.tensor(encoded[:max_len])
    else:
        return torch.cat([torch.tensor(encoded), torch.zeros(max_len - len(encoded))])

# 5. Custom Transformer Encoder
class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=4, num_layers=2):
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        embedded = self.embedding(x.long())  # [batch_size, seq_len, d_model]
        x = embedded.permute(1, 0, 2)        # transformer expects [seq_len, batch, d_model]
        out = self.transformer(x)
        out = out.permute(1, 2, 0)           # [batch, d_model, seq_len]
        pooled = self.pool(out).squeeze(-1)  # [batch, d_model]
        return pooled

# 6. Similarity Computation
def compute_similarity(text1, text2):
    tokens1 = preprocess(text1)
    tokens2 = preprocess(text2)
    vocab = build_vocab([tokens1, tokens2])
    input1 = encode(tokens1, vocab).unsqueeze(0)
    input2 = encode(tokens2, vocab).unsqueeze(0)

    model = SimpleTransformer(vocab_size=len(vocab))
    with torch.no_grad():
        emb1 = model(input1)
        emb2 = model(input2)
        sim = F.cosine_similarity(emb1, emb2).item()
    return round(sim, 4)

# 7. Similar Line Matching
from difflib import SequenceMatcher

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

def find_similar_lines(text1, text2, threshold=0.4):
    lines1 = [l.strip() for l in sent_tokenize(text1) if l.strip()]
    lines2 = [l.strip() for l in sent_tokenize(text2) if l.strip()]
    similar = []
    for line1 in lines1:
        for line2 in lines2:
            score = SequenceMatcher(None, line1, line2).ratio()
            if score >= threshold:
                hl1, hl2 = highlight_words(line1, line2)
                similar.append((hl1, hl2, round(score, 2)))
    return similar
