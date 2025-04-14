import random
import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Function to preprocess text (same as in utils.py)
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    return tokens

# Function to generate a random sentence (simple for dataset purposes)
def generate_sentence():
    subjects = ['The dog', 'A cat', 'The teacher', 'The student', 'A bird']
    verbs = ['eats', 'sits on', 'runs near', 'jumps over', 'flies across']
    objects = ['the table', 'the chair', 'the park', 'the roof', 'the garden']

    subject = random.choice(subjects)
    verb = random.choice(verbs)
    object_ = random.choice(objects)

    return f'{subject} {verb} {object_}.'

# Generate dataset of 100 sentence pairs
def generate_dataset():
    dataset = []

    for _ in range(100):
        # Generate two sentences
        sentence1 = generate_sentence()
        sentence2 = generate_sentence()

        # Calculate similarity (using a simple random similarity for now)
        similarity = random.uniform(0.5, 1.0)  # Similarity between 0.5 and 1.0 for random pairs
        if sentence1[:20] == sentence2[:20]:  # If the sentences are very similar, make the similarity high
            similarity = random.uniform(0.8, 1.0)
        elif sentence1.split()[0] == sentence2.split()[0]:  # Same subject, might be more similar
            similarity = random.uniform(0.7, 0.9)
        
        # Add to dataset
        dataset.append((sentence1, sentence2, round(similarity, 2)))
    
    return dataset

# Save dataset to a JSON file
def save_dataset(dataset, filename='dataset.json'):
    with open(filename, 'w') as f:
        json.dump(dataset, f, indent=4)

# Generate and save the dataset
dataset = generate_dataset()
save_dataset(dataset)
print(f"Dataset of {len(dataset)} samples saved to 'dataset.json'.")
