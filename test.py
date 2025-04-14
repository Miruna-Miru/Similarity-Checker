# test.py
import torch
from utils import SimpleTransformer, preprocess, encode, build_vocab  # Added build_vocab import
import json

# Load the dataset (optional, but you can use the same vocab)
with open('dataset.json', 'r') as f:
    train_data = json.load(f)

# Prepare Vocabulary (you should use the same vocab used during training)
tokens_list = [preprocess(text1) + preprocess(text2) for text1, text2, _ in train_data]
vocab = build_vocab(tokens_list)

# Load the trained model
model = SimpleTransformer(vocab_size=len(vocab))
model.load_state_dict(torch.load('similarity_model.pth'))
model.eval()  # Set the model to evaluation mode

def predict_similarity(model, sentence1, sentence2, vocab):
    # Preprocess and encode the sentences
    tokens1 = preprocess(sentence1)
    tokens2 = preprocess(sentence2)
    encoded1 = encode(tokens1, vocab)
    encoded2 = encode(tokens2, vocab)

    # Forward pass
    with torch.no_grad():  # No need to compute gradients during inference
        output1 = model(encoded1.unsqueeze(0))  # Add batch dimension
        output2 = model(encoded2.unsqueeze(0))  # Add batch dimension
        cosine_sim = torch.nn.functional.cosine_similarity(output1, output2).item()

    return round(cosine_sim, 4)

# Example usage:
sentence1 = "The dog eats the table."
sentence2 = "A dog is eating on the table."
similarity_score = predict_similarity(model, sentence1, sentence2, vocab)
print(f"Similarity score between the sentences: {similarity_score}")
