# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
from utils import SimpleTransformer, preprocess, encode, build_vocab

# Load the dataset from the JSON file
with open('dataset.json', 'r') as f:
    train_data = json.load(f)

# 2. Custom Dataset for Pairwise Similarity
class TextPairDataset(torch.utils.data.Dataset):
    def __init__(self, data, vocab):
        self.data = data
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text1, text2, similarity = self.data[idx]
        tokens1 = preprocess(text1)
        tokens2 = preprocess(text2)
        encoded1 = encode(tokens1, self.vocab)
        encoded2 = encode(tokens2, self.vocab)
        return encoded1, encoded2, torch.tensor(similarity)

# 3. Prepare Vocabulary
tokens_list = [preprocess(text1) + preprocess(text2) for text1, text2, _ in train_data]
vocab = build_vocab(tokens_list)

# 4. Create DataLoader
train_dataset = TextPairDataset(train_data, vocab)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# 5. Initialize Model, Loss Function, and Optimizer
model = SimpleTransformer(vocab_size=len(vocab))
loss_fn = nn.MSELoss()  # Mean Squared Error loss for similarity
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6. Training Loop
def train_model(model, train_loader, loss_fn, optimizer, epochs=10):
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        total_loss = 0
        
        for batch_idx, (input1, input2, labels) in enumerate(train_loader):
            optimizer.zero_grad()  # Clear previous gradients
            
            # Forward pass
            output1 = model(input1)
            output2 = model(input2)
            
            # Calculate cosine similarity between the two outputs
            cosine_sim = torch.nn.functional.cosine_similarity(output1, output2).unsqueeze(1)
            
            # Compute loss
            loss = loss_fn(cosine_sim, labels.view(-1, 1))  # Reshape labels to match output
            
            # Backward pass
            loss.backward()
            
            # Update model parameters
            optimizer.step()
            
            total_loss += loss.item()

        # Print the loss after each epoch
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(train_loader)}")

# 7. Start training
train_model(model, train_loader, loss_fn, optimizer, epochs=10)
# Save the trained model
torch.save(model.state_dict(), 'similarity_model.pth')
print("Model saved to 'similarity_model.pth'")
