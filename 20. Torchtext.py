# -*- coding: utf-8 -*-

# 1. Specify how preprocessing should be done
# 2. Use Dataset to load data
# 3. Construct an iterator to do batching & padding

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import Field, TabularDataset, BucketIterator
import spacy


# python -m spacy download en
spacy_en = spacy.load('en')

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# tokenize = lambda x: x.split()
def tokenize(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

quote = Field(sequential=True, use_vocab=True, tokenize=tokenize, lower=True)
score = Field(sequential=False, use_vocab=False)

fields = { "quote": ("q", quote), "score": ("s", score)}
train_data, test_data = TabularDataset.splits(
    path='datasets',
    train='quotes_train.csv',
    test='quotes_test.csv',
    format='csv',
    fields=fields
    )

# print(train_data[0].__dict__.keys())
# print(train_data[0].__dict__.values())

# quote.build_vocab(train_data, max_size=10000, min_freq=1)

# with pretrained word embeddings - # 1 GB (glove.6B.100d)
quote.build_vocab(train_data, max_size=10000, min_freq=1, vectors="glove.6B.100d")

train_iterator, test_iterator = BucketIterator.splits(
    (train_data, test_data),
    batch_size=2,
    device=device)

## Train a simple LSTM ###
class RNN_LSTM(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, num_layers):
        super(RNN_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(input_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers)
        self.fc1 = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # set initial hidden and cell state
        h0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size).to(device)
        
        embeded = self.embedding(x)
        outputs, _ = self.rnn(embeded, (h0, c0))
        print(outputs.shape, embeded.shape)
        prediction = self.fc1(outputs[-1, :, :])
        # print(prediction.shape)
        return prediction
        

# Hyper parameters
input_size = len(quote.vocab)
hidden_size = 512
num_layers = 2
embedding_size = 100
lr = 0.005
num_epochs = 10

print(f'Input size: {input_size}')

# Initialize network
model = RNN_LSTM(input_size, embedding_size, hidden_size, num_layers).to(device)

# Load the pre-trained embeddings onto our model
pretrained_embeddings = quote.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Train network
for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(train_iterator):
        data = batch.q.to(device)
        targets = batch.s.to(device)
        
        # forward
        scores = model(data)
        print(scores.shape)
        loss = criterion(scores.squeeze(1), targets.type_as(scores))
        
        print(f'Loss : {loss.item():.3f}')

        #backward
        optimizer.zero_grad()
        loss.backward()
    
        optimizer.step()

