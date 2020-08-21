# -*- coding: utf-8 -*-
import torch
import random
import torch.nn as nn
import torch.optim as optim
import numpy as np
import spacy
from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint
from torch.utils.tensorboard import SummaryWriter
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

# python -m spacy download en
# python -m spacy download de

spacy_ger = spacy.load('de')
spacy_eng = spacy.load('en')

def tokenize_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]

def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]


german = Field(tokenize=tokenize_ger, lower=True, init_token='<sos>', eos_token='<eos>')
english = Field(tokenize=tokenize_eng, lower=True, init_token='<sos>', eos_token='<eos>')

fields = (german, english)
train_data, valid_data, test_data = Multi30k.splits(exts=(".de", ".en"), fields=fields)

german.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        embedding = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.rnn(embedding)
        return hidden, cell
    
class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, p):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)        
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p)
    
    def forward(self, x, hidden, cell):
        x = x.unsqueeze(0)
        embedding = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))
        
        predictions = self.fc(outputs).squeeze(0)
        return predictions, hidden, cell
    
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, source ,target, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(english.vocab)
        
        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)
        
        hidden, cell = self.encoder(source)
        
        x = target[0] # first <sos> token
        
        for t in range(1, target_len):
            # use previous context vectors for prediction
            output, hidden, cell = self.decoder(x, hidden, cell)
            
            # store next output prediction
            outputs[t] = output
            
            # get the best word the Decoder predicted
            best_guess = output.argmax(1)
            x = target[t] if random.random() < teacher_force_ratio else best_guess
        
        return outputs
    
# Training Hyperparameters

num_epochs = 100
lr = 0.001
batch_size = 64

# model Hyperparameters
load_model = False
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
input_size_encoder = len(german.vocab)
input_size_decoder = len(english.vocab)
output_size = len(english.vocab)
encoder_embedding_size = 300
decoder_embedding_size = 300
hidden_size = 1024
num_layers = 2
dropout_enc = 0.5
dropout_dec = 0.5

writer = SummaryWriter(f'runs/loss_plot')
step = 0
train_iterator, val_iterator, test_iterator = BucketIterator.splits((train_data, valid_data, test_data),
                                                                    batch_size=batch_size,
                                                                    sort_within_batch=True,
                                                                    sort_key=lambda x: len(x.src),
                                                                    device=device)
encoder_net = Encoder(input_size_encoder, encoder_embedding_size, hidden_size, num_layers, dropout_enc).to(device)
decoder_net = Decoder(input_size_decoder, decoder_embedding_size, hidden_size, output_size, num_layers, dropout_dec).to(device)

model = Seq2Seq(encoder_net, decoder_net).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

pad_idx = english.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

if load_model:
    load_checkpoint(torch.load("my_chk_point.pth.tar"), model, optimizer)
    
sentence = "Ein Auto ist so gut wie sein Besitzer"

for epoch in range(num_epochs):
    print(f'[Epoch {epoch}/ {num_epochs}]')
    
    checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
    save_checkpoint(checkpoint)
    
    model.eval()
    
    translated_sent = translate_sentence(model, sentence, german, english, device, max_length=50)
    print(f'Translated example sentence: \n {translated_sent}')
    
    model.train()
    
    for batch_idx, batch in enumerate(train_iterator):
        inp_data = batch.src.to(device)
        target = batch.trg.to(device)
        
        output = model(inp_data, target)
        output = output[1:].reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)
        
        optimizer.zero_grad()
        loss = criterion(output, target)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        
        optimizer.step()
        step += 1
        
score = bleu(test_data[1:100]. model, german, english, device)
print(f'Blue score {score * 100:.2f}')
        




        
        