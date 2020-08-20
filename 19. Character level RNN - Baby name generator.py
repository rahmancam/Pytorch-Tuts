# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import string
import random
import sys
import unidecode
from torch.utils.tensorboard import SummaryWriter

# device config
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

# Get characters from string printable
all_characters = string.printable
n_characters = len(all_characters)

data = unidecode.unidecode(open('./datasets/baby_names.txt').read())
print(data[1:100])

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embed = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hidden, cell):
        out = self.embed(x)
        out, (hidden, cell) = self.lstm(out.unsqueeze(1), (hidden, cell))
        out = self.fc(out.reshape(out.shape[0], -1))
        return out, (hidden, cell)
    
    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return hidden, cell
        

class Generator:
    def __init__(self):
        # Hyper parameters
        self.chunk_len = 250
        self.num_epochs = 5000
        self.batch_size = 1
        self.print_every = 50
        self.hidden_size = 256
        self.num_layers = 2
        self.lr = 0.003
    
    def char_toTensor(self, string):
        tensor = torch.zeros(len(string)).long()
        for c in range(len(string)):
            if string[c] in all_characters:
                tensor[c] = all_characters.index(string[c])
        return tensor
    
    def get_random_batch(self):
        start_idx = random.randint(0, len(data) - self.chunk_len)
        end_idx = start_idx + self.chunk_len + 1
        sample_text = data[start_idx:end_idx]
        text_input = torch.zeros(self.batch_size, self.chunk_len)
        text_target = torch.zeros(self.batch_size, self.chunk_len)
        
        for i in range(self.batch_size):
            text_input[i,:] = self.char_toTensor(sample_text[:-1])
            text_target[i,:] = self.char_toTensor(sample_text[1:])
            
        return text_input.long(), text_target.long()
    
    def generate(self, initial_str='Ab', predict_len=100, temperature=0.85):
        hidden, cell = self.model.init_hidden(batch_size=self.batch_size)
        initial_input = self.char_toTensor(initial_str)
        predicted = initial_str
        
        for p in range(len(initial_str) - 1):
            _, (hidden, cell) = self.model(initial_input[p].view(1).to(device), hidden, cell)
            
        last_char = initial_input[-1]
        
        for p in range(predict_len):
            output, (hidden, cell) = self.model(last_char.view(1).to(device), hidden, cell)
            output_dist = output.data.view(-1).div(temperature).exp()
            top_char_idx = torch.multinomial(output_dist, 1)[0]
            predicted_char = all_characters[top_char_idx]
            predicted += predicted_char
            last_char = self.char_toTensor(predicted_char)
            
        return predicted
        
    def train(self):
        self.model = RNN(n_characters, self.hidden_size, self.num_layers, n_characters)
        self.model = self.model.to(device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        writer = SummaryWriter(f'runs/babynames0')
        
        print('#=> Starting training')
        
        for epoch in range(1, self.num_epochs + 1):
            inp, target = self.get_random_batch()
            hidden, cell = self.model.init_hidden(self.batch_size)
            
            self.model.zero_grad()
            loss = 0
            inp = inp.to(device)
            target = target.to(device)
            
            for c in range(self.chunk_len):
                output, (hidden, cell) = self.model(inp[:, c], hidden, cell)
                loss += criterion(output, target[:, c])
                
            loss.backward()
            optimizer.step()
            loss = loss.item() / self.chunk_len
            
            if epoch % self.print_every == 0:
                print(f'Loss: {loss}')
                print(self.generate())
            
            writer.add_scalar('Training Loss', loss, global_step=epoch)
            

babynames_generator = Generator()
babynames_generator.train()
        
    
    
    

