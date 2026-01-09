import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

class ANNSwap(nn.Module):
    def __init__(self, n_jobs, hidden_dim = 128):
        super(ANNSwap, self).__init__()
        self.n_jobs = n_jobs
        input_dim = n_jobs * n_jobs
        self.Layer1 = nn.Linear(input_dim, hidden_dim)
        self.Layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.index_i = nn.Linear(hidden_dim, n_jobs)
        self.index_j = nn.Linear(hidden_dim, n_jobs)

    def forward(self, x):
        x_hidden1 = F.relu(self.Layer1(x))
        x_hidden2 = F.relu(self.Layer2(x_hidden1))
        logits_i = self.index_i(x_hidden2)
        logits_j = self.index_j(x_hidden2)
        return logits_i, logits_j
    

def train_ann_epoch(model, optimizer, dataset, batch_size = 32):
    model.train()
    total_loss = 0
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)

    for chromosome, target_index_i, target_index_j in dataloader:
        optimizer.zero_grad()
        
        logits_i, logits_j = model(chromosome)

        loss_i = F.cross_entropy(logits_i, target_index_i)
        loss_j = F.cross_entropy(logits_j, target_index_j)
        loss = loss_i + loss_j
        loss.backward()

        optimizer.step()

        total_loss += loss.item()   

    average_loss = total_loss / len(dataloader)
    return average_loss

def predict_swap_indices(model, individual, temperature = 2.0):
    model.eval()
    with torch.no_grad():
        chromosome_before_swap = torch.tensor(individual.chromosome)
        chromosome_onehot = F.one_hot(chromosome_before_swap, model.n_jobs).float()
        chromosome_flatten = torch.flatten(chromosome_onehot, start_dim = 0)
        logits_i, logits_j = model(chromosome_flatten.unsqueeze(0)) # THIS NEEDS AN UNSQUEEZE(0) SO IT BECOMES BATCH SIZE 1

        prob_i = F.softmax(logits_i / temperature, dim=1)
        i = torch.multinomial(prob_i, num_samples=1).item()

        logits_j[0][i] = -float('inf')  
        prob_j = F.softmax(logits_j / temperature, dim=1)
        j = torch.multinomial(prob_j, num_samples=1).item()

        return i,j


