import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class SwapDataset(Dataset):
    def __init__(self, n_jobs, max_size = 100):
        self.n_jobs = n_jobs
        self.max_size = max_size
        self.swap_storage = []

    def __len__(self):
        return len(self.swap_storage)

    def __getitem__(self, idx):
        individual = self.swap_storage[idx]
        chromosome_before_swap, (index_i, index_j) = torch.tensor(individual.get_memory(), dtype=torch.long), individual.get_swap() 
        #index_i_onehot = self._one_hot_torch(self.n_jobs, index_i)
        #index_j_onehot = self._one_hot_torch(self.n_jobs, index_j)
        chromosome_onehot = F.one_hot(chromosome_before_swap, self.n_jobs).float()
        chromosome_flatten = torch.flatten(chromosome_onehot, start_dim = 0)
        return chromosome_flatten, index_i, index_j
    
    def add(self, data):
        if len(self.swap_storage) >= self.max_size:
            self.swap_storage.pop(0)
        self.swap_storage.append(data)

    def _one_hot_torch(index, length):
        x = torch.zeros(length, dtype = torch.float)
        x[index] = 1
        return x
    