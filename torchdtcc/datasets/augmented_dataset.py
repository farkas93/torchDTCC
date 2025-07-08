import torch
from torch.utils.data import Dataset
from abc import abstractmethod

class AugmentedDataset(Dataset):
    def __init__(self, dataframe, feature_cols, target_col):
        self.X = dataframe[feature_cols].values.astype('float32')
        self.y = dataframe[target_col].astype('int64').values

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx])
        if x.ndim == 1:
            x = x.unsqueeze(-1)  # add feature dimension if only batch and seq_len provided
        y = torch.tensor(self.y[idx])
        return x, y
    
    @abstractmethod
    def augmentation(self, batch_x):
        pass