import torch
from torch.utils.data import Dataset
from abc import abstractmethod
import numpy as np # <-- Add this import

class AugmentedDataset(Dataset):
    def __init__(self, dataframe, feature_cols, target_col):
        # Original data loading
        self.X_raw = dataframe[feature_cols].values.astype('float32')
        self.y = dataframe[target_col].astype('int64').values

        # --- NEW: Compute normalization parameters and normalize data ---
        # Assuming X_raw is [num_samples, seq_len] or [num_samples, seq_len, features]
        # For univariate (like your Meat dataset), it's [num_samples, seq_len]
        
        # Calculate mean and std deviation across the entire dataset for normalization
        # Flatten the data to compute global mean and std for all time series values
        # This ensures all samples are normalized consistently based on the overall data distribution.
        all_feature_values = self.X_raw.flatten()
        self.mean = np.mean(all_feature_values)
        self.std = np.std(all_feature_values)
        
        # Add a small epsilon to std to prevent division by zero for constant features
        if self.std < 1e-8:
            self.std = 1.0 # Or raise an error if this indicates problematic data
            # logging.warning("Standard deviation is very small, setting to 1.0 for normalization stability.")

        # Apply standardization
        self.X_normalized = (self.X_raw - self.mean) / self.std
        # --- END NEW ---

    def __len__(self):
        return len(self.X_normalized) # Use the normalized data's length

    def __getitem__(self, idx):
        # Return the normalized data
        x = torch.tensor(self.X_normalized[idx])
        if x.ndim == 1:
            x = x.unsqueeze(-1)  # add feature dimension if only batch and seq_len provided
        y = torch.tensor(self.y[idx])
        return x, y
    
    @abstractmethod
    def augmentation(self, batch_x):
        # This method receives already normalized batch_x and should return normalized x_aug
        pass
