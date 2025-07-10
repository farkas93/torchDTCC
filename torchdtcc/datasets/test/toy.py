import torch
from torch.utils.data import Dataset
import numpy as np
from ..augmented_dataset import AugmentedDataset  # Adjust import as needed
from torchdtcc.augmentations.basic import jitter
from torchdtcc.augmentations.helper import torch_augmentation_wrapper

class ToyAugmentedDataset(AugmentedDataset):
    """
    Toy dataset for time series clustering compatible with AugmentedDataset.
    Generates K clusters of simple signals.
    """
    def __init__(self,
                 path=None,
                 num_samples=300, 
                 seq_len=100, 
                 num_clusters=3, 
                 noise_std=0.1, 
                 normalize=True, 
                 seed=42):
        import pandas as pd

        np.random.seed(seed)
        torch.manual_seed(seed)

        # Generate
        X = []
        y = []
        base_freqs = np.linspace(1.0, 3.0, num_clusters)
        base_phases = np.linspace(0, np.pi, num_clusters)
        t = np.linspace(0, 2 * np.pi, seq_len)
        for i in range(num_samples):
            label = i % num_clusters
            freq = base_freqs[label]
            phase = base_phases[label]
            signal = np.sin(freq * t + phase)
            signal += np.random.normal(0, noise_std, size=signal.shape)
            X.append(signal)
            y.append(label)

        X = np.stack(X)  # [num_samples, seq_len]
        y = np.array(y, dtype=np.int64)

        # Build a DataFrame to match ARFF datasets
        feature_cols = [f'att{i}' for i in range(seq_len)]
        df = pd.DataFrame(X, columns=feature_cols)
        df['target'] = y

        # Call parent constructor
        super().__init__(df, feature_cols, 'target', normalize=normalize)

    def augmentation(self, batch_x):
        # Optionally add toy augmentations here, or just return batch_x
        return batch_x  # No augmentation by default (safe for debugging) 
        assert batch_x.ndim == 3, f"Input must be 3D, got {batch_x.shape}"
        
        augmentations = [jitter]
        x_aug = batch_x
        applied = False
        
        # Shuffle augmentations to randomize order
        import random
        random.shuffle(augmentations)
        
        for aug in augmentations:
            if not applied or torch.rand(1).item() > 0.5:  # Apply first augmentation, then 50% chance for others
                x_aug = torch_augmentation_wrapper(aug, x_aug)
                applied = True
        return x_aug