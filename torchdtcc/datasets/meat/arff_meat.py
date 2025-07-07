from datasets.arff_dataset import ArffDataset
from scipy.io import arff
from torchdtcc.augmentations.basic import jitter
from torchdtcc.augmentations.helper import torch_augmentation_wrapper
import pandas as pd

class MeatArffDataset(ArffDataset):
    def __init__(self, path):
        data_train, _ = arff.loadarff(path + 'Meat_TRAIN.arff')
        data_test, _ = arff.loadarff(path + 'Meat_TEST.arff')
        df_train = pd.DataFrame(data_train)
        df_test = pd.DataFrame(data_test)
        df = pd.concat([df_train, df_test], ignore_index=True)
        for col in df.select_dtypes([object]):
            df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
        feature_cols = [col for col in df.columns if col.startswith('att')]
        target_col = 'target'
        # Ensure target is int
        df[target_col] = df[target_col].astype(int)
        super().__init__(df, feature_cols, target_col)

    def augmentation(self, batch_x):        
        # Ensure [batch, seq_len, features]
        if batch_x.ndim == 2:
            batch_x = batch_x.unsqueeze(-1)  # add feature dim
        
        x_aug = torch_augmentation_wrapper(jitter, batch_x)
        # x_augx = torch_augmentation_wrapper(scaling, x_aug) # Import scaling first

        # Remove feature dim if univariate
        if x_aug.shape[-1] == 1:
            x_aug = x_aug.squeeze(-1)
        return x_aug