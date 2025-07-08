from scipy.io import arff
import pandas as pd
from torchdtcc.augmentations.basic import jitter
from torchdtcc.augmentations.helper import torch_augmentation_wrapper
from ..augmented_dataset import AugmentedDataset

class MeatArffDataset(AugmentedDataset):
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
        assert batch_x.ndim == 3, f"Input must be 3D, got {batch_x.shape}"
        
        x_aug = torch_augmentation_wrapper(jitter, batch_x)
        # x_augx = torch_augmentation_wrapper(scaling, x_aug) # Import scaling first

        return x_aug