# pytorch implementation of Deep Temporal Contrastive Clustering

Implementation of the [Deep Temporal Contrastive Clustering](https://arxiv.org/abs/2212.14366).
Inspired by [07zy's](https://github.com/07zy/DTCC/tree/main) bit older Tensorflow based approach (unfortunately using deprecated Tensorflow version).
Therefore I decided to implement it myself in pytorch.

# Config
Easiest is to configure it using a YAML file. Example:
```
# torchdtcc config.yaml

model:
  input_dim:  1
  num_layers:  3
  num_clusters: 3
  hidden_dims:  [100, 50, 50]
  dilation_rates: [1, 4, 16]
  tau_I: 1.0
  tau_C: 1.0
  stable_svd: false

trainer:
  save_path: "dtcc_model.pth"
  learning_rate: 0.005
  weight_decay: 0
  lambda_cd: 0.001
  num_epochs: 200
  update_interval: 5

data:
  dataset_class: "torchdtcc.datasets.meat.MeatArffDataset"
  dataset_args:   # Arguments to initialize your dataset class
    files_path: "./data/meat/"
    # add more args as needed

  batch_size: 64


output:
  soft_clusters: "soft_clusters.npy"
  hard_clusters_argmax: "hard_clusters_argmax.npy"
  hard_clusters_kmeans: "hard_clusters_kmeans.npy"

device: "cuda"  # or "cpu"
```

# Training

```
import yaml
from torchdtcc.dtcc.trainer import DTCCTrainer
from torchdtcc.dtcc.clustering import Clusterer
from torch.utils.data import DataLoader
from torchdtcc.datasets.meat.arff_meat import MeatArffDataset

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Prepare dataset
data_cfg = config.get("data", {})
dataset = MeatArffDataset(path=data_cfg['dataset_args']['files_path'])

model_cfg = config.get("model", {})
logging.info(f"STABLE SVD: {model_cfg['stable_svd']}")
 
trainer = DTCCTrainer.from_config(config, dataset)
save_path = config.get("trainer", {}).get("save_path", "")
model = trainer.run(save_path=save_path)
```

# Usage

## After training
```
...
# Assuming the training script above

dataloader = DataLoader(dataset, batch_size=data_cfg.get("batch_size", 64), shuffle=False)
clusterer = Clusterer(config["device"])
clusterer.set_model(model)
labels = clusterer.cluster(dataloader, method="kmeans")  # or "soft", "argmax"
print(f"resulting predictions:\n{labels}")

```

## Load model
```
import yaml
from torchdtcc.dtcc.trainer import DTCCTrainer
from torchdtcc.dtcc.clustering import Clusterer
from torch.utils.data import DataLoader
from torchdtcc.datasets.meat.arff_meat import MeatArffDataset

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Prepare dataset and dataloader
data_cfg = config.get("data", {})
dataset = MeatArffDataset(path=data_cfg['dataset_args']['files_path'])
dataloader = DataLoader(dataset, batch_size=data_cfg.get("batch_size", 64), shuffle=False)

model_path = config.get("trainer", {}).get("save_path", "")
model_cfg = config.get("model", {})

clusterer = Clusterer()
clusterer.load_model(
    model_path=model_path,
    model_kwargs=model_cfg,
    device=config.get("device", "cuda")
)
labels = clusterer.cluster(dataloader, method="kmeans")  # or "soft", "argmax"
print(f"resulting predictions:\n{labels}")
```

# Run evaluation
To run the clusterer and print the accuracy, NMI, ARI and RI scores.
```
# assuming you have run some clustering example above
clusterer.evaluate(dataloader, method="kmeans")
```

# Use your own dataset
This is the definition of the base class for an augmented dataset. DTCC always expects an AugmentedDataset and you need to implement the augmentation.
```
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
```

You can use the augmentations submodule for implementing the augmentations.
See the meat class as reference:
```
from scipy.io import arff
import pandas as pd
from torchdtcc.augmentations.basic import jitter
from torchdtcc.augmentations.helper import torch_augmentation_wrapper
from torchdtcc.datasets.augmented_dataset import AugmentedDataset

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
```