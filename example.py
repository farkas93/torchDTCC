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

model_cfg = config.get("model", {})
logging.info(f"STABLE SVD: {model_cfg['stable_svd']}")

def load_model_clustering_example():
    model_path = config.get("trainer", {}).get("save_path", "")

    clusterer = Clusterer()
    clusterer.load_model(
        model_path=model_path,
        model_kwargs=model_cfg,
        device=config.get("device", "cuda")
    )
    return clusterer

def use_model_clustering_example(model):
    clusterer = Clusterer(config["device"])
    clusterer.set_model(model)
    return clusterer

def run_training():    
    trainer = DTCCTrainer.from_config(config, dataset)
    save_path = config.get("trainer", {}).get("save_path", "")
    return trainer.run(save_path=save_path)

if __name__ == "__main__":
    model = run_training()
    
    # For clustering in production
    dataloader = DataLoader(dataset, batch_size=data_cfg.get("batch_size", 64), shuffle=False)
    clusterer = use_model_clustering_example(model)
    labels = clusterer.cluster(dataloader, method="kmeans")  # or "soft", "argmax"
    print(f"resulting predictions:\n{labels}")
