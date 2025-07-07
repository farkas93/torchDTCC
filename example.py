import yaml
from torchdtcc.dtcc.trainer import DTCCTrainer
from torchdtcc.dtcc.clustering import Clusterer
from torch.utils.data import DataLoader
from torchdtcc.datasets.meat.arff_meat import MeatArffDataset 


with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Prepare dataset and dataloader
dataset = MeatArffDataset(file_path="./data/meat/")
dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

def load_model_clustering_example():
    model_kwargs = config.get("model", {}).copy()
    model_path = model_kwargs.pop("path", "")

    clusterer = Clusterer()
    clusterer.load_model(
        model_path=model_path,
        model_kwargs=model_kwargs,
        device=config.get("device", "cuda")
    )
    return clusterer

def use_model_clustering_example(model):
    clusterer = Clusterer()
    num_clusters = config.get("model", {}).copy().pop("num_clusters", 3)
    clusterer.set_model(model, num_clusters)
    return clusterer

def run_training():    
    trainer = DTCCTrainer.from_config("config.yaml", dataset.augmentation)
    model_path = config.get("model", {}).copy().pop("path", "")
    return trainer.run(save_path=model_path)

if __name__ == "__main__":
    model = run_training()
    clusterer = use_model_clustering_example(model)    
    labels = clusterer.cluster(dataloader, method="kmeans")  # or "soft", "argmax"
