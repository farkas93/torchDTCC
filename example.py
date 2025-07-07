import yaml
from dtcc.trainer import DTCCTrainer
from dtcc.clustering import Clusterer


with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)