import itertools
from copy import deepcopy
import torch
from typing import Dict, List, Tuple, Callable
import logging
from .trainer import DTCCTrainer
from torchdtcc.dtcc.clustering import Clusterer
from torchdtcc.datasets.augmented_dataset import AugmentedDataset

class HyperparameterSearch:
    def __init__(
            self, 
            base_config: Dict, 
            dataset: AugmentedDataset, 
            tau_I_values: List[float], 
            tau_C_values: List[float],
            trainer_factory: Callable[[Dict, AugmentedDataset], DTCCTrainer]
        ):
        self.base_config = base_config
        self.dataset = dataset
        self.tau_I_values = tau_I_values
        self.tau_C_values = tau_C_values
        self.device = torch.device(base_config["device"] if torch.cuda.is_available() else "cpu")
        self.results = []
        self.trainer_factory = trainer_factory

    def grid_search(self, metric: str = "acc") -> List[Dict]:
        param_combinations = list(itertools.product(self.tau_I_values, self.tau_C_values))
        logging.info(f"Starting grid search over {len(param_combinations)} combinations of tau_I and tau_C")

        for tau_I, tau_C in param_combinations:
            logging.info(f"Testing tau_I={tau_I}, tau_C={tau_C}")
            # Update config with current hyperparameters
            config = deepcopy(self.base_config)
            config["model"]["tau_I"] = tau_I
            config["model"]["tau_C"] = tau_C
            config["trainer"]["mlflow"]["run"] += f"_tau_I_{tau_I}_tau_C_{tau_C}"

            # Initialize trainer
            trainer = self.trainer_factory(config, self.dataset)
            save_path = config.get("trainer", {}).get("save_path", f"dtcc_model_tauI_{tau_I}_tauC_{tau_C}.pth")

            # Run training
            model = trainer.run(save_path=save_path)

            # Evaluate clustering metrics (assuming last update_interval metrics are available)
            clusterer = Clusterer(self.device)
            clusterer.set_model(model)
            metrics = clusterer.evaluate(trainer.dataloader)

            # Store results
            result = {
                "tau_I": tau_I,
                "tau_C": tau_C,
                "metrics": metrics,
                "save_path": save_path
            }
            self.results.append(result)
            logging.info(f"Results for tau_I={tau_I}, tau_C={tau_C}: ACC={metrics['acc']:.4f}, NMI={metrics['nmi']:.4f}")

        return self.get_best_params(metric)

    def get_best_params(self, metric: str = "acc") -> Dict:
        if not self.results:
            return {}
        return max(self.results, key=lambda x: x["metrics"][metric])