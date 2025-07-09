import mlflow
import mlflow.pytorch
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from typing import Dict
from torchdtcc.dtcc.dtcc import DTCC
from torchdtcc.dtcc.clustering import Clusterer
from torchdtcc.datasets.augmented_dataset import AugmentedDataset
from .trainer import DTCCTrainer

class MlFlowDTCCTrainer(DTCCTrainer):
    def __init__(
        self,
        model: DTCC,
        dataloader: DataLoader,
        augment_time_series,
        optimizer,
        lambda_cd,
        num_epochs,
        update_interval=5,
        device="cpu",
        experiment_name: str = "MLflow_DTCC_Training",
        run_name: str = "default_run"
    ):
        super().__init__(model, dataloader, augment_time_series, optimizer, lambda_cd, num_epochs, update_interval, device)
        self.experiment_name = experiment_name
        self.run_name = run_name
        mlflow.set_experiment(experiment_name)
        logging.info(f"MLflow experiment set to: {experiment_name}")

    def run(self, save_path=None):
        with mlflow.start_run(run_name=self.run_name):
            mlflow.log_param("lambda_cd", self.lambda_cd)
            mlflow.log_param("num_epochs", self.num_epochs)
            mlflow.log_param("update_interval", self.update_interval)
            if hasattr(self.optimizer, 'param_groups'):
                mlflow.log_param("learning_rate", self.optimizer.param_groups[0]['lr'])
                mlflow.log_param("weight_decay", self.optimizer.param_groups[0].get('weight_decay', 0))
            result = super().run(save_path)
        return result
    
    def log_loss(self, epoch, avg_recon, avg_instance, avg_cd, avg_cluster, avg_total):
        # Log epoch metrics to MLflow
        mlflow.log_metric("avg_recon_loss", avg_recon, step=epoch + 1)
        mlflow.log_metric("avg_instance_loss", avg_instance, step=epoch + 1)
        mlflow.log_metric("avg_cd_loss", avg_cd, step=epoch + 1)
        mlflow.log_metric("avg_cluster_loss", avg_cluster, step=epoch + 1)
        mlflow.log_metric("avg_total_loss", avg_total, step=epoch + 1)

        logging.info(
            f"Epoch {epoch+1}/{self.num_epochs} | avg recon: {avg_recon:.4f} | avg instance: {avg_instance:.4f} | avg cd: {avg_cd:.4f} | avg cluster: {avg_cluster:.4f} | avg total: {avg_total:.4f}"
        )
    
    def log_evaluation(self, epoch, metrics):
        # Log clustering metrics to MLflow
        mlflow.log_metric("ACC", metrics['acc'], step=epoch + 1)
        mlflow.log_metric("NMI", metrics['nmi'], step=epoch + 1)
        mlflow.log_metric("ARI", metrics['ari'], step=epoch + 1)
        mlflow.log_metric("RI", metrics['ri'], step=epoch + 1)
        print(f"Epoch {epoch+1}: ACC={metrics['acc']:.4f} NMI={metrics['nmi']:.4f} ARI={metrics['ari']:.4f} RI={metrics['ri']:.4f}")

    def save_model(self, save_path):
        if save_path is not None:
            torch.save(self.model.state_dict(), save_path)
            mlflow.log_artifact(save_path)
            mlflow.pytorch.log_model(self.model, "model")
            logging.info(f"Model saved to {save_path} and logged to MLflow")
    
    @staticmethod
    def from_config(config: Dict, dataset: AugmentedDataset):
        trainer_cfg = config.get("trainer", {})
        mlflow_cfg = trainer_cfg.get("mlflow", {})
        env = DTCCTrainer._setup_model_environment(config, dataset)

        return MlFlowDTCCTrainer(
            model=env["model"],
            dataloader=env["dataloader"],
            augment_time_series=dataset.augmentation,
            optimizer=env["optimizer"],
            lambda_cd=trainer_cfg.get("lambda_cd", 1.0),
            num_epochs=trainer_cfg.get("num_epochs", 100),
            update_interval=trainer_cfg.get("update_interval", 5),
            device=env["device"],
            experiment_name=mlflow_cfg.get("experiment", "MLflow_DTCC_Training"),
            run_name=mlflow_cfg.get("run", "default_run")
        )