import mlflow
import mlflow.pytorch
import tempfile
import torch
from torch.utils.data import DataLoader
import logging
from typing import Dict, List
from torchdtcc.dtcc.dtcc import DTCC
from torchdtcc.datasets.augmented_dataset import AugmentedDataset
from .trainer import DTCCAutoencoderTrainer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import numpy as np

class MlFlowAutoencoderTrainer(DTCCAutoencoderTrainer):
    def __init__(
        self,
        model: DTCC,
        dataloader: DataLoader,
        augment_time_series,
        optimizer,
        num_epochs,
        gradient_clip = None,
        device="cpu",
        server_uri:str = "databricks",
        experiment_name: str = "MLflow_DTCC_Training",
        run_name: str = "default_run"
    ):
        super().__init__(model, dataloader, augment_time_series, optimizer, num_epochs, gradient_clip, device)
        if not server_uri == "databricks":
            mlflow.set_tracking_uri(server_uri)
        self.experiment_name = experiment_name
        self.run_name = run_name
        mlflow.set_experiment(experiment_name)
        logging.info(f"MLflow experiment set to: {experiment_name}")

    def run(self, save_path=None):
        with mlflow.start_run(run_name=self.run_name):
            mlflow.log_param("num_epochs", self.num_epochs)
            if hasattr(self.optimizer, 'param_groups'):
                mlflow.log_param("learning_rate", self.optimizer.param_groups[0]['lr'])
                mlflow.log_param("weight_decay", self.optimizer.param_groups[0].get('weight_decay', 0))
            result = super().run(save_path)
        return result


    def log_loss(self, epoch, avg_recon):
        # Log epoch metrics to MLflow
        mlflow.log_metric("avg_recon_loss", avg_recon, step=epoch + 1)

        logging.info(
            f"Epoch {epoch+1}/{self.num_epochs} | avg recon: {avg_recon:.4f}"
        )
    
    def log_evaluation(self, epoch, metrics):
        # Log clustering metrics to MLflow
        mlflow.log_metric("ACC", metrics['acc'], step=epoch + 1)
        mlflow.log_metric("NMI", metrics['nmi'], step=epoch + 1)
        mlflow.log_metric("ARI", metrics['ari'], step=epoch + 1)
        mlflow.log_metric("RI", metrics['ri'], step=epoch + 1)
        print(f"Epoch {epoch+1}: ACC={metrics['acc']:.4f} NMI={metrics['nmi']:.4f} ARI={metrics['ari']:.4f} RI={metrics['ri']:.4f}")
        self.create_tsne_plot(epoch)
    
    def create_tsne_plot(self, epoch):
        all_z, all_y = [], []
        for x, y in self.dataloader:
            x = x.to(self.device)
            z = self.model.encoder(x)
            all_z.append(z.detach().cpu().numpy())
            all_y.append(y.cpu().numpy())
        all_z = np.concatenate(all_z, axis=0)
        all_y = np.concatenate(all_y, axis=0)

        # t-SNE
        z_embedded = TSNE(n_components=2).fit_transform(all_z)
        plt.figure()
        plt.scatter(z_embedded[:,0], z_embedded[:,1], c=all_y, cmap='tab10')
        plt.title(f'Latent space t-SNE epoch {epoch+1}')
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            plt.savefig(tmp_file.name, format='png')
            mlflow.log_artifact(tmp_file.name, f"tsne_epoch_{epoch+1}.png")
        os.unlink(tmp_file.name)  # Clean up the temporary file
        plt.close()

    def save_model(self, save_path):
        if save_path is not None:
            torch.save(self.model.state_dict(), save_path)
            mlflow.log_artifact(save_path)
            mlflow.pytorch.log_model(self.model, "model")
            logging.info(f"Model saved to {save_path} and logged to MLflow")
    
    @staticmethod
    def from_config(config: Dict, dataset: AugmentedDataset):
        trainer_cfg = config.get("warmup", {})
        mlflow_cfg = config.get("mlflow", {})
        env = DTCCAutoencoderTrainer._setup_model_environment(config, dataset)
        
        lr_scheduler = None
        update_interval = 100
        if "lr_scheduler" in trainer_cfg:
            scheduler_cfg = trainer_cfg["lr_scheduler"]
            if scheduler_cfg["type"] == "StepLR":
                update_interval = scheduler_cfg.get("step_size", 100)
                lr_scheduler = torch.optim.lr_scheduler.StepLR(
                    env["optimizer"],
                    step_size=scheduler_cfg.get("step_size", 800),
                    gamma=scheduler_cfg.get("gamma", 0.1)
                )

        experiment = mlflow_cfg.get("experiment", "MLflow_DTCC_Training")
        if mlflow_cfg.get("server_uri") == "databricks":
            experiment = mlflow_cfg.get("experiment_path").format(experiment)

        run = "dtcc_ae_" + mlflow_cfg.get("run", "default_run")
        return MlFlowAutoencoderTrainer(
            model=env["model"],
            dataloader=env["dataloader"],
            augment_time_series=dataset.augmentation,
            optimizer=env["optimizer"],
            num_epochs=trainer_cfg.get("num_epochs", 100),
            gradient_clip=trainer_cfg.get("gradient_clip", None),
            device=env["device"],
            lr_scheduler=lr_scheduler,
            patience=trainer_cfg.get("patience", 100),
            update_interval=update_interval,
            server_uri=mlflow_cfg.get("server_uri", "databricks"),
            experiment_name=mlflow_cfg.get("experiment", "MLflow_DTCC_Training"),
            run_name=run
        )