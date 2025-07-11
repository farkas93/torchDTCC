import mlflow
import mlflow.pytorch
import torch
from torch.utils.data import DataLoader
import logging
from typing import Dict, List
from torchdtcc.dtcc.dtcc import DTCC
from torchdtcc.datasets.augmented_dataset import AugmentedDataset
from .trainer import DTCCTrainer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import io
import numpy as np

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
        gradient_clip = None,
        device="cpu",
        server_uri:str = "databricks",
        experiment_name: str = "MLflow_DTCC_Training",
        run_name: str = "default_run",
        ablation: List = []
    ):
        super().__init__(model, dataloader, augment_time_series, optimizer, lambda_cd, num_epochs, update_interval, gradient_clip, device, ablation)
        if not server_uri == "databricks":
            mlflow.set_tracking_uri(server_uri)
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
    
    def debug_svd(self, epoch, Q, svds):
        if hasattr(self.model, 'compute_cluster_distribution_loss'):
            # After calling compute_cluster_distribution_loss(z, z_aug)
            with torch.no_grad():
                Q_hist = torch.argmax(Q, dim=1).cpu().numpy()
                hist, _ = np.histogram(Q_hist, bins=np.arange(self.model.get_num_clusters() + 1))
                for i, count in enumerate(hist):
                    mlflow.log_metric(f"cluster_count_{i}", int(count), step=epoch+1)
        with torch.no_grad():
            for i, s in enumerate(svds['S'].cpu().numpy()):
                mlflow.log_metric(f"svd_singular_{i}", float(s), step=epoch+1)
            for i, s in enumerate(svds['S_aug'].cpu().numpy()):
                mlflow.log_metric(f"svd_singular_{i}", float(s), step=epoch+1)


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
        self.create_tsne_plot(epoch)
    
    def create_tsne_plot(self, epoch):
        if epoch % 10 == 0:
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
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            mlflow.log_figure(buf, f"tsne_epoch_{epoch+1}.png")
            mlflow.log_artifact(buf, f"tsne_epoch_{epoch+1}.png")
            plt.close()

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
            gradient_clip=trainer_cfg.get("gradient_clip", None),
            device=env["device"],
            server_uri=mlflow_cfg.get("server_uri", "databricks"),
            experiment_name=mlflow_cfg.get("experiment", "MLflow_DTCC_Training"),
            run_name=mlflow_cfg.get("run", "default_run"),
            ablation=mlflow_cfg.get("ablation", [])
        )