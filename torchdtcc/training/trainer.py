
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from typing import Dict
from ..dtcc.clustering import Clusterer
from ..dtcc.dtcc import DTCC
from torchdtcc.datasets.augmented_dataset import AugmentedDataset

class DTCCTrainer:
    def __init__(
        self, 
        model: DTCC, 
        dataloader: DataLoader, 
        augment_time_series, 
        optimizer, 
        lambda_cd, 
        num_epochs, 
        update_interval=5, 
        device="cpu"
    ):
        self.model = model
        self.device = device
        self.clusterer = Clusterer(self.device)
        self.dataloader = dataloader
        self.augment_time_series = augment_time_series
        self.optimizer = optimizer
        self.lambda_cd = lambda_cd
        self.num_epochs = num_epochs
        self.update_interval = update_interval

    def run(self, save_path=None):
        self.model.train()
        for epoch in range(self.num_epochs):
            epoch_loss = 0
            recon_losses, instance_losses, cd_losses, cluster_losses, total_losses = [], [], [], [], []
            with tqdm(total=len(self.dataloader), desc=f'Epoch {epoch+1}/{self.num_epochs}') as pbar:
                for i, batch in enumerate(self.dataloader):
                    x, y = batch
                    x = x.to(self.device)
                    x_aug = self.augment_time_series(x)
                    z, z_aug, x_recon, x_aug_recon = self.model(x, x_aug)
                    recon_loss = self.model.compute_reconstruction_loss(x, x_recon, x_aug, x_aug_recon)
                    instance_loss = self.model.compute_instance_contrastive_loss(z, z_aug)
                    cd_loss, Q, Q_aug = self.model.compute_cluster_distribution_loss(z, z_aug)
                    cluster_loss = self.model.compute_cluster_contrastive_loss(Q, Q_aug)
                    loss = recon_loss + instance_loss + cluster_loss + self.lambda_cd * cd_loss

                    recon_losses.append(recon_loss.item())
                    instance_losses.append(instance_loss.item())
                    cd_losses.append(cd_loss.item())
                    cluster_losses.append(cluster_loss.item())
                    total_losses.append(loss.item())

                    logging.debug(f"Step {i} | recon: {recon_loss.item():.4f} | instance: {instance_loss.item():.4f} | cd: {cd_loss.item():.4f} | cluster: {cluster_loss.item():.4f} | total: {loss.item():.4f}")

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss.item()

                    pbar.set_postfix({'recon': f"{recon_loss.item():.3f}", 'inst': f"{instance_loss.item():.3f}", 'cd': f"{cd_loss.item():.3f}", 'clust': f"{cluster_loss.item():.3f}", 'total': f"{loss.item():.3f}"})
                    pbar.update(1)

            avg_recon = sum(recon_losses) / len(recon_losses)
            avg_instance = sum(instance_losses) / len(instance_losses)
            avg_cd = sum(cd_losses) / len(cd_losses)
            avg_cluster = sum(cluster_losses) / len(cluster_losses)
            avg_total = sum(total_losses) / len(total_losses)

            self.log_loss(epoch, avg_recon, avg_instance, avg_cd, avg_cluster, avg_total)

            if (epoch + 1) % self.update_interval == 0:
                self.clusterer.set_model(self.model)
                metrics = self.clusterer.evaluate(self.dataloader)
                self.log_evaluation(epoch, metrics)  # Fixed method name

        self.save_model(save_path)  # Moved outside loop to save only final model
        return self.model

    def log_loss(self, epoch, avg_recon, avg_instance, avg_cd, avg_cluster, avg_total):
        logging.info(f"Epoch {epoch+1}/{self.num_epochs} | avg recon: {avg_recon:.4f} | avg instance: {avg_instance:.4f} | avg cd: {avg_cd:.4f} | avg cluster: {avg_cluster:.4f} | avg total: {avg_total:.4f}")

    def log_evaluation(self, epoch, metrics):  # Fixed method name
        print(f"Epoch {epoch+1}: ACC={metrics['acc']:.4f} NMI={metrics['nmi']:.4f} ARI={metrics['ari']:.4f} RI={metrics['ri']:.4f}")

    def save_model(self, save_path):
        if save_path is not None:
            torch.save(self.model.state_dict(), save_path)
            logging.info(f"Model saved to {save_path}")

    @staticmethod
    def _setup_model_environment(config: Dict, dataset: AugmentedDataset):
        model_cfg = config["model"]
        data_cfg = config["data"]
        trainer_cfg = config.get("trainer", {})
        device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

        dataloader = DataLoader(dataset, batch_size=data_cfg["batch_size"], shuffle=True)

        model = DTCC(
            input_dim=model_cfg["input_dim"],
            num_layers=model_cfg["num_layers"],
            num_clusters=model_cfg["num_clusters"],
            hidden_dims=model_cfg["hidden_dims"],
            dilation_rates=model_cfg["dilation_rates"],
            tau_I=model_cfg["tau_I"],
            tau_C=model_cfg["tau_C"],
            stable_svd=model_cfg["stable_svd"]
        ).to(device)

        optimizer = optim.Adam(
            model.parameters(),
            lr=trainer_cfg.get("learning_rate", 1e-3),
            weight_decay=trainer_cfg.get("weight_decay", 0)
        )
        return {
            "dataloader": dataloader,
            "model": model,
            "optimizer": optimizer,
            "device": device
        }

    @staticmethod
    def from_config(config: Dict, dataset: AugmentedDataset):
        trainer_cfg = config.get("trainer", {})
        env = DTCCTrainer.setup_model_environment(config, dataset)

        return DTCCTrainer(
            model=env["model"],
            dataloader=env["dataloader"],
            augment_time_series=dataset.augmentation,
            optimizer=env["optimizer"],
            lambda_cd=trainer_cfg.get("lambda_cd", 1.0),
            num_epochs=trainer_cfg.get("num_epochs", 100),
            update_interval=trainer_cfg.get("update_interval", 5),
            device=env["device"]
        )