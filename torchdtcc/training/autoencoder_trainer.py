
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from typing import Dict
from ..dtcc.clustering import Clusterer
from ..dtcc.autoencoder import DTCCAutoencoder
from torchdtcc.datasets.augmented_dataset import AugmentedDataset

class DTCCAutoencoderTrainer:
    def __init__(
        self, 
        model: DTCCAutoencoder, 
        dataloader: DataLoader, 
        augment_time_series, 
        optimizer,
        num_epochs,
        gradient_clip = None,
        device="cpu"
    ):
        self.model = model
        self.device = device
        self.dataloader = dataloader
        self.augment_time_series = augment_time_series
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.grad_clip_max = gradient_clip

    def run(self, save_path=None):
        self.model.train()
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            recon_losses = []
            with tqdm(total=len(self.dataloader), desc=f'Epoch {epoch+1}/{self.num_epochs}') as pbar:
            
                try:

                    for i, batch in enumerate(self.dataloader):
                        x, y = batch
                        x = x.to(self.device)
                        x_aug = self.augment_time_series(x)
                        z, x_recon = self.model(x)
                        recon_loss = self.model.compute_reconstruction_loss(x, x_recon)
                        z, x_aug_recon = self.model(x_aug)
                        recon_loss += self.model.compute_reconstruction_loss(x_aug, x_aug_recon)

                        recon_loss *= 0.5

                        recon_losses.append(recon_loss.item())

                        logging.debug(f"Step {i} | recon: {recon_loss.item():.4f}")

                        self.optimizer.zero_grad()
                        recon_loss.backward()
                        for param in self.model.parameters():
                            if param.grad is not None:
                                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                    pbar.write(f"Exploding gradients detected at step {i}, epoch {epoch+1}!")
                                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_max)

                        if self.grad_clip_max:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_max)
                        self.optimizer.step()
                        epoch_loss += recon_loss.item()

                        pbar.set_postfix({'recon': f"{recon_loss.item():.3f}"})
                        pbar.update(1)
                except Exception as e:
                    pbar.write(f"Error: {e}")
                    raise

            avg_recon = sum(recon_losses) / len(recon_losses)

            self.log_loss(epoch, avg_recon)

        self.save_model(save_path)  # Moved outside loop to save only final model
        return self.model


    def debug_svd(self, epoch, Q, Q_aug, svds):
        pass

    def log_loss(self, epoch, avg_recon):
        logging.info(f"Epoch {epoch+1}/{self.num_epochs} | avg recon: {avg_recon:.4f}")

    def log_evaluation(self, epoch, metrics):  # Fixed method name
        pass

    def save_model(self, save_path):
        pass

    @staticmethod
    def _setup_model_environment(config: Dict, dataset: AugmentedDataset):
        model_cfg = config["model"]
        data_cfg = config["data"]
        trainer_cfg = config.get("trainer", {})
        device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

        dataloader = DataLoader(dataset, batch_size=data_cfg["batch_size"], shuffle=True)

        model = DTCCAutoencoder(
            input_dim=model_cfg["input_dim"],
            num_layers=model_cfg["num_layers"],
            hidden_dims=model_cfg["hidden_dims"],
            dilation_rates=model_cfg["dilation_rates"]
        ).to(device)

        optimizer = optim.Adam(
            model.parameters(),
            lr=trainer_cfg.get("learning_rate", 1e-3),
            weight_decay=float(trainer_cfg.get("weight_decay", 0))
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
        env = DTCCAutoencoderTrainer._setup_model_environment(config, dataset)

        return DTCCAutoencoderTrainer(
            model=env["model"],
            dataloader=env["dataloader"],
            augment_time_series=dataset.augmentation,
            optimizer=env["optimizer"],
            num_epochs=trainer_cfg.get("num_epochs", 100),
            gradient_clip=trainer_cfg.get("gradient_clip", None),
            device=env["device"]
        )