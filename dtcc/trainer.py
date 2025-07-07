import torch
import importlib
from dtcc.dtcc import DTCC
from torch.utils.data import DataLoader
import torch.optim as optim

class DTCCTrainer:
    def __init__(
        self,
        model : DTCC,
        dataloader : DataLoader,
        augment_time_series,
        optimizer,
        lambda_cd,
        num_epochs,
        update_interval=5,
        device="cpu"
    ):
        self.model = model
        self.dataloader = dataloader
        self.augment_time_series = augment_time_series
        self.optimizer = optimizer
        self.lambda_cd = lambda_cd
        self.num_epochs = num_epochs
        self.update_interval = update_interval
        self.device = device

    def run(self):
        self.model.train()
        for epoch in range(self.num_epochs):
            epoch_loss = 0
            for i, batch in enumerate(self.dataloader):
                batch = batch.to(self.device)
                x = batch
                x_aug = self.augment_time_series(x)
                z, z_aug, x_recon, x_aug_recon = self.model(x, x_aug)
                recon_loss = self.model.compute_reconstruction_loss(x, x_recon, x_aug, x_aug_recon)
                instance_loss = self.model.compute_instance_contrastive_loss(z, z_aug)
                cd_loss, Q, Q_aug = self.model.compute_cluster_distribution_loss(z, z_aug)
                cluster_loss = self.model.compute_cluster_contrastive_loss(Q, Q_aug)
                loss = recon_loss + instance_loss + cluster_loss + self.lambda_cd * cd_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            print(f"Epoch {epoch+1}/{self.num_epochs}, Avg Loss: {epoch_loss / len(self.dataloader):.4f}")

        # Extract Q for the full dataset
        self.model.eval()
        z_all = []
        with torch.no_grad():
            for batch in self.dataloader:
                batch = batch.to(self.device)
                z = self.model.encoder(batch)
                z_all.append(z)
        z_all = torch.cat(z_all, dim=0)
        U, S, Vh = torch.linalg.svd(z_all, full_matrices=False)
        Q_final = U[:, :self.model.num_clusters]
        return Q_final

    @staticmethod
    def from_config(config, augment_time_series):
        model_cfg = config["model"]
        data_cfg = config["data"]
        trainer_cfg = config.get("trainer", {})
        device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

        # Dynamically import dataset class
        module_path, class_name = data_cfg["dataset_class"].rsplit(".", 1)
        DatasetClass = getattr(importlib.import_module(module_path), class_name)
        dataset = DatasetClass(**data_cfg["dataset_args"])
        dataloader = DataLoader(dataset, batch_size=data_cfg["batch_size"], shuffle=True)

        model = DTCC(
            model_cfg["input_dim"],
            model_cfg["hidden_dim"],
            model_cfg["latent_dim"],
            model_cfg["num_layers"],
            model_cfg["num_clusters"]
        ).to(device)

        optimizer = optim.Adam(
            model.parameters(),
            lr=trainer_cfg.get("learning_rate", 1e-3),
            weight_decay=trainer_cfg.get("weight_decay", 0)
        )

        return DTCCTrainer(
            model=model,
            dataloader=dataloader,
            augment_time_series=augment_time_series,
            optimizer=optimizer,
            lambda_cd=trainer_cfg.get("lambda_cd", 1.0),
            num_epochs=trainer_cfg.get("num_epochs", 100),
            update_interval=trainer_cfg.get("update_interval", 5),
            device=device
        )