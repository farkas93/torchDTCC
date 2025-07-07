import torch
from dtcc import DTCC
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset

class DTCCTrainer:
    def __init__(self, dataloader, augment_time_series, optimizer, lambda_cd, num_epochs, update_interval=5):
        self.dataloader = dataloader
        self.augment_time_series = augment_time_series
        self.optimizer = optimizer
        self.lambda_cd = lambda_cd
        self.num_epochs = num_epochs
        self.update_interval = update_interval


    def run(self, model : DTCC):
        for epoch in range(self.num_epochs):
            total_loss = 0
            
            for i, batch in enumerate(self.dataloader):
                x = batch  # Assuming batch is already time series data
                
                # Apply augmentation to create augmented view
                x_aug = self.augment_time_series(x)  # You need to implement this function
                
                # Forward pass
                z, z_aug, x_recon, x_aug_recon = model(x, x_aug)
                
                # Compute losses
                recon_loss = model.compute_reconstruction_loss(x, x_recon, x_aug, x_aug_recon)
                instance_loss = model.compute_instance_contrastive_loss(z, z_aug)
                cd_loss, Q, Q_aug = model.compute_cluster_distribution_loss(z, z_aug)
                cluster_loss = model.compute_cluster_contrastive_loss(Q, Q_aug)
                
                # Total loss
                total_loss = recon_loss + instance_loss + cluster_loss + self.lambda_cd * cd_loss
                
                # Backward and optimize
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                
                # Update cluster indicators periodically
                if epoch % self.update_interval == 0 and i == 0:
                    # This would be handled internally in compute_cluster_distribution_loss
                    pass
                    
            print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {total_loss.item()}")
        
        # Return final clustering result
        z_all = []
        model.eval()
        with torch.no_grad():
            for batch in self.dataloader:
                z = model.encoder(batch)
                z_all.append(z)
        
        z_all = torch.cat(z_all, dim=0)
        gram_matrix = torch.matmul(z_all.T, z_all)
        U, S, V = torch.svd(gram_matrix)
        Q_final = U[:, :model.num_clusters]
        
        return Q_final