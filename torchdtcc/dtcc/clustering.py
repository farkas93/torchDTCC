import torch
import numpy as np
import logging
from sklearn.cluster import KMeans
from .dtcc import DTCC
from .helper import stablize
from ..training.metrics import all_metrics

class Clusterer:
    def __init__(self, device):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        pass

    def set_model(self, model: DTCC):
        """
        Args:
            model (DTCC): trained DTCC model object
            num_clusters (int): number of clusters the model is able to cluster
        """
        self.model = model
        self.num_clusters = model.get_num_clusters()
        self.stable_svd = model.get_stable_svd()

    def load_model(self, model_path, model_kwargs, device="cpu"):
        """
        Args:
            model_path (str): Path to model weights (.pth)
            model_class (type): Model class (e.g., DTCC)
            model_kwargs (dict): Model init kwargs
            device (str): 'cpu' or 'cuda'
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = DTCC(**model_kwargs)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.num_clusters = model_kwargs.get("num_clusters", 3)
        self.stable_svd = model_kwargs.get("stable_svd", False)

    def encode(self, dataloader):
        zs = []
        with torch.no_grad():
            for batch in dataloader:
                X, y = batch
                X = X.to(self.device)
                z = self.model.encoder(X)
                zs.append(z)
        return torch.cat(zs, dim=0)  # shape [N, d]

    def soft_clusters(self, z_all):
        if self.stable_svd:
            z_all = stablize(z_all)
        U, S, V = torch.linalg.svd(z_all)
        Q = U[:, :self.num_clusters]
        return Q

    def hard_clusters_argmax(self, Q):
        return Q.argmax(dim=1)

    def hard_clusters_kmeans(self, Q):
        Q_np = Q.cpu().numpy()
        kmeans = KMeans(n_clusters=self.num_clusters, n_init=10)
        return kmeans.fit_predict(Q_np)

    def cluster(self, dataloader, method="kmeans"):
        z_all = self.encode(dataloader)
        Q = self.soft_clusters(z_all)
        if method == "soft":
            return Q
        elif method == "argmax":
            return self.hard_clusters_argmax(Q)
        elif method == "kmeans":
            return self.hard_clusters_kmeans(Q)
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
    def evaluate(self, dataloader, metrics="all", method="kmeans"):
        predicted = self.cluster(dataloader=dataloader, method=method)

        ground_truth = []
        with torch.no_grad():
            for batch in dataloader:
                X, y = batch
                ground_truth.append(y)
        ground_truth = torch.cat(ground_truth, dim=0)

        # Ensure both are 1D numpy arrays
        ground_truth = ground_truth.cpu().numpy().reshape(-1)
        if isinstance(predicted, torch.Tensor):
            predicted = predicted.cpu().numpy().reshape(-1)
        else:
            predicted = np.array(predicted).reshape(-1)
        logging.debug(f"Shapes: {ground_truth.shape} vs {predicted.shape}")
        assert ground_truth.shape == predicted.shape, f"Shape mismatch: {ground_truth.shape} vs {predicted.shape}"

        return all_metrics(ground_truth, predicted)