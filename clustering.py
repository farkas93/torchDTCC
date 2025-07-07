import yaml
import importlib
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.cluster import KMeans

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_class(full_class_string):
    """Dynamically import a class from a string."""
    module_path, class_name = full_class_string.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def load_model(model_cfg, device):
    from dtcc import DTCC  # or update path if needed
    model = DTCC(
        model_cfg["input_dim"],
        model_cfg["hidden_dim"],
        model_cfg["latent_dim"],
        model_cfg["num_layers"],
        model_cfg["num_clusters"]
    )
    model.load_state_dict(torch.load(model_cfg["path"], map_location=device))
    model.to(device)
    model.eval()
    return model

def encode_all_data(model, dataloader, device):
    zs = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            z = model.encoder(batch)
            zs.append(z)
    return torch.cat(zs, dim=0)

def get_soft_clusters(z_all, k):
    U, S, Vh = torch.linalg.svd(z_all, full_matrices=False)
    Q = U[:, :k]
    return Q

def get_hard_clusters(Q):
    return Q.argmax(dim=1)

def get_kmeans_clusters(Q, k):
    Q_np = Q.cpu().numpy()
    kmeans = KMeans(n_clusters=k, n_init=10)
    return kmeans.fit_predict(Q_np)

if __name__ == "__main__":
    # ==== Load config ====
    config = load_config("config.yaml")
    model_cfg = config["model"]
    data_cfg = config["data"]
    output_cfg = config["output"]
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    # ==== Load dataset ====
    DatasetClass = get_class(data_cfg["dataset_class"])
    dataset = DatasetClass(**data_cfg["dataset_args"])
    dataloader = DataLoader(dataset, batch_size=data_cfg["batch_size"], shuffle=False)

    # ==== Load model ====
    model = load_model(model_cfg, device)

    # ==== Encode data ====
    z_all = encode_all_data(model, dataloader, device)

    # ==== Soft clustering ====
    Q = get_soft_clusters(z_all, model_cfg["num_clusters"])

    # ==== Hard clustering ====
    labels_argmax = get_hard_clusters(Q)
    labels_kmeans = get_kmeans_clusters(Q, model_cfg["num_clusters"])

    # ==== Output ====
    np.save(output_cfg["soft_clusters"], Q.cpu().numpy())
    np.save(output_cfg["hard_clusters_argmax"], labels_argmax.cpu().numpy())
    np.save(output_cfg["hard_clusters_kmeans"], labels_kmeans)
    print("Saved soft and hard clustering results.")