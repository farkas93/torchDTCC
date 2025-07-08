import torch
import torch.nn as nn
from typing import List
from .constants import EPS

class DilatedRNN(nn.Module):
    def __init__(self, input_dim, num_layers, hidden_dim, dilation_rates, bidirectional=True):
        super(DilatedRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_dims = hidden_dim
        self.dilation_rates = dilation_rates
        self.bidirectional = bidirectional
        
        self.rnn_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.rnn_layers.append(nn.GRU(input_dim, self.hidden_dims[i], batch_first=True, 
                                             bidirectional=bidirectional))
            else:
                input_size = self.hidden_dims[i-1] * (2 if bidirectional else 1)
                self.rnn_layers.append(nn.GRU(input_size, self.hidden_dims[i], batch_first=True, 
                                             bidirectional=bidirectional))
                
    def forward(self, x):
        assert x.ndim == 3, f"Input must be 3D, got {x.shape}"        
        outputs = []
        current_input = x
        
        for i, layer in enumerate(self.rnn_layers):
            # Apply dilation
            dilated_input = self._apply_dilation(current_input, self.dilation_rates[i])
            output, hidden = layer(dilated_input)
            outputs.append(hidden.view(hidden.size(1), -1))  # Extract last hidden state
            current_input = output
            
        # Concatenate the last hidden state from each layer
        return torch.cat(outputs, dim=1)
    
    def _apply_dilation(self, x, dilation_rate):
        """
        Apply dilation to input sequence.
        Args:
            x: [batch_size, seq_len, feature_dim]
            dilation_rate: int
        Returns:
            Dilated x: [batch_size, new_seq_len, feature_dim]
        """
        batch_size, seq_len, feature_dim = x.size()
        if seq_len < dilation_rate:
            # Fallback: just return the last time step (or pad)
            last_step = x[:, -1:, :]  # shape: [batch, 1, feature_dim]
            return last_step
        else:
            # Usual dilation
            indices = torch.arange(0, seq_len, dilation_rate, device=x.device)
            dilated_x = torch.index_select(x, 1, indices)
            return dilated_x


class Encoder(nn.Module):
    def __init__(self, input_dim, num_layers, hidden_dim, dilation_rates, latent_dim):
        super(Encoder, self).__init__()
        self.dilated_rnn = DilatedRNN(input_dim, num_layers, hidden_dim, dilation_rates)
        self.norm = nn.LayerNorm(latent_dim)
        
    def forward(self, x):
        assert x.ndim == 3, f"Input must be 3D, got {x.shape}"
        z = self.dilated_rnn(x)
        return self.norm(z)


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.rnn = nn.GRU(latent_dim, latent_dim, batch_first=True)
        self.output_layer = nn.Linear(latent_dim, output_dim)
        
    def forward(self, z, seq_len):
        # Expand z to sequence length
        z_expanded = z.unsqueeze(1).repeat(1, seq_len, 1)
        output, _ = self.rnn(z_expanded)
        return self.output_layer(output)

class DTCC(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 num_layers: int, 
                 num_clusters: int,
                 hidden_dims: List[int], 
                 dilation_rates: List[int], 
                 tau_I: float = 0.5,
                 tau_C: float = 0.5):
        super(DTCC, self).__init__()
        latent_dim = sum(hidden_dims) * 2 # the hidden dims times 2 for being bidirectional
        self.encoder = Encoder(input_dim, num_layers, hidden_dims, dilation_rates, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)
        self.aug_encoder = Encoder(input_dim, num_layers, hidden_dims, dilation_rates, latent_dim)
        self.aug_decoder = Decoder(latent_dim, input_dim)
        self.num_clusters = num_clusters
        self.tau_I = tau_I  # Instance temperature
        self.tau_C = tau_C  # Cluster temperature
        
    def forward(self, x, x_aug):
        assert x.ndim == 3, f"Input must be 3D, got {x.shape}"
        # Original view
        z = self.encoder(x)
        x_recon = self.decoder(z, x.size(1))
        
        # Augmented view
        z_aug = self.aug_encoder(x_aug)
        x_aug_recon = self.aug_decoder(z_aug, x_aug.size(1))
        
        return z, z_aug, x_recon, x_aug_recon
    
    def compute_reconstruction_loss(self, x, x_recon, x_aug, x_aug_recon):
        recon_org = torch.mean((x - x_recon) ** 2)
        recon_aug = torch.mean((x_aug - x_aug_recon) ** 2)
        return recon_org + recon_aug
    
    def compute_instance_contrastive_loss(self, z, z_aug):
        n = z.size(0)
        z_norm = nn.functional.normalize(z, dim=1)
        z_aug_norm = nn.functional.normalize(z_aug, dim=1)
        
        # z -> z_aug
        sim_matrix = torch.matmul(z_norm, z_aug_norm.T) / self.tau_I
        max_sim_mat = sim_matrix.max(dim=1, keepdim=True)[0]
        sim_matrix = sim_matrix - max_sim_mat  # shift for stability

        positives = torch.diag(sim_matrix)
        exp_sim = torch.exp(sim_matrix)
        mask = torch.eye(n, device=z.device)
        neg_mask = 1 - mask
        
        # denominator: all except the positive
        denominator = torch.sum(exp_sim * neg_mask, dim=1) + torch.exp(positives)
        # Correct: log(exp(x)/y) = x - log(y)
        instance_loss = -torch.mean(positives - torch.log(denominator + EPS))
        
        # z_aug -> z
        sim_matrix_aug = torch.matmul(z_aug_norm, z_norm.T) / self.tau_I
        max_sim_mat_aug = sim_matrix_aug.max(dim=1, keepdim=True)[0]
        sim_matrix_aug = sim_matrix_aug - max_sim_mat_aug

        positives_aug = torch.diag(sim_matrix_aug)
        exp_sim_aug = torch.exp(sim_matrix_aug)
        denominator_aug = torch.sum(exp_sim_aug * neg_mask, dim=1) + torch.exp(positives_aug)
        instance_loss_aug = -torch.mean(positives_aug - torch.log(denominator_aug + EPS))
        
        return 0.5 * (instance_loss + instance_loss_aug) 
    
    def compute_cluster_distribution_loss(self, z, z_aug):
        # SVD for original view
        #print(f"original z:\n {z}")
        z = z + 1e-6 * torch.randn_like(z)
        z = torch.nan_to_num(z, nan=0.0, posinf=5.0, neginf=-5.0)
        z = torch.clamp(z, -5, 5)
        #print(f"original z:\n {z}")
        U, S, Vh = torch.linalg.svd(z)  # z: [batch_size, latent_dim]
        Q = U[:, :self.num_clusters]  # [batch_size, num_clusters]

        # SVD for augmented view
        #print(f"original z_aug:\n {z_aug}")
        z_aug = z_aug + 1e-6 * torch.randn_like(z_aug)        
        z_aug = torch.nan_to_num(z_aug, nan=0.0, posinf=5.0, neginf=-5.0)
        z_aug = torch.clamp(z_aug, -5, 5)
        #print(f"stablized z_aug:\n {z_aug}")
        U_aug, S_aug, Vh_aug = torch.linalg.svd(z_aug)
        Q_aug = U_aug[:, :self.num_clusters]

        # Compute k-means loss
        gram_matrix = torch.matmul(z.T, z)           # [latent_dim, latent_dim]
        gram_matrix_aug = torch.matmul(z_aug.T, z_aug)  # [latent_dim, latent_dim]

        # Compute QTZQ = (z.T @ Q).T @ (z.T @ Q)
        ZQ = torch.matmul(z.T, Q)        # [latent_dim, num_clusters]
        QTZQ = torch.matmul(ZQ.T, ZQ)    # [num_clusters, latent_dim] @ [latent_dim, num_clusters] = [num_clusters, num_clusters]

        ZQ_aug = torch.matmul(z_aug.T, Q_aug)
        QTZQ_aug = torch.matmul(ZQ_aug.T, ZQ_aug)

        km_org = torch.trace(gram_matrix) - torch.trace(QTZQ)
        km_aug = torch.trace(gram_matrix_aug) - torch.trace(QTZQ_aug)

        return 0.5 * (km_org + km_aug), Q, Q_aug
    
    def compute_cluster_contrastive_loss(self, Q, Q_aug):
        k = Q.size(1)
        
        # Normalize Q and Q_aug
        Q_norm = nn.functional.normalize(Q, dim=0)
        Q_aug_norm = nn.functional.normalize(Q_aug, dim=0)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(Q_norm.T, Q_aug_norm) / self.tau_C
        
        # Positive pairs are on the diagonal
        positives = torch.diag(sim_matrix)
        
        # For each q_i, compute denominator
        exp_sim = torch.exp(sim_matrix)
        mask = torch.eye(k, device=Q.device)
        neg_mask = 1 - mask
        
        denominator = torch.sum(exp_sim * neg_mask, dim=1) + torch.exp(positives)
        cluster_loss = -torch.mean(torch.log(torch.exp(positives) / (denominator+EPS) + EPS))
        
        # Compute entropy term
        P_q = torch.mean(Q, dim=0)
        P_q_aug = torch.mean(Q_aug, dim=0)
        entropy = -torch.sum(P_q * torch.log(P_q + EPS)) - torch.sum(P_q_aug * torch.log(P_q_aug + EPS))
        
        return cluster_loss - entropy