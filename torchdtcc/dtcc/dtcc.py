import logging
import torch
import torch.nn as nn
from typing import List
from .dilated_rnn import DilatedRNN
from .helper import EPS, stablize

class Encoder(nn.Module):
    def __init__(self, input_dim, num_layers, hidden_dim, dilation_rates, latent_dim):
        super(Encoder, self).__init__()
        self.dilated_rnn = DilatedRNN(input_dim, num_layers, hidden_dim, dilation_rates)
        #self.norm = nn.LayerNorm(latent_dim)
        
    def forward(self, x):
        assert x.ndim == 3, f"Input must be 3D, got {x.shape}"
        z = self.dilated_rnn(x)
        return z


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
                 tau_C: float = 0.5,
                 weight_sharing = False,
                 stable_svd: bool = False):
        super(DTCC, self).__init__()
        latent_dim = sum(hidden_dims) * 2 # the hidden dims times 2 for being bidirectional
        self.encoder = Encoder(input_dim, num_layers, hidden_dims, dilation_rates, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)
        if weight_sharing:
            self.aug_encoder = Encoder(input_dim, num_layers, hidden_dims, dilation_rates, latent_dim)
            self.aug_decoder = Decoder(latent_dim, input_dim)
        self.num_clusters = num_clusters
        self.tau_I = tau_I  # Instance temperature
        self.tau_C = tau_C  # Cluster temperature
        self.weight_sharing = weight_sharing
        self.stable_svd = stable_svd
        self._init_weights()
    
    def _init_weights(self):
        # Xavier/Glorot init
        for m in self.modules():
            if isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def get_num_clusters(self):
        return self.num_clusters

    def get_stable_svd(self):
        return self.stable_svd

    def forward(self, x, x_aug):
        assert x.ndim == 3, f"Input must be 3D, got {x.shape}"
        # Original view
        z = self.encoder(x)
        x_recon = self.decoder(z, x.size(1))
        
        # Augmented view
        if self.weight_sharing:
            z_aug = self.aug_encoder(x_aug)
            x_aug_recon = self.aug_decoder(z_aug, x_aug.size(1))
        else:
            z_aug = self.encoder(x)
            x_aug_recon = self.decoder(z_aug, x_aug.size(1))
        return z, z_aug, x_recon, x_aug_recon
    
    def compute_reconstruction_loss(self, x, x_recon, x_aug, x_aug_recon):
        # L_recon-org = 1/n ∑ ||x_i - x̂_i||_2^2
        # ||x_i - x̂_i||_2^2 sums over sequence length and feature dimension for each sample.
        # Then, this sum is averaged over the batch (1/n).
        recon_org = torch.sum((x - x_recon) ** 2, dim=(1, 2)).mean()
        recon_aug = torch.sum((x_aug - x_aug_recon) ** 2, dim=(1, 2)).mean()
        return recon_org + recon_aug
        recon_org = torch.mean((x - x_recon) ** 2)
        recon_aug = torch.mean((x_aug - x_aug_recon) ** 2)
        return recon_org + recon_aug
    
    def calculate_Q(self, z, z_aug):
        assert not torch.isnan(z).any(), "NaN in z before SVD"
        assert not torch.isinf(z).any(), "Inf in z before SVD"
        assert not torch.isnan(z_aug).any(), "NaN in z_aug before SVD"
        assert not torch.isinf(z_aug).any(), "Inf in z_aug before SVD"
        
        logging.debug(f"original z:\n {z}")
        logging.debug(f"original z_aug:\n {z_aug}")
        if self.stable_svd:
            z = stablize(z)
            logging.debug(f"stablized z:\n {z}")
            z_aug = stablize(z_aug)
            logging.debug(f"stablized z_aug:\n {z_aug}")

        # SVD for original view
        U, S, V = torch.linalg.svd(z)  # z: [batch_size, latent_dim]
        Q = U[:, :self.num_clusters]  # [batch_size, num_clusters]
        assert not torch.isnan(Q).any(), "NaN in Q after SVD"

        # SVD for augmented view
        
        U_aug, S_aug, V_aug = torch.linalg.svd(z_aug)
        Q_aug = U_aug[:, :self.num_clusters]
        assert not torch.isnan(Q_aug).any(), "NaN in Q_aug after SVD"
        return Q, Q_aug, {"U": U, "S": S, "V": V, "U_aug": U_aug, "S_aug": S_aug, "V_aug": V_aug}

    def compute_cluster_distribution_loss(self, z, z_aug, Q, Q_aug):
        # If one does not transpose z, you don't have the dimensionalities described in the paper and the math
        # does not work.
        z = z.T
        z_aug = z_aug.T

        # Compute k-means loss
        gram_matrix = torch.matmul(z.T, z)           # [batch_size, batch_size]
        gram_matrix_aug = torch.matmul(z_aug.T, z_aug)  # [batch_size, batch_size]

        # Compute QTZQ = (z.T @ Q).T @ (z.T @ Q)
        ZQ = torch.matmul(z, Q)        # [latent_dim, num_clusters]
        QZTZQ = torch.matmul(ZQ.T, ZQ)    # [num_clusters, latent_dim] @ [latent_dim, num_clusters] = [num_clusters, num_clusters]

        ZQ_aug = torch.matmul(z_aug, Q_aug)
        QZTZQ_aug = torch.matmul(ZQ_aug.T, ZQ_aug)

        km_org = torch.trace(gram_matrix) - torch.trace(QZTZQ)
        km_aug = torch.trace(gram_matrix_aug) - torch.trace(QZTZQ_aug)

        return (0.5 * (km_org + km_aug))
    
    def _calculate_info_nce_loss(self, query_features_matrix, positive_features_matrix, all_features_view1, all_features_view2, temperature):
        """
        Calculates InfoNCE-like loss as defined in Eq. 14 and 16.
        
        Args:
            query_features_matrix (Tensor): Matrix where each row is a query vector (e.g., z for instance, Q.T for cluster).
                                            Shape: [num_queries, feature_dim]
            positive_features_matrix (Tensor): Matrix where each row is the positive key for the corresponding query.
                                               Shape: [num_queries, feature_dim]
            all_features_view1 (Tensor): Matrix of all potential negative keys from view 1.
                                         Shape: [num_queries, feature_dim]
            all_features_view2 (Tensor): Matrix of all potential negative keys from view 2.
                                         Shape: [num_queries, feature_dim]
            temperature (float): Temperature parameter.
            
        Returns:
            Tensor: Loss terms for each query. Shape: [num_queries].
        """
        num_queries = query_features_matrix.size(0)
        
        # Normalize features along the feature dimension (dim=1)
        query_norm = nn.functional.normalize(query_features_matrix, dim=1)
        positive_norm = nn.functional.normalize(positive_features_matrix, dim=1)
        all_v1_norm = nn.functional.normalize(all_features_view1, dim=1)
        all_v2_norm = nn.functional.normalize(all_features_view2, dim=1)

        # Numerator: exp(sim(query_i, positive_i) / temperature)
        # This is a dot product of two matrices, then take the diagonal for (query_i, positive_i) pairs
        sim_pos = torch.sum(query_norm * positive_norm, dim=1) / temperature # Element-wise product then sum for each row

        # Denominator terms
        # sim(query_i, all_j_from_view1) -> for term exp(sim(z_i, z_j)) or exp(sim(q_i, q_j))
        sim_v1 = torch.matmul(query_norm, all_v1_norm.T) / temperature # [num_queries, num_queries]

        # sim(query_i, all_j_from_view2) -> for term 1_[i≠j] exp(sim(z_i, z_j^a)) or 1_[i≠j] exp(sim(q_i, q_j^a))
        sim_v2 = torch.matmul(query_norm, all_v2_norm.T) / temperature # [num_queries, num_queries]

        # Apply 1_[i≠j] mask: for sim(query_i, view2_j), when i=j, it should be excluded (set to -inf before exp)
        mask_diag = torch.eye(num_queries, device=query_features_matrix.device, dtype=torch.bool)
        sim_v2.masked_fill_(mask_diag, float('-inf')) 

        # Exponentiate similarities
        exp_sim_pos = torch.exp(sim_pos) # [num_queries]
        exp_sim_v1 = torch.exp(sim_v1) # [num_queries, num_queries]
        exp_sim_v2 = torch.exp(sim_v2) # [num_queries, num_queries]

        # Sum for denominator
        # The sum is over j. So sum along dim=1
        denominator = torch.sum(exp_sim_v1, dim=1) + torch.sum(exp_sim_v2, dim=1)
        
        # Add a small epsilon for numerical stability
        denominator = denominator.clamp(min=EPS)

        loss_terms = -torch.log(exp_sim_pos / denominator)
        return loss_terms

    
    def compute_instance_contrastive_loss(self, z, z_aug):
        # z: [batch_size, latent_dim]
        # z_aug: [batch_size, latent_dim]
        n = z.size(0) # batch_size

        # For ℓ_{z_i}: query_features_matrix = z, positive_features_matrix = z_aug,
        # all_features_view1 = z, all_features_view2 = z_aug
        loss_z = self._calculate_info_nce_loss(z, z_aug, z, z_aug, self.tau_I)
        
        # For ℓ_{z_i^a}: query_features_matrix = z_aug, positive_features_matrix = z,
        # all_features_view1 = z_aug, all_features_view2 = z
        loss_z_aug = self._calculate_info_nce_loss(z_aug, z, z_aug, z, self.tau_I)
        
        scalar = 1.0 / (2*n)
        return scalar * (torch.sum(loss_z) + torch.sum(loss_z_aug))
    
    def compute_cluster_contrastive_loss(self, Q, Q_aug):
        # Q: [batch_size, num_clusters]
        # Q_aug: [batch_size, num_clusters]
        k = Q.size(1) # num_clusters
        
        # For cluster contrastive, the queries are the columns of Q (or Q_aug).
        # So, we need to transpose Q and Q_aug to make columns into rows for _calculate_info_nce_loss.
        # Q_t: [num_clusters, batch_size]
        Q_t = Q.T
        Q_aug_t = Q_aug.T

        # For ℓ_{q_i}: query_features_matrix = Q_t, positive_features_matrix = Q_aug_t,
        # all_features_view1 = Q_t, all_features_view2 = Q_aug_t
        loss_q = self._calculate_info_nce_loss(Q_t, Q_aug_t, Q_t, Q_aug_t, self.tau_C)
        
        # For ℓ_{q_i^a}: query_features_matrix = Q_aug_t, positive_features_matrix = Q_t,
        # all_features_view1 = Q_aug_t, all_features_view2 = Q_t
        loss_q_aug = self._calculate_info_nce_loss(Q_aug_t, Q_t, Q_aug_t, Q_t, self.tau_C)
        
        scalar = 1.0 / (2*k)
        contrastive_loss = scalar * (torch.sum(loss_q) + torch.sum(loss_q_aug))
        
        # Compute entropy term H(q) = − ∑ [P (q_i) log P (q_i) + P (q_i^a) log P (q_i^a)]
        P_q = torch.mean(Q, dim=0) # P(q_i) = 1/n ∑ q_ji
        P_q_aug = torch.mean(Q_aug, dim=0)
        
        # Apply clamp for numerical stability
        P_q = torch.clamp(P_q, min=EPS)
        P_q_aug = torch.clamp(P_q_aug, min=EPS)
        
        entropy = torch.sum(P_q * torch.log(P_q)) + torch.sum(P_q_aug * torch.log(P_q_aug))
        
        # The paper's L_cluster = ... - H(q). Your `entropy` variable is already `-sum(P log P)`.
        # So, `contrastive_loss + entropy` correctly implements `contrastive_loss - |H(q)|`.
        return (contrastive_loss + entropy)
