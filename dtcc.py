import torch
import torch.nn as nn

class DilatedRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, bidirectional=True):
        super(DilatedRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        
        self.rnn_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.rnn_layers.append(nn.GRU(input_dim, hidden_dim, batch_first=True, 
                                             bidirectional=bidirectional))
            else:
                self.rnn_layers.append(nn.GRU(hidden_dim * (2 if bidirectional else 1), 
                                             hidden_dim, batch_first=True, 
                                             bidirectional=bidirectional))
                
    def forward(self, x, dilation_rates=None):
        if dilation_rates is None:
            dilation_rates = [2**i for i in range(self.num_layers)]
            
        outputs = []
        current_input = x
        
        for i, layer in enumerate(self.rnn_layers):
            # Apply dilation
            dilated_input = self._apply_dilation(current_input, dilation_rates[i])
            output, hidden = layer(dilated_input)
            outputs.append(hidden.view(hidden.size(1), -1))  # Extract last hidden state
            current_input = output
            
        # Concatenate the last hidden state from each layer
        return torch.cat(outputs, dim=1)
    
    def _apply_dilation(self, x, dilation_rate):
        """
        Apply dilation to the input sequence.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, feature_dim]
            dilation_rate: Dilation rate (skipping factor)
            
        Returns:
            Dilated sequence
        """
        batch_size, seq_len, feature_dim = x.size()
        
        # Calculate new sequence length after dilation
        new_seq_len = seq_len // dilation_rate
        if new_seq_len == 0:
            new_seq_len = 1
        
        # Create dilated sequence by selecting time steps with the dilation rate
        dilated_indices = torch.arange(0, new_seq_len, device=x.device) * dilation_rate
        dilated_x = torch.index_select(x, 1, dilated_indices)
        
        return dilated_x


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers):
        super(Encoder, self).__init__()
        self.dilated_rnn = DilatedRNN(input_dim, hidden_dim, num_layers)
        
    def forward(self, x):
        return self.dilated_rnn(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.rnn = nn.GRU(latent_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, z, seq_len):
        # Expand z to sequence length
        z_expanded = z.unsqueeze(1).repeat(1, seq_len, 1)
        output, _ = self.rnn(z_expanded)
        return self.output_layer(output)


class DTCC(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers, num_clusters):
        super(DTCC, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, num_layers)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
        self.num_clusters = num_clusters
        self.tau_I = 0.5  # Instance temperature
        self.tau_C = 0.5  # Cluster temperature
        
    def forward(self, x, x_aug):
        # Original view
        z = self.encoder(x)
        x_recon = self.decoder(z, x.size(1))
        
        # Augmented view
        z_aug = self.encoder(x_aug)
        x_aug_recon = self.decoder(z_aug, x_aug.size(1))
        
        return z, z_aug, x_recon, x_aug_recon
    
    def compute_reconstruction_loss(self, x, x_recon, x_aug, x_aug_recon):
        recon_org = torch.mean((x - x_recon) ** 2)
        recon_aug = torch.mean((x_aug - x_aug_recon) ** 2)
        return recon_org + recon_aug
    
    def compute_instance_contrastive_loss(self, z, z_aug):
        n = z.size(0)
        z_norm = nn.functional.normalize(z, dim=1)
        z_aug_norm = nn.functional.normalize(z_aug, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(z_norm, z_aug_norm.T) / self.tau_I
        
        # Positive pairs are on the diagonal
        positives = torch.diag(sim_matrix)
        
        # For each z_i, compute denominator (sum over all possible negatives)
        exp_sim = torch.exp(sim_matrix)
        mask = torch.eye(n, device=z.device)
        neg_mask = 1 - mask
        
        denominator = torch.sum(exp_sim * neg_mask, dim=1) + torch.exp(positives)
        instance_loss = -torch.mean(torch.log(torch.exp(positives) / denominator))
        
        # Repeat for the augmented view
        sim_matrix_aug = torch.matmul(z_aug_norm, z_norm.T) / self.tau_I
        positives_aug = torch.diag(sim_matrix_aug)
        exp_sim_aug = torch.exp(sim_matrix_aug)
        denominator_aug = torch.sum(exp_sim_aug * neg_mask, dim=1) + torch.exp(positives_aug)
        instance_loss_aug = -torch.mean(torch.log(torch.exp(positives_aug) / denominator_aug))
        
        return (instance_loss + instance_loss_aug) / 2
    
    def compute_cluster_distribution_loss(self, z, z_aug):
        # SVD for original view
        gram_matrix = torch.matmul(z.T, z)
        U, S, V = torch.svd(gram_matrix)
        Q = U[:, :self.num_clusters]
        
        # SVD for augmented view
        gram_matrix_aug = torch.matmul(z_aug.T, z_aug)
        U_aug, S_aug, V_aug = torch.svd(gram_matrix_aug)
        Q_aug = U_aug[:, :self.num_clusters]
        
        # Compute k-means loss
        km_org = torch.trace(torch.matmul(z.T, z)) - torch.trace(torch.matmul(torch.matmul(Q.T, torch.matmul(z.T, z)), Q))
        km_aug = torch.trace(torch.matmul(z_aug.T, z_aug)) - torch.trace(torch.matmul(torch.matmul(Q_aug.T, torch.matmul(z_aug.T, z_aug)), Q_aug))
        
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
        cluster_loss = -torch.mean(torch.log(torch.exp(positives) / denominator))
        
        # Compute entropy term
        P_q = torch.mean(Q, dim=0)
        P_q_aug = torch.mean(Q_aug, dim=0)
        entropy = -torch.sum(P_q * torch.log(P_q + 1e-8)) - torch.sum(P_q_aug * torch.log(P_q_aug + 1e-8))
        
        return cluster_loss - entropy