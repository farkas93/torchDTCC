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
        self.norm = nn.LayerNorm(latent_dim)
        
    def forward(self, x):
        assert x.ndim == 3, f"Input must be 3D, got {x.shape}"
        z = self.dilated_rnn(x)
        z = self.norm(z)
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

class DTCCAutoencoder(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 num_layers: int,
                 hidden_dims: List[int], 
                 dilation_rates: List[int],
                 init="x/g"):
        super(DTCCAutoencoder, self).__init__()
        latent_dim = sum(hidden_dims) * 2 # the hidden dims times 2 for being bidirectional
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.hidden_dims = hidden_dims
        self.dilation_rates = dilation_rates
        self.encoder = Encoder(input_dim, num_layers, hidden_dims, dilation_rates, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)
        if init == "x/g":
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
    

    def forward(self, x):
        assert x.ndim == 3, f"Input must be 3D, got {x.shape}"
        # Original view
        z = self.encoder(x)
        x_recon = self.decoder(z, x.size(1))        
        return z, x_recon
    
    def compute_reconstruction_loss(self, x, x_recon):
        # L_recon-org = 1/n ∑ ||x_i - x̂_i||_2^2
        # ||x_i - x̂_i||_2^2 sums over sequence length and feature dimension for each sample.
        # Then, this sum is averaged over the batch (1/n).
        recon_org = torch.sum((x - x_recon) ** 2, dim=(1, 2)).mean()
        return recon_org