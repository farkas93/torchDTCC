import torch
import torch.nn as nn
import torch.nn.functional as F

class DilatedRNN(nn.Module):
    def __init__(self, input_dim, num_layers, hidden_dims, dilation_rates, bidirectional=True):
        super(DilatedRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_dims = hidden_dims
        self.dilation_rates = dilation_rates
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        self.rnn_layers = nn.ModuleList()
        for i in range(num_layers):
            current_input_dim = input_dim if i == 0 else self.hidden_dims[i-1] * self.num_directions
            # Each GRU in the list is a single-layer GRU
            self.rnn_layers.append(nn.GRU(current_input_dim, self.hidden_dims[i], batch_first=True, 
                                         bidirectional=bidirectional))
                
    def forward(self, x):
        assert x.ndim == 3, f"Input must be 3D, got {x.shape}"        
        
        layer_last_hidden_states = [] # To store the concatenated last hidden states for each layer
        current_input = x # Input for the current layer, will be updated iteratively
        
        for i, layer_gru in enumerate(self.rnn_layers): # Renamed `layer` to `layer_gru` for clarity
            dilation_rate = self.dilation_rates[i]
            batch_size, seq_len, feature_dim = current_input.size()

            if dilation_rate == 1: # Standard RNN layer (no dilation applied)
                output_seq, hidden_state = layer_gru(current_input)
                # hidden_state: [num_directions, batch_size, hidden_size]
                # Flatten the hidden state to [batch_size, num_directions * hidden_size]
                layer_last_hidden_states.append(hidden_state.permute(1, 0, 2).contiguous().view(batch_size, -1))
                current_input = output_seq # Output of this layer becomes input for the next
                continue

            # --- Dilated RNN logic (parallelization trick from paper Figure 1, right) ---
            # 1. Pad sequence to be divisible by dilation_rate
            pad_len = (dilation_rate - (seq_len % dilation_rate)) % dilation_rate
            if pad_len > 0:
                x_padded = F.pad(current_input, (0, 0, 0, pad_len))
            else:
                x_padded = current_input
            padded_seq_len = x_padded.size(1)

            # 2. Reshape into `dilation_rate` parallel streams
            # [batch_size, padded_seq_len, feature_dim] -> 
            # [batch_size, padded_seq_len / dilation_rate, dilation_rate, feature_dim] (conceptual split)
            # -> [batch_size, dilation_rate, padded_seq_len / dilation_rate, feature_dim] (permute to group streams)
            # -> [batch_size * dilation_rate, padded_seq_len / dilation_rate, feature_dim] (view for batch_first GRU)
            x_interleaved = x_padded.view(batch_size, padded_seq_len // dilation_rate, dilation_rate, feature_dim)
            x_interleaved = x_interleaved.permute(0, 2, 1, 3).contiguous().view(
                batch_size * dilation_rate, padded_seq_len // dilation_rate, feature_dim
            )

            # 3. Process each stream with a standard GRU layer (weights are shared across streams)
            output_interleaved, hidden_interleaved = layer_gru(x_interleaved) 
            # output_interleaved: [batch_size * dilation_rate, new_seq_len (padded_seq_len/dilation_rate), hidden_dim * num_directions]

            # 4. Reshape back and interleave outputs to get the full sequence output for the next layer
            output_reshaped = output_interleaved.view(
                batch_size, dilation_rate, padded_seq_len // dilation_rate, self.hidden_dims[i] * self.num_directions
            )
            output_seq = output_reshaped.permute(0, 2, 1, 3).contiguous().view(
                batch_size, padded_seq_len, self.hidden_dims[i] * self.num_directions
            )
            # Remove padding from the output sequence
            output_seq = output_seq[:, :seq_len, :]
            
            # 5. Extract the "last hidden state" for this layer
            # This takes the output of the GRU at the last time step of the (unpadded) sequence.
            layer_last_hidden_states.append(output_seq[:, -1, :])
            
            current_input = output_seq # Output of this layer becomes input for the next layer
            
        # Concatenate the last hidden state from each layer to form the final latent representation `z`
        # This will be [batch_size, sum(hidden_dims[i] * num_directions)]
        return torch.cat(layer_last_hidden_states, dim=1)