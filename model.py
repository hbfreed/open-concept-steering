import torch
import torch.nn as nn
import torch.nn.functional as F

class SAE(nn.Module):
    def __init__(self, input_size, hidden_size, init_scale=0.1):
        super().__init__()
        
        # Store dimensions
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Initialize as before 
        self.encode = nn.Linear(input_size, hidden_size, bias=True)
        self.decode = nn.Linear(hidden_size, input_size, bias=True)
                
        with torch.no_grad():
            # Random directions
            decoder_weights = torch.randn(input_size, hidden_size) 
            # Normalize columns
            decoder_weights = decoder_weights / torch.linalg.vector_norm(decoder_weights, dim=0, keepdim=True)
            # Scale by random values between 0.05 and 1.0
            scales = torch.rand(hidden_size) * 0.95 + 0.05
            decoder_weights = decoder_weights * scales
            
            self.decode.weight.data = decoder_weights
            self.encode.weight.data = decoder_weights.T.contiguous()
            self.encode.bias.data.zero_() #zero in place
            self.decode.bias.data.zero_()

        self.constrain_weights()

    @property
    def device(self):
        """Return the device the model parameters are on"""
        return next(self.parameters()).device

    def constrain_weights(self):
        """Constrain the decoder weights to have unit norm."""
        with torch.no_grad():
            decoder_norm = torch.linalg.vector_norm(self.decode.weight, dim=0, keepdim=True)
            self.decode.weight.data = self.decode.weight.data / decoder_norm

    def forward(self, x):
        features = F.relu(self.encode(x))
        reconstruction = self.decode(features)
        return reconstruction, features

    def get_decoder_norms(self):
        # returns a 1-D tensor (hidden_size,) on the right device/dtype
        return torch.linalg.vector_norm(self.decode.weight, dim=0)

        
    @property
    def W_dec(self):
        """Return decoder weights for easier access during analysis"""
        return self.decode.weight
        
    def compute_loss(self, x, recon, feats, _lambda):
        # reconstruction term — sum over feature-dim, mean over batch
        recon_mse = (recon - x).pow(2).sum(-1).mean()

        # sparsity term — L1 on feature activations * current decoder-column norms
        sparsity = (feats.abs() * self.get_decoder_norms()).sum(1).mean()

        return recon_mse + _lambda * sparsity
