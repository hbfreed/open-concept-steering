import torch
import torch.nn as nn
import torch.nn.functional as F

class SAE(nn.Module):
    def __init__(self, input_size, hidden_size, init_scale=0.1):
        super().__init__()
        
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
            self.encode.bias.data.zero_()
            self.decode.bias.data.zero_()
    
    def forward(self, x):
        features = self.encode(x)
        features = F.relu(features)
        reconstruction = self.decode(features)
        return reconstruction, features

    def get_decoder_norms(self):
        """Return L2 norms of decoder columns for loss calculation"""
        return torch.linalg.vector_norm(self.decode.weight, dim=0)
        
    def compute_loss(self, x, reconstruction, features, lambda_=5.0):
        """        
        Args:
            x: Input tensor: one of our residual stream vectors
            reconstruction: Reconstructed input
            features: Feature activations (after ReLU)
            lambda_: Sparsity coefficient (default 5.0)
        """
        mse_loss = F.mse_loss(x, reconstruction)

        # L1 sparsity loss with decoder norms
        decoder_norms = self.get_decoder_norms()
        l1_loss = torch.sum(torch.abs(features) * decoder_norms[None, :]) / (x.shape[0] * features.shape[1]) #Mean over batch and features
        # Total loss
        total_loss = mse_loss + (lambda_ * l1_loss) #Compute SAE loss: MSE + λ * L1 with decoder norms
        return total_loss