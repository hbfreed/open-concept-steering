import torch
import torch.nn as nn
import torch.nn.functional as F

class SAE(nn.Module):
    def __init__(self, input_size, hidden_size, init_scale=0.1):
        super().__init__()
        
        # Initialize as before 
        self.encode = nn.Linear(input_size, hidden_size, bias=True)
        self.decode = nn.Linear(hidden_size, input_size, bias=True)
                
        #initializing weights not working for now, so commented out
        # with torch.no_grad():
        #     # Initialize decoder weights
        #     decoder_weights = torch.randn(hidden_size, input_size)
        #     decoder_weights = decoder_weights.T
        #     decoder_weights = init_scale * decoder_weights / torch.linalg.vector_norm(decoder_weights, dim=0, keepdim=True)
            
        #     self.decode.weight.data = decoder_weights
        #     self.encode.weight.data = decoder_weights.T
        #     self.encode.bias.data.zero_()
        #     self.decode.bias.data.zero_()
            
        #     # Assert that all column norms are approximately init_scale
        #     column_norms = torch.linalg.vector_norm(self.decode.weight.data, dim=0)
        #     assert torch.allclose(column_norms, torch.full_like(column_norms, init_scale), rtol=1e-5), \
        #         f"Column norms should be {init_scale}, but got norms ranging from {column_norms.min().item()} to {column_norms.max().item()}"
    
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