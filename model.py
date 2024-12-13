import torch
import torch.nn as nn
import torch.nn.functional as F

class SAE(nn.Module):
    def __init__(self, input_size, hidden_size, init_scale=0.1):
        super(SAE, self).__init__()
        
        # Size check
        self.input_size = input_size
        self.hidden_size = hidden_size  # This is typically larger than input_size
        
        # Create the layers
        self.encode = nn.Linear(input_size, hidden_size, bias=True)
        self.decode = nn.Linear(hidden_size, input_size, bias=True)
        
        # Initialize weights according to Anthropic's scheme
        with torch.no_grad():
            # Initialize decoder weights with random directions and small norms
            decoder_weights = torch.randn(hidden_size, input_size)
            decoder_weights = init_scale * decoder_weights / torch.linalg.vector_norm(decoder_weights, dim=1, keepdim=True)
            self.decode.weight.data = decoder_weights.T
            
            # Initialize encoder as transpose of decoder
            self.encode.weight.data = decoder_weights
            
            # Initialize biases to zero
            self.encode.bias.data.zero_()
            self.decode.bias.data.zero_()
    
    def forward(self, x):
        encoded = self.encode(x)
        features = F.relu(encoded)
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
        # MSE loss
        mse_loss = F.mse_loss(reconstruction, x)
        
        # L1 sparsity loss with decoder norms
        decoder_norms = self.get_decoder_norms()
        l1_loss = torch.sum(torch.abs(features) * decoder_norms[None, :])  # Broadcasting the norms
        
        # Total loss
        total_loss = mse_loss + (lambda_ * l1_loss) #Compute SAE loss: MSE + λ * L1 with decoder norms

        
        return total_loss