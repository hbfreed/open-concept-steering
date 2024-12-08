import pytest
import torch
from open_concept_steering.sae.model import SAE

def test_sparsity_penalty_increases_with_dense_features():
    model = SAE(input_size=10, hidden_size=20)
    
    # Create two feature tensors - one sparse, one dense
    sparse_features = torch.zeros(1, 20)
    sparse_features[0, 0] = 1.0  # Only one active feature
    
    dense_features = torch.ones(1, 20) * 0.05  # All features slightly active
    
    # Both should give similar reconstructions but different losses
    x = torch.randn(1, 10)
    loss_sparse = model.compute_loss(x, x, sparse_features)  # Should be lower
    loss_dense = model.compute_loss(x, x, dense_features)    # Should be higher
    
    assert loss_sparse < loss_dense

def test_perfect_reconstruction_still_has_sparsity_penalty():
    model = SAE(input_size=10, hidden_size=20)
    x = torch.randn(1, 10)
    
    # Even with perfect reconstruction, should have non-zero loss due to L1 penalty
    features = torch.ones(1, 20)  # Dense features
    loss = model.compute_loss(x, x, features)  # x == reconstruction
    
    assert loss > 0  # Loss should not be zero due to sparsity penalty

