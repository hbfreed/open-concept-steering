import torch
from transformers import PreTrainedModel

class ResidualStreamCollector:
    """Collect the residual stream activations from a model."""

    def __init__(self, model: PreTrainedModel, layer_idx: int): #note: anthropic uses the middle layer
        self.model = model
        self.layer_idx = layer_idx
        self.activations = []
        self.hook_handle = None

    def _hook(self, module, input, output):
        """Hook function to capture residual stream activations.
        In LlamaDecoderLayer, the input to input_layernorm is the residual stream.
        """
        # input[0] is the residual stream tensor: [batch_size, seq_len, hidden_size]
        self.activations.append(input[0].detach())

    def attach_hook(self):
        """Attach hook to the specified layer."""
        target_layer = self.model.model.layers[self.layer_idx].input_layernorm
        self.hook_handle = target_layer.register_forward_hook(self._hook)

    def remove_hook(self):
        """Remove hook."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

    def clear_activations(self):
        """Clear the collected activations."""
        self.activations = []

    def get_activations(self) -> torch.Tensor:
        """Get the collected activations."""
        if not self.activations:
            raise ValueError("No activations collected.")
        return self.activations
    