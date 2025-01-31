import torch
from transformers import PreTrainedModel

class ResidualStreamCollector:
    """Collect the residual stream from a model."""
    def __init__(self, model: PreTrainedModel, layer_idx: int):
        self.model = model
        self.layer_idx = layer_idx
        self.residual_stream = None  # Changed from list to single tensor
        self.hook_handle = None

    def _hook(self, module, input, output):
        """Hook function to capture residual stream."""
        # Store directly as tensor instead of appending to list
        self.residual_stream = input[0].detach()

    def attach_hook(self):
        """Attach hook to the specified layer."""
        base_model = self.model.module if hasattr(self.model, "module") else self.model
        if hasattr(base_model, "model"):
            target_layer = base_model.model.layers[self.layer_idx].input_layernorm
        else:
            target_layer = base_model.layers[self.layer_idx].input_layernorm
        self.hook_handle = target_layer.register_forward_hook(self._hook)

    def remove_hook(self):
        """Remove hook."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

    def clear_residual_stream(self):
        """Clear the collected residual stream."""
        self.residual_stream = None

    def get_residual_stream(self) -> torch.Tensor:
        """Get the collected residual stream."""
        if self.residual_stream is None:
            raise ValueError("No residual stream collected.")
        return self.residual_stream