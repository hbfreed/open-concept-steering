import torch
from transformers import PreTrainedModel
from typing import Optional

class ResidualStreamCollector:
    """
    Collect the residual stream from various transformer model architectures.
    Currently supports:
    - Llama (input_layernorm)
    - OLMo (post_attention_layernorm)
    - More architectures can be added as needed
    
    The collector attaches a hook to collect the residual stream at a specified layer.
    Different model architectures have different layer structures, but we want to capture
    the same conceptual point in the residual stream across architectures.
    """
    
    def __init__(self, model: PreTrainedModel, layer_idx: int):
        """
        Initialize the collector.
        
        Args:
            model (PreTrainedModel): The transformer model to collect from
            layer_idx (int): Which layer to collect from
        """
        self.model = model
        self.layer_idx = layer_idx
        self.residual_stream = None
        self.hook_handle = None
        
    def _hook(self, module, input, output):
        """
        Hook function to capture residual stream.
        Stores the first input to the layer, which is the residual stream.
        
        Args:
            module: The PyTorch module
            input: Input to the module
            output: Output of the module
        """
        self.residual_stream = input[0].detach()

    def _get_base_model(self) -> PreTrainedModel:
        """
        Get the base model, handling various wrapper cases.
        
        Returns:
            The base model without wrappers
        """
        base_model = self.model
        # Handle DistributedDataParallel and similar wrappers
        if hasattr(base_model, "module"):
            base_model = base_model.module
        # Handle the common pattern where the base model is in a .model attribute
        if hasattr(base_model, "model"):
            base_model = base_model.model
        return base_model

    def _get_target_layer(self, base_model: PreTrainedModel) -> Optional[torch.nn.Module]:
        """
        Get the appropriate layer to hook into based on model architecture.
        
        Args:
            base_model: The base transformer model
            
        Returns:
            The specific layer to attach the hook to, or None if architecture not supported
            
        This is where we handle different model architectures. Add new architectures here.
        """
        try:
            # Get the transformer layer
            layer = base_model.layers[self.layer_idx]
            
            # Check for different model architectures
            # Llama-style architectures
            if hasattr(layer, "input_layernorm"):
                return layer.input_layernorm
                
            # OLMo-style architectures
            elif hasattr(layer, "post_attention_layernorm"):
                return layer.post_attention_layernorm
                
            # Add more architecture patterns here as needed
            # elif hasattr(layer, "other_architecture_pattern"):
            #     return layer.other_pattern
                
            else:
                model_name = base_model.__class__.__name__
                raise ValueError(
                    f"Unsupported model architecture: {model_name}. "
                    "Could not find a supported layer normalization pattern. "
                    "Please add support for this architecture in _get_target_layer()."
                )
                
        except Exception as e:
            raise ValueError(
                f"Error accessing layer {self.layer_idx}. "
                f"Original error: {str(e)}"
            )

    def attach_hook(self):
        """
        Attach hook to the specified layer.
        Will raise ValueError if model architecture is not supported.
        """
        base_model = self._get_base_model()
        target_layer = self._get_target_layer(base_model)
        
        if target_layer is None:
            raise ValueError(
                "Could not find appropriate layer to hook into. "
                "Is this model architecture supported?"
            )
            
        self.hook_handle = target_layer.register_forward_hook(self._hook)

    def remove_hook(self):
        """Remove the hook if it exists."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

    def clear_residual_stream(self):
        """Clear the collected residual stream."""
        self.residual_stream = None

    def get_residual_stream(self) -> torch.Tensor:
        """
        Get the collected residual stream.
        
        Returns:
            torch.Tensor: The collected residual stream
            
        Raises:
            ValueError: If no residual stream has been collected
        """
        if self.residual_stream is None:
            raise ValueError(
                "No residual stream collected. "
                "Did you run the model with the hook attached?"
            )
        return self.residual_stream

    def __del__(self):
        """Cleanup by removing hook on deletion."""
        self.remove_hook()