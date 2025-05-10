import os
import torch
import time
from pathlib import Path

class SAETracker:
    """Ultra-simple tracker that just handles basic metrics without complex analysis"""
    
    def __init__(
        self, 
        model, 
        config, 
        out_dir,
        progress_bar=None,
        use_wandb=False,
        log_period=10
    ):
        self.model = model
        self.config = config
        self.out_dir = Path(out_dir)
        self.progress_bar = progress_bar
        self.use_wandb = use_wandb
        self.log_period = log_period
        
        # Simple metrics
        self.current_metrics = {}
        self.start_time = time.time()
        self.last_log_time = self.start_time
        
        # Dead feature tracking (very basic)
        self.feature_activity = None
        self.reset_tracking()
        
        # Create output dir
        os.makedirs(self.out_dir, exist_ok=True)
    
    def reset_tracking(self):
        """Reset basic tracking"""
        self.feature_activity = torch.zeros(self.config['hidden_size'], dtype=torch.bool, device='cpu')
    
    def track_step(self, step, epoch, data, reconstruction, features, loss, lambda_value, lr):
        """Track a single training step with minimal overhead"""
        # Update feature activity (just for dead feature tracking)
        dims_to_sum = tuple(range(features.ndim - 1))
        active_features = (features.detach().abs().sum(dim=dims_to_sum) > 0)
        self.feature_activity = self.feature_activity.to(features.device)
        self.feature_activity |= active_features
        self.feature_activity = self.feature_activity.cpu()
        
        # Calculate only the most basic metrics
        with torch.no_grad():
            recon_loss = ((reconstruction - data) ** 2).mean().item()
            l0_norm = (features.abs() > 0).float().sum(1).mean().item()
            
            # Compute time metrics
            current_time = time.time()
            step_time = current_time - self.last_log_time
            self.last_log_time = current_time
        
        # Store minimal metrics
        self.current_metrics = {
            'loss': loss.item(),
            'recon_loss': recon_loss,
            'l0_norm': l0_norm,
            'lambda': lambda_value,
            'learning_rate': lr,
            'step': step,
            'epoch': epoch,
            'step_time': step_time
        }
        
        # Update progress bar
        if self.progress_bar is not None:
            self.progress_bar.set_description(
                f"Loss: {loss.item():.2f}, "
                f"Recon: {recon_loss:.4f}, "
                f"L0: {l0_norm:.1f}, "
                f"λ: {lambda_value:.4f}"
            )
        
        # Log to wandb (minimal metrics only)
        if self.use_wandb and step % self.log_period == 0:
            import wandb
            wandb.log(self.current_metrics)
    
    def identify_dead_features(self):
        """Return indices of features that haven't activated yet"""
        return torch.where(~self.feature_activity)[0].tolist()
    
    def save_final_metrics(self, model_path):
        """Save minimal final metrics"""
        import json
        metrics_path = os.path.join(os.path.dirname(model_path), "final_metrics.json")
        
        metrics = {
            "final_recon_loss": self.current_metrics.get('recon_loss', 0.0),
            "final_loss": self.current_metrics.get('loss', 0.0),
            "final_l0_norm": self.current_metrics.get('l0_norm', 0.0),
            "lambda_final": self.config.get("lambda_final", 0.0),
            "learning_rate": self.config.get("learning_rate", 0.0),
            "training_steps": self.current_metrics.get('step', 0),
        }
        
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        if self.use_wandb:
            import wandb
            wandb.log({f"final/{k}": v for k, v in metrics.items()})