import os
import optuna
import importlib.util
import subprocess
import json
from datetime import datetime

def objective(trial):
    """Optuna objective function to minimize"""
    # Define parameter space
    lambda_final = trial.suggest_float("lambda_final", 0.001, 0.5, log=True)
    lambda_warmup_pct = trial.suggest_float("lambda_warmup_pct", 0.3, 0.8)
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 5e-5, log=True)
    
    # Load base config
    spec = importlib.util.spec_from_file_location("config", args.base_config)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    base_config = config_module.config
    
    # Create experiment directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "tmp_configs"), exist_ok=True)
    
    # Create a unique name for this run
    run_name = f"trial_{trial.number}_lambda_{lambda_final:.4f}_warmup_{int(lambda_warmup_pct*100)}_lr_{learning_rate:.6f}"
    
    # Create config dict for this run
    config = dict(base_config)
    config["lambda_final"] = lambda_final
    config["lambda_warmup_pct"] = lambda_warmup_pct
    config["learning_rate"] = learning_rate
    config["wandb_name"] = f"optuna_{run_name}"
    
    # Update output directory
    run_output_dir = os.path.join(args.output_dir, run_name)
    config["out_dir"] = run_output_dir
    
    # Create temporary config file
    config_path = os.path.join(args.output_dir, "tmp_configs", f"{run_name}.py")
    with open(config_path, "w") as f:
        f.write(f"config = {repr(config)}")
    
    # Build command
    cmd = ["python3", "train.py", "--config_path", config_path]
    
    try:
        # Run the training
        subprocess.run(cmd, check=True)
        
        # Process results
        # Let's assume you save some metrics in a JSON file at the end of training
        # If not, you'll need to modify your train.py to save metrics
        metrics_path = os.path.join(run_output_dir, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
                
            # Calculate score (lower is better)
            # For example, weighting reconstruction loss, sparsity, and dead features
            recon_loss = metrics.get("final_recon_loss", 1.0)
            sparsity = metrics.get("final_sparsity", 1.0)
            dead_features_pct = metrics.get("final_dead_features_pct", 100.0)
            
            # You can adjust this formula based on what you care about
            score = recon_loss + (0.1 * sparsity) + (0.5 * dead_features_pct)
            
            # If dead features exceed 20%, penalize heavily
            if dead_features_pct > 20:
                score += 100
                
            return score
        else:
            # If metrics file doesn't exist, assume training failed
            return float("inf")
            
    except subprocess.CalledProcessError:
        # If subprocess fails, return a large value
        return float("inf")

def run_optuna_optimization():
    """Run the Optuna optimization process"""
    parser = argparse.ArgumentParser(description="Run Optuna optimization for SAE hyperparameters")
    parser.add_argument("--base_config", type=str, default="config/train_sae_8k.py", 
                       help="Base configuration file to use")
    parser.add_argument("--output_dir", type=str, default="optuna_results", 
                       help="Directory to store results")
    parser.add_argument("--n_trials", type=int, default=20,
                       help="Number of trials to run")
    parser.add_argument("--timeout", type=int, default=None,
                       help="Timeout in seconds (optional)")
    global args
    args = parser.parse_args()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = os.path.join(args.output_dir, f"study_{timestamp}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create and run study
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout, n_jobs=20)
    
    # Print results
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Save study
    study_path = os.path.join(args.output_dir, "study.pkl")
    with open(study_path, "wb") as f:
        optuna.study.Study._pickle_save_study(study, f)
    
    # Save best parameters
    best_params = {
        "value": trial.value,
        "params": trial.params
    }
    with open(os.path.join(args.output_dir, "best_params.json"), "w") as f:
        json.dump(best_params, f, indent=2)
    
    print(f"Study saved to {study_path}")
    print(f"Best parameters saved to {os.path.join(args.output_dir, 'best_params.json')}")

if __name__ == "__main__":
    import argparse
    run_optuna_optimization()