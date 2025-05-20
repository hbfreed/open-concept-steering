config = {
    # ── Data params ─────────────────────────────────────
    "data_dir": "/media/henry/MoreFiles/olmo_dataset",
    "out_dir":  "out/sae_65k_lambda26_ramp30",   # new folder

    # ── Architecture params ────────────────────────────
    "input_size":  4096,
    "hidden_size": 65_536,
    "init_scale":  0.1,

    # ── Training params ────────────────────────────────
    "batch_size":    4096,
    "learning_rate": 5e-5,
    "num_epochs":    1,          # 200 k steps at BS=4096 ≈ 600 M tokens
    "lambda_final":        26,   # stronger but not brutal
    "lambda_warmup_pct":   0.30, # 30 % linear ramp (60 k steps)

    # ── Wandb params ───────────────────────────────────
    "wandb_project": "sae-training",
    "wandb_name":    "sae_65k_lambda26_ramp30",
    "wandb_entity":  "hbfreed",
}
