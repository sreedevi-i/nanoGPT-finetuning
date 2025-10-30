# generate_configs_128.py
import itertools, os

# ===============================
# 1) Fixed “batch” pairs (4 × 32)
# ===============================
# Batch 1: (64, 4) 
# Batch 2: (64, 6)
# Batch 3: (128, 4)
# Batch 4: (128, 6)
fixed_pairs = [
    (1, 64, 4),    # (batch_id, block_size, n_layer)
    (2, 64, 6),
    (3, 128, 4),
    (4, 128, 6),
]

# ===============================
# 2) Variables to sweep (32 combos per batch)
# ===============================
grid = {
    "n_head": [4, 8],
    "n_embd": [128, 256],
    "batch_size": [8, 16],
    "max_iters": [1000, 2000],
    "dropout": [0.1, 0.2],
}

# ===============================
# 3) Static defaults
# ===============================
base_config = {
    "eval_interval": 250,
    "eval_iters": 200,
    "log_interval": 10,
    "always_save_checkpoint": False,
    "wandb_log": False,
    "wandb_project": "shakespeare",
    "wandb_run_name": "mini-gpt",
    "dataset": "shakespeare",
    "gradient_accumulation_steps": 1,
    "learning_rate": 1e-3,
    "lr_decay_iters": None,  # will match max_iters per combo
    "min_lr": 1e-4,
    "beta2": 0.99,
    "warmup_iters": 100,
}

keys = list(grid.keys())
combos = list(itertools.product(*grid.values()))  # 32

global_idx = 0
for batch_id, block_size, n_layer in fixed_pairs:
    cfg_dir = f"{batch_id}_configs"
    os.makedirs(cfg_dir, exist_ok=True)

    for j, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        lr_decay_iters = params["max_iters"]
        exp_idx = global_idx  # 0..127
        out_dir = f"out/exp_{exp_idx:03d}"

        cfg_text = f"""# Auto-generated nanoGPT experiment #{exp_idx:03d}
# Batch {batch_id}: block_size={block_size}, n_layer={n_layer}
# ------------------------------------------------------------

out_dir = '{out_dir}'
eval_interval = {base_config["eval_interval"]}
eval_iters = {base_config["eval_iters"]}
log_interval = {base_config["log_interval"]}

always_save_checkpoint = {base_config["always_save_checkpoint"]}
wandb_log = {base_config["wandb_log"]}
wandb_project = '{base_config["wandb_project"]}'
wandb_run_name = '{base_config["wandb_run_name"]}'

dataset = '{base_config["dataset"]}'
gradient_accumulation_steps = {base_config["gradient_accumulation_steps"]}
batch_size = {params["batch_size"]}
block_size = {block_size}

n_layer = {n_layer}
n_head = {params["n_head"]}
n_embd = {params["n_embd"]}
dropout = {params["dropout"]}

learning_rate = {base_config["learning_rate"]}
max_iters = {params["max_iters"]}
lr_decay_iters = {lr_decay_iters}
min_lr = {base_config["min_lr"]}
beta2 = {base_config["beta2"]}
warmup_iters = {base_config["warmup_iters"]}

# device = 'cpu'
# compile = False
"""

        with open(f"{cfg_dir}/exp_{exp_idx:03d}.py", "w") as f:
            f.write(cfg_text)

        global_idx += 1

print(f" Generated {global_idx} experiment config files across 4 folders.")
