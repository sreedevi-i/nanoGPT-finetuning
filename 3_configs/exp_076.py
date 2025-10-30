# Auto-generated nanoGPT experiment #076
# Batch 3: block_size=128, n_layer=4
# ------------------------------------------------------------

out_dir = 'out/exp_076'
eval_interval = 250
eval_iters = 200
log_interval = 10

always_save_checkpoint = False
wandb_log = False
wandb_project = 'shakespeare'
wandb_run_name = 'mini-gpt'

dataset = 'shakespeare'
gradient_accumulation_steps = 1
batch_size = 16
block_size = 128

n_layer = 4
n_head = 4
n_embd = 256
dropout = 0.1

learning_rate = 0.001
max_iters = 1000
lr_decay_iters = 1000
min_lr = 0.0001
beta2 = 0.99
warmup_iters = 100

# device = 'cpu'
# compile = False
