import os, glob, subprocess, re

# Point to all config folders
CONFIG_FOLDERS = ["1_configs", "2_configs", "3_configs", "4_configs"]


PERSIST_OUT_ROOT = os.environ.get("PERSIST_OUT_ROOT", "").strip() 

def read_out_dir_from_config(cfg_path):
    # read the line "out_dir = '...'"
    with open(cfg_path, "r") as f:
        for line in f:
            m = re.match(r"out_dir\s*=\s*['\"](.+?)['\"]", line.strip())
            if m:
                return m.group(1)
    raise ValueError(f"out_dir not found in {cfg_path}")

def run_config(cfg_path):
    out_dir = read_out_dir_from_config(cfg_path)

    # redirect to persistent root
    if PERSIST_OUT_ROOT:
        # rename to /persist_root/exp_xxx keeping the exp_xxx part
        exp_name = os.path.basename(out_dir.rstrip("/"))
        out_dir = os.path.join(PERSIST_OUT_ROOT, exp_name)

    os.makedirs(out_dir, exist_ok=True)
    ckpt = os.path.join(out_dir, "ckpt.pt")
    ckpt_best = os.path.join(out_dir, "ckpt_best.pt")

    # Skip if a best checkpoint already exists
    if os.path.exists(ckpt_best):
        print(f"SKIP (best exists): {out_dir}")
        return

    # If any checkpoint exists, resume. Otherwise start fresh.
    init_from = "resume" if os.path.exists(ckpt) else "scratch"

    print(f" RUN: {cfg_path}  →  out_dir={out_dir}  ({init_from})")
    # Pass overrides via CLI to ensure correct out_dir and proper resume behavior
    cmd = [
        "python", "train.py", cfg_path,
        f"--out_dir={out_dir}",
        f"--init_from={init_from}",
    ]
    subprocess.run(cmd, check=False)

if __name__ == "__main__":
    cfgs = []
    for folder in CONFIG_FOLDERS:
        cfgs.extend(sorted(glob.glob(os.path.join(folder, "exp_*.py"))))

    for cfg in cfgs:
        run_config(cfg)
