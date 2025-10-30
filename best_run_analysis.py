# finding best run config
import os, glob, re, json, argparse, math
import pandas as pd
import torch
import tiktoken

def exp_idx_from_dir(path: str):
    m = re.search(r"exp_(\d+)$", os.path.basename(path.rstrip("/")))
    return int(m.group(1)) if m else None

def load_loss_csv(exp_dir: str):
    csv = os.path.join(exp_dir, "loss_log.csv")
    if not os.path.exists(csv):
        return None
    try:
        df = pd.read_csv(csv)
        if df.empty or "val_loss" not in df.columns:
            return None
        return df
    except Exception:
        return None

def pick_top_runs(root: str, top_k: int = 1):
    rows = []
    for d in sorted(glob.glob(os.path.join(root, "exp_*"))):
        df = load_loss_csv(d)
        if df is None or df["val_loss"].isna().all():
            continue
        min_val = float(df["val_loss"].min())
        idxmin = int(df["val_loss"].idxmin())
        iter_at_min = int(df.loc[idxmin, "iter"])
        elapsed_at_min = float(df.loc[idxmin, "elapsed_s"]) if "elapsed_s" in df.columns else math.inf
        rows.append({
            "exp_dir": d,
            "exp_idx": exp_idx_from_dir(d),
            "min_val_loss": min_val,
            "iter_at_min": iter_at_min,
            "elapsed_s_at_min": elapsed_at_min
        })
    if not rows:
        raise SystemExit(f"No runs with valid loss_log.csv found under {root}")
    df = pd.DataFrame(rows).sort_values(
        ["min_val_loss", "elapsed_s_at_min"], ascending=[True, True]
    )
    return df.head(top_k).reset_index(drop=True), df

def load_ckpt(exp_dir: str):
    """Prefer ckpt_best.pt, fall back to ckpt.pt."""
    best = os.path.join(exp_dir, "ckpt_best.pt")
    latest = os.path.join(exp_dir, "ckpt.pt")
    ckpt_path = best if os.path.exists(best) else (latest if os.path.exists(latest) else None)
    if ckpt_path is None:
        return None, None
    ckpt = torch.load(ckpt_path, map_location="cpu")
    return ckpt, ckpt_path

def extract_sweep_knobs(cfg_dict: dict):
    fields = ["block_size", "n_layer", "n_head", "n_embd", "batch_size", "max_iters", "dropout"]
    return {k: cfg_dict.get(k) for k in fields}

# ---- generation (standalone, no need to call sample.py) ---------------------

def build_model_from_ckpt(ckpt, device):
    from model import GPTConfig, GPT
    # Prefer model_args, fall back to config
    conf_src = ckpt.get("model_args") or ckpt.get("config") or {}
    gptconf = GPTConfig(**conf_src)
    model = GPT(gptconf)
    state_dict = ckpt["model"]
    # handle unwanted prefix if present
    unwanted_prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)
    return model

@torch.no_grad()
def generate_text(ckpt, device, start_text, num_samples=2, max_new_tokens=200, temperature=0.8, top_k=200):
    # encoder/decoder: default to GPT-2 BPE (same as sample.py fallback)
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

    model = build_model_from_ckpt(ckpt, device)
    x = torch.tensor([encode(start_text)], dtype=torch.long, device=device)

    # Use the model's own generate for simplicity
    y = model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    text = decode(y[0].tolist())
    return text

# ---- main -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str,
        default=os.environ.get("PERSIST_OUT_ROOT", "").strip() or "out",
        help="Folder containing exp_*/")
    ap.add_argument("--top_k", type=int, default=1, help="Show/sample top-K runs")
    ap.add_argument("--start_text", type=str, default="ROMEO:", help="Prompt to start generation")
    ap.add_argument("--num_samples", type=int, default=2)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--topk_logits", type=int, default=200, help="top-k sampling cutoff")
    args = ap.parse_args()

    print(f"Scanning: {args.root}")
    top_df, all_df = pick_top_runs(args.root, top_k=args.top_k)

    # Save overall ranking for reference
    all_csv = os.path.join(args.root, "all_runs_ranked.csv")
    all_df.to_csv(all_csv, index=False)
    print(f"Saved full ranking to: {all_csv}\n")

    print("=== Top runs ===")
    print(top_df.to_string(index=False))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for i, row in top_df.iterrows():
        exp_dir = row["exp_dir"]
        ckpt, ckpt_path = load_ckpt(exp_dir)
        if ckpt is None:
            print(f"\n{exp_dir}: no checkpoint found, skipping generation.")
            continue

        # Extract + display the 7 knobs from the saved config
        cfg = ckpt.get("config", {}) or ckpt.get("model_args", {}) or {}
        knobs = extract_sweep_knobs(cfg)

        print(f"\n--- Best #{i+1} ---")
        print(f"exp_dir        : {exp_dir}")
        print(f"ckpt           : {os.path.basename(ckpt_path)}")
        print(f"min val loss   : {row['min_val_loss']:.4f}  at iter {row['iter_at_min']}")
        if math.isfinite(row['elapsed_s_at_min']):
            print(f"time to min    : {row['elapsed_s_at_min']:.2f}s")
        print("config (sweep knobs):")
        for k, v in knobs.items():
            print(f"  {k}: {v}")

        # Generate samples
        samples_dir = os.path.join(exp_dir, "samples")
        os.makedirs(samples_dir, exist_ok=True)
        all_texts = []
        for s in range(args.num_samples):
            txt = generate_text(
                ckpt, device,
                start_text=args.start_text,
                num_samples=1,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.topk_logits,
            )
            all_texts.append(txt)
            # Save each to a file
            out_txt = os.path.join(
                samples_dir,
                f"sample_{s+1}_T{args.temperature}_topk{args.topk_logits}_len{args.max_new_tokens}.txt",
            )
            with open(out_txt, "w", encoding="utf-8") as f:
                f.write(txt)
        print(f"Saved {len(all_texts)} sample(s) to: {samples_dir}")

if __name__ == "__main__":
    main()
