# nanoGPT Hyperparameter Study on Tiny Shakespeare

A reproducible nanoGPT sweep over **128 configurations** on the Tiny Shakespeare dataset — DDP-ready training, CSV loss/time logging, Drive-friendly batching for Colab, automatic best-run analysis, and sampling utilities.

---

## Contents

* [Overview](#overview)
* [Repo Structure](#repo-structure)
* [Setup](#setup)
* [Data Preparation (Tiny Shakespeare)](#data-preparation-tiny-shakespeare-gpt-2-bpe)
* [Generate All 128 Configs](#generate-all-128-configs)
* [Run Experiments](#run-experiments)
* [Logging & Artifacts](#logging--artifacts)
* [Pick the Best Run + Plots](#pick-the-best-run--plots)
* [Generate Samples](#generate-samples)
* [Reproducibility Notes](#reproducibility-notes)
* [License & Acknowledgments](#license--acknowledgments)

---

## Overview

* **Goal:** Compare nanoGPT variants by systematically sweeping architecture and training hyperparameters on a small shakespeare dataset.
* **Grid:** `block_size × n_layer × n_head × n_embd × batch_size × max_iters × dropout = 2⁷ = 128` configs.
* **Metrics:** Train/val loss curves, wall-clock time, throughput, qualitative samples for coherence/style.

---

## Repo Structure

```
.
├─ model.py                   # nanoGPT (decoder-only Transformer)
├─ train.py                   # modified: CSV logging, timing, optional in-train sampling
├─ sample.py                  # text generation from a checkpoint (best config checkpoint)
├─ prepare.py                 # builds  {train, val}.bin
├─ generate_configs_128.py    # creates 128 config files across 4 folders
├─ run_launcher.py            # runs/auto-resumes configs, optional Drive persistence
├─ best_run_analysis.py       # finds best run, plots loss/time curves
├─ 1_configs/exp_000.py … exp_031.py
├─ 2_configs/exp_032.py … exp_063.py
├─ 3_configs/exp_064.py … exp_095.py
├─ 4_configs/exp_096.py … exp_127.py
├─ out/exp_000 - exp_127  # consists of all configurations outputs(loss log, loss curves, analysis curves,exp_config.txt)

```

---

## Setup

```bash
# Python 3.10+ recommended
pip install -r requirements.txt
# Install PyTorch for your CUDA/OS from https://pytorch.org
```

**requirements.txt**

```
tiktoken
pandas
matplotlib
numpy
```

---

## Data Preparation (Tiny Shakespeare, GPT-2 BPE)

```bash
python shakespeare/prepare.py
# writes: shakespeare/train.bin and shakespeare/val.bin
```

In configs, set:

```
dataset = 'shakespeare'
```

---

## Generate All 128 Configs

```bash
python generate_configs_128.py
```

This produces four groups (32 each):

* **Group 1:** block_size=64,  n_layer=4
* **Group 2:** block_size=64,  n_layer=6
* **Group 3:** block_size=128, n_layer=4
* **Group 4:** block_size=128, n_layer=6

---

## Run Experiments

### Batch launcher (auto-resume, skip finished)

**Colab + Drive (persist across timeouts):**

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
export PERSIST_OUT_ROOT=/content/drive/MyDrive/nanogpt_runs
python run_launcher.py
```

Each experiment writes to `$PERSIST_OUT_ROOT/exp_xxx` (or `out/exp_xxx` if `PERSIST_OUT_ROOT` is unset).


## Logging & Artifacts

Each run folder (`out/exp_xxx/` or Drive) contains:

* **`loss_log.csv`** with columns:

  * `iter`, `train_loss`, `val_loss`, `lr`
  * `elapsed_s` (wall-clock since start), `iter_ms` (per-iter time)
  * `throughput_tok_s` (tokens/sec)
* **Checkpoints**

  * `ckpt.pt` at each eval (latest)
  * Optional `ckpt_best.pt` if you enabled best-checkpoint saving in `train.py`

---

## Pick the Best Run + Plots

```bash
# If drive was used:
export PERSIST_OUT_ROOT=/content/drive/MyDrive/nanogpt_runs
python best_run_analysis.py
```

Outputs:

* `out/all_runs_summary.csv` (or Drive root) ranking runs by **min val loss**
* In the **best** run’s folder: `analysis_plots/`

  * `loss_vs_iter.png`, `loss_vs_time.png`, `lr_vs_iter.png`, `throughput_vs_iter.png`

---

## Generate Samples

From any finished run:

```bash
python sample.py \
  --init_from=resume \
  --out_dir=out/exp_000 \
  --start "ROMEO:" \
  --num_samples=3 \
  --max_new_tokens=200 \
  --temperature=0.8 \
  --top_k=200
```


---

## Reproducibility Notes

* Seeds set per DDP rank: `random`, `numpy`, `torch`, `torch.cuda` with `1337 + rank`.
* TF32 is enabled for speed, results are repeatable in spirit on the same setup.


## License & Acknowledgments

* Built on the nanoGPT codebase (https://github.com/karpathy/nanoGPT.git) concept and Tiny Shakespeare dataset.
