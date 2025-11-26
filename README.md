CLALM Continual VQA (without data)
===================================

Project
-------
Continual multilingual VQA on ALM-Bench using Qwen3-VL-2B with LoRA adapters. Supports Naive and EWC, three orders (clustered, interleaved, random:42), and lightweight metrics (TF/mcq accuracy, short EM, long F1).

What’s inside
-------------
- Codebase (`continual_vqa`) and trained runs (`runs_three_orders_*`).
- No dataset cache (add your own `cache/local_dataset.jsonl` + `cache/images`).
- `ALM_demonstration/` — Gradio demo (no cache inside; see “Demo”).

Install requirements
---------------------
pip install -r requirements.txt

Prefetch dataset  (Takes time to prefetch data and store locally)
-------------------------------------------------------------------------------
python -m continual_vqa.cli.prefetch \
  --languages "English" "German" "Spanish" "Portuguese" "Russian" "Ukrainian" "Hindi" "Bengali" "Amharic" "Japanese" "Chinese (Simplified)" "Emirati Arabic" \
  --image-cache cache/images \
  --hf-cache cache/hf \
  --seed 42

Train commands
--------------
Run from repo root (`/l/users/rishabh.lalla/CLALM_without_data`) after adding `cache/`.

Naive:
```
python -m continual_vqa.cli.run_three_orders \
  --mode train --method naive \
  --orders clustered interleaved random:42 \
  --output-root runs_three_orders_naive \
  --languages "English" "German" "Spanish" "Portuguese" "Russian" "Ukrainian" "Hindi" "Bengali" "Amharic" "Japanese" "Chinese (Simplified)" "Emirati Arabic" \
  --train-per-type 12 --test-per-type 6 \
  --epochs 10 --bits 16 \
  --param-budget 2000000 --param-budget-mode adjust_r \
  --per-device-train-batch-size 2 --grad-accum 4 \
  --per-device-eval-batch-size 4 \
  --verbosity verbose --log-interval 10 \
  --local-dataset cache/local_dataset.jsonl \
  --image-cache cache/images \
  --hf-cache cache/hf \
  --prefetch-cache-only \
  --no-bertscore
```

EWC:
```
python -m continual_vqa.cli.run_three_orders \
  --mode train --method ewc \
  --orders clustered interleaved random:42 \
  --ewc-lambda 2000 --ewc-gamma 0.7 --fisher-batches 64 \
  --output-root runs_three_orders_ewc \
  --languages "English" "German" "Spanish" "Portuguese" "Russian" "Ukrainian" "Hindi" "Bengali" "Amharic" "Japanese" "Chinese (Simplified)" "Emirati Arabic" \
  --train-per-type 12 --test-per-type 6 \
  --epochs 10 --bits 16 \
  --param-budget 2000000 --param-budget-mode adjust_r \
  --per-device-train-batch-size 2 --grad-accum 4 \
  --per-device-eval-batch-size 4 \
  --verbosity verbose --log-interval 10 \
  --local-dataset cache/local_dataset.jsonl \
  --image-cache cache/images \
  --hf-cache cache/hf \
  --prefetch-cache-only \
  --no-bertscore
```

Test commands (trajectory)
--------------------------
Naive:
```
python -m continual_vqa.cli.run_three_orders \
  --mode test --eval-mode trajectory \
  --orders clustered interleaved random:42 \
  --output-root runs_three_orders_naive \
  --languages "English" "German" "Spanish" "Portuguese" "Russian" "Ukrainian" "Hindi" "Bengali" "Amharic" "Japanese" "Chinese (Simplified)" "Emirati Arabic" \
  --train-per-type 12 --test-per-type 6 \
  --per-device-eval-batch-size 4 \
  --verbosity verbose --log-interval 10 \
  --local-dataset cache/local_dataset.jsonl \
  --image-cache cache/images \
  --hf-cache cache/hf \
  --prefetch-cache-only \
  --no-bertscore
```

EWC:
```
python -m continual_vqa.cli.run_three_orders \
  --mode test --eval-mode trajectory \
  --orders clustered interleaved random:42 \
  --output-root runs_three_orders_ewc \
  --languages "English" "German" "Spanish" "Portuguese" "Russian" "Ukrainian" "Hindi" "Bengali" "Amharic" "Japanese" "Chinese (Simplified)" "Emirati Arabic" \
  --train-per-type 12 --test-per-type 6 \
  --per-device-eval-batch-size 4 \
  --verbosity verbose --log-interval 10 \
  --local-dataset cache/local_dataset.jsonl \
  --image-cache cache/images \
  --hf-cache cache/hf \
  --prefetch-cache-only \
  --no-bertscore
```

Notes
-----
- Metrics simplified: short → EM, long → F1 (no ROUGE/BERTScore). TF/MCQ use option-aware matching.
- Reduce `--test-per-type` to speed eval if desired.

Demo (Colab)
------------
Folder: `ALM_demonstration` (no cache inside). Upload to Drive with:
- `cache/` (local_dataset.jsonl + images)
- `adapters/` (6 finals: clustered/interleaved/random × naive/EWC)

Colab steps:
```
# 1) GPU check
!nvidia-smi

# 2) Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# 3) Copy demo folder locally (faster I/O)
!rm -rf /content/ALM_demonstration
!cp -r /content/drive/MyDrive/ALM_demonstration /content/ALM_demonstration

# 4) Install deps
%cd /content/ALM_demonstration
!pip install -q -r requirements.txt

# 5) Open demo_app.py and set paths (edit these if your Drive layout differs):
BASE_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
CACHE_ROOT = "/content/ALM_demonstration/cache"
LOCAL_DATASET = f"{CACHE_ROOT}/local_dataset.jsonl"
IMAGE_CACHE = f"{CACHE_ROOT}/images"
ADAPTERS = {
  "Clustered-Naive": "/content/ALM_demonstration/adapters/clustered_naive",
  "Clustered-EWC": "/content/ALM_demonstration/adapters/clustered_ewc",
  "Interleaved-Naive": "/content/ALM_demonstration/adapters/interleaved_naive",
  "Interleaved-EWC": "/content/ALM_demonstration/adapters/interleaved_ewc",
  "Random-Naive": "/content/ALM_demonstration/adapters/random_naive",
  "Random-EWC": "/content/ALM_demonstration/adapters/random_ewc",
}

# 6) Run the Gradio app (will give you a share URL)
!python demo_app.py

```
Pick question type + adapter, run 12 samples; predictions stream row-by-row; metrics show acc/EM/F1 per type.
