# trainllm: NRI Confessions story generator

Fine-tune an open Hugging Face LLM on a confession page's posts, then generate new
fictional confessions in the same voice.

```
Instagram (official API or data export)
   -> images + captions            confession_llm/ingest_*.py
   -> OCR text from slides         confession_llm/ocr.py          (Qwen3-VL-8B)
   -> cleaned, PII-scrubbed SFT    confession_llm/build_dataset.py
   -> LoRA fine-tune               confession_llm/train_lora.py   (Qwen3-4B-Instruct-2507)
   -> new stories + copy filter    confession_llm/generate.py
   -> quality/safety gate          confession_llm/laya_filter.py  (convaiinnovations/laya, optional)
```

## Models and why

| Role | Model | Why |
|---|---|---|
| Story writer (fine-tuned) | `Qwen/Qwen3-4B-Instruct-2507` | Strong multilingual writer that handles Telugu/Hindi-mixed English, Apache-2.0, and fits a free Colab T4 with QLoRA. Move up to `Qwen/Qwen3-30B-A3B-Instruct-2507` on an A100 if you want better stories. |
| OCR of slides | `Qwen/Qwen3-VL-8B-Instruct` (or `-2B` on a T4) | The confessions are text rendered on images. A VLM reads mixed scripts better than Tesseract. |
| Gate (optional) | `convaiinnovations/laya` | A **classifier, not a generator**. It can't write stories, but it gives calibrated yes/no answers and scores ("names a real person?", "harmful?", "engaging?"), so it filters what Qwen writes. |

## Step 0: data rights (read first)

* **The Instagram API only reads accounts you control.** The Instagram Platform API
  returns media only for the Business/Creator account the token belongs to. It can't
  read someone else's page, and scraping other accounts breaks Instagram's Terms.
  So you need admin access to `nri_confessions` (owner, or an admin on the linked Page),
  or written permission from the owner to use an export they give you.
* Confessions are real people's private stories. The pipeline removes phone numbers,
  emails, @handles and links. `generate.py` throws away any output that copies long
  verbatim runs from a real confession. Keep `data/` private (it's git-ignored) and
  push models to the Hub as **private**.
* Label generated posts as AI-generated/fictional. Don't present them as real
  submissions.

## Step 1: get the posts (pick one)

**Option A: Instagram data export (easiest, no developer app needed)**
1. Instagram app → Settings → Accounts Center → *Your information and permissions* →
   *Download your information* → select the page account → *Some of your information* → *Content* →
   Format **JSON**, Media quality **High** → Create files.
2. Download and unzip the archive, then run:
   ```bash
   python -m confession_llm.ingest_export --export-dir ~/Downloads/instagram-nri_confessions --out data/raw
   ```

**Option B: official Instagram API (repeatable, good for pulling new posts later)**
1. Make sure the account is **Professional** (Business or Creator).
2. Go to developers.facebook.com → *Create app* → use case *Manage messaging & content on Instagram*
   → add the **Instagram API with Instagram Login** product.
3. Under *API setup*, add the IG account as a tester/role and **Generate token** with
   `instagram_business_basic`. Exchange it for a long-lived token (60 days).
4. Run:
   ```bash
   export IG_ACCESS_TOKEN=...
   python -m confession_llm.ingest_graph_api --out data/raw
   ```
   (For the Facebook Login flow, pass `--base-url https://graph.facebook.com/v23.0 --user-id <IG_USER_ID>`.)

Both options write `data/raw/posts.jsonl`: one line per post with its caption and downloaded slide images.

## Step 2: set up a GPU machine

Free: Google Colab (T4, 16 GB) or Kaggle (2×T4). Paid: Colab Pro L4/A100, Lambda, RunPod, or an Azure
`NC A100 v4` VM.
```bash
git clone https://github.com/DandaAkhilReddy/trainllm && cd trainllm
pip install -r requirements.txt
huggingface-cli login        # token from huggingface.co/settings/tokens (write access to push)
```
Upload `data/raw/` to the machine (or run Step 1 there).

## Step 3: OCR the slides
```bash
python -m confession_llm.ocr --in data/raw/posts.jsonl --out data/ocr/posts_ocr.jsonl
# on a T4: add  --model Qwen/Qwen3-VL-2B-Instruct
```
This step can be resumed. Spot-check ~10 rows of `data/ocr/posts_ocr.jsonl` against the real posts.

## Step 4: build the training set
```bash
python -m confession_llm.build_dataset --in data/ocr/posts_ocr.jsonl --out data/sft
```
This strips cover slides (`#DesiLolliConfessions #2026 - #88`), hashtags and PII, removes
duplicates and near-duplicates, tags themes, and writes `train.jsonl`, `val.jsonl` and
`stories.txt`. ~400 posts should give ~300–380 usable stories, enough for style transfer
with LoRA.

## Step 5: fine-tune
```bash
# L4 / A10G / A100 / 4090
python -m confession_llm.train_lora --push-to-hub <hf-username>/nri-confessions-lora
# free T4
python -m confession_llm.train_lora --qlora --batch-size 1 --grad-accum 16
```
Watch eval loss. If it starts rising after epoch 2, use `--epochs 2`, because small datasets overfit
and start memorizing.

## Step 6: generate new stories
```bash
python -m confession_llm.generate --n 10 --theme "visa & immigration"
python -m confession_llm.generate --n 20            # random themes
```
Output goes to `outputs/generated.jsonl`. Any story with more than 15% of its 8-grams copied
from the training data is rejected automatically.

## Step 7 (optional): gate with Laya, then review by hand
```bash
pip install laya
python -m confession_llm.laya_filter --in outputs/generated.jsonl --out outputs/approved.jsonl
```
Read every approved story yourself before posting.

## Tests
```bash
pytest -q     # data pipeline tests run on CPU, no GPU or model downloads needed
```
