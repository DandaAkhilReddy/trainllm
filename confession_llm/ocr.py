"""Step 2: transcribe the text written on each carousel slide with a vision-language model.

The confessions on the page are text rendered onto images, so captions alone are not
enough. Qwen3-VL reads English, Telugu/Hindi script and Tenglish/Hinglish well.

    python -m confession_llm.ocr --in data/raw/posts.jsonl --out data/ocr/posts_ocr.jsonl

Use --model Qwen/Qwen3-VL-2B-Instruct on smaller GPUs (e.g. a free Colab T4).
Resumable: posts already present in the output file are skipped.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

OCR_PROMPT = (
    "Transcribe ALL text visible in this image exactly as written, preserving line breaks "
    "and the original language/script. Do not summarize, translate or add commentary. "
    "If the image contains no text, reply with NONE."
)


def load_model(model_id: str):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def transcribe(model, processor, image_path: str, max_new_tokens: int = 1024) -> str:
    import torch
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": OCR_PROMPT}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new_tokens = out[0, inputs["input_ids"].shape[1] :]
    return processor.decode(new_tokens, skip_special_tokens=True).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/raw/posts.jsonl")
    ap.add_argument("--out", default="data/ocr/posts_ocr.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(l)["id"] for l in out.read_text(encoding="utf-8").splitlines() if l.strip()}

    posts = [json.loads(l) for l in Path(args.inp).read_text(encoding="utf-8").splitlines() if l.strip()]
    todo = [p for p in posts if p["id"] not in done]
    print(f"{len(posts)} posts, {len(done)} already done, {len(todo)} to OCR")
    if not todo:
        return

    model, processor = load_model(args.model)
    with open(out, "a", encoding="utf-8") as f:
        for n, post in enumerate(todo, 1):
            post["ocr"] = [transcribe(model, processor, p) for p in post["images"]]
            f.write(json.dumps(post, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{n}/{len(todo)}] {post['id']}: {len(post['images'])} slides")


if __name__ == "__main__":
    main()
