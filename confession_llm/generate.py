"""Step 5: generate new confessions with the fine-tuned adapter.

    python -m confession_llm.generate --n 10 --theme "visa & immigration"

Every output is checked against the training stories; anything that copies long
verbatim runs from a real confession is discarded instead of saved.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from .build_dataset import SYSTEM_PROMPT, build_user_prompt
from .textutils import THEME_KEYWORDS, clean_story, ngram_overlap, word_ngrams


def load_reference_ngrams(path: Path, n: int) -> set:
    if not path.exists():
        print(f"warning: {path} not found; memorization filter disabled")
        return set()
    grams = set()
    for story in path.read_text(encoding="utf-8").split("<|story|>"):
        grams |= word_ngrams(story, n)
    return grams


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--adapter", default="outputs/qwen3-4b-nri-confessions-lora")
    ap.add_argument("--stories", default="data/sft/stories.txt")
    ap.add_argument("--out", default="outputs/generated.jsonl")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--theme", default=None, help=f"one of: {', '.join(THEME_KEYWORDS)} (or free text)")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=800)
    ap.add_argument("--max-overlap", type=float, default=0.15, help="max share of 8-grams copied from training data")
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.adapter if Path(args.adapter).exists() else args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", device_map="auto")
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    ref = load_reference_ngrams(Path(args.stories), 8)
    themes = list(THEME_KEYWORDS)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    kept = rejected = 0
    with open(out, "a", encoding="utf-8") as f:
        while kept < args.n and rejected < args.n * 3:
            theme = args.theme or random.choice(themes)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt([theme])},
            ]
            inputs = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
            ).to(model.device)
            with torch.inference_mode():
                ids = model.generate(
                    **inputs, do_sample=True, temperature=args.temperature, top_p=args.top_p,
                    repetition_penalty=1.1, max_new_tokens=args.max_new_tokens,
                )
            text = tokenizer.decode(ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
            story = clean_story(text)
            overlap = ngram_overlap(story, ref, 8)
            if overlap > args.max_overlap:
                rejected += 1
                print(f"rejected (copied {overlap:.0%} from training data)")
                continue
            kept += 1
            rec = {
                "theme": theme,
                "story": story,
                "train_overlap": round(overlap, 4),
                "ai_generated": True,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"\n=== [{kept}/{args.n}] {theme} (overlap {overlap:.1%}) ===\n{story}\n")
    print(f"kept {kept}, rejected {rejected} -> {out}")


if __name__ == "__main__":
    main()
