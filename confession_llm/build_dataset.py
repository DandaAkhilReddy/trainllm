"""Step 3: turn OCR'd posts into a cleaned, de-duplicated SFT dataset.

    python -m confession_llm.build_dataset --in data/ocr/posts_ocr.jsonl --out data/sft

Writes prompt/completion chat JSONL (loss is computed on the story only) plus
`stories.txt`, which generate.py uses to reject outputs that copy real confessions.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .textutils import clean_story, content_hash, detect_themes, jaccard, word_ngrams

SYSTEM_PROMPT = (
    "You are a creative writer for an entertainment page that publishes anonymous, fictional "
    "first-person confessions from Indian students and professionals living abroad (NRIs). "
    "Write vivid, original stories in the page's casual Tenglish/Hinglish-friendly English voice. "
    "Never use real names, handles, employers, phone numbers or other identifying details."
)


def build_user_prompt(themes: list[str]) -> str:
    if themes:
        return f"Write a new NRI confession about {' and '.join(themes)}."
    return "Write a new NRI confession."


def post_to_story(post: dict) -> str:
    ocr_parts = [t for t in post.get("ocr", []) if t and t.strip().upper() != "NONE"]
    story = clean_story("\n\n".join(ocr_parts))
    caption = clean_story(post.get("caption", ""))
    # Some posts put the confession in the caption instead of on the slides
    return story if len(story) >= len(caption) else caption


def build(posts: list[dict], min_chars: int, near_dup: float) -> list[dict]:
    rows, hashes, shingles = [], set(), []
    for post in posts:
        story = post_to_story(post)
        if len(story) < min_chars:
            continue
        h = content_hash(story)
        if h in hashes:
            continue
        sh = word_ngrams(story, 5)
        if any(jaccard(sh, other) >= near_dup for other in shingles):
            continue
        hashes.add(h)
        shingles.append(sh)
        themes = detect_themes(story)
        rows.append(
            {
                "id": post["id"],
                "themes": themes,
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(themes)},
                ],
                "completion": [{"role": "assistant", "content": story}],
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/ocr/posts_ocr.jsonl")
    ap.add_argument("--out", default="data/sft")
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--near-dup", type=float, default=0.6)
    ap.add_argument("--val-frac", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    posts = [json.loads(l) for l in Path(args.inp).read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = build(posts, args.min_chars, args.near_dup)
    random.Random(args.seed).shuffle(rows)
    n_val = max(1, int(len(rows) * args.val_frac)) if len(rows) > 10 else 0
    splits = {"val": rows[:n_val], "train": rows[n_val:]}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, split in splits.items():
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in split:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "stories.txt").write_text(
        "\n\n<|story|>\n\n".join(r["completion"][0]["content"] for r in rows), encoding="utf-8"
    )
    print(f"{len(posts)} posts -> {len(rows)} stories ({len(splits['train'])} train / {n_val} val)")


if __name__ == "__main__":
    main()
