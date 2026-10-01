"""Step 1b: read posts from Instagram's "Download your information" export (JSON format).

Instagram app -> Settings -> Accounts Center -> Your information and permissions ->
Download your information -> pick the account -> Format: JSON, Media quality: High.
Unzip it, then:

    python -m confession_llm.ingest_export --export-dir ~/Downloads/instagram-export --out data/raw
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .textutils import fix_mojibake

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic"}


def iter_posts(export_dir: Path):
    files = sorted(export_dir.rglob("posts_*.json"))
    if not files:
        raise SystemExit(f"No posts_*.json found under {export_dir}. Did you choose JSON format?")
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        for i, post in enumerate(data if isinstance(data, list) else data.get("posts", [])):
            media = post.get("media", [])
            caption = post.get("title") or (media[0].get("title") if media else "") or ""
            ts = post.get("creation_timestamp") or (media[0].get("creation_timestamp") if media else None)
            images = []
            for m in media:
                uri = m.get("uri", "")
                full = export_dir / uri
                if full.suffix.lower() in IMAGE_EXTS and full.exists():
                    images.append(str(full))
            yield {
                "id": f"{path.stem}_{i}",
                "timestamp": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat() if ts else None,
                "permalink": None,
                "caption": fix_mojibake(caption),
                "images": images,
            }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-dir", required=True)
    ap.add_argument("--out", default="data/raw")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(out / "posts.jsonl", "w", encoding="utf-8") as f:
        for rec in iter_posts(Path(args.export_dir).expanduser()):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    print(f"Wrote {count} posts to {out / 'posts.jsonl'}")


if __name__ == "__main__":
    main()
