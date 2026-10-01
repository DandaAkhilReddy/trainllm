"""Step 1c: build posts.jsonl from manually taken screenshots (shared with the page admin's permission).

Put one folder per post, slides in swipe order:

    data/screenshots/
        post_001/  01.png 02.png 03.png ...
        post_002/  01.jpg 02.jpg ...

or keep everything flat with a post prefix: post001_01.png, post001_02.png, post002_01.png ...

    python -m confession_llm.ingest_screenshots --dir data/screenshots --out data/raw
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic"}
_FLAT_RE = re.compile(r"^(?P<post>.+?)[_\- ](?P<slide>\d+)$")


def natural_key(path: Path) -> list:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]


def group_screenshots(root: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in root.rglob("*"):
        if path.suffix.lower() not in IMAGE_EXTS or not path.is_file():
            continue
        if path.parent != root:
            groups[path.parent.relative_to(root).as_posix()].append(path)
        else:
            m = _FLAT_RE.match(path.stem)
            groups[m.group("post") if m else path.stem].append(path)
    return {post: sorted(files, key=natural_key) for post, files in groups.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/screenshots")
    ap.add_argument("--out", default="data/raw")
    args = ap.parse_args()

    groups = group_screenshots(Path(args.dir))
    if not groups:
        raise SystemExit(f"No images found under {args.dir}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "posts.jsonl", "w", encoding="utf-8") as f:
        for post_id in sorted(groups, key=lambda p: natural_key(Path(p))):
            rec = {"id": post_id, "timestamp": None, "permalink": None, "caption": "",
                   "images": [str(p) for p in groups[post_id]]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    n_imgs = sum(len(v) for v in groups.values())
    print(f"{len(groups)} posts, {n_imgs} screenshots -> {out / 'posts.jsonl'}")


if __name__ == "__main__":
    main()
