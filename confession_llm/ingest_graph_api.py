"""Step 1a: pull posts from YOUR OWN Instagram professional account via the official API.

The Instagram API only returns media for accounts the token's owner controls
(Business/Creator accounts). It cannot read other people's pages.

Usage:
    export IG_ACCESS_TOKEN=...   # long-lived token for the account
    python -m confession_llm.ingest_graph_api --out data/raw

Instagram Login flow (default):  --base-url https://graph.instagram.com --user-id me
Facebook Login flow:             --base-url https://graph.facebook.com/v23.0 --user-id <IG_USER_ID>
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import requests

FIELDS = "id,caption,media_type,media_url,permalink,timestamp,children{id,media_type,media_url}"


def _get(url: str, params: dict | None = None, retries: int = 4) -> dict:
    for attempt in range(retries):
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 429 or resp.status_code >= 500:
            time.sleep(2 ** (attempt + 1))
            continue
        resp.raise_for_status()
        return resp.json()
    resp.raise_for_status()
    return {}


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        return
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    dest.write_bytes(resp.content)


def fetch_media(base_url: str, user_id: str, token: str, limit: int | None):
    url = f"{base_url}/{user_id}/media"
    params = {"fields": FIELDS, "access_token": token, "limit": 50}
    seen = 0
    while url:
        page = _get(url, params)
        for item in page.get("data", []):
            yield item
            seen += 1
            if limit and seen >= limit:
                return
        url = page.get("paging", {}).get("next")
        params = None  # `next` already carries the query string


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--base-url", default="https://graph.instagram.com")
    ap.add_argument("--user-id", default="me")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    token = os.environ.get("IG_ACCESS_TOKEN")
    if not token:
        raise SystemExit("Set IG_ACCESS_TOKEN (long-lived token for your own professional account).")

    out = Path(args.out)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(out / "posts.jsonl", "w", encoding="utf-8") as f:
        for item in fetch_media(args.base_url, args.user_id, token, args.limit):
            slides = item.get("children", {}).get("data") or [item]
            images = []
            for idx, slide in enumerate(slides):
                if slide.get("media_type") != "IMAGE" or not slide.get("media_url"):
                    continue  # skip reels/videos; stories here are text-on-image slides
                dest = img_dir / f"{item['id']}_{idx:02d}.jpg"
                _download(slide["media_url"], dest)
                images.append(str(dest))
            record = {
                "id": item["id"],
                "timestamp": item.get("timestamp"),
                "permalink": item.get("permalink"),
                "caption": item.get("caption") or "",
                "images": images,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    print(f"Wrote {count} posts to {out / 'posts.jsonl'}")


if __name__ == "__main__":
    main()
