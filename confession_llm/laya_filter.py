"""Step 6 (optional): score generated stories with convaiinnovations/laya before posting.

Laya is a classifier (it cannot write text), so it is used here as a calibrated
quality/safety gate on what the generator produced.

    pip install laya
    python -m confession_llm.laya_filter --in outputs/generated.jsonl --out outputs/approved.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .textutils import THEME_KEYWORDS

QUESTIONS = {
    "is_confession": {
        "type": "noul",
        "instructions": "Is this a coherent first-person personal confession story?",
    },
    "identifies_real_person": {
        "type": "noul",
        "instructions": "Does the text name or identify a specific real person, company, or contact detail?",
    },
    "harmful": {
        "type": "noul",
        "instructions": "Does the text contain hate, harassment, sexual content involving minors, or encourage self-harm or violence?",
    },
    "engagement": {
        "type": "score",
        "instructions": "How engaging and relatable would this be to Indian students and professionals abroad?",
        "criteria": ["boring", "okay", "very engaging"],
    },
    "theme": {
        "type": "choice",
        "instructions": "What is the main theme of this story?",
        "criteria": {t: t for t in THEME_KEYWORDS} | {"other": "anything else"},
    },
}


def decide(answers: dict, min_confession: float, max_risk: float, min_engagement: float) -> tuple[bool, str]:
    if answers["harmful"]["noul"] > max_risk:
        return False, "harmful"
    if answers["identifies_real_person"]["noul"] > max_risk:
        return False, "identifies a real person"
    if answers["is_confession"]["noul"] < min_confession:
        return False, "not a coherent confession"
    if answers["engagement"]["score"] < min_engagement:
        return False, "low engagement"
    return True, "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="outputs/generated.jsonl")
    ap.add_argument("--out", default="outputs/approved.jsonl")
    ap.add_argument("--min-confession", type=float, default=0.6)
    ap.add_argument("--max-risk", type=float, default=0.3)
    ap.add_argument("--min-engagement", type=float, default=1.0, help="0-2 scale")
    args = ap.parse_args()

    from laya import Router

    router = Router(preload=True)
    rows = [json.loads(l) for l in Path(args.inp).read_text(encoding="utf-8").splitlines() if l.strip()]
    kept = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for row in rows:
            # multilingual checkpoint handles Tenglish/Hinglish and long stories
            res = router.predict(row["story"], QUESTIONS, model="multilingual", max_len=4096)
            ok, reason = decide(res["answers"], args.min_confession, args.max_risk, args.min_engagement)
            row["laya"] = {k: v for k, v in res["answers"].items()}
            row["laya_verdict"] = reason
            print(f"{'KEEP' if ok else 'DROP'} ({reason}): {row['story'][:70]!r}")
            if ok:
                kept += 1
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    print(f"approved {kept}/{len(rows)} -> {args.out}")


if __name__ == "__main__":
    main()
