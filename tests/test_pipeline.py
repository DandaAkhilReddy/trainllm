import json

from confession_llm.build_dataset import build, post_to_story
from confession_llm.ingest_export import iter_posts
from confession_llm.textutils import clean_story, detect_themes, fix_mojibake, ngram_overlap, word_ngrams

STORY = (
    "I came to the US for my masters in 2022. My roommate and I shared everything, "
    "but when the H1B lottery results came out he got picked and I didn't. "
    "My amma kept calling every day asking about the visa and I lied to her for months. "
    "Text me at +1 (469) 555-0199 or mail me at someone@gmail.com, follow @real.person on insta."
)


def test_clean_story_drops_cover_header_and_pii():
    raw = "#DesiLolliConfessions\n#2026 - #88\n\n" + STORY + "\n#nriconfessions #desi\nforms.gle/abc123"
    out = clean_story(raw)
    assert "#2026" not in out and "DesiLolli" not in out and "#" not in out
    assert "555-0199" not in out and "someone@gmail.com" not in out and "@real.person" not in out
    assert "forms.gle" not in out
    assert out.startswith("I came to the US")


def test_detect_themes():
    themes = detect_themes(STORY)
    assert "visa & immigration" in themes


def test_post_to_story_prefers_ocr_and_skips_none():
    post = {"caption": "#nriconfessions follow @desi.lolli", "ocr": ["#DesiLolliConfessions\n#2026 - #88", "NONE", STORY]}
    assert post_to_story(post).startswith("I came to the US")


def test_build_dedupes_and_filters_short():
    posts = [
        {"id": "1", "caption": "", "ocr": [STORY]},
        {"id": "2", "caption": "", "ocr": [STORY + " "]},  # exact dup after normalization
        {"id": "3", "caption": "", "ocr": ["too short"]},
    ]
    rows = build(posts, min_chars=100, near_dup=0.6)
    assert [r["id"] for r in rows] == ["1"]
    assert rows[0]["completion"][0]["role"] == "assistant"
    assert rows[0]["prompt"][-1]["role"] == "user"


def test_ngram_overlap_flags_copies():
    ref = word_ngrams(STORY, 8)
    assert ngram_overlap(STORY, ref, 8) > 0.9
    assert ngram_overlap("A totally different story about a coffee shop in Dallas and a lost wallet near the station.", ref, 8) == 0


def test_fix_mojibake():
    assert fix_mojibake("cafÃ©") == "café"
    assert fix_mojibake("already fine") == "already fine"


def test_ingest_export(tmp_path):
    (tmp_path / "media/posts").mkdir(parents=True)
    (tmp_path / "media/posts/a.jpg").write_bytes(b"x")
    content = tmp_path / "your_instagram_activity/content"
    content.mkdir(parents=True)
    (content / "posts_1.json").write_text(json.dumps([
        {"title": "caption here", "creation_timestamp": 1700000000,
         "media": [{"uri": "media/posts/a.jpg"}, {"uri": "media/posts/missing.jpg"}]}
    ]))
    posts = list(iter_posts(tmp_path))
    assert len(posts) == 1
    assert posts[0]["caption"] == "caption here"
    assert posts[0]["images"] == [str(tmp_path / "media/posts/a.jpg")]


def test_laya_decide():
    from confession_llm.laya_filter import decide

    good = {"harmful": {"noul": 0.01}, "identifies_real_person": {"noul": 0.05},
            "is_confession": {"noul": 0.95}, "engagement": {"score": 1.6}}
    assert decide(good, 0.6, 0.3, 1.0) == (True, "ok")
    assert decide(good | {"identifies_real_person": {"noul": 0.8}}, 0.6, 0.3, 1.0)[0] is False
    assert decide(good | {"engagement": {"score": 0.4}}, 0.6, 0.3, 1.0) == (False, "low engagement")


def test_group_screenshots_folders_and_flat(tmp_path):
    from confession_llm.ingest_screenshots import group_screenshots

    for name in ["post_2/10.png", "post_2/2.png", "post_10/1.jpg", "post001_02.png", "post001_1.png", "notes.txt"]:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    groups = group_screenshots(tmp_path)
    assert set(groups) == {"post_2", "post_10", "post001"}
    assert [p.name for p in groups["post_2"]] == ["2.png", "10.png"]  # natural sort, not lexical
    assert [p.name for p in groups["post001"]] == ["post001_1.png", "post001_02.png"]
