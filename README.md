# The Powerful Name — Shorts pipeline

Builds the daily devotional Short for the YouTube channel **The Powerful Name**. Each night a scheduled run builds the Short that publishes the next day (Eastern time) so Tommy can review it around 8 PM.

## Layout

| Path | What it is |
|---|---|
| `queue/queue.json` | One devotional page per publish date. The source of every verse. |
| `nightly.py` | Step 1: builds the ambient bed for the next publish date. |
| `ambient/` | The ambient-bed generator, mood tagger and uniqueness ledger. See [`ambient/README.md`](ambient/README.md). |
| `ambient/ledger.json` | Record of every bed that has shipped. **Never edit by hand.** |
| `render/render_short.py` | Step 2: renders the finished vertical Short from the queue entry and the bed. |
| `renders/<verse>/` | Output per Short: `ambient.m4a`, `ambient.json`, `short.mp4`. Committed. |
| `build/` | Scratch files (full-quality WAVs, temp images). Gitignored. |

## Adding devotional pages

Add one entry per publish date to `queue/queue.json`:

```json
"2026-11-02": {
  "ref": "Psalm 145:4-5",
  "title": "One Generation Shall Praise Your Works",
  "text": "One Generation Shall Praise Your Works. One generation shall praise Your works to another...",
  "format": "short"
}
```

- `text` may start with the title, as the pages do now. The render step removes it so the title isn't shown twice.
- `long_video: true` marks a Sunday page that also gets the weekly long video. The Short is still rendered as usual. The long video isn't built here yet.
- Optional `mood` (praise, awe, comfort, hope, lament, repentance, warning, victory, rest, wisdom) overrides the detected tone.

If a date has no page, the nightly run stops and asks for it. It never makes up a verse.

## Nightly run

From the repo root:

```bash
pip install --break-system-packages -r requirements.txt   # numpy, scipy, Pillow; ffmpeg must be installed

python3 nightly.py                  # 1. ambient bed  -> renders/<verse>/ambient.m4a
python3 render/render_short.py      # 2. finished Short -> renders/<verse>/short.mp4
```

Both take `--date YYYY-MM-DD` or `--lead-days N` (default 1 = tomorrow, Eastern) and print one line of JSON.

| Exit | Status | Meaning / what to do |
|---|---|---|
| 0 | `ok` | Done. |
| 2 | `missing_page` | No queue entry for that date. Ask Tommy for that day's devotional page and stop. |
| 3 | `missing_bed` | `render_short.py` ran before `nightly.py`. Run `nightly.py` first. |
| 4 | `qa_failed` | The video came out the wrong size or length, or has no sound. `problems` says which. Don't ship it; report it. |

Then commit `renders/<verse>/` and `ambient/ledger.json` (never `build/`), pull/fetch `origin main`, push to `main`, and send Tommy `renders/<verse>/short.mp4` with the publish date, verse reference, mood and key.

Rules:

- Never edit `ambient/ledger.json` by hand, and never pass `--fresh` or `--variant` in the nightly run. The ledger guarantees every Short gets its own bed, and it must change only through `nightly.py`.
- The render step only reads the bed. It never touches the ledger, so re-running it to fix the visuals is always safe.

## What the Short looks like

- 1080×1920, 30 fps, 54 s (the bed's length), H.264 + AAC, about 4–5 MB.
- Five sections, lined up with the bed's five sections:
  1. title
  2–4. the verse, split into up to three parts at natural pauses
  5. reference (gold) and "The Powerful Name"
- Background: a slow, softly glowing gradient in colours picked from the bed's mood (blue/violet for awe, amber for praise, and so on).
- Text fades in and out. The first and last frames are the same text-free background, and the bed loops seamlessly, so the Short loops cleanly too.
- Font: Lora (falls back to DejaVu Serif if Lora isn't installed). To pin a font, put the `.ttf` files in `render/fonts/`.
- Logo: if `render/assets/logo.png` exists (transparent PNG), it's shown small at the top. There's no logo in the repo yet.

## Not built yet

- Voiceover. `ambient_bed.py` can already duck the bed under a VO (`--vo` / `--mix-out`). The render step would need a VO source first.
- The Sunday long video.
- Uploading to YouTube. Tommy reviews and publishes by hand.
