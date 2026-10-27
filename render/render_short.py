#!/usr/bin/env python3
"""Render the finished vertical Short for The Powerful Name.

Runs after nightly.py. Takes the queue entry for the publish date and the
ambient bed nightly.py just built, and writes renders/<verse>/short.mp4:
1080x1920, 30 fps, the same length as the bed (54 s), five sections lined up
with the bed's five sections:

  1. title   2-4. the verse, in up to three parts   5. reference + channel name

The background is a slow gradient in the bed's mood colours. Both the picture
and the sound start and end on the same frame, so the Short loops cleanly.

This step never touches ambient/ledger.json: it only reads the bed.

  python3 render/render_short.py                    # tomorrow's Short
  python3 render/render_short.py --date 2026-10-28
"""
import argparse
import json
import math
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from nightly import QUEUE, RENDERS, WORK, slug  # noqa: E402

W, H, FPS = 1080, 1920, 30
BG_SCALE = 4                      # background is drawn at 1/4 size and scaled up
FADE = 0.8                        # seconds for each text fade in/out
GAP = 0.3                         # text-free time at each section edge
TEXT_W = 880                      # max text width in pixels
CHANNEL = "The Powerful Name"
LOGO = ROOT / "render" / "assets" / "logo.png"   # optional; shown at the top if present

FONT_DIRS = [ROOT / "render" / "fonts", Path("/usr/share/fonts/truetype/google-fonts"),
             Path("/usr/share/fonts/truetype/dejavu")]

# top colour, bottom colour, glow colour (RGB) for each mood
MOOD_COLOURS = {
    "praise":     ((58, 32, 8),   (14, 8, 4),   (240, 170, 60)),
    "awe":        ((18, 22, 58),  (4, 4, 16),   (150, 140, 255)),
    "comfort":    ((22, 44, 52),  (6, 12, 16),  (120, 200, 200)),
    "hope":       ((52, 36, 20),  (10, 10, 22), (255, 190, 120)),
    "lament":     ((26, 28, 36),  (6, 6, 10),   (120, 130, 160)),
    "repentance": ((34, 22, 34),  (8, 4, 10),   (170, 120, 170)),
    "warning":    ((50, 16, 12),  (10, 4, 4),   (220, 90, 60)),
    "victory":    ((48, 34, 6),   (12, 6, 20),  (255, 210, 90)),
    "rest":       ((16, 36, 30),  (4, 10, 10),  (130, 210, 170)),
    "wisdom":     ((20, 30, 48),  (6, 8, 14),   (200, 180, 120)),
}


# ---------- text ----------

def clean_verse(item):
    """Queue text starts with the title; drop it so it isn't shown twice."""
    text, title = item["text"].strip(), item.get("title", "").strip()
    if title and text.startswith(title):
        text = text[len(title):].lstrip(" .:-—")
    return text.strip()


def split_verse(text, max_parts=3, words_per_part=18):
    """Split the verse into 1-3 parts of similar length, breaking after punctuation when possible."""
    words = text.split()
    n = max(1, min(max_parts, math.ceil(len(words) / words_per_part)))
    cuts, start = [], 0
    for k in range(1, n):
        target = round(len(words) * k / n)
        best = target
        for d in range(0, 6):                              # look up to 5 words either side
            for c in (target + d, target - d):
                if start < c < len(words) and words[c - 1][-1] in ",.;:!?”\"":
                    best = c
                    break
            else:
                continue
            break
        cuts.append(best)
        start = best
    bounds = [0] + cuts + [len(words)]
    return [" ".join(words[a:b]) for a, b in zip(bounds, bounds[1:]) if b > a]


def find_font(names):
    for d in FONT_DIRS:
        for n in names:
            p = d / n
            if p.exists():
                return p
    raise SystemExit(f"No font found among {names}")


def load_font(path, size, variation=None):
    f = ImageFont.truetype(str(path), size)
    if variation:
        try:
            f.set_variation_by_name(variation)
        except Exception:
            pass
    return f


def wrap(draw, text, font, width):
    lines, line = [], ""
    for w in text.split():
        trial = f"{line} {w}".strip()
        if draw.textlength(trial, font=font) <= width or not line:
            line = trial
        else:
            lines.append(line)
            line = w
    if line:
        lines.append(line)
    return lines


def card(blocks, path):
    """Draw centred blocks of text on a transparent 1080x1920 card.

    blocks: list of (text, font_path, start_size, min_size, variation, colour, spacing_after)
    """
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    laid = []
    for text, fpath, size, min_size, var, colour, after in blocks:
        while True:
            font = load_font(fpath, size, var)
            lines = wrap(d, text, font, TEXT_W)
            lh = int(size * 1.32)
            if lh * len(lines) <= 1000 or size <= min_size:
                break
            size -= 4
        laid.append((lines, font, lh, colour, after))
    total = sum(lh * len(lines) + after for lines, _, lh, _, after in laid)
    y = (H - total) // 2 - 40
    text_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    td = ImageDraw.Draw(text_layer)
    for lines, font, lh, colour, after in laid:
        for ln in lines:
            td.text((W // 2, y + lh // 2), ln, font=font, fill=colour + (255,), anchor="mm")
            y += lh
        y += after
    # soft shadow so text reads on any part of the gradient
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    shadow.putalpha(text_layer.getchannel("A").point(lambda a: a * 0.75))
    shadow = shadow.filter(ImageFilter.GaussianBlur(10))
    img.alpha_composite(shadow)
    img.alpha_composite(text_layer)
    img.save(path)


# ---------- background ----------

def background_frames(mood, dur):
    """Yield low-res RGB frames. Motion is periodic over dur, so the last frame meets the first."""
    top, bottom, glow = (np.array(c, dtype=np.float32) for c in MOOD_COLOURS.get(mood, MOOD_COLOURS["awe"]))
    w, h = W // BG_SCALE, H // BG_SCALE
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    v = (yy / (h - 1))[..., None]
    base = top * (1 - v) + bottom * v
    n = int(round(dur * FPS))
    for i in range(n):
        ph = 2 * math.pi * i / n
        cx = w * (0.5 + 0.16 * math.sin(ph))
        cy = h * (0.38 + 0.07 * math.sin(2 * ph + 0.6))
        r = w * (0.62 + 0.06 * math.cos(ph))
        g = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (r * r)))[..., None]
        intensity = 0.30 + 0.06 * math.cos(ph)
        frame = base + glow * g * intensity
        yield np.clip(frame, 0, 255).astype(np.uint8).tobytes()


# ---------- main ----------

def ffprobe(path, entries):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", entries, "-of", "json", str(path)],
                         check=True, capture_output=True, text=True).stdout
    return json.loads(out)


def render(day, item, out_dir):
    name = slug(item["ref"])
    meta = json.loads((out_dir / "ambient.json").read_text())
    dur = float(meta["seconds"])
    sections = int(meta.get("sections", 5))
    sec = dur / sections
    wav = WORK / f"{name}_ambient.wav"
    audio = wav if wav.exists() else out_dir / "ambient.m4a"

    serif = find_font(["Lora-Variable.ttf", "DejaVuSerif.ttf"])
    serif_it = find_font(["Lora-Italic-Variable.ttf", "DejaVuSerif-Italic.ttf"])
    ivory, gold = (246, 240, 228), (226, 196, 128)

    verse_parts = split_verse(clean_verse(item))
    tmp = Path(tempfile.mkdtemp(prefix="short_", dir=WORK))
    cards = []   # (png, start, end)

    def add(blocks, start, end):
        p = tmp / f"card{len(cards)}.png"
        card(blocks, p)
        cards.append((p, start + GAP, end - GAP))

    add([(item["title"], serif, 96, 60, "Bold", ivory, 0)], 0, sec)
    middle_start, middle_end = sec, sec * (sections - 1)
    step = (middle_end - middle_start) / len(verse_parts)
    for k, part in enumerate(verse_parts):
        add([(part, serif, 78, 48, "Regular", ivory, 0)], middle_start + k * step, middle_start + (k + 1) * step)
    add([(item["ref"], serif, 88, 60, "SemiBold", gold, 40),
         (CHANNEL, serif_it, 54, 40, "Italic", ivory, 0)], middle_end, dur)

    cmd = ["ffmpeg", "-y", "-v", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W // BG_SCALE}x{H // BG_SCALE}", "-r", str(FPS), "-i", "-",
           "-i", str(audio)]
    for p, s, e in cards:
        cmd += ["-loop", "1", "-framerate", str(FPS), "-t", f"{e - s:.3f}", "-i", str(p)]
    has_logo = LOGO.exists()
    if has_logo:
        cmd += ["-loop", "1", "-framerate", str(FPS), "-t", f"{dur:.3f}", "-i", str(LOGO)]

    f = [f"[0:v]scale={W}:{H}:flags=bicubic,format=yuv420p,noise=alls=2:allf=t[bg0]"]
    last = "bg0"
    for i, (p, s, e) in enumerate(cards):
        ln = e - s
        f.append(f"[{i + 2}:v]format=rgba,fade=t=in:st=0:d={FADE}:alpha=1,"
                 f"fade=t=out:st={ln - FADE:.3f}:d={FADE}:alpha=1,setpts=PTS-STARTPTS+{s:.3f}/TB[c{i}]")
        f.append(f"[{last}][c{i}]overlay=eof_action=pass:format=auto[bg{i + 1}]")
        last = f"bg{i + 1}"
    if has_logo:
        li = len(cards) + 2
        f.append(f"[{li}:v]scale=220:-1,format=rgba,colorchannelmixer=aa=0.85[logo]")
        f.append(f"[{last}][logo]overlay=(W-w)/2:150:format=auto[vlogo]")
        last = "vlogo"
    f.append(f"[{last}]format=yuv420p[v]")

    out = out_dir / "short.mp4"
    cmd += ["-filter_complex", ";".join(f), "-map", "[v]", "-map", "1:a",
            "-t", f"{dur:.3f}", "-r", str(FPS),
            "-c:v", "libx264", "-preset", "medium", "-crf", "24", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out)]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for fr in background_frames(meta["mood"], dur):
            proc.stdin.write(fr)
        proc.stdin.close()
    except BrokenPipeError:
        pass
    if proc.wait() != 0:
        raise SystemExit("ffmpeg failed")

    # QA: right size, right length, has sound
    info = ffprobe(out, "format=duration:stream=codec_type,width,height")
    streams = info["streams"]
    v = next(s for s in streams if s["codec_type"] == "video")
    got = float(info["format"]["duration"])
    problems = []
    if (v["width"], v["height"]) != (W, H):
        problems.append(f"size {v['width']}x{v['height']}")
    if abs(got - dur) > 0.1:
        problems.append(f"length {got:.2f}s, expected {dur:.2f}s")
    if not any(s["codec_type"] == "audio" for s in streams):
        problems.append("no audio")
    for p, _, _ in cards:
        p.unlink()
    tmp.rmdir()
    return out, got, verse_parts, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lead-days", type=int, default=1)
    ap.add_argument("--date", help="Publish date to render (YYYY-MM-DD), overrides --lead-days")
    a = ap.parse_args()
    day = a.date or (datetime.now(ZoneInfo("America/New_York")).date() + timedelta(days=a.lead_days)).isoformat()

    item = json.loads(QUEUE.read_text()).get(day)
    if not item:
        print(json.dumps({"status": "missing_page", "publish_date": day}))
        sys.exit(2)
    if item.get("format", "short") != "short":
        print(json.dumps({"status": "skipped", "publish_date": day, "reason": "long-format video, not rendered here"}))
        return
    out_dir = RENDERS / slug(item["ref"])
    if not (out_dir / "ambient.json").exists():
        print(json.dumps({"status": "missing_bed", "publish_date": day,
                          "reason": "run nightly.py first"}))
        sys.exit(3)
    WORK.mkdir(exist_ok=True)
    out, got, parts, problems = render(day, item, out_dir)
    status = "ok" if not problems else "qa_failed"
    print(json.dumps({"status": status, "publish_date": day, "ref": item["ref"],
                      "video": str(out.relative_to(ROOT)), "seconds": round(got, 2),
                      "verse_parts": len(parts), "problems": problems}))
    sys.exit(0 if not problems else 4)


if __name__ == "__main__":
    main()
