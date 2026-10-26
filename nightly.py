#!/usr/bin/env python3
"""Nightly build for The Powerful Name.

Looks up the Short that publishes `--lead-days` days from today (Eastern time)
in queue/queue.json, renders its ambient bed through the uniqueness ledger, and
writes a phone preview to renders/<verse>/. Exit code 2 means the devotional
page for that day hasn't been added to the queue yet.

  python nightly.py                 # tomorrow's Short
  python nightly.py --lead-days 2   # once the batches of 10 start
  python nightly.py --date 2026-10-27
"""
import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
QUEUE = ROOT / "queue" / "queue.json"
LEDGER = ROOT / "ambient" / "ledger.json"
RENDERS = ROOT / "renders"
WORK = ROOT / "build"


def slug(ref):
    return re.sub(r"[^a-z0-9]+", "-", ref.lower()).strip("-")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lead-days", type=int, default=1)
    ap.add_argument("--date", help="Publish date to build (YYYY-MM-DD), overrides --lead-days")
    a = ap.parse_args()

    day = a.date or (datetime.now(ZoneInfo("America/New_York")).date() + timedelta(days=a.lead_days)).isoformat()
    queue = json.loads(QUEUE.read_text()) if QUEUE.exists() else {}
    item = queue.get(day)
    if not item:
        print(json.dumps({"status": "missing_page", "publish_date": day}))
        sys.exit(2)

    name = slug(item["ref"])
    out_dir = RENDERS / name
    out_dir.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(exist_ok=True)
    wav = WORK / f"{name}_ambient.wav"
    cmd = [sys.executable, str(ROOT / "ambient" / "ambient_bed.py"), "--ref", item["ref"], "--text", item["text"],
           "--format", item.get("format", "short"), "--ledger", str(LEDGER), "--out", str(wav), "--preview"]
    if item.get("mood"):
        cmd += ["--mood", item["mood"]]
    meta = json.loads(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip().splitlines()[-1])
    (out_dir / "ambient.m4a").write_bytes(wav.with_suffix(".m4a").read_bytes())
    (out_dir / "ambient.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({"status": "ok", "publish_date": day, "ref": item["ref"], "mood": meta["mood"],
                      "key": meta["key"], "preview": str((out_dir / "ambient.m4a").relative_to(ROOT)),
                      "wav": str(wav.relative_to(ROOT))}))


if __name__ == "__main__":
    main()
