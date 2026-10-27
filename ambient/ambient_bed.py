#!/usr/bin/env python3
"""Ambient bed generator for The Powerful Name.

Reads the verse, picks its tone, and synthesizes an original ambient bed
for that Short or Sunday long video. Nothing is sampled or downloaded;
every sound is built here from sine/wavetable oscillators, filtered noise
and a synthetic reverb, so there is no copyright-claim risk.

Shorts: 54 s, five chord sections (one per logo turn), seamless loop.
Long:   any length (--duration or --match a VO/video file), gentle arc in and out.

Usage:
  python ambient_bed.py --ref "Psalm 145:4-5" --text "One generation shall praise..." \
      --format short --out build/ambient.wav --preview
  python ambient_bed.py ... --vo build/vo.wav --mix-out build/vo_mix.wav
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve, lfilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from moods import PALETTES, SCALES, tag_mood  # noqa: E402

SHORT_SECONDS = 54.0
SHORT_SECTIONS = 5          # one chord section per logo turn
TARGET_RMS_DB = -21.0       # bed level before it goes under the VO


# ---------------------------------------------------------------- helpers
def midi_hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


class Ctx:
    def __init__(self, dur, sr, rng, loop):
        self.sr = sr
        self.n = int(round(dur * sr))
        self.dur = self.n / sr
        self.rng = rng
        self.loop = loop
        self.t = np.arange(self.n, dtype=np.float64) / sr

    def snap(self, f):
        """Round a frequency so it completes whole cycles over the bed (seamless loop)."""
        return max(1, round(f * self.dur)) / self.dur

    def lfo_k(self, hz):
        """Whole number of LFO cycles closest to hz, at least one."""
        return max(1, round(hz * self.dur))


def wavetable(bright, size=4096, harmonics=14):
    p = 2.4 - 1.8 * float(np.clip(bright, 0.0, 1.0))
    x = np.arange(size) / size
    w = np.zeros(size)
    for h in range(1, harmonics + 1):
        w += np.sin(2 * np.pi * h * x) / h ** p
    return w / np.max(np.abs(w))


def osc(table, phase):
    idx = (phase % 1.0) * len(table)
    i0 = idx.astype(np.int64)
    frac = idx - i0
    n = len(table)
    return table[i0 % n] * (1 - frac) + table[(i0 + 1) % n] * frac


def pan(sig, p):
    th = (np.clip(p, -1, 1) + 1) * np.pi / 4
    return np.stack([sig * np.cos(th), sig * np.sin(th)])


def add_at(buf, sig, start, wrap):
    n, L = buf.shape[1], sig.shape[1]
    if not wrap:
        end = min(n, start + L)
        if start < n:
            buf[:, start:end] += sig[:, : end - start]
        return
    pos, off = start % n, 0
    while off < L:
        k = min(L - off, n - pos)
        buf[:, pos:pos + k] += sig[:, off:off + k]
        off += k
        pos = 0


def rms(x):
    return float(np.sqrt(np.mean(x ** 2)) + 1e-12)


def chord_midis(root, scale, degrees, inversion=0, octave=0):
    out = []
    for d in degrees:
        out.append(root + scale[d % 7] + 12 * (d // 7) + 12 * octave)
    out = sorted(out)
    for _ in range(inversion):
        out = out[1:] + [out[0] + 12]
    return out


def fft_filter(x, sr, gain_fn):
    """Zero-phase circular filter (keeps the loop seamless)."""
    X = np.fft.rfft(x, axis=-1)
    f = np.fft.rfftfreq(x.shape[-1], 1 / sr)
    return np.fft.irfft(X * gain_fn(f), n=x.shape[-1], axis=-1)


# ---------------------------------------------------------------- layers
def sections(c, chords, root, scale, inversions):
    L = c.dur / len(chords)
    X = min(3.5, 0.4 * L)
    voiced = [chord_midis(root, scale, ch, inversion=inv) for ch, inv in zip(chords, inversions)]
    return L, X, voiced


def seg_env(seglen, fade, tt, total):
    env = np.ones(seglen)
    x = np.linspace(0, 1, fade)
    env[:fade] = np.sin(np.pi / 2 * x)
    env[-fade:] = np.cos(np.pi / 2 * x)
    return env * (0.86 + 0.14 * np.sin(np.pi * tt / total))


def layer_pad(c, P, sec, amp, rng):
    L, X, voiced = sec
    table = wavetable(P["bright"] * 0.75)
    seglen = int(round((L + 2 * X) * c.sr))
    fade = int(round(2 * X * c.sr))
    tt = np.arange(seglen) / c.sr
    env = seg_env(seglen, fade, tt, L + 2 * X)
    out = np.zeros((2, c.n))
    for i, notes in enumerate(voiced):
        seg = np.zeros((2, seglen))
        for m in notes:
            f = midi_hz(m)
            for v in range(3):
                cents = rng.uniform(-9, 9)
                ff = f * 2 ** (cents / 1200)
                ph = rng.random()
                s = osc(table, ph + ff * tt) / (len(notes) * 3)
                seg += pan(s, rng.uniform(-0.75, 0.75))
        add_at(out, seg * env, int(round(i * L * c.sr)), c.loop)
    return out / rms(out) * amp


def layer_strings(c, P, sec, amp, rng):
    L, X, voiced = sec
    table = wavetable(min(1.0, P["bright"] + 0.2))
    seglen = int(round((L + 2 * X) * c.sr))
    fade = int(round(2 * X * c.sr))
    tt = np.arange(seglen) / c.sr
    env = seg_env(seglen, fade, tt, L + 2 * X)
    swell = np.sin(np.pi * tt / (L + 2 * X)) ** 1.6   # one breath per section
    out = np.zeros((2, c.n))
    for i, notes in enumerate(voiced):
        seg = np.zeros((2, seglen))
        top = [m + 12 for m in notes[-2:]]
        for m in top:
            f = midi_hz(m)
            for v in range(2):
                vib = 1 + 0.0028 * np.sin(2 * np.pi * rng.uniform(4.6, 5.6) * tt + rng.random() * 6.28)
                ff = f * 2 ** (rng.uniform(-6, 6) / 1200)
                phase = np.cumsum(ff * vib) / c.sr + rng.random()
                seg += pan(osc(table, phase) / 4, rng.uniform(-0.6, 0.6))
        # soften the bow edge
        seg = lfilter([0.18], [1, -0.82], seg, axis=1)
        add_at(out, seg * env * swell, int(round(i * L * c.sr)), c.loop)
    return out / rms(out) * amp


def layer_drone(c, P, root, amp, rng):
    table = wavetable(0.15)
    out = np.zeros((2, c.n))
    for m, side in ((root - 12, -0.3), (root - 12, 0.3), (root - 5, 0.0)):
        f = c.snap(midi_hz(m) * 2 ** (rng.uniform(-5, 5) / 1200))
        s = osc(table, rng.random() + f * c.t)
        k = c.lfo_k(rng.uniform(0.03, 0.08))
        s *= 0.75 + 0.25 * np.sin(2 * np.pi * k * c.t / c.dur + rng.random() * 6.28)
        out += pan(s, side)
    return out / rms(out) * amp


def bell(f, sr, vel, length=7.0, decay=2.4):
    t = np.arange(int(length * sr)) / sr
    s = np.zeros_like(t)
    for ratio, a, dk in ((1, 1, 1), (2.0, .42, .6), (2.76, .28, .45), (5.40, .11, .3), (8.93, .05, .2)):
        if f * ratio < 16000:
            s += a * np.sin(2 * np.pi * f * ratio * t) * np.exp(-t / (decay * dk))
    att = int(0.004 * sr)
    s[:att] *= np.linspace(0, 1, att)
    return s * vel


def layer_bells(c, P, sec, amp, rng, density):
    L, X, voiced = sec
    out = np.zeros((2, c.n))
    grid = L / 8
    slots = int(round(c.dur / grid))
    p = min(0.9, density * grid)
    hits = 0
    for k in range(slots):
        if rng.random() > p:
            continue
        t0 = k * grid + rng.uniform(-0.03, 0.03)
        if not c.loop:
            t0 = max(0.0, t0)
        i = min(len(voiced) - 1, int((k * grid) // L))
        m = int(rng.choice(voiced[i])) + P["bell_octave"]
        s = bell(midi_hz(m), c.sr, rng.uniform(0.35, 1.0), decay=rng.uniform(1.8, 3.2))
        add_at(out, pan(s, rng.uniform(-0.6, 0.6)), int(round(t0 * c.sr)) % c.n, c.loop)
        hits += 1
    if hits == 0:
        return out
    return out / rms(out) * amp


def layer_shimmer(c, P, root, amp, rng):
    out = np.zeros((2, c.n))
    for iv in (36, 43, 40 + 12 * int(rng.integers(0, 2))):
        f = c.snap(midi_hz(root + iv))
        s = np.sin(2 * np.pi * f * c.t + rng.random() * 6.28)
        k = c.lfo_k(rng.uniform(0.08, 0.25))
        s *= (0.5 + 0.5 * np.sin(2 * np.pi * k * c.t / c.dur + rng.random() * 6.28)) ** 2
        out += pan(s, rng.uniform(-0.8, 0.8))
    return out / rms(out) * amp


def layer_wind(c, P, amp, rng):
    fc = rng.uniform(350, 900) * (0.7 + 0.6 * P["bright"])
    bw = rng.uniform(0.55, 0.9)
    noise = rng.standard_normal((2, c.n))

    def g(f):
        f = np.maximum(f, 1.0)
        return np.exp(-0.5 * (np.log(f / fc) / bw) ** 2)

    w = fft_filter(noise, c.sr, g)
    for ch in range(2):
        k1, k2 = c.lfo_k(rng.uniform(0.04, 0.09)), c.lfo_k(rng.uniform(0.11, 0.2))
        w[ch] *= (0.55 + 0.30 * np.sin(2 * np.pi * k1 * c.t / c.dur + rng.random() * 6.28)
                  + 0.15 * np.sin(2 * np.pi * k2 * c.t / c.dur + rng.random() * 6.28))
    return w / rms(w) * amp


def layer_pulse(c, P, sec, root, amp, rng):
    style, per_section = P["pulse"]
    L = sec[0]
    out = np.zeros((2, c.n))
    f0 = midi_hz(root - 24)
    t = np.arange(int(1.2 * c.sr)) / c.sr
    inst = f0 * (1 + 0.9 * np.exp(-t * 28))
    hit = np.sin(2 * np.pi * np.cumsum(inst) / c.sr) * np.exp(-t / (0.32 if style == "heart" else 0.6))
    hit = pan(hit, 0.0)
    step = L / per_section
    n_hits = int(round(c.dur / step))
    for k in range(n_hits):
        start = int(round(k * step * c.sr))
        vel = 1.0 if k % per_section == 0 else 0.75
        add_at(out, hit * vel, start, c.loop)
        if style == "heart":
            add_at(out, hit * vel * 0.55, start + int(0.27 * c.sr), c.loop)
    return out / rms(out) * amp


# ---------------------------------------------------------------- mastering
def reverb(x, c, rt, wet, rng):
    Lr = int(rt * c.sr)
    t = np.arange(Lr) / c.sr
    ir = rng.standard_normal((2, Lr)) * np.exp(-6.9 * t / rt)
    ir = lfilter([0.35], [1, -0.65], ir, axis=1)     # darker tail
    ir /= np.sqrt(np.sum(ir ** 2, axis=1, keepdims=True))
    ir = np.pad(ir, ((0, 0), (int(0.022 * c.sr), 0)))
    mono = x.mean(axis=0)
    rev = np.zeros_like(x)
    for ch in range(2):
        y = fftconvolve(mono, ir[ch])
        o = y[:c.n].copy()
        if c.loop:                                     # wrap the tail to the top
            tail = y[c.n:]
            while len(tail):
                k = min(c.n, len(tail))
                o[:k] += tail[:k]
                tail = tail[k:]
        rev[ch] = o
    rev *= rms(x) / rms(rev)
    return (1 - wet) * x + wet * rev


def master(x, c, long_arc):
    x = fft_filter(x, c.sr, lambda f: 1 / np.sqrt(1 + (35 / np.maximum(f, 1)) ** 8))   # rumble cut
    x = fft_filter(x, c.sr, lambda f: 1 / np.sqrt(1 + (f / 11000) ** 4))              # air trim
    if long_arc:
        g = 0.72 + 0.28 * np.sin(np.pi * c.t / c.dur)
        fi, fo = int(4 * c.sr), int(7 * c.sr)
        g[:fi] *= np.linspace(0, 1, fi) ** 2
        g[-fo:] *= np.linspace(1, 0, fo) ** 2
        x = x * g
    x *= 10 ** (TARGET_RMS_DB / 20) / rms(x)
    peak = np.max(np.abs(x))
    if peak > 0.89:
        x = 0.89 * np.tanh(x / 0.89)
    return x


# ---------------------------------------------------------------- build
NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def plan(ref, fmt, variant, mood, dur):
    """Every musical choice for this bed, decided from the seed before any audio is made."""
    P = dict(PALETTES[mood])
    seed_src = f"{ref}|{fmt}|{variant}|{mood}"
    seed = int(hashlib.sha256(seed_src.encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    loop = fmt == "short"

    # per-render variation inside the palette
    P["bright"] = float(np.clip(P["bright"] + rng.uniform(-0.08, 0.08), 0.05, 0.95))
    root = int(rng.choice(P["roots"]))
    scale = SCALES[P["scale"]]
    prog = [list(ch) for ch in P["progressions"][int(rng.integers(len(P["progressions"])))]]
    if len(prog) > 3 and rng.random() < 0.4:                     # swap two inner chords
        a, b = rng.choice(np.arange(1, len(prog) - 1), 2, replace=False)
        prog[a], prog[b] = prog[b], prog[a]
    n_sec = SHORT_SECTIONS if loop else max(4, int(round(dur / 13)))
    chords = list(prog)
    while len(chords) < n_sec:          # long videos keep wandering through the palette
        chords += [list(ch) for ch in P["progressions"][int(rng.integers(len(P["progressions"])))]]
    chords = chords[:n_sec]
    if not loop:
        chords[-1] = list(prog[0])      # land home at the end
    inversions = [int(rng.integers(0, 2)) for _ in chords]

    layers ={k: v * rng.uniform(0.82, 1.18) for k, v in P["layers"].items()}
    for k, v in P.get("optional", {}).items():
        if rng.random() < 0.5:
            layers[k] = v * rng.uniform(0.8, 1.2)
    density = P["bell_density"] * rng.uniform(0.7, 1.3)
    rt = P["rt"] * rng.uniform(0.85, 1.15)
    key = f"{NOTE_NAMES[root % 12]} {P['scale']}"
    signature = key + " | " + " ".join("-".join(map(str, ch)) for ch in chords[:5]) + " | " + ",".join(sorted(layers))
    return dict(P=P, rng=rng, seed_src=seed_src, loop=loop, root=root, scale=scale, chords=chords,
                inversions=inversions,
                n_sec=n_sec, layers=layers, density=density, rt=rt, key=key, signature=signature)


def detect_mood(text, mood_override):
    mood, scores = tag_mood(text) if text else ("awe", {})
    return (mood_override or mood), scores


def build(ref, text, fmt, duration, variant, mood_override, sr):
    mood, scores = detect_mood(text, mood_override)
    dur = SHORT_SECONDS if fmt == "short" else float(duration)
    pl = plan(ref, fmt, variant, mood, dur)
    P, rng, loop, root = pl["P"], pl["rng"], pl["loop"], pl["root"]
    layers, density, rt, n_sec = pl["layers"], pl["density"], pl["rt"], pl["n_sec"]
    c = Ctx(dur, sr, rng, loop)
    sec = sections(c, pl["chords"], root, pl["scale"], pl["inversions"])

    mix = np.zeros((2, c.n))
    for name, amp in layers.items():
        if name == "pad":
            mix += layer_pad(c, P, sec, amp, rng)
        elif name == "strings":
            mix += layer_strings(c, P, sec, amp, rng)
        elif name == "drone":
            mix += layer_drone(c, P, root, amp, rng)
        elif name == "bells":
            mix += layer_bells(c, P, sec, amp, rng, density)
        elif name == "shimmer":
            mix += layer_shimmer(c, P, root, amp, rng)
        elif name == "wind":
            mix += layer_wind(c, P, amp, rng)
        elif name == "pulse" and P["pulse"]:
            mix += layer_pulse(c, P, sec, root, amp, rng)

    mix = reverb(mix, c, rt, P["wet"], rng)
    mix = master(mix, c, long_arc=not loop)

    meta = {
        "ref": ref, "format": fmt, "variant": variant, "mood": mood,
        "mood_scores": {k: v for k, v in scores.items() if v},
        "key": pl["key"], "seconds": round(c.dur, 3),
        "sections": n_sec, "layers": sorted(layers), "reverb_s": round(rt, 2),
        "seamless_loop": loop, "seed": pl["seed_src"], "signature": pl["signature"],
        "audio_fingerprint": hashlib.sha256(np.round(mix * 32767).astype(np.int16).tobytes()).hexdigest()[:16],
    }
    return mix, meta


# ---------------------------------------------------------------- uniqueness ledger
RECENT_SIGNATURES = 30   # a new bed may not repeat any of the last 30 key + progression + layer combos
RECENT_KEYS = 2          # and may not share a key with the two Shorts right before it


def choose_variant(ledger, ref, fmt, mood, dur, fresh=False):
    """Same Short re-rendered -> same bed. New Short -> a bed unlike the recent ones."""
    entries = ledger.setdefault("entries", [])
    mine = [e for e in entries if e["ref"] == ref and e["format"] == fmt]
    if mine and not fresh:
        return mine[-1]["variant"], False
    start = (max(e["variant"] for e in mine) + 1) if mine else 0
    same_fmt = [e for e in entries if e["format"] == fmt]
    recent_sigs = {e["signature"] for e in same_fmt[-RECENT_SIGNATURES:]}
    recent_keys = {e["key"] for e in same_fmt[-RECENT_KEYS:]}
    used_sigs = {e["signature"] for e in same_fmt}
    fallback = None
    for v in range(start, start + 200):
        pl = plan(ref, fmt, v, mood, dur)
        if pl["signature"] in recent_sigs or pl["key"] in recent_keys:
            continue
        if pl["signature"] not in used_sigs:
            return v, True
        fallback = v if fallback is None else fallback
    return (fallback if fallback is not None else start), True


def load_ledger(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {"entries": []}


def record(ledger, path, meta):
    ledger["entries"].append({k: meta[k] for k in ("ref", "format", "variant", "mood", "key",
                                                  "signature", "audio_fingerprint")})
    Path(path).write_text(json.dumps(ledger, indent=1))


def write_wav(path, x, sr):
    rng = np.random.default_rng(0)
    d = (rng.random(x.shape) - rng.random(x.shape)) / 32768      # TPDF dither
    pcm = np.clip((x + d) * 32767, -32768, 32767).astype(np.int16)
    wavfile.write(path, sr, pcm.T)


def ffprobe_seconds(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "default=nw=1:nk=1", str(path)], capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


LOOP_SHIELD = 0.008   # seconds faded to silence at both ends of the final mix.
# Loop-point pop (found Oct 27, 2026 on Psalm 76:4,7): the limiter delayed the whole mix by
# ~5 ms, leaving a dropout at the top of the file while the end played at full level, so the
# player wrapped from full sound into a gap. latency=1 removes the delay; the short fades make
# the wrap meet silence even with the ~13 ms of padding the AAC encoder adds at the end.


def loop_seam_qa(path, max_step=0.02):
    """Fail if the encoded file clicks when the player wraps from the end back to the start."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-f", "f32le", "-ac", "2",
                          "-ar", "48000", "-"], capture_output=True, check=True).stdout
    x = np.frombuffer(raw, np.float32).reshape(-1, 2)
    lvl = np.abs(x).max(axis=1)
    head, tail = lvl[:96].max(), lvl[-96:].max()                  # first/last 2 ms
    lead_gap = int(np.argmax(lvl > 0.005)) if lvl.max() > 0.005 else 0   # silence before sound starts
    ok = max(head, tail) <= max_step and lead_gap <= 96           # quiet edges, no dropout at the top
    return {"loop_seam_ok": bool(ok), "loop_head_level": round(float(head), 4),
            "loop_tail_level": round(float(tail), 4), "loop_lead_gap_ms": round(lead_gap / 48, 2)}


def mix_under_vo(vo, bed, out, bed_db=-4.0):
    """Lay the bed under the voiceover and duck it whenever Tommy's VO speaks."""
    total = max(ffprobe_seconds(bed), ffprobe_seconds(vo))      # bed runs the full Short
    fc = (f"[1:a]aformat=channel_layouts=stereo,volume={bed_db}dB[bed];"
          f"[0:a]aformat=sample_rates=48000:channel_layouts=stereo,apad=whole_dur={total:.3f},asplit=2[vo][sc];"
          "[bed][sc]sidechaincompress=threshold=0.025:ratio=6:attack=60:release=650[duck];"
          "[vo][duck]amix=inputs=2:normalize=0:duration=longest,alimiter=limit=0.95:latency=1,"
          f"atrim=end={total:.3f},afade=t=in:d={LOOP_SHIELD},"
          f"afade=t=out:st={total - LOOP_SHIELD:.3f}:d={LOOP_SHIELD}[m]")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(vo), "-i", str(bed),
                    "-filter_complex", fc, "-map", "[m]", "-ar", "48000", str(out)], check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True, help='Verse reference, e.g. "Psalm 145:4-5"')
    ap.add_argument("--text", help="Verse text")
    ap.add_argument("--text-file", help="File holding the verse text")
    ap.add_argument("--mood", choices=sorted(PALETTES), help="Override the detected tone")
    ap.add_argument("--format", choices=["short", "long"], default="short")
    ap.add_argument("--duration", type=float, help="Seconds (long format)")
    ap.add_argument("--match", help="Media file whose length the long bed should match")
    ap.add_argument("--variant", type=int, default=0, help="Bump for a fresh take on the same verse")
    ap.add_argument("--sr", type=int, default=48000)
    ap.add_argument("--out", required=True, help="Output WAV")
    ap.add_argument("--preview", action="store_true", help="Also write a phone-friendly .m4a")
    ap.add_argument("--ledger", help="Uniqueness ledger (JSON). Picks a variant unlike recent Shorts")
    ap.add_argument("--fresh", action="store_true", help="With --ledger: force a new bed for a verse used before")
    ap.add_argument("--vo", help="Voiceover to mix the bed under")
    ap.add_argument("--mix-out", help="Where to write the VO + bed mix")
    a = ap.parse_args()

    text = a.text or (Path(a.text_file).read_text() if a.text_file else "")
    duration = a.duration
    if a.format == "long" and duration is None:
        if not a.match:
            ap.error("long format needs --duration or --match")
        duration = ffprobe_seconds(a.match)

    variant, is_new = a.variant, False
    if a.ledger:
        ledger = load_ledger(a.ledger)
        mood, _ = detect_mood(text, a.mood)
        dur = SHORT_SECONDS if a.format == "short" else duration
        variant, is_new = choose_variant(ledger, a.ref, a.format, mood, dur, a.fresh)

    x, meta = build(a.ref, text, a.format, duration, variant, a.mood, a.sr)
    if a.ledger and is_new:
        clash = [e for e in ledger["entries"] if e["audio_fingerprint"] == meta["audio_fingerprint"]]
        if clash:
            sys.exit(f"audio identical to an earlier bed ({clash[0]['ref']}); refusing to reuse it")
        record(ledger, a.ledger, meta)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_wav(out, x, a.sr)
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    if a.preview:
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(out), "-c:a", "aac", "-b:a", "192k",
                        str(out.with_suffix(".m4a"))], check=True)
    if a.vo:
        mix_path = a.mix_out or out.with_name(out.stem + "_vo_mix.wav")
        mix_under_vo(a.vo, out, mix_path)
        if a.format == "short":
            meta.update(loop_seam_qa(mix_path))
            if not meta["loop_seam_ok"]:
                print(json.dumps(meta))
                sys.exit("loop seam QA failed: the Short would pop where it loops")
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
