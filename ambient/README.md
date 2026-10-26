# Ambient beds

Every Powerful Name Short and Sunday long video gets its own ambient bed, built from the verse's tone. Nothing is sampled or downloaded. The sound is synthesized here, so there's nothing for a Content ID claim to match.

Approved for autonomous mode: the nightly build runs this step on its own. The bed is reviewed as part of the finished Short, same as the visuals and VO.

## Nightly step

```bash
# Short (54 s, five sections lined up with the five logo turns, loops seamlessly)
python ambient/ambient_bed.py --ref "Psalm 145:4-5" --text-file build/verse.txt \
    --format short --out build/ambient.wav --vo build/vo.wav --mix-out build/audio_final.wav

# Sunday long video (matches the VO length, eases in and out)
python ambient/ambient_bed.py --ref "Psalm 145:4-5" --text-file build/verse.txt \
    --format long --match build/long_vo.wav --out build/long_ambient.wav \
    --vo build/long_vo.wav --mix-out build/long_audio_final.wav
```

`audio_final.wav` goes to the render step. A sidecar `ambient.json` records the mood, key and layers used.

## How it decides the sound

1. `moods.py` scores the verse text against ten tones: praise, awe, comfort, hope, lament, repentance, warning, victory, rest, wisdom.
2. Each tone has a palette: scale, chord progressions, brightness, reverb size, and which layers play (pad, strings, drone, bells, shimmer, wind, low pulse).
3. The verse reference and format seed the variations inside that palette (key, progression order, voicings, bell placement, layer balance, room size). A Short and a long video on the same verse sound different. The same verse re-rendered for a fix sounds the same.
4. The bed sits under the VO and ducks whenever the voice speaks.

## Overrides

- `--mood lament` forces a tone if the tagger guesses wrong.
- `--variant 1` gives a fresh take on the same verse, e.g. when a verse comes back months later.

Requires Python 3 with numpy and scipy, plus ffmpeg.
