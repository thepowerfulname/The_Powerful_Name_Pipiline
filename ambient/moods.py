"""Verse-tone tagging and sound palettes for The Powerful Name ambient beds.

tag_mood() reads the verse text and returns the closest mood.
PALETTES holds the sound recipe for each mood. The generator draws
random variations inside each recipe, so every Short and every Sunday
long video gets its own bed while staying in the channel's sound family.
"""
import re

# Scale intervals in semitones
SCALES = {
    "major":      [0, 2, 4, 5, 7, 9, 11],
    "lydian":     [0, 2, 4, 6, 7, 9, 11],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "dorian":     [0, 2, 3, 5, 7, 9, 10],
    "minor":      [0, 2, 3, 5, 7, 8, 10],
    "phrygian":   [0, 1, 3, 5, 7, 8, 10],
}

# Chords are written as scale-degree stacks (0 = tonic).
I, ii, iii, IV, V, vi, vii = ([d, d + 2, d + 4] for d in range(7))
Isus2, Isus4 = [0, 1, 4], [0, 3, 4]
Iadd9 = [0, 2, 4, 8]
IVadd9 = [3, 5, 7, 11]
IImaj = [1, 3, 5]          # lydian II (major)
bVII = [6, 8, 10]          # mixolydian/minor flat seven
i_, iv_, v_, VI_, III_, VII_ = I, IV, V, vi, iii, vii  # minor-scale readings


PALETTES = {
    "praise": {
        "scale": "major", "roots": [50, 52, 55, 53],
        "progressions": [[I, V, vi, IV, Iadd9], [I, IV, vi, V, I], [Iadd9, vi, IV, V, I],
                         [I, iii, IV, V, Iadd9]],
        "bright": 0.68, "rt": 4.5, "wet": 0.38,
        "layers": {"pad": 0.55, "strings": 0.30, "drone": 0.22, "bells": 0.42, "shimmer": 0.12},
        "optional": {"wind": 0.10},
        "bell_density": 0.32, "bell_octave": 24, "pulse": None,
    },
    "awe": {
        "scale": "lydian", "roots": [48, 50, 53],
        "progressions": [[I, IImaj, I, vi, Iadd9], [Isus2, IImaj, vii, I, IImaj],
                         [I, iii, IImaj, I, Isus2]],
        "bright": 0.52, "rt": 7.0, "wet": 0.52,
        "layers": {"pad": 0.60, "strings": 0.22, "drone": 0.32, "bells": 0.22, "shimmer": 0.20},
        "optional": {"wind": 0.12},
        "bell_density": 0.10, "bell_octave": 24, "pulse": None,
    },
    "comfort": {
        "scale": "major", "roots": [48, 53, 51],
        "progressions": [[I, IV, I, vi, IVadd9], [I, vi, IV, I, Isus4], [IVadd9, I, vi, IV, I]],
        "bright": 0.34, "rt": 4.2, "wet": 0.40,
        "layers": {"pad": 0.65, "strings": 0.28, "drone": 0.20, "bells": 0.24, "shimmer": 0.06},
        "optional": {"wind": 0.08},
        "bell_density": 0.14, "bell_octave": 12, "pulse": None,
    },
    "hope": {
        "scale": "major", "roots": [50, 52, 55],
        "progressions": [[vi, IV, I, V, Iadd9], [IV, V, vi, IV, I], [ii, IV, V, I, Iadd9]],
        "bright": 0.56, "rt": 4.8, "wet": 0.40,
        "layers": {"pad": 0.55, "strings": 0.36, "drone": 0.18, "bells": 0.32, "shimmer": 0.12},
        "optional": {"wind": 0.08},
        "bell_density": 0.22, "bell_octave": 24, "pulse": None,
    },
    "lament": {
        "scale": "minor", "roots": [45, 47, 50],
        "progressions": [[i_, VI_, iv_, v_, i_], [i_, iv_, VI_, III_, i_], [i_, VII_, VI_, iv_, i_]],
        "bright": 0.24, "rt": 5.8, "wet": 0.48,
        "layers": {"pad": 0.55, "strings": 0.34, "drone": 0.38, "bells": 0.12, "wind": 0.20},
        "optional": {"shimmer": 0.05},
        "bell_density": 0.05, "bell_octave": 12, "pulse": None,
    },
    "repentance": {
        "scale": "dorian", "roots": [45, 50, 48],
        "progressions": [[i_, IV, i_, bVII, i_], [i_, iv_, IV, i_, bVII], [i_, bVII, IV, i_, Isus2]],
        "bright": 0.30, "rt": 5.2, "wet": 0.46,
        "layers": {"pad": 0.60, "strings": 0.22, "drone": 0.32, "bells": 0.14, "wind": 0.14},
        "optional": {"shimmer": 0.06},
        "bell_density": 0.07, "bell_octave": 12, "pulse": None,
    },
    "warning": {
        "scale": "phrygian", "roots": [43, 45],
        "progressions": [[i_, [1, 3, 5], i_, VII_, i_], [i_, VI_, [1, 3, 5], i_, iv_]],
        "bright": 0.20, "rt": 5.0, "wet": 0.40,
        "layers": {"pad": 0.45, "strings": 0.30, "drone": 0.45, "wind": 0.18, "pulse": 0.38},
        "optional": {"bells": 0.08},
        "bell_density": 0.04, "bell_octave": 12, "pulse": ("heart", 2),
    },
    "victory": {
        "scale": "mixolydian", "roots": [50, 52, 55],
        "progressions": [[I, bVII, IV, I, Iadd9], [I, IV, bVII, IV, I], [I, V, bVII, IV, I]],
        "bright": 0.74, "rt": 4.0, "wet": 0.34,
        "layers": {"pad": 0.48, "strings": 0.42, "drone": 0.25, "bells": 0.30, "pulse": 0.34},
        "optional": {"shimmer": 0.10},
        "bell_density": 0.20, "bell_octave": 24, "pulse": ("timpani", 2),
    },
    "rest": {
        "scale": "major", "roots": [48, 53, 55],
        "progressions": [[I, IVadd9, I, Isus2, IVadd9], [Iadd9, IV, I, vi, I], [I, IVadd9, vi, IVadd9, I]],
        "bright": 0.30, "rt": 5.5, "wet": 0.46,
        "layers": {"pad": 0.62, "drone": 0.18, "bells": 0.20, "wind": 0.16, "shimmer": 0.08},
        "optional": {"strings": 0.14},
        "bell_density": 0.08, "bell_octave": 24, "pulse": None,
    },
    "wisdom": {
        "scale": "dorian", "roots": [50, 52, 48],
        "progressions": [[i_, IV, i_, bVII, IV], [i_, ii, IV, i_, Isus2], [i_, bVII, IV, IV, i_]],
        "bright": 0.44, "rt": 4.6, "wet": 0.40,
        "layers": {"pad": 0.55, "strings": 0.20, "drone": 0.26, "bells": 0.34, "shimmer": 0.08},
        "optional": {"wind": 0.08},
        "bell_density": 0.24, "bell_octave": 12, "pulse": None,
    },
}

LEXICON = {
    "praise": ["praise", "praises", "sing", "song", "songs", "rejoice", "glory", "glorious", "worship",
               "bless", "blessed", "thanks", "thanksgiving", "exalt", "hallelujah", "alleluia", "magnify",
               "joy", "joyful", "declare", "honour", "honor", "extol", "shout", "celebrate", "generation"],
    "awe": ["majesty", "majestic", "mighty", "wondrous", "wonders", "wonderful", "marvelous", "marvellous",
            "holy", "heavens", "throne", "awesome", "splendor", "splendour", "almighty", "stars",
            "created", "creation", "firmament", "fear of the lord", "great is", "power", "works"],
    "comfort": ["comfort", "shepherd", "refuge", "fear not", "do not fear", "be not afraid", "help",
                "heal", "healeth", "carry", "beside", "protect", "shelter", "compassion", "lovingkindness",
                "steadfast love", "near", "upholds", "keep thee", "keeps", "with you", "with thee"],
    "hope": ["hope", "promise", "future", "new", "morning", "wait", "renew", "rise", "dawn",
             "faithful", "faithfulness", "trust", "light", "eagles", "plans"],
    "lament": ["why", "forsaken", "sorrow", "weep", "mourn", "grief", "cry", "how long", "darkness",
               "afflicted", "affliction", "broken", "trouble", "distress", "pit", "tears", "groan", "sigh"],
    "repentance": ["sin", "sins", "iniquity", "transgression", "transgressions", "forgive", "forgiveness",
                   "repent", "wash", "clean", "confess", "contrite", "create in me", "turn from", "return unto"],
    "warning": ["wrath", "judgment", "judgement", "wicked", "destroy", "destruction", "beware", "perish",
                "fire", "flee", "sword", "vengeance", "woe", "abomination", "terror"],
    "victory": ["victory", "overcome", "overcometh", "triumph", "conquer", "conquerors", "battle", "shield",
                "deliver", "deliverer", "salvation", "warrior", "armor", "armour", "banner", "strength",
                "fortress", "enemies", "rock"],
    "rest": ["rest", "peace", "still", "quiet", "quietness", "sleep", "green pastures", "still waters",
             "lie down", "be still", "sabbath", "gentle", "calm", "restoreth", "restores"],
    "wisdom": ["wisdom", "wise", "understanding", "knowledge", "instruction", "teach", "path", "paths",
               "counsel", "law", "statutes", "precepts", "learn", "discern", "lamp", "commandments", "word"],
}

# Phrases count double: they carry more tone than a single word.
def tag_mood(text):
    """Return (mood, scores) for a verse. Ties fall back to the order in LEXICON."""
    low = " " + re.sub(r"[^a-z' ]+", " ", text.lower()) + " "
    scores = {}
    for mood, words in LEXICON.items():
        s = 0.0
        for w in words:
            hits = len(re.findall(r"(?<![a-z])" + re.escape(w) + r"(?![a-z])", low))
            s += hits * (2.0 if " " in w else 1.0)
        scores[mood] = s
    best = max(scores, key=lambda m: scores[m])
    if scores[best] == 0:
        best = "awe"
    return best, scores
