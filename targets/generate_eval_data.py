#!/usr/bin/env python3
"""
Generate labeled evaluation dataset for swipe keyboard accuracy testing.

Creates synthetic swipe paths for words from the vocabulary by tracing
through QWERTY key center positions with interpolation and optional noise.

The keyboard layout matches KeyboardView.kt (model-space coordinates):
  Row 0 (QWERTYUIOP): y = 1/6,  x starts at 0.05, spacing 0.10
  Row 1 (ASDFGHJKL):  y = 1/2,  x starts at 0.10, spacing 0.10
  Row 2 (ZXCVBNM):    y = 5/6,  x starts at 0.20, spacing 0.10

Output: binary file for eval_accuracy (see format in eval_accuracy.cpp header).

Usage:
  python3 targets/generate_eval_data.py <vocab_file> <output.bin> [OPTIONS]

Options:
  --num-words N       Number of words to sample (default: 1000)
  --min-length N      Minimum word length (default: 3)
  --max-length N      Maximum word length (default: 12)
  --points-per-key N  Interpolated points between keys (default: 8)
  --noise F           Gaussian noise stddev in normalized coords (default: 0.015)
  --seed N            Random seed (default: 42)
  --top-freq          Sample from most frequent words (default: uniform)
  --word-list FILE    Use specific word list (one word per line) instead of sampling
"""

import argparse
import random
import struct
import sys
from pathlib import Path

# QWERTY layout in model-space coordinates [0, 1]
# Matches KeyboardView.kt: x = (offset + col + 0.5) / 10, y = (row + 0.5) / 3
KEY_POSITIONS = {}

ROW0 = "qwertyuiop"
ROW1 = "asdfghjkl"
ROW2 = "zxcvbnm"

for i, ch in enumerate(ROW0):
    KEY_POSITIONS[ch] = ((0.0 + i + 0.5) / 10.0, (0 + 0.5) / 3.0)

for i, ch in enumerate(ROW1):
    KEY_POSITIONS[ch] = ((0.5 + i + 0.5) / 10.0, (1 + 0.5) / 3.0)

for i, ch in enumerate(ROW2):
    KEY_POSITIONS[ch] = ((1.5 + i + 0.5) / 10.0, (2 + 0.5) / 3.0)


def load_vocab_combined(path):
    """Load words + frequencies from no_abbr2.combined format.

    Lines look like: ' word=the,f=222,flags=,originalFreq=222'
    (leading space, then 'word=<word>,f=<freq>,...')
    """
    words = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("dictionary="):
                continue
            # Parse ' word=<word>,<key>=<val>,<key>=<val>,...'
            if not line.startswith("word="):
                continue
            # Split into comma-separated fields
            fields = line.split(",")
            word = None
            freq = 1.0
            for field in fields:
                kv = field.split("=", 1)
                if len(kv) != 2:
                    continue
                key, val = kv
                if key == "word":
                    word = val
                elif key == "f":
                    try:
                        freq = float(val)
                    except ValueError:
                        pass
            if not word:
                continue
            # Extract alpha-only version for matching
            alpha = "".join(c for c in word.lower() if c.isalpha())
            if not alpha:
                continue
            # Only keep words that are pure lowercase alpha (swipeable)
            if alpha == word.lower():
                words[alpha] = freq
    return words


def load_vocab_plain(path):
    """Load words from tab-separated vocab (word<TAB>freq)."""
    words = {}
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            word = parts[0].lower().strip()
            freq = float(parts[1]) if len(parts) > 1 else 1.0
            alpha = "".join(c for c in word if c.isalpha())
            if alpha == word and alpha:
                words[alpha] = freq
    return words


def load_vocab(path):
    """Auto-detect and load vocabulary."""
    with open(path, "r") as f:
        first = f.read(11)
    if first.startswith("dictionary="):
        return load_vocab_combined(path)
    return load_vocab_plain(path)


def generate_swipe_path(word, points_per_key, noise_std, rng):
    """Generate a synthetic swipe path for a word.

    Produces a smooth path through key centers with Catmull-Rom interpolation
    and optional Gaussian noise.
    """
    # Get key positions
    centers = []
    for ch in word:
        if ch not in KEY_POSITIONS:
            return None  # Skip words with non-QWERTY chars
        centers.append(KEY_POSITIONS[ch])

    if len(centers) < 2:
        return None

    # Generate interpolated path using Catmull-Rom spline
    xs, ys, ts = [], [], []
    time_ms = 0.0
    dt_per_point = 8.0  # ~120 Hz touch sampling

    for seg in range(len(centers) - 1):
        # Control points for Catmull-Rom
        p0 = centers[max(0, seg - 1)]
        p1 = centers[seg]
        p2 = centers[seg + 1]
        p3 = centers[min(len(centers) - 1, seg + 2)]

        n = points_per_key
        for j in range(n):
            t = j / n
            t2 = t * t
            t3 = t2 * t

            # Catmull-Rom basis
            x = 0.5 * (
                (2 * p1[0])
                + (-p0[0] + p2[0]) * t
                + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2
                + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3
            )
            y = 0.5 * (
                (2 * p1[1])
                + (-p0[1] + p2[1]) * t
                + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2
                + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3
            )

            # Add noise
            if noise_std > 0:
                x += rng.gauss(0, noise_std)
                y += rng.gauss(0, noise_std)

            # Clamp
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))

            xs.append(x)
            ys.append(y)
            ts.append(time_ms)
            time_ms += dt_per_point

    # Add final key center
    fx, fy = centers[-1]
    if noise_std > 0:
        fx += rng.gauss(0, noise_std)
        fy += rng.gauss(0, noise_std)
    xs.append(max(0.0, min(1.0, fx)))
    ys.append(max(0.0, min(1.0, fy)))
    ts.append(time_ms)

    return xs, ys, ts


def main():
    parser = argparse.ArgumentParser(description="Generate swipe evaluation dataset")
    parser.add_argument("vocab", help="Vocabulary file (combined or plain)")
    parser.add_argument("output", help="Output binary file")
    parser.add_argument("--num-words", type=int, default=1000)
    parser.add_argument("--min-length", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=12)
    parser.add_argument("--points-per-key", type=int, default=8)
    parser.add_argument("--noise", type=float, default=0.015)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--top-freq", action="store_true", help="Sample most frequent words"
    )
    parser.add_argument("--word-list", help="Use specific word list file")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Load vocabulary
    print(f"Loading vocabulary: {args.vocab}")
    vocab = load_vocab(args.vocab)
    print(f"  {len(vocab)} swipeable words")

    # Select words
    if args.word_list:
        with open(args.word_list) as f:
            words = [w.strip().lower() for w in f if w.strip()]
        words = [w for w in words if w in vocab and args.min_length <= len(w) <= args.max_length]
        print(f"  Using {len(words)} words from {args.word_list}")
    else:
        eligible = [
            (w, f)
            for w, f in vocab.items()
            if args.min_length <= len(w) <= args.max_length
        ]
        print(f"  {len(eligible)} eligible words (length {args.min_length}-{args.max_length})")

        if args.top_freq:
            eligible.sort(key=lambda x: -x[1])
            words = [w for w, _ in eligible[: args.num_words]]
        else:
            words = [w for w, _ in eligible]
            rng.shuffle(words)
            words = words[: args.num_words]

    # Generate paths
    print(f"Generating {len(words)} swipe paths (noise={args.noise}, ppk={args.points_per_key})...")
    samples = []
    for word in words:
        result = generate_swipe_path(word, args.points_per_key, args.noise, rng)
        if result is not None:
            xs, ys, ts = result
            samples.append((word, xs, ys, ts))

    print(f"  Generated {len(samples)} valid samples")

    # Write binary
    with open(args.output, "wb") as f:
        f.write(struct.pack("<I", len(samples)))
        for word, xs, ys, ts in samples:
            word_bytes = word.encode("ascii")
            f.write(struct.pack("<I", len(word_bytes)))
            f.write(word_bytes)
            n = len(xs)
            f.write(struct.pack("<I", n))
            f.write(struct.pack(f"<{n}f", *xs))
            f.write(struct.pack(f"<{n}f", *ys))
            f.write(struct.pack(f"<{n}f", *ts))

    size_kb = Path(args.output).stat().st_size / 1024
    print(f"Wrote {args.output} ({size_kb:.1f} KB, {len(samples)} samples)")


if __name__ == "__main__":
    main()
