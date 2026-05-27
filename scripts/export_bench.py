#!/usr/bin/env python3
"""
export_bench - run benchmark_e2e on-device and output benchmark.json files.

The universal-swipe model-card renderer (scripts/model_card_pdf.py) reads an
optional `benchmark.json` sitting next to each model's `metadata.json`. This
script is the bridge: it runs the on-device benchmark for every selected CPU
affinity (big, little) in both modes (encoder-only and encoder+decoder), then
drops a sidecar into both `models/<codename>/` here and the matching
`exports/<codename>/` directory in ../universal-swipe.

Schema (dual: flat fields match the current universal-swipe card renderer,
`affinities` carries the per-cluster detail for richer cards later):
  {
    "device": "google Pixel 4", "soc": "flame", "abi": "arm64-v8a",
    "precision": "fp16", "threads": 1, "beam_width": 100, "runs": 500,
    "mode": "encoder_only" | "encoder_plus_decoder",

    # Primary affinity (little), flat for compatibility:
    "resample_us": {p50, p99}, "model_us": {p50, p99},
    "beam_search_us": {p50, p99}, "total_us": {p50, p99},
    "init_ms": float, "rss_model_mb": float,
    "primary_affinity": "little",

    # Per-affinity detail:
    "affinities": {
      "little": { stage dicts + init_ms + rss_model_mb },
      "big":    { ... }
    }
  }

Field mapping (ml-inference benchmark_e2e --json -> model card):
  encoder card (encoder-only run):
    resample_us      ← resample_us
    model_us         ← encoder_us
    beam_search_us   ← beam_us
    total_us         ← total_us
  decoder card (encoder+decoder run):
    resample_us      ← resample_us
    encoder_us       ← encoder_us (paired encoder stage)
    model_us         ← decoder_us (only the refiner cost)
    beam_search_us   ← beam_us
    total_us         ← total_us (full pipeline)

Assumes `make push-demo` has staged the models on device. Usage:
  scripts/export_bench.py --encoder honorable_sturgeon --decoder magic_macaw
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEVICE_DIR = "/data/local/tmp/ml-inference"


def adb(*args: str) -> str:
    cmd = ["adb", *args]
    try:
        out = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        sys.stderr.write(f"adb failed: {' '.join(cmd)}\n{e.stderr}\n")
        raise
    return out.stdout.strip()


def device_info() -> dict:
    brand = adb("shell", "getprop", "ro.product.brand")
    model = adb("shell", "getprop", "ro.product.model")
    soc = adb("shell", "getprop", "ro.hardware") or adb("shell", "getprop", "ro.board.platform")
    abi = adb("shell", "getprop", "ro.product.cpu.abi")
    return {"device": f"{brand} {model}".strip(), "soc": soc, "abi": abi}


def run_benchmark(encoder: str, encoder_precision: str,
                  decoder: str | None, decoder_precision: str,
                  vocab: str, layout: str,
                  threads: int, warmup: int, runs: int, affinity: str) -> dict:
    args = [
        f"--encoder {encoder}/model_{encoder_precision}.pte",
        f"--vocab {vocab}",
        f"--layout {layout}",
        f"--threads {threads}",
        f"--warmup {warmup}",
        f"--runs {runs}",
        f"--affinity {affinity}",
        "--json",
    ]
    if decoder:
        args.append(f"--decoder {decoder}/model_{decoder_precision}.pte")
    cmd = f"cd {DEVICE_DIR} && LD_LIBRARY_PATH=. ./benchmark_e2e " + " ".join(args)
    out = adb("shell", cmd)
    start = out.find("{")
    if start < 0:
        raise RuntimeError(f"no JSON found in benchmark output:\n{out}")
    return json.loads(out[start:])


def _stats(s: dict | None) -> dict:
    if not s:
        return {}
    return {"p50": float(s.get("p50", 0.0)), "p99": float(s.get("p99", 0.0))}


def _encoder_affinity_block(bench: dict) -> dict:
    return {
        "resample_us": _stats(bench.get("resample_us")),
        "model_us": _stats(bench.get("encoder_us")),
        "beam_search_us": _stats(bench.get("beam_us")),
        "total_us": _stats(bench.get("total_us")),
        "init_ms": bench.get("init_ms"),
        "rss_model_mb": bench.get("rss_after_init_mb"),
    }


def _decoder_affinity_block(bench: dict) -> dict:
    return {
        "resample_us": _stats(bench.get("resample_us")),
        "encoder_us": _stats(bench.get("encoder_us")),
        "model_us": _stats(bench.get("decoder_us")),
        "beam_search_us": _stats(bench.get("beam_us")),
        "total_us": _stats(bench.get("total_us")),
        "init_ms": bench.get("init_ms"),
        "rss_model_mb": bench.get("rss_after_init_mb"),
    }


def _card_header(bench: dict, dev: dict, precision: str, mode: str) -> dict:
    return {
        "device": dev["device"],
        "soc": dev["soc"],
        "abi": dev["abi"],
        "precision": precision,
        "threads": bench.get("threads"),
        "beam_width": bench.get("beam_width"),
        "runs": bench.get("runs"),
        "mode": mode,
    }


def write_sidecar(card: dict, codename: str, universal_swipe: Path | None) -> list[Path]:
    written = []
    local = REPO_ROOT / "models" / codename / "benchmark.json"
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text(json.dumps(card, indent=2) + "\n")
    written.append(local)
    if universal_swipe is not None:
        remote = universal_swipe / "exports" / codename / "benchmark.json"
        if remote.parent.is_dir():
            shutil.copy2(local, remote)
            written.append(remote)
        else:
            sys.stderr.write(f"skip {remote} (export dir not found)\n")
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", default="honorable_sturgeon", help="encoder codename")
    ap.add_argument("--decoder", default="magic_macaw", help="decoder codename (empty to skip)")
    ap.add_argument("--vocab", default="vocabs/en.combined")
    ap.add_argument("--layout", default="layouts/en_qwerty.json")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--runs", type=int, default=500)
    ap.add_argument("--affinities", default="big,little",
                    help="comma-separated CPU clusters to benchmark (default: big,little)")
    ap.add_argument("--encoder-precision", default="mixed",
                    help="encoder .pte precision suffix - model_<prec>.pte (default: mixed)")
    ap.add_argument("--decoder-precision", default="fp16",
                    help="decoder .pte precision suffix - model_<prec>.pte (default: fp16)")
    ap.add_argument("--universal-swipe", type=Path, default=REPO_ROOT.parent / "universal-swipe",
                    help="path to universal-swipe checkout (benchmark.json copied into exports/ if present)")
    args = ap.parse_args()

    affinities = [a.strip() for a in args.affinities.split(",") if a.strip()]
    for a in affinities:
        if a not in ("big", "little"):
            sys.exit(f"unknown affinity: {a}")

    dev = device_info()
    print(f"device: {dev['device']} ({dev['soc']}, {dev['abi']})")
    print(f"affinities: {affinities}")

    us_dir = args.universal_swipe if args.universal_swipe and args.universal_swipe.is_dir() else None
    if us_dir is None:
        print(f"note: universal-swipe dir {args.universal_swipe} not found - writing local output only")

    # Primary affinity (flat fields use this). "little" matches production.
    primary = "little" if "little" in affinities else affinities[0]

    # Encoder-only runs (encoder card reports its own precision)
    print(f"\n== encoder-only ({args.encoder}, {args.encoder_precision}) ==")
    enc_card = None
    for aff in affinities:
        print(f"  run: affinity={aff}")
        b = run_benchmark(args.encoder, args.encoder_precision,
                          None, args.decoder_precision,
                          args.vocab, args.layout,
                          args.threads, args.warmup, args.runs, aff)
        if enc_card is None:
            enc_card = _card_header(b, dev, args.encoder_precision, "encoder_only")
            enc_card["affinities"] = {}
        enc_card["affinities"][aff] = _encoder_affinity_block(b)
    enc_card.update(enc_card["affinities"][primary])
    enc_card["primary_affinity"] = primary
    for p in write_sidecar(enc_card, args.encoder, us_dir):
        print(f"  wrote {p}")

    # Encoder + decoder runs (decoder card reports its own precision; full pipeline
    # runs the encoder at encoder_precision so decoder numbers are realistic)
    if args.decoder:
        print(f"\n== encoder ({args.encoder_precision}) + decoder ({args.decoder}, {args.decoder_precision}) ==")
        dec_card = None
        for aff in affinities:
            print(f"  run: affinity={aff}")
            b = run_benchmark(args.encoder, args.encoder_precision,
                              args.decoder, args.decoder_precision,
                              args.vocab, args.layout,
                              args.threads, args.warmup, args.runs, aff)
            if dec_card is None:
                dec_card = _card_header(b, dev, args.decoder_precision, "encoder_plus_decoder")
                dec_card["encoder_precision"] = args.encoder_precision
                dec_card["affinities"] = {}
            dec_card["affinities"][aff] = _decoder_affinity_block(b)
        dec_card.update(dec_card["affinities"][primary])
        dec_card["primary_affinity"] = primary
        for p in write_sidecar(dec_card, args.decoder, us_dir):
            print(f"  wrote {p}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
