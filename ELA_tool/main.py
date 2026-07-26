"""
main.py
-------
Command-line interface for the ELA-Tool (SoSe 2026).

Usage examples
--------------
# Run the full ELA pipeline on a JPEG image:
    python main.py ela input.jpg --quality 75 --multiplier 30

# Encode + decode only (with intermediate text files saved):
    python main.py encode input.png --quality 50 --save-intermediates

# Run built-in module self-tests:
    python main.py test
"""

import argparse
import sys
import os


def cmd_encode(args):
    from encoder import encode
    from decoder import decode

    enc = encode(
        args.image,
        quality=args.quality,
        save_intermediates=args.save_intermediates,
        output_dir=args.output_dir,
    )
    base = os.path.splitext(os.path.basename(args.image))[0]
    repro_name = f"{base}_reproduced_q{args.quality:.0f}"
    decode(enc, output_dir=args.output_dir, image_name=repro_name)


def cmd_ela(args):
    from ela import generate_ela

    generate_ela(
        args.image,
        quality=args.quality,
        multiplier=args.multiplier,
        output_dir=args.output_dir,
        save_intermediates=args.save_intermediates,
    )


def cmd_test(_args):
    """Run self-tests for each module."""
    import subprocess, sys

    modules = [
        "quantization",
        "colorspace",
        "dct_blocks",
    ]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for mod in modules:
        print(f"\n{'─'*50}")
        print(f"Running self-test: {mod}.py")
        print(f"{'─'*50}")
        result = subprocess.run(
            [sys.executable, os.path.join(script_dir, f"{mod}.py")],
            capture_output=False,
        )
        if result.returncode != 0:
            print(f"  ✗ {mod} self-test FAILED (exit code {result.returncode})")
        else:
            print(f"  ✓ {mod} self-test passed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ela_tool",
        description="JPEG-based ELA Tool – Datenkompression Praktikum SoSe 2026",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── encode sub-command ──────────────────────────────────────────────────
    p_enc = sub.add_parser("encode", help="Encode and decode an image")
    p_enc.add_argument("image",           help="Path to input image (PNG or JPEG)")
    p_enc.add_argument("--quality", "-q", type=float, default=50.0,
                       help="Quality value Q in (0, 100]  (default: 50)")
    p_enc.add_argument("--save-intermediates", "-s", action="store_true",
                       help="Save intermediate text files and component images")
    p_enc.add_argument("--output-dir", "-o", default="output",
                       help="Output directory  (default: ./output)")
    p_enc.set_defaults(func=cmd_encode)

    # ── ela sub-command ─────────────────────────────────────────────────────
    p_ela = sub.add_parser("ela", help="Generate an ELA image")
    p_ela.add_argument("image",             help="Path to input image (JPEG recommended)")
    p_ela.add_argument("--quality",  "-q",  type=float, default=75.0,
                       help="Quality value Q for re-encoding  (default: 75)")
    p_ela.add_argument("--multiplier", "-m", type=float, default=30.0,
                       help="Contrast multiplier M  (default: 30)")
    p_ela.add_argument("--save-intermediates", "-s", action="store_true",
                       help="Also save encoder intermediate text files")
    p_ela.add_argument("--output-dir", "-o", default="output",
                       help="Output directory  (default: ./output)")
    p_ela.set_defaults(func=cmd_ela)

    # ── test sub-command ────────────────────────────────────────────────────
    p_test = sub.add_parser("test", help="Run module self-tests")
    p_test.set_defaults(func=cmd_test)

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
