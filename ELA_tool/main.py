"""
main.py
-------
Kommandozeilenschnittstelle fuer das ELA-Tool
"""

import argparse
import sys
import os


def cmd_encode(args):
    from encoder import encode
    from decoder import decode

    """
    Importiert encode und decode und kodiert das Eingabebild mit 
    gegebener Qualitaetsstufe (quality). 
    Dekodiert das Ergebnis direkt wieder und speichert es als 
    'reproduziertes' Bild mit dem Dateinamen wie 'bild_reproduced_q60.png'.
    --save_intermediates speichert Zwischenschritte
    --output_dir bestimmt  Ausgabeorder (default: ./output)
    """

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

    """
    Importiert ela und erzeugt das ELA-Bild. 
    Das Originalbild wird mit fester Qualitaet (--quality, Default: 75)
    komprimiert und die Differenz zum Original wird mit einem Kontrast-Multiplikator
    (--multiplier, Default: 30) sichtbar gemacht. 
    """

    generate_ela(
        args.image,
        quality=args.quality,
        multiplier=args.multiplier,
        output_dir=args.output_dir,
        save_intermediates=args.save_intermediates,
    )

def cmd_test(_args):
    """
    Fuehrt Selbsttest fuer die einzelnen Module (quantization, colorspace, dct_blocks)
    durch. 
    Jedes Modul wird als eigene Python_Subprozess ausgefuehrt (subprocess.run).
    Gibt eine Statusmeldung aus je nach Exit-Code ()
    """
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

    """
    Baut das komplette ArgumentParser mit allen Subcommands und Optionen auf.
    Jeder Subparser bekommt via 'set_defaults(func=..)' seine zugehoerige Funktion zugewiesen. 
    """

    parser = argparse.ArgumentParser(
        prog="ela_tool",
        description="JPEG-based ELA Tool – Datenkompression Praktikum SoSe 2026",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── encode sub-command ──────────────────────────────────────────────────
    p_enc = sub.add_parser("encode", help="Bild kodieren und dekodieren")
    p_enc.add_argument("image",           help="Pfad zum Eingabebild (PNG oder JPEG)")
    p_enc.add_argument("--quality", "-q", type=float, default=50.0,
                       help="Qualitätswert Q in (0, 100]  (Standard: 50)")
    p_enc.add_argument("--save-intermediates", "-s", action="store_true",
                       help="Zwischendateien und Komponentenbilder speichern")
    p_enc.add_argument("--output-dir", "-o", default="output",
                       help="Ausgabeordner  (Standard: ./output)")
    p_enc.set_defaults(func=cmd_encode)

    # ── ela sub-command ─────────────────────────────────────────────────────
    p_ela = sub.add_parser("ela", help="ELA-Bild erzeugen")
    p_ela.add_argument("image",             help="Pfad zum Eingabebild (JPEG empfohlen)")
    p_ela.add_argument("--quality",  "-q",  type=float, default=75.0,
                       help="Qualitätswert Q für die erneute Kodierung  (Standard: 75)")
    p_ela.add_argument("--multiplier", "-m", type=float, default=30.0,
                       help="Kontrastmultiplikator M  (Standard: 30)")
    p_ela.add_argument("--save-intermediates", "-s", action="store_true",
                       help="Zusätzlich Zwischendateien der Kodierung speichern")
    p_ela.add_argument("--output-dir", "-o", default="output",
                       help="Ausgabeordner  (Standard: ./output)")
    p_ela.set_defaults(func=cmd_ela)

    # ── test sub-command ────────────────────────────────────────────────────
    p_test = sub.add_parser("test", help="Selbsttests der Module ausführen")
    p_test.set_defaults(func=cmd_test)

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
