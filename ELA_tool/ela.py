"""
ela.py
------
Error-Level Analysis (ELA) image generation for the ELA-Tool (SoSe 2026).

Section 2.4 of the assignment spec:

    ELA_pixel = clip( |input_pixel - reproduced_pixel| * M, 0, 255 )

where M is an adjustable contrast multiplier.

The ELA image is saved with the quality value and multiplier encoded in
the filename.
"""

import os
import numpy as np
from PIL import Image

from colorspace import load_image
from encoder    import encode
from decoder    import decode, compute_psnr


def generate_ela(image_path: str,
                quality: float = 75.0,
                multiplier: float = 30.0,
                output_dir: str = "ELA_tool\output",
                save_intermediates: bool = False,
                save_output: bool = True) -> np.ndarray: # NEUER PARAMETER damit die GUI nicht automatisch speichert nach jedem Slider-Update
    """
    Generiert das ELA-Bild fuer das Input-Bild

    Ablauf:
        1. Encoded das Input-Bild mit der angegeben Qualitaet G (quality)
        2. Decodiert um das reproduzierte Bild zu erhalten
        3. Berechnet die pixel-weise abolute Differenz, skaliert durch M (multiplier)
        4. Begrenzen auf [0, 255] und abbspeichern als PNG

    Parameter
    ----------
    image_path         : str -> Dateipfad des Input-Bildes (JPEG ist empfohlen)
    quality            : float -> JPEG-Qualitaets-Wert Q (0, 100]
    multiplier         : float -> Kontrastverstaerkungsmultiplier M
    output_dir         : str -> Pfad fuer die Output-Bilder
    save_intermediates : bool -> speichert auch temporaere Techtdateien des Encoders
    save_output        : bool -> speichert das erzeugte ELA-Bild
    Ausgabe
    -------
    ela_image : np.ndarray, shape (H, W, 3), dtype uint8
    """
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    print(f"\n{'='*60}")
    print(f"ELA: {base_name}  |  Q={quality:.0f}%  |  M={multiplier:.0f}")
    print(f"{'='*60}")

 
    # --------- 1 & 2. Encoden + Decoden -------------------------------------
    """
    Beginn des Encoding-Prozess anhand der uebergebenen Parameter, das Ergebnis
    wird in Variable (enc_result) gespeichert und anschliessend ausgegeben.

    """

    enc_result = encode(
        image_path,
        quality=quality,
        save_intermediates=save_intermediates,
        output_dir=output_dir,
    )

    print(enc_result)

    repro_name = f"{base_name}_reproduced_q{quality:.0f}"
    reproduced_rgb, psnr = decode(enc_result,
                                   output_dir=output_dir,
                                   image_name=repro_name)

    # --------- 3. ELA Bild erstellen ------------------------------------------
    #   ELA_pixel = clip( |input - reproduced| * M, 0, 255 )

    original_rgb = enc_result.original_rgb.astype(np.float64)
    reproduced_f = reproduced_rgb.astype(np.float64)

    ela_float = np.abs(original_rgb - reproduced_f) * multiplier  # Eq. 2.8
    ela_image = np.clip(ela_float, 0.0, 255.0).astype(np.uint8)

    # --------- 4. ELA-Bild speichern -------------------------------------------

    if save_output:
        ela_name = f"{base_name}_ela_q{quality:.0f}_m{multiplier:.0f}.png"
        ela_path = os.path.join(output_dir, ela_name)
        Image.fromarray(ela_image, mode="RGB").save(ela_path)
        print(f"\n[ELA] ELA-Bild gespeichert im Ordner: {ela_path}")
        print(f"[ELA] PSNR (original vs reproduced): {psnr:.4f} dB")

    else:
        # Hinweis, wenn ELA-Bild nicht gespeichert wird (z.B. bei GUI-Slider-Updates)
        print(f"\n[ELA] save_output=False -> ELA image not written to disk")
        print(f"[ELA] PSNR (original vs reproduced): {psnr:.4f} dB")

    return ela_image
