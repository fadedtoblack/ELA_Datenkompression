"""
ela.py
------
Erzeugung von Error-Level-Analysis-(ELA-)Bildern für das ELA-Tool.

    ELA_pixel = clip( |input_pixel - reproduced_pixel| * M, 0, 255 )

wobei M einen einstellbaren Kontrastmultiplikator darstellt.

Das ELA-Bild wird gespeichert, wobei der Qualitaetswert und der
Multiplikator im Dateinamen angegeben werden.
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
                save_output: bool = True) -> np.ndarray: # damit die GUI nicht automatisch speichert nach jedem Slider-Update
    """
    Erzeugt das ELA-Bild für das Eingabebild.

    Ablauf:
        1. Kodiert das Eingabebild mit der angegebenen Qualitaet Q (quality)
        2. Dekodiert das Bild, um das rekonstruierte Bild zu erhalten
        3. Berechnet die pixelweise absolute Differenz, skaliert mit M (multiplier)
        4. Begrenzt die Werte auf [0, 255] und speichert das Ergebnis als PNG

    Parameter
    ----------
    image_path         : str -> Dateipfad des Eingabebildes (JPEG empfohlen)
    quality            : float -> JPEG-Qualitaetswert Q (0, 100]
    multiplier         : float -> Kontrastverstaerkungsmultiplikator M
    output_dir         : str -> Pfad für die Ausgabebilder
    save_intermediates : bool -> speichert zusätzlich temporaere Textdateien des Encoders
    save_output        : bool -> speichert das erzeugte ELA-Bild

    Ausgabe
    -------
    ela_image : np.ndarray, shape (H, W, 3), dtype uint8
    psnr      : float, PSNR-Wert in dB zwischen Originalbild und rekonstruiertem Bild
    """
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    print(f"\n{'='*60}")
    print(f"ELA: {base_name}  |  Q={quality:.0f}%  |  M={multiplier:.0f}")
    print(f"{'='*60}")

    def _intermediates_dir(output_dir: str, base_name:str, quality: float) -> str:
        """
        Erstellt den Pfad für den Unterordner der Zwischenschritte und legt 
        diesen bei Bedarf an, z.B.:
            <output_dir>/base_name>_intermediates_q75/
        """

        path = os.path.join(output_dir, f"{base_name}_intermediates_q{quality:.0f}")
        os.makedirs(path, exist_ok=True)
        return path

 
    # --------- 1 & 2. Kodierung + Dekodierung -------------------------------------
    """
    Startet den Kodierungsprozess anhand der übergebenen Parameter. Das Ergebnis 
    wird in der Variable enc_result gespeichert und anschließend ausgegeben.
    """

    enc_result = encode(
        image_path,
        quality=quality,
        save_intermediates=save_intermediates,
        output_dir=_intermediates_dir(output_dir, base_name, quality) if save_intermediates else output_dir
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
        print(f"\n[ELA] ELA-Bild gespeichert: {ela_path}")
        print(f"[ELA] PSNR (Originalbild vs. rekonstruiertes Bild): {psnr:.4f} dB")

    else:
        # Hinweis, wenn ELA-Bild nicht gespeichert wird (z.B. bei GUI-Slider-Updates)
        print(f"\n[ELA] save_output=False -> ELA wird nicht auf der Festplatte gespeichert.")
        print(f"[ELA] PSNR (Originalbild vs. rekonstruiertes Bild): {psnr:.4f} dB")

    return ela_image, psnr
