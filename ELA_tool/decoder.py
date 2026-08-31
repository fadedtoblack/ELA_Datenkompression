"""
decoder.py
----------
Pipeline:
    1. Quantisierte DCT-Koeffizienten vom Encoder uebernehmen
    2. Dequantisierung
    3. Inverse 2-D-DCT pro Block (IDCT, Level-Shift +128 wird innerhalb
       von apply_idct_to_blocks angewendet)
    4. 8×8-Bloecke wieder zu vollstaendigen Komponenten-Arrays zusammensetzen
    5. YCbCr -> RGB  (inverse Transformation gemaess ITU-R BT.601-4)
    6. Rekonstruiertes Bild als PNG speichern
    7. PSNR gegenueber dem Originalbild berechnen und anzeigen
"""

import os
import numpy as np
from PIL import Image

from colorspace import ycbcr_to_rgb
from dct_blocks import (dequantize_blocks, apply_idct_to_blocks,
                         merge_blocks)
from encoder    import EncoderResult


def decode(enc_result: EncoderResult,
           output_dir: str = "ELA_tool\output",
           image_name: str = "reproduced",
           save_output: bool = True) -> tuple[np.ndarray, float]: # damit die GUI nicht automatisch speichert nach jedem Slider-Update -> Default True wird nur bei benutzen der GUI auf False gesetzt
    """
    Fuehrt die vollstaendige Decoder-Pipeline unter Verwendung der Ausgabe des Encoders aus.

    Parameter
    ----------
    enc_result  : EncoderResult – Ausgabe von encoder.encode()
    output_dir  : str           – Verzeichnis fuer die Ausgabedateien
    image_name  : str           – Basisname für die gespeicherte PNG-Datei
                                  (ohne Dateiendung)

    Rueckgabe
    --------
    reproduced_rgb : np.ndarray, Form (H, W, 3), Datentyp uint8
    psnr           : float – PSNR in dB gegenüber dem Originalbild
    """
    os.makedirs(output_dir, exist_ok=True)

    print("[Decoder] ekodierungspipeline wird gestartet ...")

    qtables        = enc_result.qtables
    q_dct          = enc_result.q_dct
    orig_shapes    = enc_result.original_shapes

    components_out = {}

    for comp_name in ("Y", "Cb", "Cr"):

        # Dequantisierung
        dq_blks = dequantize_blocks(q_dct[comp_name], qtables[comp_name])

        # Inverse DCT  (Level-Shift +128 wird innerhalb von apply_idct_to_blocks angewendet)
        spatial_blks = apply_idct_to_blocks(dq_blks)

        # 8×8-Bloecke wieder zur vollstaendigen Komponente zusammensetzen
        component = merge_blocks(spatial_blks, orig_shapes[comp_name])
        components_out[comp_name] = component

    # YCbCr -> RGB  (inverse Transformation gemäß ITU-R BT.601-4)
    print("[Decoder] Inverse Farbraumtransformation YCbCr -> RGB ...")
    reproduced_rgb = ycbcr_to_rgb(
        components_out["Y"],
        components_out["Cb"],
        components_out["Cr"],
    )

    # Rekonstruiertes Bild als PNG speichern
    if save_output:
        out_path = os.path.join(output_dir, f"{image_name}.png")
        Image.fromarray(reproduced_rgb, mode="RGB").save(out_path)
        print(f"[Decoder] Rekonstruiertes Bild gespeichert: {out_path}")
    else:
         print("[Decoder] save_output=False -> Rekonstruiertes Bild wird nicht auf der Festplatte gespeichert")

    # 6. PSNR berechnen
    if enc_result.original_rgb is not None:
        psnr = compute_psnr(enc_result.original_rgb, reproduced_rgb)
        print(f"[Decoder] PSNR: {psnr:.4f} dB")
    else:
        psnr = float("nan")
        print("[Decoder] Warnung: Originalbild nicht verfügbar, PSNR wird nicht berechnet.")

    print("[Decoder] Abgeschlossen.")
    return reproduced_rgb, psnr


def compute_psnr(original: np.ndarray, reproduced: np.ndarray,
                 peak: float = 255.0) -> float:
    """
    Berechnet das Peak Signal-to-Noise Ratio (PSNR) zwischen zwei RGB-Bildern.

    Gleichungen:

        MSE  = (1 / (H*W*3)) * sum( (original - reproduced)^2 )
        PSNR = 10 * log10( peak^2 / MSE )   [dB]

    Parameter
    ----------
    original   : np.ndarray, Form (H, W, 3), Datentyp uint8
    reproduced : np.ndarray, Form (H, W, 3), Datentyp uint8
    peak       : float - maximaler Pixelwert (255 für uint8)

    Rueckgabe
    --------
    float – PSNR in dB (inf, wenn die Bilder identisch sind)
    """
    orig  = original.astype(np.float64)
    repro = reproduced.astype(np.float64)

    # Gleiche Form sicherstellen (bei Padding auf die kleinere Groeße zuschneiden)
    H = min(orig.shape[0], repro.shape[0])
    W = min(orig.shape[1], repro.shape[1])
    orig  = orig[:H, :W, :]
    repro = repro[:H, :W, :]

    mse = np.mean((orig - repro) ** 2)            

    if mse == 0.0:
        return float("inf")

    psnr = 10.0 * np.log10(peak ** 2 / mse)  
    return float(psnr)
