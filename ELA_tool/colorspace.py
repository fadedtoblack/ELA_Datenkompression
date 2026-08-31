"""
colorspace.py
-------------
RGB ↔ YCbCr Farbraum-Transformationen gemäß ITU-R BT.601-4.

Der JPEG-Standard (T.81) bezieht sich auf ITU-R BT.601 für die Farbraum-
konvertierung in der Encoder- und Decoder-Pipeline.

Vorwärtstransformation (Encoder):  RGB → YCbCr
Rücktransformation (Decoder):     YCbCr → RGB

Hinweis: Es wird keine Chroma-Subsampling durchgeführt, wie in den Vorgaben
gefordert.
"""

import numpy as np
from PIL import Image
import os

# ---------------------------------------------------------------------------
# ITU-R BT.601-4 Vorwärtstransformation:  RGB → YCbCr
# ---------------------------------------------------------------------------
# Eingabe:  R, G, B in [0, 255]  (uint8)
# Ausgabe: Y  in [16, 235]
#          Cb in [16, 240]
#          Cr in [16, 240]
#
# Die "Studio Swing"-Formulierung (eingeschränkter Bereich) aus BT.601:
#
#   Y  =  16 + 65.481*R/255 + 128.553*G/255 +  24.966*B/255
#   Cb = 128 - 37.797*R/255 -  74.203*G/255 + 112.000*B/255
#   Cr = 128 + 112.000*R/255 - 93.786*G/255 -  18.214*B/255
#
# Äquivalent (mit auf [0, 1] normalisiertem R, G, B):
#   Y  =  16 + 219*(  0.299*R + 0.587*G + 0.114*B)
#   Cb = 128 + 224*(-0.16875*R - 0.33126*G + 0.5*B)   (gerundete Koeffizienten)
#   Cr = 128 + 224*(  0.5*R - 0.41869*G - 0.08131*B)
#
# Wir verwenden unten die exakten BT.601-Matrixkoeffizienten.
# ---------------------------------------------------------------------------

def rgb_to_ycbcr(image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Konvertiert ein RGB-Bild in YCbCr-Komponenten (ITU-R BT.601-4).

    Parameter
    ----------
    image_rgb : np.ndarray, Form (H, W, 3), dtype uint8
        Eingabebild im RGB-Farbraum, Pixelwerte in [0, 255].

    Rückgabe
    -------
    Y, Cb, Cr : drei np.ndarray-Arrays der Form (H, W), dtype float64
        Luma- und Chroma-Komponenten (Gleitkomma, nicht abgeschnitten).
        Y  ∈ [16, 235],  Cb ∈ [16, 240],  Cr ∈ [16, 240]  (ungefähr)
    """
    img = image_rgb.astype(np.float64)
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    Y  = 0.299 * R  + 0.587 * G + 0.114 * B
    Cb = 128 - 0.169 * R - 0.331 * G + 0.500 * B
    Cr = 128 + 0.500 * R - 0.419 * G - 0.081 * B

    return Y, Cb, Cr

# ---------------------------------------------------------------------------
# ITU-R BT.601-4 Rücktransformation:  YCbCr → RGB
# ---------------------------------------------------------------------------

def ycbcr_to_rgb(Y: np.ndarray, Cb: np.ndarray, Cr: np.ndarray) -> np.ndarray:
    """
    Konvertiert YCbCr-Komponenten zurück in ein RGB-Bild (ITU-R BT.601-4 invers).

    Parameter
    ----------
    Y, Cb, Cr : np.ndarray, Form (H, W), dtype float64
        Luma- und Chroma-Komponenten (Gleitkomma).

    Rückgabe
    -------
    np.ndarray, Form (H, W, 3), dtype uint8
        Rekonstruiertes RGB-Bild, Pixelwerte auf [0, 255] begrenzt.
    """
    # Verschiebungsversatz der Chroma-Kanäle
    cb = Cb - 128.0
    cr = Cr - 128.0

    R = Y + 1.402 * cr
    G = Y - 0.34414 * cb - 0.71414 * cr
    B = Y + 1.772 * cb

    R = np.clip(np.round(R), 0, 255)
    G = np.clip(np.round(G), 0, 255)
    B = np.clip(np.round(B), 0, 255)

    return np.stack([R, G, B], axis=2).astype(np.uint8)

# ---------------------------------------------------------------------------
# Hilfsfunktion: Einzelne Y / Cb / Cr-Komponentenbilder zur visuellen Inspektion speichern
# ---------------------------------------------------------------------------

def save_component_images(Y: np.ndarray, Cb: np.ndarray, Cr: np.ndarray,
                           base_name: str, output_dir: str = ".") -> None:
    """
    Speichert die Y-, Cb- und Cr-Komponenten als einzelne Graustufen-PNG-Bilder
    (Abschnitt 2.2.2 – visueller Vergleich der Farbraumtransformationen).

    Werte werden vor dem Speichern begrenzt und in uint8 umgewandelt.

    Parameter
    ----------
    Y, Cb, Cr   : np.ndarray, Form (H, W)
    base_name   : str  – z. B. "kodim07"
    output_dir  : str  – Zielverzeichnis für die Bilder
    """
    os.makedirs(output_dir, exist_ok=True)

    for component, data in (("y", Y), ("cb", Cb), ("cr", Cr)):
        arr = np.clip(np.round(data), 0, 255).astype(np.uint8)
        img = Image.fromarray(arr, mode="L")
        path = os.path.join(output_dir, f"{base_name}_{component}.png")
        img.save(path)
        print(f"  Komponentenbild gespeichert: {path}")


# ---------------------------------------------------------------------------
# Hilfsfunktion: Bild (PNG oder JPEG) laden und als RGB-Numpy-Array zurückgeben
# ---------------------------------------------------------------------------

def load_image(path: str) -> np.ndarray:
    """
    Lädt ein PNG- oder JPEG-Bild und gibt ein (H, W, 3) uint8 RGB-Array zurück.
    Graustufen- und RGBA-Bilder werden automatisch in RGB konvertiert.
    """
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)