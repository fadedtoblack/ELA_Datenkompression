"""
colorspace.py
-------------
RGB <-> YCbCr Farbraum-Transformationen gemäß ITU-R BT.601-4.

Der JPEG-Standard (T.81) bezieht sich auf ITU-R BT.601 fuer die Farbraum-
konvertierung in der Encoder- und Decoder-Pipeline.

Vorwärtstransformation (Encoder):  RGB -> YCbCr
Ruecktransformation (Decoder):     YCbCr -> RGB
"""

import numpy as np
from PIL import Image
import os

# ---------------------------------------------------------------------------
# ITU-R BT.601-4 Vorwärtstransformation:  RGB -> YCbCr
# ---------------------------------------------------------------------------

def rgb_to_ycbcr(image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Konvertiert ein RGB-Bild in YCbCr-Komponenten (ITU-R BT.601-4).

    Parameter
    ----------
    image_rgb : np.ndarray, Form (H, W, 3), dtype uint8
        Eingabebild im RGB-Farbraum, Pixelwerte in [0, 255].

    Rueckgabe
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
# ITU-R BT.601-4 Ruecktransformation:  YCbCr -> RGB
# ---------------------------------------------------------------------------

def ycbcr_to_rgb(Y: np.ndarray, Cb: np.ndarray, Cr: np.ndarray) -> np.ndarray:
    """
    Konvertiert YCbCr-Komponenten zurueck in ein RGB-Bild (ITU-R BT.601-4 invers).

    Parameter
    ----------
    Y, Cb, Cr : np.ndarray, Form (H, W), dtype float64
        Luma- und Chroma-Komponenten (Gleitkomma).

    Rueckgabe
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
    Werte werden vor dem Speichern begrenzt und in uint8 umgewandelt.

    Parameter
    ----------
    Y, Cb, Cr   : np.ndarray, Form (H, W)
    base_name   : str  
    output_dir  : str 
    """
    os.makedirs(output_dir, exist_ok=True)

    for component, data in (("y", Y), ("cb", Cb), ("cr", Cr)):
        arr = np.clip(np.round(data), 0, 255).astype(np.uint8)
        img = Image.fromarray(arr, mode="L")
        path = os.path.join(output_dir, f"{base_name}_{component}.png")
        img.save(path)
        print(f"  Komponentenbild gespeichert: {path}")


# ---------------------------------------------------------------------------
# Hilfsfunktion: Bild (PNG oder JPEG) laden und als RGB-Numpy-Array zurueckgeben
# ---------------------------------------------------------------------------

def load_image(path: str) -> np.ndarray:
    """
    Lädt ein PNG- oder JPEG-Bild und gibt ein (H, W, 3) uint8 RGB-Array zurueck.
    Graustufen- und RGBA-Bilder werden automatisch in RGB konvertiert.
    """
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)