"""
colorspace.py
-------------
RGB ↔ YCbCr color space transformations following ITU-R BT.601-4.

The JPEG standard (T.81) references ITU-R BT.601 for the color space
conversion used in the encoder and decoder pipeline.

Forward transform (encoder):  RGB → YCbCr
Inverse transform (decoder):  YCbCr → RGB

Note: No chroma subsampling is performed, as required by the assignment spec
(Section 2.1, bullet 2).
"""

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# ITU-R BT.601-4 forward transform:  RGB → YCbCr
# ---------------------------------------------------------------------------
# Input:  R, G, B in [0, 255]  (uint8)
# Output: Y  in [16, 235]
#         Cb in [16, 240]
#         Cr in [16, 240]
#
# The "studio swing" (limited range) formulation from BT.601:
#
#   Y  =  16 + 65.481*R/255 + 128.553*G/255 +  24.966*B/255
#   Cb = 128 - 37.797*R/255 -  74.203*G/255 + 112.000*B/255
#   Cr = 128 + 112.000*R/255 - 93.786*G/255 -  18.214*B/255
#
# Equivalently (with R, G, B normalised to [0, 1]):
#   Y  =  16 + 219*(  0.299*R + 0.587*G + 0.114*B)
#   Cb = 128 + 224*(-0.16875*R - 0.33126*G + 0.5*B)   (rounded coefficients)
#   Cr = 128 + 224*(  0.5*R - 0.41869*G - 0.08131*B)
#
# We use the exact BT.601 matrix coefficients below.
# ---------------------------------------------------------------------------

def rgb_to_ycbcr(image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert an RGB image to YCbCr components (ITU-R BT.601-4).

    Parameters
    ----------
    image_rgb : np.ndarray, shape (H, W, 3), dtype uint8
        Input image in RGB colour space, pixel values in [0, 255].

    Returns
    -------
    Y, Cb, Cr : three np.ndarray arrays of shape (H, W), dtype float64
        Luma and chroma components (floating-point, not clipped).
        Y  ∈ [16, 235],  Cb ∈ [16, 240],  Cr ∈ [16, 240]  (approximately)
    """
    img = image_rgb.astype(np.float64)
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    # ITU-R BT.601-4 forward transform (studio swing / limited range)
    Y  =  16.0 + 65.481 * R / 255.0 + 128.553 * G / 255.0 +  24.966 * B / 255.0
    Cb = 128.0 - 37.797 * R / 255.0 -  74.203 * G / 255.0 + 112.000 * B / 255.0
    Cr = 128.0 + 112.000 * R / 255.0 - 93.786 * G / 255.0 -  18.214 * B / 255.0

    return Y, Cb, Cr


# ---------------------------------------------------------------------------
# ITU-R BT.601-4 inverse transform:  YCbCr → RGB
# ---------------------------------------------------------------------------

def ycbcr_to_rgb(Y: np.ndarray, Cb: np.ndarray, Cr: np.ndarray) -> np.ndarray:
    """
    Convert YCbCr components back to an RGB image (ITU-R BT.601-4 inverse).

    Parameters
    ----------
    Y, Cb, Cr : np.ndarray, shape (H, W), dtype float64
        Luma and chroma components (floating-point).

    Returns
    -------
    np.ndarray, shape (H, W, 3), dtype uint8
        Reconstructed RGB image, pixel values clipped to [0, 255].
    """
    # Shift chroma channels
    cb = Cb - 128.0
    cr = Cr - 128.0
    y  = Y  -  16.0

    # ITU-R BT.601-4 inverse transform
    R = 255.0 / 219.0 * y                          + 255.0 / 112.0 * 0.701       * cr
    G = 255.0 / 219.0 * y - 255.0 / 112.0 * 0.886 * 0.114 / 0.587 * cb \
                           - 255.0 / 112.0 * 0.701 * 0.299 / 0.587 * cr
    B = 255.0 / 219.0 * y + 255.0 / 112.0 * 0.886                  * cb

    R = np.clip(np.round(R), 0, 255)
    G = np.clip(np.round(G), 0, 255)
    B = np.clip(np.round(B), 0, 255)

    return np.stack([R, G, B], axis=2).astype(np.uint8)


# ---------------------------------------------------------------------------
# Helper: save individual Y / Cb / Cr component images for visual inspection
# ---------------------------------------------------------------------------

def save_component_images(Y: np.ndarray, Cb: np.ndarray, Cr: np.ndarray,
                           base_name: str, output_dir: str = ".") -> None:
    """
    Save the Y, Cb, and Cr components as individual greyscale PNG images
    (Section 2.2.2 – visual comparison of colour space transform results).

    Values are clipped and cast to uint8 before saving.

    Parameters
    ----------
    Y, Cb, Cr   : np.ndarray, shape (H, W)
    base_name   : str  – e.g. "kodim07"
    output_dir  : str  – directory to write images into
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    for component, data in (("y", Y), ("cb", Cb), ("cr", Cr)):
        arr = np.clip(np.round(data), 0, 255).astype(np.uint8)
        img = Image.fromarray(arr, mode="L")
        path = os.path.join(output_dir, f"{base_name}_{component}.png")
        img.save(path)
        print(f"  Saved component image: {path}")


# ---------------------------------------------------------------------------
# Helper: load an image (PNG or JPEG) and return an RGB numpy array
# ---------------------------------------------------------------------------

def load_image(path: str) -> np.ndarray:
    """
    Load a PNG or JPEG image and return an (H, W, 3) uint8 RGB array.
    Greyscale and RGBA images are converted to RGB automatically.
    """
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Colour space module self-test ===\n")

    # Build a small synthetic test image with known pixel values
    test_pixels = np.array([
        [[0,   0,   0  ]],   # black
        [[255, 255, 255]],   # white
        [[255, 0,   0  ]],   # red
        [[0,   255, 0  ]],   # green
        [[0,   0,   255]],   # blue
    ], dtype=np.uint8)                              # shape (5, 1, 3)

    Y, Cb, Cr = rgb_to_ycbcr(test_pixels)
    rgb_back  = ycbcr_to_rgb(Y, Cb, Cr)

    print("Pixel        Original →  Y     Cb    Cr   →  Reconstructed  Match?")
    labels = ["black", "white", "red", "green", "blue"]
    for i, label in enumerate(labels):
        orig  = test_pixels[i, 0]
        recon = rgb_back[i, 0]
        y_val  = Y[i, 0]
        cb_val = Cb[i, 0]
        cr_val = Cr[i, 0]
        match  = np.allclose(orig, recon, atol=2)
        print(f"  {label:5s}  {orig}  →  "
              f"Y={y_val:6.1f} Cb={cb_val:6.1f} Cr={cr_val:6.1f}  →  "
              f"{recon}  ✓" if match else f"{recon}  ✗ diff={orig.astype(int)-recon.astype(int)}")

    print("\nRound-trip max pixel error:",
          np.max(np.abs(test_pixels.astype(int) - rgb_back.astype(int))))
