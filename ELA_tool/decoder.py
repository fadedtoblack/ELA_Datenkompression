"""
decoder.py
----------
JPEG-style decoder for the ELA-Tool (SoSe 2026).

Pipeline (Section 2.3.1 / Figure 3):
    1. Receive quantized DCT coefficients from the encoder
    2. De-quantize
    3. Inverse 2-D DCT per block (IDCT, level shift +128 applied inside)
    4. Reassemble 8×8 blocks into full component arrays
    5. YCbCr → RGB  (ITU-R BT.601-4 inverse)
    6. Save reproduced image as PNG (Section 2.3.1)
    7. Compute and display PSNR vs. the original image (Section 2.3.2)
"""

import os
import numpy as np
from PIL import Image

from colorspace import ycbcr_to_rgb
from dct_blocks import (dequantize_blocks, apply_idct_to_blocks,
                         merge_blocks)
from encoder    import EncoderResult


def decode(enc_result: EncoderResult,
           output_dir: str = "output",
           image_name: str = "reproduced") -> tuple[np.ndarray, float]:
    """
    Run the full decoder pipeline using the output of the encoder.

    Parameters
    ----------
    enc_result  : EncoderResult – output of encoder.encode()
    output_dir  : str           – directory for output files
    image_name  : str           – base name for the saved PNG (without extension)

    Returns
    -------
    reproduced_rgb : np.ndarray, shape (H, W, 3), dtype uint8
    psnr           : float – PSNR in dB vs. the original image
    """
    os.makedirs(output_dir, exist_ok=True)

    print("[Decoder] Starting decoding pipeline …")

    qtables        = enc_result.qtables
    q_dct          = enc_result.q_dct
    orig_shapes    = enc_result.original_shapes

    components_out = {}

    for comp_name in ("Y", "Cb", "Cr"):
        # ------------------------------------------------------------------
        # 1. De-quantize  (T.81, A.3.6)
        # ------------------------------------------------------------------
        dq_blks = dequantize_blocks(q_dct[comp_name], qtables[comp_name])

        # ------------------------------------------------------------------
        # 2. Inverse DCT  (level shift +128 applied inside apply_idct_to_blocks)
        # ------------------------------------------------------------------
        spatial_blks = apply_idct_to_blocks(dq_blks)

        # ------------------------------------------------------------------
        # 3. Reassemble 8×8 blocks into full component
        # ------------------------------------------------------------------
        component = merge_blocks(spatial_blks, orig_shapes[comp_name])
        components_out[comp_name] = component

    # ------------------------------------------------------------------
    # 4. YCbCr → RGB  (ITU-R BT.601-4 inverse)
    # ------------------------------------------------------------------
    print("[Decoder] YCbCr → RGB inverse colour space transformation …")
    reproduced_rgb = ycbcr_to_rgb(
        components_out["Y"],
        components_out["Cb"],
        components_out["Cr"],
    )

    # ------------------------------------------------------------------
    # 5. Save reproduced image as PNG  (Section 2.3.1)
    # ------------------------------------------------------------------
    out_path = os.path.join(output_dir, f"{image_name}.png")
    Image.fromarray(reproduced_rgb, mode="RGB").save(out_path)
    print(f"[Decoder] Reproduced image saved: {out_path}")

    # ------------------------------------------------------------------
    # 6. Compute PSNR  (Section 2.3.2, Equations 2.6 / 2.7)
    # ------------------------------------------------------------------
    if enc_result.original_rgb is not None:
        psnr = compute_psnr(enc_result.original_rgb, reproduced_rgb)
        print(f"[Decoder] PSNR: {psnr:.4f} dB")
    else:
        psnr = float("nan")
        print("[Decoder] Warning: original image not available, PSNR not computed.")

    print("[Decoder] Done.")
    return reproduced_rgb, psnr


def compute_psnr(original: np.ndarray, reproduced: np.ndarray,
                 peak: float = 255.0) -> float:
    """
    Compute the Peak Signal-to-Noise Ratio (PSNR) between two RGB images.

    Equations (2.6) and (2.7) from the assignment spec:

        MSE  = (1 / (H*W*3)) * sum( (original - reproduced)^2 )
        PSNR = 10 * log10( peak^2 / MSE )   [dB]

    Parameters
    ----------
    original   : np.ndarray, shape (H, W, 3), dtype uint8
    reproduced : np.ndarray, shape (H, W, 3), dtype uint8
    peak       : float – maximum pixel value (255 for uint8)

    Returns
    -------
    float – PSNR in dB  (inf if images are identical)
    """
    orig  = original.astype(np.float64)
    repro = reproduced.astype(np.float64)

    # Ensure same shape (crop to smaller if padding was applied)
    H = min(orig.shape[0], repro.shape[0])
    W = min(orig.shape[1], repro.shape[1])
    orig  = orig[:H, :W, :]
    repro = repro[:H, :W, :]

    mse = np.mean((orig - repro) ** 2)             # Eq. 2.7

    if mse == 0.0:
        return float("inf")

    psnr = 10.0 * np.log10(peak ** 2 / mse)        # Eq. 2.6
    return float(psnr)
