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
                 save_intermediates: bool = False) -> np.ndarray:
    """
    Generate an ELA image for a given input image.

    Steps:
        1. Encode the input image at the given quality.
        2. Decode to obtain the reproduced image.
        3. Compute the pixel-wise absolute difference, scaled by M.
        4. Clip to [0, 255] and save as PNG.

    Parameters
    ----------
    image_path         : str   – path to the input image (JPEG recommended)
    quality            : float – JPEG quality value Q in (0, 100]
    multiplier         : float – contrast enhancement multiplier M
    output_dir         : str   – directory for all output files
    save_intermediates : bool  – also save encoder intermediate text files

    Returns
    -------
    ela_image : np.ndarray, shape (H, W, 3), dtype uint8
    """
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    print(f"\n{'='*60}")
    print(f"ELA: {base_name}  |  Q={quality:.0f}%  |  M={multiplier:.0f}")
    print(f"{'='*60}")

    # -----------------------------------------------------------------------
    # 1 & 2. Encode + decode
    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # 3. Compute ELA image  (Equation 2.8)
    #
    #   ELA_pixel = clip( |input - reproduced| * M, 0, 255 )
    # -----------------------------------------------------------------------
    original_rgb = enc_result.original_rgb.astype(np.float64)
    reproduced_f = reproduced_rgb.astype(np.float64)

    ela_float = np.abs(original_rgb - reproduced_f) * multiplier  # Eq. 2.8
    ela_image = np.clip(ela_float, 0.0, 255.0).astype(np.uint8)

    # -----------------------------------------------------------------------
    # 4. Save ELA image
    # -----------------------------------------------------------------------
    ela_name = f"{base_name}_ela_q{quality:.0f}_m{multiplier:.0f}.png"
    ela_path = os.path.join(output_dir, ela_name)
    Image.fromarray(ela_image, mode="RGB").save(ela_path)
    print(f"\n[ELA] ELA image saved: {ela_path}")
    print(f"[ELA] PSNR (original vs reproduced): {psnr:.4f} dB")

    return ela_image
