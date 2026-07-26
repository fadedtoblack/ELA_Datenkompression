"""
encoder.py
----------
JPEG-style encoder for the ELA-Tool (SoSe 2026).

Pipeline (Section 2.2.1 / Figure 1):
    1. Load input image (PNG or JPEG)
    2. RGB → YCbCr  (ITU-R BT.601-4)
    3. Split each component into 8×8 blocks
    4. Forward 2-D DCT per block (level-shift -128 applied inside dct_blocks)
    5. Quantize using quality-scaled tables
    6. (Optional) Save intermediate results to text files
    7. Compute and display entropy values (12 total, Section 2.2.6)
    8. Return quantized DCT coefficients to the decoder

No entropy coding or JPEG header generation is performed, as specified.
"""

import os
import numpy as np
from PIL import Image

from colorspace   import rgb_to_ycbcr, load_image, save_component_images
from dct_blocks   import (split_into_blocks, apply_dct_to_blocks,
                           quantize_blocks, save_blocks_to_file, compute_entropy)
from quantization import (compute_all_qtables, save_qtables_to_file,
                           estimate_quality)


class EncoderResult:
    """Container for all outputs produced by the encoder."""

    def __init__(self):
        # Quantized DCT coefficients for each component
        self.q_dct: dict[str, np.ndarray] = {}        # 'Y', 'Cb', 'Cr'
        # Original block shapes (needed by decoder to reassemble)
        self.original_shapes: dict[str, tuple] = {}   # 'Y', 'Cb', 'Cr'
        # Quantization tables used
        self.qtables: dict[str, np.ndarray] = {}
        # Original image array (for PSNR / ELA reference)
        self.original_rgb: np.ndarray | None = None
        # Entropy values (12 total): keys described in print_entropies()
        self.entropies: dict[str, float] = {}
        # Quality value used
        self.quality: float = 50.0
        # Verified (back-estimated) quality
        self.estimated_quality: float = 0.0


def encode(image_path: str,
           quality: float = 50.0,
           save_intermediates: bool = False,
           output_dir: str = "output") -> EncoderResult:
    """
    Run the full encoder pipeline on an image file.

    Parameters
    ----------
    image_path         : str   – path to PNG or JPEG input image
    quality            : float – quality value Q in (0, 100]
    save_intermediates : bool  – if True, write text files and component
                                 images as described in Section 2.2.2
    output_dir         : str   – directory for all output files

    Returns
    -------
    EncoderResult
    """
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    result = EncoderResult()
    result.quality = quality

    # ------------------------------------------------------------------
    # 1. Load image
    # ------------------------------------------------------------------
    print(f"[Encoder] Loading image: {image_path}")
    rgb = load_image(image_path)
    result.original_rgb = rgb
    print(f"  Image size: {rgb.shape[1]}×{rgb.shape[0]}  (W×H)")

    # ------------------------------------------------------------------
    # 2. Compute quantization tables for the given quality value
    # ------------------------------------------------------------------
    qtables = compute_all_qtables(quality)
    result.qtables = qtables
    result.estimated_quality = estimate_quality(qtables)
    print(f"  Quality: Q={quality:.1f}%  "
          f"(back-estimated: {result.estimated_quality:.2f}%)")

    if save_intermediates:
        qt_path = os.path.join(output_dir, f"{base_name}_qtables_q{quality:.0f}.txt")
        save_qtables_to_file(qtables, quality, qt_path)
        print(f"  Saved quantization tables: {qt_path}")

    # ------------------------------------------------------------------
    # 3. RGB → YCbCr colour space transformation
    # ------------------------------------------------------------------
    print("[Encoder] RGB → YCbCr colour space transformation …")
    Y, Cb, Cr = rgb_to_ycbcr(rgb)

    # Entropy of R, G, B channels
    result.entropies["H_R"]  = compute_entropy(rgb[:, :, 0])
    result.entropies["H_G"]  = compute_entropy(rgb[:, :, 1])
    result.entropies["H_B"]  = compute_entropy(rgb[:, :, 2])

    # Entropy of Y, Cb, Cr channels
    result.entropies["H_Y"]  = compute_entropy(Y)
    result.entropies["H_Cb"] = compute_entropy(Cb)
    result.entropies["H_Cr"] = compute_entropy(Cr)

    if save_intermediates:
        # Save Y/Cb/Cr text files (blockwise – we use the raw component
        # values arranged into 8×8 tiles for the colour-space step)
        save_component_images(Y, Cb, Cr, base_name, output_dir)
        for comp_name, comp_data in (("y", Y), ("cb", Cb), ("cr", Cr)):
            blks, _ = split_into_blocks(comp_data)
            txt_path = os.path.join(output_dir,
                                    f"{base_name}_{comp_name}.txt")
            save_blocks_to_file(blks, txt_path)
            print(f"  Saved colour-space blocks: {txt_path}")

    # ------------------------------------------------------------------
    # 4. 8×8 block decomposition  +  5. Forward DCT
    # ------------------------------------------------------------------
    print("[Encoder] Block decomposition and DCT …")
    components = {"Y": Y, "Cb": Cb, "Cr": Cr}
    dct_results  = {}
    qdct_results = {}

    for comp_name, comp_data in components.items():
        # Split into blocks (padding handled inside split_into_blocks)
        blocks, orig_shape = split_into_blocks(comp_data)
        result.original_shapes[comp_name] = orig_shape

        # Forward DCT (level shift applied inside apply_dct_to_blocks)
        dct_blks = apply_dct_to_blocks(blocks)
        dct_results[comp_name] = dct_blks

        # Entropy of DCT coefficients
        result.entropies[f"H_{comp_name}_dct"] = compute_entropy(dct_blks)

        # ------------------------------------------------------------------
        # 6. Quantize
        # ------------------------------------------------------------------
        q_key = "Y" if comp_name == "Y" else "Cb"  # Cb and Cr share chroma table
        q_blks = quantize_blocks(dct_blks, qtables[comp_name])
        qdct_results[comp_name] = q_blks

        # Entropy of quantized DCT coefficients
        result.entropies[f"H_{comp_name}_qdct"] = compute_entropy(q_blks)

        if save_intermediates:
            comp_lower = comp_name.lower()
            # DCT text file
            dct_txt = os.path.join(output_dir,
                                   f"{base_name}_{comp_lower}_dct.txt")
            save_blocks_to_file(dct_blks, dct_txt)
            print(f"  Saved DCT blocks:  {dct_txt}")

            # Quantized DCT text file
            qdct_txt = os.path.join(output_dir,
                                    f"{base_name}_{comp_lower}_qdct.txt")
            save_blocks_to_file(q_blks, qdct_txt)
            print(f"  Saved QDCT blocks: {qdct_txt}")

    result.q_dct = qdct_results

    # ------------------------------------------------------------------
    # 7. Print entropy summary (12 values, Section 2.2.6)
    # ------------------------------------------------------------------
    _print_entropies(result.entropies)

    # Optionally save entropy to file
    if save_intermediates:
        ent_path = os.path.join(output_dir,
                                f"{base_name}_entropy_q{quality:.0f}.txt")
        _save_entropies(result.entropies, ent_path, quality)
        print(f"  Saved entropy values: {ent_path}")

    print("[Encoder] Done.")
    return result


def _print_entropies(entropies: dict) -> None:
    print("\n[Encoder] Entropy values (bits per symbol):")
    order = [
        ("H_R",        "R channel"),
        ("H_G",        "G channel"),
        ("H_B",        "B channel"),
        ("H_Y",        "Y component"),
        ("H_Cb",       "Cb component"),
        ("H_Cr",       "Cr component"),
        ("H_Y_dct",    "Y  DCT coefficients"),
        ("H_Cb_dct",   "Cb DCT coefficients"),
        ("H_Cr_dct",   "Cr DCT coefficients"),
        ("H_Y_qdct",   "Y  quantized DCT"),
        ("H_Cb_qdct",  "Cb quantized DCT"),
        ("H_Cr_qdct",  "Cr quantized DCT"),
    ]
    for key, label in order:
        val = entropies.get(key, float("nan"))
        print(f"  {label:<30s}: {val:.4f} bit")


def _save_entropies(entropies: dict, filepath: str, quality: float) -> None:
    with open(filepath, "w") as f:
        f.write(f"Entropy values for Q={quality:.1f}%\n\n")
        for key, val in entropies.items():
            f.write(f"{key}: {val:.6f}\n")
