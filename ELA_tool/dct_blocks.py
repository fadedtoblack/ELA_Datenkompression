"""
dct_blocks.py
-------------
8×8 block decomposition, 2-D DCT / IDCT, quantization and de-quantization.

All operations follow JPEG standard T.81 (Section A.3.3 for the DCT,
Section A.3.6 for quantization).

The 2-D DCT-II (forward) and DCT-III (inverse) are computed via
scipy.fft.dctn / scipy.fft.idctn, which implement the orthonormal variant
used by the JPEG standard.
"""

import numpy as np

# 8×8 block decomposition and reassembly -------------------------------

def split_into_blocks(component: np.ndarray) -> tuple[np.ndarray, tuple]:
    """
    Split a 2-D component array into 8×8 blocks.

    If the image dimensions are not multiples of 8 the component is
    symmetrically padded (edge replication) before splitting.

    Parameters
    ----------
    component : np.ndarray, shape (H, W)

    Returns
    -------
    blocks       : np.ndarray, shape (n_blocks_h, n_blocks_w, 8, 8)
    original_shape : (H, W) – needed to crop the reconstructed image
    """
    H, W = component.shape
    original_shape = (H, W)

    # Pad to a multiple of 8 if necessary
    pad_h = (8 - H % 8) % 8
    pad_w = (8 - W % 8) % 8
    if pad_h > 0 or pad_w > 0:
        component = np.pad(component,
                           ((0, pad_h), (0, pad_w)),
                           mode="edge")

    H_pad, W_pad = component.shape
    n_h = H_pad // 8
    n_w = W_pad // 8

    # Reshape into (n_h, 8, n_w, 8) then transpose to (n_h, n_w, 8, 8)
    blocks = (component
              .reshape(n_h, 8, n_w, 8)
              .transpose(0, 2, 1, 3))          # → (n_h, n_w, 8, 8)
    return blocks, original_shape


def merge_blocks(blocks: np.ndarray, original_shape: tuple) -> np.ndarray:
    """
    Reassemble 8×8 blocks back into a 2-D component, then crop to the
    original image size.

    Parameters
    ----------
    blocks         : np.ndarray, shape (n_h, n_w, 8, 8)
    original_shape : (H, W)

    Returns
    -------
    np.ndarray, shape (H, W)
    """
    n_h, n_w, _, _ = blocks.shape
    # Transpose back and reshape
    component = (blocks
                 .transpose(0, 2, 1, 3)        # → (n_h, 8, n_w, 8)
                 .reshape(n_h * 8, n_w * 8))
    H, W = original_shape
    return component[:H, :W]

# DCT and IDCT (JPEG-Standard T.81, A.3.3)---------

# Vorberechnung der DCT-Transformationsmatrix (8x8)
def get_dct_matrix():
    T = np.zeros((8, 8))
    for u in range(8):
        c = 1 / np.sqrt(2) if u == 0 else 1.0
        for x in range(8):
            T[u, x] = 0.5 * c * np.cos((2 * x + 1) * u * np.pi / 16)
    return T

DCT_MATRIX = get_dct_matrix()

# DCT anhand der Formel:

def c(k):
    """Berechnet den Faktor C_u bzw. C_v."""
    return 1 / np.sqrt(2) if k == 0 else 1

def dct2(block: np.ndarray) -> np.ndarray:
    # Manuelle DCT mittels Matrixmultiplikation
    return DCT_MATRIX @ block @ DCT_MATRIX.T

# IDCT nach der Formel:

def idct2(S: np.ndarray) -> np.ndarray:
    # Manuelle IDCT mittels Matrixmultiplikation
    return DCT_MATRIX.T @ S @ DCT_MATRIX


def apply_dct_to_blocks(blocks: np.ndarray) -> np.ndarray:
    """
    Apply the forward DCT to every 8×8 block in the array, with the
    required level shift of -128.

    Parameters
    ----------
    blocks : np.ndarray, shape (n_h, n_w, 8, 8)

    Returns
    -------
    np.ndarray, shape (n_h, n_w, 8, 8) – DCT coefficients (float64)
    """
    shifted = blocks.astype(np.float64) - 128.0    # level shift (T.81 A.3.1)
    n_h, n_w = blocks.shape[:2]
    dct_blocks = np.zeros_like(shifted)
    for i in range(n_h):
        for j in range(n_w):
            dct_blocks[i, j] = dct2(shifted[i, j])
    return dct_blocks


def apply_idct_to_blocks(dct_blocks: np.ndarray) -> np.ndarray:
    """
    Apply the inverse DCT to every 8×8 block and undo the level shift.

    Parameters
    ----------
    dct_blocks : np.ndarray, shape (n_h, n_w, 8, 8)

    Returns
    -------
    np.ndarray, shape (n_h, n_w, 8, 8) – reconstructed spatial values (float64)
    """
    n_h, n_w = dct_blocks.shape[:2]
    spatial = np.zeros_like(dct_blocks)
    for i in range(n_h):
        for j in range(n_w):
            spatial[i, j] = idct2(dct_blocks[i, j])
    spatial += 128.0                                # undo level shift
    return spatial

# Quantisierung und Dequantisierung ------------------------------------------

def quantize_blocks(dct_blocks: np.ndarray, qtable: np.ndarray) -> np.ndarray:
    """
    Quantize DCT coefficients by dividing by the quantization table and
    rounding to the nearest integer (T.81, Eq. A-4).

    Parameters
    ----------
    dct_blocks : np.ndarray, shape (n_h, n_w, 8, 8)
    qtable     : np.ndarray, shape (8, 8)

    Returns
    -------
    np.ndarray, shape (n_h, n_w, 8, 8), dtype int32
    """
    return np.round(dct_blocks / qtable.astype(np.float64)).astype(np.int32)


def dequantize_blocks(q_blocks: np.ndarray, qtable: np.ndarray) -> np.ndarray:
    """
    De-quantize coefficients by multiplying by the quantization table
    (T.81, Eq. A-5).

    Parameters
    ----------
    q_blocks : np.ndarray, shape (n_h, n_w, 8, 8), dtype int32
    qtable   : np.ndarray, shape (8, 8)

    Returns
    -------
    np.ndarray, shape (n_h, n_w, 8, 8), dtype float64
    """
    return q_blocks.astype(np.float64) * qtable.astype(np.float64)

# Auslesen der 8×8 Blöcke in Textdatei ----------------------------

def save_blocks_to_file(blocks: np.ndarray, filepath: str) -> None:
    """
    Write all 8×8 blocks to a text file in the format required by the
    assignment (Section 2.2.2):

        Block: 1
        1: value; 2: value; … ; 64: value
        Block: 2
        …

    Parameters
    ----------
    blocks   : np.ndarray, shape (n_h, n_w, 8, 8)
    filepath : str
    """
    n_h, n_w = blocks.shape[:2]
    block_num = 1
    with open(filepath, "w") as f:
        for i in range(n_h):
            for j in range(n_w):
                flat = blocks[i, j].flatten()
                values = "; ".join(f"{k+1}: {flat[k]:.4g}" for k in range(64))
                f.write(f"Block: {block_num}\n{values}\n")
                block_num += 1

# Entropieberechnung (Section 2.2.6)-----------

def compute_entropy(data: np.ndarray) -> float:
    """
    Compute the mean information content (Shannon entropy) of an array,
    treating all values as symbols in a discrete alphabet.

    H = -sum( p_i * log2(p_i) )   for all non-zero probabilities p_i

    Parameters
    ----------
    data : np.ndarray – any shape; values are flattened and discretized.

    Returns
    -------
    float – entropy in bits per symbol
    """
    flat = data.flatten()
    # Round floating-point values to integer bins for frequency counting
    if np.issubdtype(flat.dtype, np.floating):
        flat = np.round(flat).astype(np.int64)
    else:
        flat = flat.astype(np.int64)

    _, counts = np.unique(flat, return_counts=True)
    probs = counts / counts.sum()
    # Avoid log2(0)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))

# Quick self-test (drin lassen?) ------------------

if __name__ == "__main__":
    print("=== DCT / block module self-test ===\n")

    # 1. Block splitting round-trip
    dummy = np.arange(512 * 768, dtype=np.float64).reshape(512, 768) % 256
    blocks, orig_shape = split_into_blocks(dummy)
    recovered = merge_blocks(blocks, orig_shape)
    print(f"Block split / merge round-trip error (512×768 → {blocks.shape[:2]} blocks): "
          f"{np.max(np.abs(dummy - recovered)):.6f}  (expected 0)")

    # 2. DCT → IDCT round-trip on a single block
    block = np.random.rand(8, 8) * 255.0 - 128.0
    coeffs = dct2(block)
    reconstructed = idct2(coeffs)
    print(f"DCT/IDCT round-trip max error: {np.max(np.abs(block - reconstructed)):.2e}")

    # 3. Quantize / de-quantize
    from quantization import LUMA_REF, compute_qtable
    q50 = compute_qtable(LUMA_REF, 50)
    q_coef = quantize_blocks(np.expand_dims(np.expand_dims(coeffs, 0), 0), q50)
    dq_coef = dequantize_blocks(q_coef, q50)
    print(f"Quantization introduces error (expected > 0): "
          f"{np.max(np.abs(coeffs - dq_coef)):.2f}")

    # 4. Entropy of a uniform random array
    arr = np.random.randint(0, 256, size=(256, 256))
    ent = compute_entropy(arr)
    print(f"Entropy of 256×256 uniform random uint8 image: {ent:.4f} bits  (expected ≈ 8.0)")

""" Kann nicht benutzt werden, da zu langsam

def dct2(block):

    Berechnet die 2D-DCT eines 8x8-Blocks.
    block: 8x8 NumPy-Array

    S = np.zeros((8, 8))

    for u in range(8):
        for v in range(8):
            s = 0.0
            for x in range(8):
                for y in range(8):
                    s += (block[y, x] *
                          np.cos((2*x + 1) * u * np.pi / 16) *
                          np.cos((2*y + 1) * v * np.pi / 16))

            S[v, u] = 0.25 * c(u) * c(v) * s

    return S

    def idct2(S):

    Berechnet die inverse 2D-DCT.
    S: 8x8 DCT-Koeffizienten

    block = np.zeros((8, 8))

    for x in range(8):
        for y in range(8):
            s = 0.0
            for u in range(8):
                for v in range(8):
                    s += (c(u) * c(v) * S[v, u] *
                          np.cos((2*x + 1) * u * np.pi / 16) *
                          np.cos((2*y + 1) * v * np.pi / 16))

            block[y, x] = 0.25 * s

    return block """

#Beispiel:
""" block = np.array([
    [52, 55, 61, 66, 70, 61, 64, 73],
    [63, 59, 55, 90,109, 85, 69, 72],
    [62, 59, 68,113,144,104, 66, 73],
    [63, 58, 71,122,154,106, 70, 69],
    [67, 61, 68,104,126, 88, 68, 70],
    [79, 65, 60, 70, 77, 68, 58, 75],
    [85, 71, 64, 59, 55, 61, 65, 83],
    [87, 79, 69, 68, 65, 76, 78, 94]
], dtype=float)

S = dct2(block)
rekonstruiert = idct2(S)

print("DCT:")
print(np.round(S, 2))

print("\nRekonstruiertes Bild:")
print(np.round(rekonstruiert)) """
