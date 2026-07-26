"""
quantization.py
---------------
JPEG quantization tables (T.81 / ITU-R BT.601 reference) and
quality-scaling as required by the ELA-Tool assignment (SoSe 2026).

Reference tables taken from JPEG standard T.81, Annex K.
Equations (2.1), (2.2), (2.3), (2.4), (2.5) from the assignment spec.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Reference quantization tables from JPEG standard T.81, Annex K (Figure 2)
# ---------------------------------------------------------------------------

# Luminance (Y) reference table – 8x8, row-major order
LUMA_REF = np.array([
    [16, 11, 10, 16,  24,  40,  51,  61],
    [12, 12, 14, 19,  26,  58,  60,  55],
    [14, 13, 16, 24,  40,  57,  69,  56],
    [14, 17, 22, 29,  51,  87,  80,  62],
    [18, 22, 37, 56,  68, 109, 103,  77],
    [24, 35, 55, 64,  81, 104, 113,  92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103,  99],
], dtype=np.float64)

# Chrominance (Cb, Cr) reference table – 8x8, row-major order
CHROMA_REF = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
], dtype=np.float64)


def _scaling_factor(quality: float) -> float:
    """
    Compute the scaling factor S for a given quality value Q (0–100).
    Equation (2.2) from the assignment spec.

    S = 5000 / Q       if Q < 50
    S = 200 - 2*Q      if Q >= 50

    Q=1  → S=5000  (maximum quantization, lowest quality)
    Q=50 → S=100   (tables ≈ reference tables)
    Q=100 → S=0    (no quantization, lossless limit)
    """
    if quality <= 0:
        raise ValueError("Quality must be > 0")
    if quality > 100:
        raise ValueError("Quality must be <= 100")
    if quality < 50:
        return 5000.0 / quality
    else:
        return 200.0 - 2.0 * quality


def compute_qtable(ref_table: np.ndarray, quality: float) -> np.ndarray:
    """
    Compute a scaled quantization table for a given quality value.
    Equation (2.1) from the assignment spec:

        Q_uv = floor((S * Q_ref_uv + 50) / 100)

    Values are clipped to [1, 255].

    Parameters
    ----------
    ref_table : np.ndarray, shape (8, 8)
        Reference quantization table (LUMA_REF or CHROMA_REF).
    quality : float
        Quality value in the range (0, 100].

    Returns
    -------
    np.ndarray, shape (8, 8), dtype int32
    """
    s = _scaling_factor(quality)
    scaled = np.floor((s * ref_table + 50.0) / 100.0)
    return np.clip(scaled, 1, 255).astype(np.int32)


def compute_all_qtables(quality: float) -> dict:
    """
    Compute luma and chroma quantization tables for a given quality value.

    Returns
    -------
    dict with keys 'Y', 'Cb', 'Cr' each containing an (8, 8) int32 array.
    Cb and Cr share the same chroma table, as per the JPEG standard.
    """
    luma_q = compute_qtable(LUMA_REF, quality)
    chroma_q = compute_qtable(CHROMA_REF, quality)
    return {"Y": luma_q, "Cb": chroma_q, "Cr": chroma_q}


# ---------------------------------------------------------------------------
# Reverse: estimate quality from computed quantization tables (Section 2.2.4)
# ---------------------------------------------------------------------------

def _mean_ac(qtable: np.ndarray) -> float:
    """
    Compute the mean of the AC coefficients only (i.e. all entries except
    the DC coefficient at position [0, 0]).
    """
    flat = qtable.flatten()
    ac_values = flat[1:]   # skip index 0 (the DC coefficient Q_{0,0})
    return float(np.mean(ac_values))


def estimate_quality(qtables: dict) -> float:
    """
    Estimate the quality value from a set of quantization tables using the
    algorithm described in Section 2.2.4 (Equations 2.3–2.5).

    Algorithm:
        1. Compute mean AC values for Y, Cb, Cr tables.
        2. mu = (mean_Y_AC + mean_Cb_AC + mean_Cr_AC) / 3      (Eq. 2.3)
        3. D  = |mean_Y_AC - mean_Cb_AC| * 0.49
                + |mean_Y_AC - mean_Cr_AC| * 0.49              (Eq. 2.5)
        4. Q  = 100 - mu + D                                    (Eq. 2.4)

    Parameters
    ----------
    qtables : dict with keys 'Y', 'Cb', 'Cr'

    Returns
    -------
    float – estimated quality value (typically 0–100)
    """
    mean_y  = _mean_ac(qtables["Y"])
    mean_cb = _mean_ac(qtables["Cb"])
    mean_cr = _mean_ac(qtables["Cr"])

    mu = (mean_y + mean_cb + mean_cr) / 3.0                    # Eq. 2.3
    d  = (abs(mean_y - mean_cb) + abs(mean_y - mean_cr)) * 0.49  # Eq. 2.5
    q  = 100.0 - mu + d                                         # Eq. 2.4
    return q


def save_qtables_to_file(qtables: dict, quality: float, filepath: str) -> None:
    """
    Save computed quantization tables to a text file (Section 2.2.5).
    Also writes the estimated quality value for verification.
    """
    estimated_q = estimate_quality(qtables)
    with open(filepath, "w") as f:
        f.write(f"Quantization tables for quality Q = {quality:.1f}%\n")
        f.write(f"Estimated quality (reverse check): {estimated_q:.2f}%\n\n")
        for component in ("Y", "Cb", "Cr"):
            f.write(f"--- {component} table ---\n")
            for row in qtables[component]:
                f.write("  " + "  ".join(f"{v:4d}" for v in row) + "\n")
            f.write("\n")


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Quantization module self-test ===\n")

    # At Q=50 the computed tables should equal the reference tables exactly
    tables_50 = compute_all_qtables(50)
    luma_match   = np.array_equal(tables_50["Y"],  LUMA_REF.astype(np.int32))
    chroma_match = np.array_equal(tables_50["Cb"], CHROMA_REF.astype(np.int32))
    print(f"Q=50  →  luma table matches reference:   {luma_match}")
    print(f"Q=50  →  chroma table matches reference: {chroma_match}")

    # Reverse quality estimate for the reference tables should be ~51%
    ref_tables = {"Y": LUMA_REF.astype(np.int32),
                  "Cb": CHROMA_REF.astype(np.int32),
                  "Cr": CHROMA_REF.astype(np.int32)}
    est = estimate_quality(ref_tables)
    print(f"\nEstimated quality from reference tables: {est:.2f}%  (expected ≈ 51%)")

    # Show tables for a few quality levels
    for q in (10, 50, 75, 95):
        tables = compute_all_qtables(q)
        est_q  = estimate_quality(tables)
        print(f"\nQ={q:3d}%  →  Y[0,0]={tables['Y'][0,0]:4d}  "
              f"Cb[0,0]={tables['Cb'][0,0]:4d}  "
              f"estimated Q={est_q:.1f}%")
        print("  Y table:")
        for row in tables["Y"]:
            print("   ", " ".join(f"{v:4d}" for v in row))
