import numpy as np

# Zerlegung in 8x8 Blöcke  -------------------------------

def split_into_blocks(component: np.ndarray) -> tuple[np.ndarray, tuple]:
    """
    Aufteilung einer 2D-Komponente in 8x8-Blöcke

    Sind die Bildabmessungen kein Vielfaches von 8, wird die Komponente
    vor der Aufteilung symmetrisch aufgefüllt (Kantenreplikation)
    """
    H, W = component.shape
    original_shape = (H, W)

    # Bei Bedarf um ein Vielfaches von 8 aufrunden
    pad_h = (8 - H % 8) % 8
    pad_w = (8 - W % 8) % 8
    if pad_h > 0 or pad_w > 0:
        component = np.pad(component,
                           ((0, pad_h), (0, pad_w)),
                           mode="edge")

    H_pad, W_pad = component.shape
    n_h = H_pad // 8
    n_w = W_pad // 8

    # Umformen in (n_h, 8, n_w, 8) und anschließend Transponierung in (n_h, n_w, 8, 8)
    blocks = (component
              .reshape(n_h, 8, n_w, 8)
              .transpose(0, 2, 1, 3))          # -> (n_h, n_w, 8, 8)
    return blocks, original_shape


def merge_blocks(blocks: np.ndarray, original_shape: tuple) -> np.ndarray:
    """
    Zusammensetzen der 8x8-Blöcke in 2D-Komponente und Zuschneiden
    auf ursprügliche Bildgröße
    """
    n_h, n_w, _, _ = blocks.shape
    # Rücktransponierung und Umformung
    component = (blocks
                 .transpose(0, 2, 1, 3)        # -> (n_h, 8, n_w, 8)
                 .reshape(n_h * 8, n_w * 8))
    H, W = original_shape
    return component[:H, :W]

# DCT und IDCT (JPEG-Standard T.81, A.3.3)----------------------

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
    #Berechnet den Faktor C_u bzw. C_v
    return 1 / np.sqrt(2) if k == 0 else 1

def dct2(block: np.ndarray) -> np.ndarray:
    # Manuelle DCT mittels Matrixmultiplikation
    return DCT_MATRIX @ block @ DCT_MATRIX.T

# IDCT nach der Formel:

def idct2(S: np.ndarray) -> np.ndarray:
    # Manuelle IDCT mittels Matrixmultiplikation
    return DCT_MATRIX.T @ S @ DCT_MATRIX


def apply_dct_to_blocks(blocks: np.ndarray) -> np.ndarray:
    #Anwendung der DCT auf jeden 8x8-Block
    shifted = blocks.astype(np.float64)    
    n_h, n_w = blocks.shape[:2]
    dct_blocks = np.zeros_like(shifted)
    for i in range(n_h):
        for j in range(n_w):
            dct_blocks[i, j] = dct2(shifted[i, j])
    return dct_blocks


def apply_idct_to_blocks(dct_blocks: np.ndarray) -> np.ndarray:
    # Anwendung der IDCT auf jeden 8x8-Block 
    n_h, n_w = dct_blocks.shape[:2]
    spatial = np.zeros_like(dct_blocks)
    for i in range(n_h):
        for j in range(n_w):
            spatial[i, j] = idct2(dct_blocks[i, j])
    return spatial

# Quantisierung und Dequantisierung ------------------------------------------

def quantize_blocks(dct_blocks: np.ndarray, qtable: np.ndarray) -> np.ndarray:
    """
    Quantisierung der DCT-Koeffizienten durch Dividierung mit der 
    Quantisierungstabelle und Rundung auf die nächste ganze Zahl
    """
    return np.round(dct_blocks / qtable.astype(np.float64)).astype(np.int32)


def dequantize_blocks(q_blocks: np.ndarray, qtable: np.ndarray) -> np.ndarray:
    """
    Dequantisierung der Koeffizienten durch Multiplikation mit der 
    Quantisierungstabelle
    """
    return q_blocks.astype(np.float64) * qtable.astype(np.float64)

# Auslesen der 8×8 Blöcke in Textdatei ----------------------------

def save_blocks_to_file(blocks: np.ndarray, filepath: str,decimals: int = 3, eps: float = 1e-6) -> None:
    
    n_h, n_w = blocks.shape[:2]
    block_num = 1
    with open(filepath, "w") as f:
        for i in range(n_h):
            for j in range(n_w):
                flat = blocks[i, j].flatten().astype(np.float64)
                cleaned = np.where(np.abs(flat) < eps, 0.0, flat)
                values = "; ".join(
                    f"{k+1}: {cleaned[k]:.{decimals}f}" for k in range(64)
                )
                f.write(f"Block: {block_num}\n{values}\n")
                block_num += 1

# Entropieberechnung -------------------------------------------

def compute_entropy(data: np.ndarray) -> float:
    """
    Berechnung des mittleren Informationsgehalts eines Arrays, 
    wobei alle Werte als Symbole eines diskreten Alphabets betrachtet werden

    H = -sum( p_i * log2(p_i) )   für alle Wahrscheinlichkeiten p_i ungleich Null
    """
    flat = data.flatten()
    # Rundung von Gleitkommawerten auf ganzzahlige Klassen für die Häufigkeitszählung
    if np.issubdtype(flat.dtype, np.floating):
        flat = np.round(flat).astype(np.int64)
    else:
        flat = flat.astype(np.int64)

    _, counts = np.unique(flat, return_counts=True)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))



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
