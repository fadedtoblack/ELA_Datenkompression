import numpy as np

# Luminanz (Y) Referenztabelle der JPEG-Empfehlung T.81
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

# Chrominanz (Cb, Cr) Referenztabelle der JPEG-Empfehlung T.81
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
    #Fehlerbehandlung und Berechnung des Skalierungsfaktors aus dem Qualitaetswert (quality)
    if quality <= 0:
        raise ValueError("Quality must be > 0")
    if quality > 100:
        raise ValueError("Quality must be <= 100")
    if quality < 50:
        return 5000.0 / quality
    else:
        return 200.0 - 2.0 * quality


def compute_qtable(ref_table: np.ndarray, quality: float) -> np.ndarray:
    # berechnet die Quantisierungstabelle für eine gegebene Referenztabelle und einen Qualitaetswert (quality)
    
    # berechnet anhand Qualitaetsfaktors den Skalierungsfaktor s
    s = _scaling_factor(quality)

    # skaliert alle Werte der Referenztabelle mit s mit Abrundung
    scaled = np.floor((s * ref_table + 50.0) / 100.0)

    # Rueckgabe der skalierten Tabelle, Werte auf [1, 255] begrenzt
    return np.clip(scaled, 1, 255).astype(np.int32)


def compute_all_qtables(quality: float) -> dict:
    # Berechne Luminanz- und Chrominanz-Quantisierungstabellen fuer einen gegebenen Qualitaetswert (quality)
    
    luma_q = compute_qtable(LUMA_REF, quality)
    chroma_q = compute_qtable(CHROMA_REF, quality)
    return {"Y": luma_q, "Cb": chroma_q, "Cr": chroma_q}


def _mean_ac(qtable: np.ndarray) -> float:
    # Berechnet den Mittelwert der AC-Koeffizienten einer Quantisierungstabelle (qtable)
    
    flat = qtable.flatten()

    # ueberspringe Q_{0,0}
    ac_values = flat[1:]   
    return float(np.mean(ac_values))


def estimate_quality(qtables: dict) -> float:
  
    # Mittelwerte bilden der AC-Koeffizienten der Tabellen
    mean_y  = _mean_ac(qtables["Y"])
    mean_cb = _mean_ac(qtables["Cb"])
    mean_cr = _mean_ac(qtables["Cr"])

    # Mittelwert aus den drei Mittelwerten der Quantisierungstabellen
    mu = (mean_y + mean_cb + mean_cr) / 3.0                 

    # Approximiere Qualitaetswert
    d  = (abs(mean_y - mean_cb) + abs(mean_y - mean_cr)) * 0.49  
    q  = 100.0 - mu + d  
    
    q = max(0.0, min(100.0, q))                                       
    return q


def save_qtables_to_file(qtables: dict, quality: float, filepath: str) -> None:
    # Quantisierungstabellen in Textdateien speichern (Kapitel 2.2.5)
    # Schreibt auch den ermittelten Qualitaetswert zur Verifikation
    
    estimated_q = estimate_quality(qtables)
    with open(filepath, "w") as f:
        f.write(f"Quantisierungstabelle fuer Qualitaetswert = {quality:.1f}%\n")
        f.write(f"Approximierter Qualitaetswert: {estimated_q:.2f}%\n\n")
        for component in ("Y", "Cb", "Cr"):
            f.write(f"--- {component} Tabelle ---\n")
            for row in qtables[component]:
                f.write("  " + "  ".join(f"{v:4d}" for v in row) + "\n")
            f.write("\n")



# schneller Selbsttest, der die Referenztabellen mit den berechneten Tabellen fuer Q=50 vergleicht
# TODO: vor Abgbabe entfernen
if __name__ == "__main__":
    print("=== Quantization module self-test ===\n")

    # At Q=50 the computed tables should equal the reference tables exactly
    tables_50 = compute_all_qtables(50)
    luma_match   = np.array_equal(tables_50["Y"],  LUMA_REF.astype(np.int32))
    chroma_match = np.array_equal(tables_50["Cb"], CHROMA_REF.astype(np.int32))
    print(f"Q=50  →  Luma Tabelle passt zur Referenz:   {luma_match}")
    print(f"Q=50  →  Chroma Tabelle passt zur Referenz: {chroma_match}")

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
