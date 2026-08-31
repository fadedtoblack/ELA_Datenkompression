"""
encoder.py
----------

Pipeline
    1. Eingabebild laden (PNG oder JPEG)
    2. RGB -> YCbCr  (ITU-R BT.601-4)
    3. Jede Komponente in 8×8 Bloecke aufteilen
    4. Vorwaerts-2-D-DCT fuer jeden Block 
       (Level-Shift -128 wird innerhalb von dct_blocks angewendet)
    5. Quantisierung mit qualitaetsabhaengig skalierten Tabellen
    6. (Optional) Zwischenergebnisse in Textdateien speichern
    7. Entropiewerte berechnen und anzeigen (insgesamt 12)
    8. Quantisierte DCT-Koeffizienten an den Decoder zurueckgeben


Bild-Kompression
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
    """
    Container fuer alle Outputs, die vom Encoder produziert werden.

        q_dct: quantisierte DCT_Koeffizient pro Kanal (Y,Cb,Cr) -> eigentliche 'komprimierte' Information
        original_shapes: urspruenglichen Bildmasse pro Komponente (wichtig beim Zerlegen in 8x8-Bloecke)
        qtables: verwendete Quantisierungstabellen
        original_rgb: Originalbild, nur fuer spaeteren Vergleich wichtig
        entropies: Statistik-Werte fuer den Bericht
        quality/estimated_quality: angeforderte vs. die aus Tabelle zurueckrerechnete Qualitaet
    """

    def __init__(self):
        # Quantisierte DCT-Koeffizienten für jede Komponente
        self.q_dct: dict[str, np.ndarray] = {}        # 'Y', 'Cb', 'Cr'
        # Urspruengliche Blockgroessen (werden vom Decoder zum Wiederzusammensetzen benoetigt)
        self.original_shapes: dict[str, tuple] = {}   # 'Y', 'Cb', 'Cr'
        # Verwendete Quantisierungstabellen
        self.qtables: dict[str, np.ndarray] = {}
        # Urspruengliches Bild-Array (als Referenz für PSNR / ELA)
        self.original_rgb: np.ndarray | None = None
        # Entropiewerte (insgesamt 12): Schluessel in print_entropies() beschrieben
        self.entropies: dict[str, float] = {}
        # Verwendeter Qualitaetswert
        self.quality: float = 50.0
        # Ueberpruefte (rueckgeschaetzte) Qualitaet
        self.estimated_quality: float = 0.0


def encode(image_path: str,
           quality: float = 50.0,
           save_intermediates: bool = False,
           output_dir: str = "output") -> EncoderResult:
    
    """
    Fuehrt die vollstaendige Encoder-Pipeline fuer eine Bilddatei aus.

    Parameter
    ----------
    image_path         : str   – Pfad zum PNG oder JPEG Eingabebild
    quality            : float – Qualitaetswert Q in (0, 100]
    save_intermediates : bool  – wenn True, werden Textdateien und 
                                 Komponentenbilder gespeichert
    output_dir         : str   – Verzeichnis fuer alle Ausgabedateien

    Rueckgabe
    -------
    EncoderResult
    """
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    result = EncoderResult()
    result.quality = quality


    # --------- 1. Bild laden -----------------------------------------
    # load_image aus colorspace.py 

    print(f"[Encoder] Bild wird geladen: {image_path}")
    rgb = load_image(image_path)  
    result.original_rgb = rgb
    print(f"  Bildgröße: {rgb.shape[1]}×{rgb.shape[0]}  (B×H)")

    
    # --------- 2. Quantisierungstabellen berechnen --------------------
    # aus dem quality-Wert werden die eigentlichen 8x8-Quantisierungstabellen berechnet 
    # estimate_quality() rechnet Tabellen zurueck, welche Qualitaet sie tatsaechlich repraesentieren

    qtables = compute_all_qtables(quality)
    result.qtables = qtables
    result.estimated_quality = estimate_quality(qtables)
    print(f"  Qualität: Q={quality:.1f}%  "
          f"(rückgeschätzt: {result.estimated_quality:.2f}%)")

    if save_intermediates:
        qt_path = os.path.join(output_dir, f"{base_name}_qtables_q{quality:.0f}.txt")
        save_qtables_to_file(qtables, quality, qt_path)
        print(f"  Quantisierungstabellen gespeichert: {qt_path}")


    # --------- 3. Farbraumtransformation RGB -> YCbCr ------------------

    print("[Encoder] Farbraumtransformation RGB -> YCbCr ...")
    Y, Cb, Cr = rgb_to_ycbcr(rgb)

    Y  = np.clip(np.round(Y),  0, 255)
    Cb = np.clip(np.round(Cb), 0, 255)
    Cr = np.clip(np.round(Cr), 0, 255)

    # Entropie der R, G, B Kanaele
    result.entropies["H_R"]  = compute_entropy(rgb[:, :, 0])
    result.entropies["H_G"]  = compute_entropy(rgb[:, :, 1])
    result.entropies["H_B"]  = compute_entropy(rgb[:, :, 2])

    # Entropien der Y (Luminanz), Cb und Cr (Chrominanz) Kanaele
    result.entropies["H_Y"]  = compute_entropy(Y)
    result.entropies["H_Cb"] = compute_entropy(Cb)
    result.entropies["H_Cr"] = compute_entropy(Cr)

    if save_intermediates:
        # Y/Cb/Cr-Textdateien speichern (blockweise -> hierfür werden die
        # Rohwerte der Komponenten in 8×8-Kacheln angeordnet)
        save_component_images(Y, Cb, Cr, base_name, output_dir)
        for comp_name, comp_data in (("y", Y), ("cb", Cb), ("cr", Cr)):
            blks, _ = split_into_blocks(comp_data)
            txt_path = os.path.join(output_dir,
                                    f"{base_name}_{comp_name}.txt")
            save_blocks_to_file(blks, txt_path)
            print(f"  Farbraumblöcke gespeichert: {txt_path}")


    
    # --------- 4. 8×8 Block-Zerlegung  +  5. DCT ----------------------

    print("[Encoder] Blockzerlegung und DCT ...")
    components = {"Y": Y, "Cb": Cb, "Cr": Cr}
    dct_results  = {}
    qdct_results = {}

    for comp_name, comp_data in components.items():
        # In Bloecke aufteilen (Padding wird innerhalb von split_into_blocks behandelt)
        blocks, orig_shape = split_into_blocks(comp_data)
        result.original_shapes[comp_name] = orig_shape

        # Vorwaerts-DCT (Level-Shift wird innerhalb von apply_dct_to_blocks angewendet)
        dct_blks = apply_dct_to_blocks(blocks)
        dct_results[comp_name] = dct_blks

        # Entropie der DCT-Koeffizienten
        result.entropies[f"H_{comp_name}_dct"] = compute_entropy(dct_blks)


        # --------- 6. Quantisierung ---------------------------------------

        q_key = "Y" if comp_name == "Y" else "Cb"  # Cb und Cr verwenden dieselbe Chrominanz-Tabelle
        q_blks = quantize_blocks(dct_blks, qtables[comp_name])
        qdct_results[comp_name] = q_blks

        # Entropie der quantisierten DCT-Koeffizienten
        result.entropies[f"H_{comp_name}_qdct"] = compute_entropy(q_blks)

        if save_intermediates:
            comp_lower = comp_name.lower()
            # DCT-Textdateien
            dct_txt = os.path.join(output_dir,
                                   f"{base_name}_{comp_lower}_dct.txt")
            save_blocks_to_file(dct_blks, dct_txt)
            print(f"  DCT-Blöcke gespeichert:  {dct_txt}")

            # Quantisierte DCT-Textdateien
            qdct_txt = os.path.join(output_dir,
                                    f"{base_name}_{comp_lower}_qdct.txt")
            save_blocks_to_file(q_blks, qdct_txt)
            print(f"  QDCT-Blöcke gespeichert: {qdct_txt}")

    result.q_dct = qdct_results


    # --------- 7. Ausgabe der Zusammenfassung aller Entropien ---------
    
    _print_entropies(result.entropies)

    # Entropie optional in einer Datei speichern
    if save_intermediates:
        ent_path = os.path.join(output_dir, f"{base_name}_entropy_q{quality:.0f}.txt")
        _save_entropies(result.entropies, ent_path, quality)
        print(f"  Entropiewert gespeichert: {ent_path}")

    print("[Encoder] Abgeschlossen.")
    return result


def _print_entropies(entropies: dict) -> None:
    print("\n[Encoder] Entropiewerte (Bits pro Symbol):")
    order = [
        ("H_R",        "R-Kanal"),
        ("H_G",        "G-Kanal"),
        ("H_B",        "B-Kanal"),
        ("H_Y",        "Y-Komponente"),
        ("H_Cb",       "Cb-Komponente"),
        ("H_Cr",       "Cr-Komponente"),
        ("H_Y_dct",    "Y-DCT-Koeffizienten"),
        ("H_Cb_dct",   "Cb-DCT-Koeffizienten"),
        ("H_Cr_dct",   "Cr-DCT-Koeffizienten"),
        ("H_Y_qdct",   "Y-quantisierte DCT"),
        ("H_Cb_qdct",  "Cb-quantisierte DCT"),
        ("H_Cr_qdct",  "Cr-quantisierte DCT"),
    ]
    for key, label in order:
        val = entropies.get(key, float("nan"))
        print(f"  {label:<30s}: {val:.4f} bit")


def _save_entropies(entropies: dict, filepath: str, quality: float) -> None:
    with open(filepath, "w") as f:
        f.write(f"Entropiewerte für Q={quality:.1f}%\n\n")
        for key, val in entropies.items():
            f.write(f"{key}: {val:.6f}\n")
