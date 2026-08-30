import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import os
from PIL import Image, ImageTk
import threading
import queue
import numpy as np
from encoder import encode
from decoder import decode

from ela import generate_ela

# FARBSCHEMA
BG_APP        = "#14161c"   # Hintergrund des Hauptfensters
BG_CARD       = "#1c1f27"   # Hintergrund der Karten
BG_CARD_ALT   = "#20242e"   # leicht abgesetzter Kartenhintergrund (z.B. PSNR-Box)
BORDER_COLOR  = "#2c303a"   # Kartenrahmen
TEXT_PRIMARY  = "#f0f1f3"
TEXT_SECOND   = "#9aa0ac"
ACCENT        = "#4a90d9"
TROUGH_COLOR  = "#2c303a"

FONT_TITLE    = ("Segoe UI", 16, "bold")
FONT_SUBTITLE = ("Segoe UI", 9)
FONT_HEADING  = ("Segoe UI", 10, "bold")
FONT_NORMAL   = ("Segoe UI", 9)
FONT_STAT_VAL = ("Segoe UI", 11, "bold")
FONT_BIG_VAL  = ("Segoe UI", 16, "bold")

def make_card(parent, **grid_opts):
    """
    Erstellt einen dunklen Hintergrund und Rahmen.
    """
    card = tk.Frame(parent, bg=BG_CARD, bd=1, relief="solid", highlightbackground=BORDER_COLOR, highlightcolor=BORDER_COLOR, highlightthickness=1)
    card.grid(**grid_opts)
    return card

# CACHE & THREAD-STEUERUNG
_cache_lock = threading.Lock()
_cache = {"path": None, "quality": None, "encoder_result": None, "ela_image": None, "psnr": None}
_compute_generation = 0 # wird bei jeder neuen Berechnunganfrage erhöht
_computing = False 
_result_queue = queue.Queue()  # Queue für Ergebnisse der Hintergrundberechnung

def request_quality_computation(path, quality):
    """
    Startet encode() + decode() im Hintergrund-Thread.
    """

    global _compute_generation, _computing
    with _cache_lock:
        _compute_generation += 1
        my_generation = _compute_generation
        _computing = True
    lbl_psnr_out.config(text="Berechnung läuft...")
    start_progress()
    threading.Thread(target=_background_compute,
                     args=(path, quality, my_generation),
                     daemon=True).start()
    
def _background_compute(path, quality, generation):
    encode_result = encode(path, quality=quality, save_intermediates=False)
    reproduced_rgb, psnr = decode(encode_result, image_name="preview", save_output=False)
    _result_queue.put((generation, path, quality, encode_result, reproduced_rgb, psnr))


def _poll_queue():
    """
    Laeuft alle 50ms im Haupt_Thread, holt fertige Ergebnisse ab.
    """

    global _computing
    updated = False
    try:
        while True:
            generation, path, quality, encode_result, reproduced_rgb, psnr = _result_queue.get_nowait()
            with _cache_lock:
                if generation == _compute_generation:
                    _cache.update(path=path, 
                                  quality=quality, 
                                  encoder_result=encode_result,
                                  reproduced_rgb=reproduced_rgb,
                                  psnr=psnr)
                    _computing = False
                    updated = True
    except queue.Empty:
        pass

    if updated:
        stop_progress()
        apply_current_multiplier()  # Cache verwenden, nur ELA-Bild neu berechnen
    root.after(50, _poll_queue)  # alle 50ms erneut prüfen

def apply_current_multiplier():
    """
    Rechnet aus dem zuletzt verfuegbaren Cache-Stand das ELA-Bild neu, unabhaengig
    davon, ob gerade im Hintergrund ein neuer Quality Wert laeuft.
    """ 

    global pic_new
    with _cache_lock:
        if _cache["encoder_result"] is None:
            return  # noch kein Ergebnis verfügbar
        original_rgb = _cache["encoder_result"].original_rgb.astype(np.float64)
        reproduced_f = _cache["reproduced_rgb"].astype(np.float64)
        psnr = _cache["psnr"]

    multiplier_val = multiplier_var.get()
    ela_float = np.abs(original_rgb - reproduced_f) * multiplier_val
    ela_array = np.clip(ela_float, 0.0, 255.0).astype(np.uint8)

    if psnr is None:
        lbl_psnr_out.config(text="-")
        return

    lbl_psnr_out.config(text=f"{psnr:.4f} dB")
    pic_new = Image.fromarray(ela_array)
    refresh_previews()  # Vorschau-Bilder aktualisieren

    # Infozeile unter dem ELA-Bild aktualisieren (Aufloesung, Q, M, PSNR)
    h, w = ela_array.shape[:2]
    lbl_ela_info.config(
        text=f"{w} × {h}px   •   Q: {quality_var.get()}   •   "
             f"M: {multiplier_val}   •   PSNR: {psnr:.2f} dB"
    )


# Datei auswählen, Dateinamen einfügen und Bild in Feld laden

def datei_auswaehlen():
    global pic_old, pic_new, dateipfad_global

    # Datei-Dialog öffnen 
    dateipfad = filedialog.askopenfilename(
        title="Bild auswählen", 
        filetypes=[
            ("Bilddateien", "*.png *.jpg *.jpeg *.gif *.bmp"),
            ("Alle Dateien", "*.*")
        ]
    )

    # wenn Datei ausgewählt
    if dateipfad:
        dateiname = os.path.basename(dateipfad)
        entry_var.set(dateiname)
        dateipfad_global = dateipfad  # Pfad merken, wird bei jedem Slider-Update erneut an generate_ela() übergeben

        #Bild laden
        pic_old = Image.open(dateipfad)
        pic_new = pic_old.copy()

        # Anzeige der Vorschaubilder aktualisieren
        refresh_previews()  # Vorschau-Bilder aktualisieren


        lbl_psnr_out.config(text="-") # PSNR-Anzeige zuruecksetzen bei neuem Bild

        # Bildinfo-Zeile unter dem Originalbild aktualisieren (Aufloesung, Dateigroesse)
        try:
            w, h = pic_old.size
            size_bytes = os.path.getsize(dateipfad)
            size_mb = size_bytes / (1024 * 1024)
            lbl_orig_info.config(text=f"{w} × {h}px   •   {size_mb:.1f} MB")
        except Exception:
            lbl_orig_info.config(text="")

        # erste ELA_Vorschau mit aktuellem Slidewerten berechnen
        update_image()


# Resize der Bilder - passt in eine (max_width, max_height)-Box, Seitenverhaeltnis bleibt erhalten
def resize_image(img, max_width, max_height):
    ratio = min(max_width / img.width, max_height / img.height)
    new_w = max(1, int(img.width * ratio))
    new_h = max(1, int(img.height * ratio))
    return img.resize((new_w, new_h))


# Update der Schieberegler anzeigen 
def update_label(raw_value, label, int_var):
    int_value = int(round(float(raw_value)))
    int_var.set(int_value)
    label.config(text=f"{int_value}")

def _current_slider_length():
    total_width = root.winfo_width()
    return max(80, int(total_width * 0.2) - 20)  # mindestens 80 Pixel, sonst 20% der Fensterbreite

def _preview_box_for(card, title_label, info_label):
    """
    Berechnet die maximal verfuegbare Breite/Hoehe fuer das Vorschaubild
    innerhalb einer Karte - basierend auf der TATSAECHLICHEN aktuellen
    Kartengroesse, nicht auf einem pauschalen Schaetzwert. Das verhindert,
    dass ein Bild ueber den Kartenrand hinausragt und dabei die Infozeile
    darunter verdeckt.
    """
    card.update_idletasks()
    total_h = card.winfo_height()
    reserved = title_label.winfo_reqheight() + info_label.winfo_reqheight() + 40
    max_h = max(100, total_h - reserved)
    max_w = max(100, card.winfo_width() - 32)  # 16px Innenabstand auf beiden Seiten
    return max_w, max_h

def refresh_previews():
    """
    Zeichnet die Original- und ELA-Vorschau-Bilder neu, wenn das 
    Fenster skaliert wird. Die maximale Groesse wird direkt aus der
    tatsaechlichen Kartengroesse abgeleitet (robuster als ein pauschaler
    Fensterhoehen-Schaetzwert, der bei unterschiedlich hohen Karten zu
    Bildern fuehren kann, die ueber den Kartenrand hinausragen).
    """
    if pic_old is not None:
        max_w, max_h = _preview_box_for(card_original, lbl_original_title, lbl_orig_info)
        img_old = resize_image(pic_old.copy(), max_w, max_h)
        tk_img_old = ImageTk.PhotoImage(img_old)
        lbl_pic_old.config(image=tk_img_old)
        lbl_pic_old.image = tk_img_old

    if pic_new is not None:
        max_w, max_h = _preview_box_for(card_ela, lbl_ela_title, lbl_ela_info)
        img_new = resize_image(pic_new.copy(), max_w, max_h)
        tk_img_new = ImageTk.PhotoImage(img_new)
        lbl_pic_new.config(image=tk_img_new)
        lbl_pic_new.image = tk_img_new

_resize_job = None  # Variable, um den geplanten Job zu speichern
_RESIZE_DELAY = 150  # Verzögerung in Millisekunden

def on_window_resize(event):
    """
    Wird bei jeder Groessenaenderung des Hauptfensters aufgerufen und 
    passt die Reglerlaenge und die Vorschau-Bilder an.
    """
    global _resize_job

    if _resize_job is not None:
        root.after_cancel(_resize_job)  # Vorherigen Job abbrechen

    _resize_job = root.after(_RESIZE_DELAY, _apply_resize)  # Neuen Job planen

def _apply_resize():
    """
    Fuehrt die eigentliche Anpassung der Vorschau-Bilder und Reglerlaenge durch,
    nachdem die Verzögerung abgelaufen ist.
    """

    global _resize_job
    _resize_job = None  # Job abgeschlossen

    new_length = _current_slider_length()
    scale_quality.config(length=new_length)
    scale_multiplier.config(length=new_length)
    progress_canvas.config(width=new_length)

    refresh_previews()  # Vorschau-Bilder aktualisieren



# Bild live bearbeiten -> ELA-Vorschau berechnen 
def update_image(*args):
    """
    Wird beim Loslassen des Quality-Reglers (und beim Bild öffnen) aufgerufen.
    """

    if dateipfad_global is None:
        return
    quality_val = quality_var.get()
    with _cache_lock:
        cached_valid = (_cache["path"] == dateipfad_global and 
                        _cache["quality"] == quality_val and
                        _cache["encoder_result"] is not None)

    if cached_valid:
        apply_current_multiplier()  # Cache verwenden, nur ELA-Bild neu berechnen
    else:
        request_quality_computation(dateipfad_global, quality_val)  # Neue Berechnung starten



# Berechnung des ELA-Bildes beim loslassen des Sliders (Quality-Regler)
def on_slider_release(event):
    update_image()

def on_multiplier_release(event):
    with _cache_lock:
        currently_computing = _computing

    if currently_computing:
        return  # Wenn gerade eine Berechnung laeuft, nicht neu starten

    apply_current_multiplier()  # nur ELA-Bild neu berechnen, kein neues Encoding/Decoding

# Ladebalken animieren
def _animate_progress_bar():
    global _progress_pos, _progress_job
    progress_canvas.delete("all")

    
    width = progress_canvas.winfo_width()
    height = 10
    # Hintergrund-Leiste
    progress_canvas.create_rectangle(0, 0, width, height,
                                     fill="#e0e0e0", outline="")

    # bewegtes Segment
    seg_width = max(20, width // 4)
    x0 = _progress_pos
    x1 = x0 + seg_width
    progress_canvas.create_rectangle(x0, 0, x1, height,
                                    fill="#4a90d9", outline="")

    _progress_pos += 3
    if _progress_pos > width:
        _progress_pos = -seg_width

    _progress_job = root.after(20, _animate_progress_bar)


def start_progress():
    global _progress_pos
    _progress_pos = -40
    progress_canvas.grid()
    _animate_progress_bar()

def stop_progress():
    global _progress_job
    if _progress_job is not None:
        root.after_cancel(_progress_job)
        _progress_job = None
    progress_canvas.delete("all")

    

# Bild Speichern 
def make_flat_button(parent, text, command, bg, fg, hover_bg, font, padx=14, pady=6):
    """
    Baut einen klickbaren 'Button' aus einem tk.Label statt tk.Button.
    """
    btn = tk.Label(parent, text=text, bg=bg, fg=fg, font=font,
                   padx=padx, pady=pady, cursor="hand2")

    def _on_enter(event):
        btn.config(bg=hover_bg)

    def _on_leave(event):
        btn.config(bg=bg)

    def _on_click(event):
        command()

    btn.bind("<Enter>", _on_enter)
    btn.bind("<Leave>", _on_leave)
    btn.bind("<Button-1>", _on_click)
    return btn


def save_pic():
    global pic_new

    if dateipfad_global is None:
        print("Kein Bild vorhanden!")
        return

    quality_val = quality_var.get()
    multiplier_val = multiplier_var.get()

    ausgabe_ordner = filedialog.askdirectory(title="Ausgabeordner auswählen")
    if not ausgabe_ordner:
        return

    try:
        generate_ela(
            dateipfad_global,
            quality=quality_val,
            multiplier=multiplier_val,
            output_dir=ausgabe_ordner,
            save_output=True,
            save_intermediates=save_intermediates_var.get()
        )
        print("ELA_Bild gepseichert im Ordner:", ausgabe_ordner)
    except Exception as e:
        print("Fehler beim Speichern des ELA-Bildes:", e)



# Erstellen GUI Fenster 

PREVIEW_WIDTH = 350

root = tk.Tk()
root.title("ELA - Tool")
root.geometry("1200x700")
root.minsize(width=1000, height=650) 
root.configure(bg=BG_APP) 

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.maxsize(width=screen_width, height=screen_height)  # Maximale Größe auf Bildschirmgröße setzen

style = ttk.Style()
style.theme_use("clam")
style.configure("TScale",
                background=BG_CARD,
                troughcolor=TROUGH_COLOR,
                lightcolor=ACCENT,
                darkcolor=ACCENT)

# globale Zustandsvariablen 
pic_old = None
pic_new = None
dateipfad_global = None
_progress_bar_id = None
_progress_pos = 0
_progress_job = None

# Einstellung Spaltengröße/Zeilenhoehe
root.columnconfigure(0, weight=1)
root.rowconfigure(2, weight=1, minsize=350)  


# HEADER (Titel)
header = tk.Frame(root, bg=BG_APP)
header.grid(column=0, row=0, sticky="ew", padx=24, pady=(20,10))
header.columnconfigure(0, weight=1)

title_box = tk.Frame(header, bg=BG_APP)
title_box.grid(column=0, row=0, sticky="w")

lbl_title = tk.Label(title_box, text="ELA-Analyzer", font=FONT_TITLE, bg=BG_APP, fg=TEXT_PRIMARY)
lbl_title.pack(anchor="w")

lbl_subtitle = tk.Label(title_box, text="Error-Level-Analyse für die digitale Bildforensik",
                        font=FONT_SUBTITLE, bg=BG_APP, fg=TEXT_SECOND)
lbl_subtitle.pack(anchor="w")


# Button und Textfeld Dateiauswahl
file_card = make_card(root, column=0, row=1, sticky="ew", padx=24, pady=(0, 14))
file_card.columnconfigure(1, weight=1)

entry_var = tk.StringVar(value="Keine Datei ausgewählt")

tk.Label(file_card, text="Bild auswählen", font=FONT_HEADING, bg=BG_CARD, fg=TEXT_PRIMARY).grid(column=0, row=0, sticky="w", padx=16, pady=(14, 0))

lbl_dataname = tk.Label(file_card, textvariable=entry_var, font=FONT_NORMAL, bg=BG_CARD, fg=TEXT_SECOND, anchor="w")
lbl_dataname.grid(column=0, row=1, columnspan=2, sticky="w", padx=16, pady=(2,14))

btn_chosedata = make_flat_button(file_card, "Durchsuchen", datei_auswaehlen,
                                 bg=BG_CARD_ALT, fg=TEXT_PRIMARY, hover_bg=ACCENT,
                                 font=FONT_NORMAL, padx=14, pady=6)
btn_chosedata.grid(column=2, row=0, rowspan=3, sticky="e", padx=16, pady=14)
file_card.columnconfigure(2, weight=0)


# HAUPTBEREICH: Originalbild, Parameter, ELA-Bild

main_area =tk.Frame(root, bg=BG_APP)
main_area.grid(column=0, row=2, sticky="nsew", padx=24, pady=(0, 14))
main_area.columnconfigure(0, weight=2)
main_area.columnconfigure(1, weight=1)
main_area.columnconfigure(2, weight=2)
main_area.rowconfigure(0, weight=1)


# Bildanzeige Originalbild 
card_original = make_card(main_area, column=0, row=0, sticky="nsew", padx=(0, 10))
card_original.columnconfigure(0, weight=1)
card_original.columnconfigure(1, weight=1)
card_original.rowconfigure(1, weight=1)

lbl_original_title = tk.Label(card_original, text="Originalbild", font=FONT_HEADING, bg=BG_CARD,
         fg=TEXT_PRIMARY)
lbl_original_title.grid(column=0, row=0, sticky="w", padx=16, pady=(14, 8))

lbl_pic_old = tk.Label(card_original, bg=BG_CARD)
lbl_pic_old.grid(column=0, row=1, padx=16, pady=(0, 8))

lbl_orig_info = tk.Label(card_original, text="", font=FONT_NORMAL, bg=BG_CARD,
                         fg=TEXT_SECOND)
lbl_orig_info.grid(column=0, row=2, sticky="w", padx=16, pady=(0, 14))


#Frame für Regler und Felder Mitte 
card_params = make_card(main_area, column=1, row=0, sticky="nsew", padx=10)
card_params.columnconfigure(0, weight=1)

tk.Label(card_params, text="Parameter", font=FONT_HEADING, bg=BG_CARD, 
         fg=TEXT_PRIMARY).grid(column=0, row=0, sticky="w", padx=16, pady=(14, 12))


# Parameter 1: Quality (Q)
tk.Label(card_params, text="Quality (Q)", font=FONT_NORMAL, bg=BG_CARD, 
         fg=TEXT_PRIMARY).grid(column=0, row=1, sticky="w", padx=16)

quality_var = tk.IntVar(value=75) # Default bei 75
quality_raw = tk.DoubleVar(value=75.0) # Raw-Wert für den Slider, um die Genauigkeit zu erhalten

scale_quality = ttk.Scale(card_params, 
                        from_=1, to=100, 
                        orient="horizontal",
                        length=150, 
                        variable=quality_raw, # Regler haengt an der kontinuierlichen Variable, die den genauen Wert speichert
                        command=lambda v: update_label(v, lbl_quality_out, quality_var)
                        )
scale_quality.grid(column=0, row=2, padx=16, pady=(4,2), sticky="ew")
scale_quality.bind("<ButtonRelease-1>", on_slider_release)  # Event-Handler für Slider loslassen

lbl_quality_out =tk.Label(card_params, text="75", font=FONT_NORMAL, bg=BG_CARD,
                          fg=TEXT_SECOND) # Default bei 75
lbl_quality_out.grid(column=0, row=3, sticky="w", padx=16, pady=(0, 16))


# Parameter 2: Multiplier (M)
tk.Label(card_params, text="Multiplier (M)", font=FONT_NORMAL, bg=BG_CARD,
         fg=TEXT_PRIMARY).grid(column=0, row=4, sticky="w", padx=16)

multiplier_var = tk.IntVar(value=30) # Default bei 30
multiplier_raw = tk.DoubleVar(value=30.0) # Raw-Wert für den Slider, um die Genauigkeit zu erhalten

scale_multiplier = ttk.Scale(card_params,
                        from_=1, to=100, 
                        orient="horizontal", 
                        length=150, 
                        variable=multiplier_raw, # Regler haengt an der kontinuierlichen Variable, die den genauen Wert speichert
                        command=lambda v: update_label(v, lbl_multiplier_out, multiplier_var))
scale_multiplier.grid(column=0, row=5, padx=16, pady=(4,2), sticky="ew")
scale_multiplier.bind("<ButtonRelease-1>", on_multiplier_release)  # Event-Handler für Slider loslassen

lbl_multiplier_out =tk.Label(card_params, text="30", font=FONT_NORMAL, 
                             bg=BG_CARD, fg=TEXT_SECOND ) # Default bei 30
lbl_multiplier_out.grid(column=0, row=6, sticky="w", padx=16, pady=(0, 16))


# PSNR - Anzeige
psnr_box = tk.Frame(card_params, bg=BG_CARD_ALT, bd=1, relief="solid",
                    highlightbackground=BORDER_COLOR, highlightthickness=1)
psnr_box.grid(column=0, row=7, sticky="ew", padx=16, pady=(0,8))

tk.Label(psnr_box, text="PSNR", font=FONT_NORMAL, bg=BG_CARD_ALT,
         fg=TEXT_SECOND).pack(anchor="w", padx=14, pady=(10,0))

lbl_psnr_out = tk.Label(psnr_box, text="-", font=FONT_BIG_VAL, bg=BG_CARD_ALT,
                       fg=ACCENT)
lbl_psnr_out.pack(anchor="w", padx=14, pady=(0, 4))  # <- fehlte komplett, Label war dadurch unsichtbar

tk.Label(psnr_box, text="Peak Signal-to-Noise Ratio", font=("Segoe UI", 8), 
         bg=BG_CARD_ALT, fg=TEXT_SECOND).pack(anchor="w", padx=14, pady=(0, 10))


# Ladebalken
progress_canvas =tk.Canvas(card_params, width=150, height=8,
                           highlightthickness=0, bg=BG_CARD)
progress_canvas.grid(column=0, row=8, sticky="ew", padx=16, pady=(5, 0))


# Bildanzeige neues Bild 
card_ela = make_card(main_area, column=2, row=0, sticky="nsew", padx=(10, 0))
card_ela.columnconfigure(0, weight=1)
card_ela.rowconfigure(1, weight=1)

lbl_ela_title = tk.Label(card_ela, text="ELA-Darstellung", font=FONT_HEADING, bg=BG_CARD, 
         fg=TEXT_PRIMARY)
lbl_ela_title.grid(column=0, row=0, sticky="w", padx=16, pady=(14, 8))

lbl_pic_new = tk.Label(card_ela, bg=BG_CARD)
lbl_pic_new.grid(column=0, row=1, padx=16, pady=(0, 8))
 
lbl_ela_info = tk.Label(card_ela, text="", font=FONT_NORMAL,
                        bg=BG_CARD, fg=TEXT_SECOND)
lbl_ela_info.grid(column=0, row=2, sticky="w", padx=16, pady=(0, 14))


# Checkbox um die Intermediates beim Speichern mit zu exportieren
footer = make_card(root, column=0, row=3, sticky="ew", padx=24, pady=(0, 20))
footer.columnconfigure(0, weight=1)

save_intermediates_var = tk.BooleanVar(value=False)  # Default: nicht speichern

chk_intermediates = tk.Checkbutton(
    footer, text="Intermediates exportieren", variable=save_intermediates_var,
    bg=BG_CARD, fg=TEXT_PRIMARY, activebackground=BG_CARD_ALT, activeforeground=TEXT_PRIMARY,
    selectcolor=BG_CARD_ALT, font=FONT_NORMAL, bd=0, highlightthickness=0
    )
chk_intermediates.grid(column=0, row=0, sticky="w", padx=16, pady=14)

# Speicherbutton 
btn_save = make_flat_button(footer, "ELA-Bild speichern", save_pic,
                            bg=ACCENT, fg="#ffffff", hover_bg="#3a7bc0",
                            font=FONT_HEADING, padx=18, pady=8)
btn_save.grid(column=1, row=0, sticky="e", padx=16, pady=14)

# Groessenänderung des Fensters abfangen, um die Vorschau-Bilder und Slider-Längen anzupassen
root.bind("<Configure>", on_window_resize)


# starten Event-Loop
root.after(50, _poll_queue)  # Queue-Polling starten
root.mainloop()