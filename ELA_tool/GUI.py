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

        lbl_psnr_out.config(text=f"{psnr:.4f} dB")
        pic_new = Image.fromarray(ela_array)
        refresh_previews()  # Vorschau-Bilder aktualisieren


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

        # erste ELA_Vorschau mit aktuellem Slidewerten berechnen
        update_image()


# Resize der Bilder 
def resize_image(img, target_width):
    ratio = target_width / img.width
    height = int(img.height * ratio)
    return img.resize((target_width, height))       


# Update der Schieberegler anzeigen 
def update_label(raw_value, label, int_var):
    int_value = int(round(float(raw_value)))
    int_var.set(int_value)
    label.config(text=f"{int_value}")

def _current_preview_width():
    total_width = root.winfo_width()
    return max(50, int(total_width * 0.4) - 20)  # mindestens 50 Pixel, sonst 30% der Fensterbreite

def _current_slider_length():
    total_width = root.winfo_width()
    return max(80, int(total_width * 0.2) - 20)  # mindestens 80 Pixel, sonst 20% der Fensterbreite

def refresh_previews():
    """
    Zeichnet die Original- und ELA-Vorschau-Bilder neu, wenn das 
    Fenster skaliert wird.
    """
    prewiew_width = _current_preview_width()

    if pic_old is not None:
        img_old = resize_image(pic_old.copy(), prewiew_width)
        tk_img_old = ImageTk.PhotoImage(img_old)
        lbl_pic_old.config(image=tk_img_old)
        lbl_pic_old.image = tk_img_old

    if pic_new is not None:
        img_new = resize_image(pic_new.copy(), prewiew_width)
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
    seg_width = 40
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

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.maxsize(width=screen_width, height=screen_height)  # Maximale Größe auf Bildschirmgröße setzen

# WIDGETS

# globale Zustandsvariablen 
pic_old = None
pic_new = None
dateipfad_global = None

# Einstellung Spaltengröße 
root.columnconfigure(0, weight=2)
root.columnconfigure(1, weight=1)
root.columnconfigure(2, weight=2)

# Label Anfangstext
lbl_text = tk.Label(root, text="ELA-Tool: Einführungstext: Bild auswählen, Quality (Q) und Multiplier (M) einstellen, um das ELA-Bild live zu berechnen. ", wraplength=600, justify="center")
lbl_text.grid(column=0, row=0, columnspan=3, padx=10, pady=10, sticky="n")

# Button und Textfeld Dateiauswahl
entry_var = tk.StringVar(value="Datei auswählen...")

frame = tk.Frame(root)
frame.grid(column=0, row=1, columnspan=3, sticky="ew", padx=40)
frame.columnconfigure(0, weight=1)
frame.columnconfigure(1, weight=0)

lbl_dataname = tk.Label(frame, textvariable=entry_var, anchor="w", relief="sunken")
lbl_dataname.grid(column=0, row=0, sticky="ew")

btn_chosedata = tk.Button(frame, text="Durchsuchen", command=datei_auswaehlen)
btn_chosedata.grid(column=1, row=0)

# Bildanzeige Originalbild 
lbl_pic_old = tk.Label(root)
lbl_pic_old.grid(column=0, row=2)   

# Bildanzeige neues Bild 
lbl_pic_new = tk.Label(root)
lbl_pic_new.grid(column=2, row=2)

#Frame für Regler und Felder Mitte 
para_frame = tk.Frame(root)
para_frame.grid(column=1, row=2)
para_frame.columnconfigure(2, weight=1)
para_frame.columnconfigure(1, weight=1)


# Parameter 1: Quality (Q)
lbl_quality = tk.Label(para_frame, anchor="center", text="Quality (Q)")
lbl_quality.grid(column=0, row=0)

quality_var = tk.IntVar(value=75) # Default bei 75
quality_raw = tk.DoubleVar(value=75.0) # Raw-Wert für den Slider, um die Genauigkeit zu erhalten

scale_quality = ttk.Scale(para_frame, 
                        from_=1, to=100, 
                        orient="horizontal",
                        length=150, 
                        variable=quality_raw, # Regler haengt an der kontinuierlichen Variable, die den genauen Wert speichert
                        command=lambda v: update_label(v, lbl_quality_out, quality_var)
                        )
scale_quality.grid(column=0, row=1)
scale_quality.bind("<ButtonRelease-1>", on_slider_release)  # Event-Handler für Slider loslassen

lbl_quality_out =tk.Label(para_frame, anchor="center", text="75") # Default bei 75
lbl_quality_out.grid(column=0, row=2)

# Parameter 2: Multiplier (M)
lbl_multiplier = tk.Label(para_frame, anchor="center", text="Multiplier (M)")
lbl_multiplier.grid(column=0, row=4)

multiplier_var = tk.IntVar(value=30) # Default bei 30
multiplier_raw = tk.DoubleVar(value=30.0) # Raw-Wert für den Slider, um die Genauigkeit zu erhalten

scale_multiplier = ttk.Scale(para_frame,
                        from_=1, to=100, 
                        orient="horizontal", 
                        length=150, 
                        variable=multiplier_raw, # Regler haengt an der kontinuierlichen Variable, die den genauen Wert speichert
                        command=lambda v: update_label(v, lbl_multiplier_out, multiplier_var))
scale_multiplier.grid(column=0, row=5)
scale_multiplier.bind("<ButtonRelease-1>", on_multiplier_release)  # Event-Handler für Slider loslassen

lbl_multiplier_out =tk.Label(para_frame, anchor="center", text="30") # Default bei 30
lbl_multiplier_out.grid(column=0, row=6)

# PSNR - Anzeige
lbl_psnr =tk.Label(para_frame, anchor="center", text="PSNR:")
lbl_psnr.grid(column=0, row=7)

lbl_psnr_out =tk.Label(para_frame, anchor="center", text="-")
lbl_psnr_out.grid(column=0, row=8)

# Ladebalken
progress_canvas =tk.Canvas(para_frame, width=150, height=10,
                           highlightthickness=0, bg=root.cget("bg"))
progress_canvas.grid(column=0, row=9, pady=(5, 0))

_progress_bar_id = None
_progress_pos = 0 
_progress_job = None

# Checkbox um die Intermediates beim Speichern mit zu exportieren
save_intermediates_var = tk.BooleanVar(value=False)  # Default: nicht speichern

chk_intermediates = tk.Checkbutton(
    root,
    text="Intermediates exportieren",
    variable=save_intermediates_var,
    anchor="center"
    )
chk_intermediates.grid(column=1, row=3)

# Speicherbutton 
btn_save = tk.Button(root, text="ELA-Bild speichern", command=save_pic, anchor="center")
btn_save.grid(column=1, row=4)

# Groessenänderung des Fensters abfangen, um die Vorschau-Bilder und Slider-Längen anzupassen
root.bind("<Configure>", on_window_resize)


# starten Event-Loop
root.after(50, _poll_queue)  # Queue-Polling starten
root.mainloop()