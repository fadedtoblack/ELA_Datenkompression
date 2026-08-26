import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import font as tkfont
import os
import threading
import queue
from PIL import Image, ImageTk

from ela import generate_ela


# ---------------------------------------------------------------------------
# FARBEN & STIL (moderne, abgerundete Optik, Indigo-Akzent)
# ---------------------------------------------------------------------------
COLOR_BG            = "#f3f4fb"   # Haupt-Hintergrund (helles Lavendel-Grau)
COLOR_PANEL         = "#ffffff"   # Karten-Hintergrund
COLOR_BORDER        = "#e4e6f2"   # Rahmenfarbe der Karten
COLOR_SHADOW        = "#dcdfee"   # weicher Schatten unter den Karten

COLOR_ACCENT        = "#6366f1"   # Akzentfarbe (Indigo)
COLOR_ACCENT_HOVER  = "#4f46e5"
COLOR_ACCENT_ACTIVE = "#4338ca"

COLOR_SECONDARY        = "#eceefa"  # sekundäre Buttons (hell, dezent)
COLOR_SECONDARY_HOVER  = "#dfe2f5"
COLOR_SECONDARY_ACTIVE = "#d1d5ef"
COLOR_SECONDARY_TEXT   = "#1f2330"

COLOR_TEXT      = "#12131a"   # Haupttext
COLOR_SUBTEXT   = "#6b6f80"   # Sekundärtext
COLOR_INFO_BG   = "#eef1ff"   # Hintergrund Info-Box
COLOR_INFO_BORDER = "#c9cff5"
COLOR_INFO_TEXT = "#3a3ca3"   # Text Info-Box

FONT_TITLE   = ("Segoe UI", 21, "bold")
FONT_HEAD    = ("Segoe UI", 10, "bold")
FONT_TEXT    = ("Segoe UI", 10)
FONT_SMALL   = ("Segoe UI", 9)
FONT_BUTTON  = ("Segoe UI", 10, "bold")


# ---------------------------------------------------------------------------
# Canvas-Hilfsfunktion: Punktliste fuer ein abgerundetes Rechteck
# (Tkinter kennt keine nativen abgerundeten Ecken -> per Polygon + smooth=True)
# ---------------------------------------------------------------------------
def rounded_rect_points(x0, y0, x1, y1, r):
    r = max(0, min(r, (x1 - x0) / 2, (y1 - y0) / 2))
    return [
        x0 + r, y0,
        x1 - r, y0,
        x1, y0,
        x1, y0 + r,
        x1, y1 - r,
        x1, y1,
        x1 - r, y1,
        x0 + r, y1,
        x0, y1,
        x0, y1 - r,
        x0, y0 + r,
        x0, y0,
    ]


# ---------------------------------------------------------------------------
# RoundedFrame: Karte mit abgerundeten Ecken + optionalem weichen Schatten.
# Eigentlicher Inhalt kommt in .body (ein normales tk.Frame).
# ---------------------------------------------------------------------------
class RoundedFrame(tk.Canvas):
    def __init__(self, parent, bg_outer, fill=COLOR_PANEL, border=None,
                 radius=18, shadow=True, **kwargs):
        super().__init__(parent, bg=bg_outer, highlightthickness=0, bd=0, **kwargs)
        self.fill = fill
        self.border = border
        self.radius = radius
        self.shadow = shadow
        self.margin = int(radius * 0.55) + 10

        self.body = tk.Frame(self, bg=fill)
        self._win = self.create_window(self.margin, self.margin, window=self.body, anchor="nw")
        self.bind("<Configure>", self._on_configure)

    def _on_configure(self, event):
        w, h = event.width, event.height
        self.delete("bg")
        if w > 4 and h > 4:
            if self.shadow:
                shadow_pts = rounded_rect_points(4, 7, w - 2, h + 1, self.radius)
                self.create_polygon(shadow_pts, smooth=True, fill=COLOR_SHADOW,
                                     outline=COLOR_SHADOW, tags="bg")
            outline_color = self.border if self.border else self.fill
            main_pts = rounded_rect_points(2, 2, w - 4, h - 4, self.radius)
            self.create_polygon(main_pts, smooth=True, fill=self.fill,
                                 outline=outline_color, width=1.2, tags="bg")
            self.tag_lower("bg")

        body_w = max(1, w - 2 * self.margin)
        body_h = max(1, h - 2 * self.margin)
        self.coords(self._win, self.margin, self.margin)
        self.itemconfig(self._win, width=body_w, height=body_h)

    # Setzt die eigene Hoehe so, dass sie genau zum tatsaechlich benoetigten
    # Platz des Inhalts (.body) passt. Notwendig, weil ein Canvas (im Gegensatz
    # zu einem tk.Frame) NICHT automatisch auf seinen Inhalt schrumpft/waechst -
    # ohne das wuerde die Karte auf Tk's Standardgroesse (defaultmaessig recht
    # gross) stehen bleiben und anderen Bereichen (z.B. der Bildvorschau) den
    # Platz wegnehmen.
    def fit_height_to_content(self):
        self.body.update_idletasks()
        needed_h = self.body.winfo_reqheight() + 2 * self.margin
        if abs(self.winfo_reqheight() - needed_h) > 1:
            self.configure(height=needed_h)


# ---------------------------------------------------------------------------
# RoundedButton: Pillenfoermiger Button (Canvas), da ttk keine echten
# abgerundeten Ecken unterstuetzt. Reagiert auf Hover/Klick wie ein normaler Button.
# ---------------------------------------------------------------------------
class RoundedButton(tk.Canvas):
    FILL_DISABLED = "#d9dbe8"
    TEXT_DISABLED = "#9599ad"

    def __init__(self, parent, text, command=None, bg_outer=COLOR_BG,
                 fill=COLOR_ACCENT, hover=COLOR_ACCENT_HOVER, active=COLOR_ACCENT_ACTIVE,
                 fg="white", font=FONT_BUTTON, padx=22, pady=12, **kwargs):
        super().__init__(parent, bg=bg_outer, highlightthickness=0, bd=0, cursor="hand2", **kwargs)
        self.command = command
        self.fill_normal = fill
        self.fill_hover = hover
        self.fill_active = active
        self.fg = fg
        self.font = font
        self.text = text
        self.enabled = True

        measure_font = tkfont.Font(font=font)
        text_w = measure_font.measure(text)
        text_h = measure_font.metrics("linespace")
        self.btn_w = text_w + 2 * padx
        self.btn_h = text_h + 2 * pady
        self.radius = self.btn_h / 2
        self.configure(width=self.btn_w, height=self.btn_h)

        self._draw(self.fill_normal)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)

    def _draw(self, color, fg=None):
        self.delete("all")
        pts = rounded_rect_points(1, 1, self.btn_w - 1, self.btn_h - 1, self.radius)
        self.create_polygon(pts, smooth=True, fill=color, outline=color)
        self.create_text(self.btn_w / 2, self.btn_h / 2, text=self.text,
                          fill=fg if fg is not None else self.fg, font=self.font)

    def _on_enter(self, event):
        if self.enabled:
            self._draw(self.fill_hover)

    def _on_leave(self, event):
        if self.enabled:
            self._draw(self.fill_normal)

    def _on_press(self, event):
        if self.enabled:
            self._draw(self.fill_active)

    def _on_release(self, event):
        if not self.enabled:
            return
        self._draw(self.fill_hover)
        in_bounds = 0 <= event.x <= self.btn_w and 0 <= event.y <= self.btn_h
        if in_bounds and self.command:
            self.command()

    # Aktiviert/deaktiviert den Button optisch und funktional
    # (z.B. waehrend im Hintergrund die ELA-Berechnung laeuft)
    def set_enabled(self, enabled):
        self.enabled = enabled
        if enabled:
            self.configure(cursor="hand2")
            self._draw(self.fill_normal)
        else:
            self.configure(cursor="arrow")
            self._draw(self.FILL_DISABLED, fg=self.TEXT_DISABLED)


# ---------------------------------------------------------------------------
# Bild-Hilfsfunktionen
# ---------------------------------------------------------------------------

# Skaliert ein Bild moeglichst gross, passend in (max_w, max_h), Seitenverhaeltnis bleibt erhalten
def fit_image(img, max_w, max_h):
    max_w = max(1, int(max_w))
    max_h = max(1, int(max_h))
    ratio = min(max_w / img.width, max_h / img.height)
    new_w = max(1, int(img.width * ratio))
    new_h = max(1, int(img.height * ratio))
    return img.resize((new_w, new_h), Image.LANCZOS)


# Rendert pic_old/pic_new passend zur aktuellen Groesse der Vorschau-Labels neu
def update_preview_images():
    root.update_idletasks()

    if pic_old is not None:
        max_w = lbl_pic_old.winfo_width() - 4
        max_h = lbl_pic_old.winfo_height() - 4
        if max_w > 10 and max_h > 10:
            img_old = fit_image(pic_old, max_w, max_h)
            tk_img_old = ImageTk.PhotoImage(img_old)
            lbl_pic_old.config(image=tk_img_old, text="")
            lbl_pic_old.image = tk_img_old

    if pic_new is not None:
        max_w = lbl_pic_new.winfo_width() - 4
        max_h = lbl_pic_new.winfo_height() - 4
        if max_w > 10 and max_h > 10:
            img_new = fit_image(pic_new, max_w, max_h)
            tk_img_new = ImageTk.PhotoImage(img_new)
            lbl_pic_new.config(image=tk_img_new, text="")
            lbl_pic_new.image = tk_img_new


# Verzoegertes Neuzeichnen bei Fenstergroessenaenderung (verhindert Ruckeln beim Ziehen)
def schedule_preview_update(event=None):
    global _resize_job
    if _resize_job is not None:
        root.after_cancel(_resize_job)
    _resize_job = root.after(120, update_preview_images)


# Passt die Zeilenumbruchbreite der Info-Box an die aktuelle Fensterbreite an
# und die Kartenhoehe an die dadurch ggf. veraenderte Anzahl Textzeilen
def update_info_wrap(event):
    new_width = event.width - 8
    if new_width > 100:
        lbl_info_text.config(wraplength=new_width)
        info_card.fit_height_to_content()


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
        dateipfad_global = dateipfad  # Pfad merken, wird bei Klick auf "Berechnen" erneut an generate_ela() übergeben

        # Bild laden
        pic_old = Image.open(dateipfad)
        pic_new = pic_old.copy()

        # Rechtes Feld zeigt zunächst nur das Original, ELA erst nach "Berechnen"
        update_preview_images()

        lbl_psnr_out.config(text="-")  # PSNR-Anzeige zuruecksetzen bei neuem Bild
        status_var.set("Bild geladen. Werte prüfen und auf \"ELA berechnen\" klicken.")


# ---------------------------------------------------------------------------
# Regler <-> manuelles Eingabefeld synchron halten (nur Ganzzahlen)
# ---------------------------------------------------------------------------

# Wird aufgerufen, wenn der Schieberegler bewegt wird -> auf Ganzzahl runden
def on_scale_move(value_str, var):
    try:
        gerundet = int(round(float(value_str)))
    except ValueError:
        return
    if var.get() != gerundet:
        var.set(gerundet)


# Wird aufgerufen, wenn im Eingabefeld manuell ein Wert eingetragen wird
def on_spinbox_commit(var, minimum=1, maximum=100):
    try:
        wert = int(round(float(var.get())))
    except (tk.TclError, ValueError):
        wert = minimum
    wert = max(minimum, min(maximum, wert))
    var.set(wert)


# ---------------------------------------------------------------------------
# ELA-Berechnung (nur noch auf Knopfdruck, nicht mehr live)
# Laeuft in einem Hintergrundthread, damit die GUI nicht einfriert und der
# Fortschrittsbalken waehrend der Berechnung live aktualisiert werden kann.
# ---------------------------------------------------------------------------

_progress_queue = queue.Queue()


def _set_controls_enabled(enabled):
    btn_start.set_enabled(enabled)
    btn_save.set_enabled(enabled)
    btn_chosedata.set_enabled(enabled)


def berechne_ela():
    if dateipfad_global is None:
        status_var.set("Bitte zuerst ein Bild auswählen!")
        return

    # Eingaben nochmal validieren, bevor gerechnet wird
    on_spinbox_commit(quality_var)
    on_spinbox_commit(multiplier_var)

    quality_val = quality_var.get()
    multiplier_val = multiplier_var.get()

    _set_controls_enabled(False)
    status_var.set("Berechne ELA-Bild ...")
    progress_bar.start(12)  # unbestimmter Fortschritt: laufender Balken, solange der Hintergrundthread rechnet

    def worker():
        try:
            ela_array = generate_ela(
                dateipfad_global,
                quality=quality_val,
                multiplier=multiplier_val,
                # output_dir="ELA_tool/output",
                save_intermediates=False,
            )
            _progress_queue.put(("done", (ela_array, quality_val, multiplier_val)))
        except Exception as exc:
            _progress_queue.put(("error", str(exc)))

    threading.Thread(target=worker, daemon=True).start()
    root.after(50, _poll_ela_progress)


# Fragt periodisch die Ergebnis-Queue ab und aktualisiert GUI-Elemente
# (darf nur im Haupt-Thread laufen -> deshalb ueber root.after() statt
# direkt aus dem Hintergrundthread heraus). generate_ela() meldet selbst
# keine Zwischenschritte, daher laeuft waehrenddessen nur ein unbestimmter
# Fortschrittsbalken (progress_bar.start()); erst bei "done"/"error" wird
# er wieder gestoppt.
def _poll_ela_progress():
    global pic_new

    try:
        while True:
            kind, payload = _progress_queue.get_nowait()

            if kind == "done":
                ela_array, quality_val, multiplier_val = payload
                pic_new = Image.fromarray(ela_array)
                update_preview_images()
                progress_bar.stop()
                status_var.set(f"ELA-Bild berechnet (Q={quality_val}, M={multiplier_val}).")
                _set_controls_enabled(True)
                return

            elif kind == "error":
                progress_bar.stop()
                status_var.set(f"Fehler bei der ELA-Berechnung: {payload}")
                _set_controls_enabled(True)
                return
    except queue.Empty:
        pass

    root.after(50, _poll_ela_progress)


# Bild Speichern
def save_pic():
    global pic_new

    if dateipfad_global is None:
        status_var.set("Kein Bild vorhanden!")
        return

    on_spinbox_commit(quality_var)
    on_spinbox_commit(multiplier_var)

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
            save_output=True
        )
        status_var.set(f"ELA-Bild gespeichert im Ordner: {ausgabe_ordner}")
        print("ELA_Bild gepseichert im Ordner:", ausgabe_ordner)
    except Exception as e:
        status_var.set("Fehler beim Speichern des ELA-Bildes.")
        print("Fehler beim Speichern des ELA-Bildes:", e)


# ---------------------------------------------------------------------------
# GUI AUFBAU
# ---------------------------------------------------------------------------

root = tk.Tk()
root.title("ELA-Tool")
root.geometry("1100x900")
root.minsize(width=780, height=640)
root.configure(bg=COLOR_BG)

# globale Zustandsvariablen
pic_old = None
pic_new = None
dateipfad_global = None
_resize_job = None

# ---- ttk-Style: modernes, flaches Aussehen (fuer Scale/Spinbox) -----------
style = ttk.Style(root)
style.theme_use("clam")

style.configure("TFrame", background=COLOR_BG)
style.configure("TLabel", background=COLOR_BG, foreground=COLOR_TEXT, font=FONT_TEXT)
style.configure("Card.TLabel", background=COLOR_PANEL, foreground=COLOR_TEXT, font=FONT_TEXT)
style.configure("Title.TLabel", background=COLOR_BG, foreground=COLOR_TEXT, font=FONT_TITLE)
style.configure("Sub.TLabel", background=COLOR_BG, foreground=COLOR_SUBTEXT, font=FONT_SMALL)

style.configure("Horizontal.TScale", background=COLOR_PANEL, troughcolor=COLOR_INFO_BORDER)
style.map("Horizontal.TScale", background=[("active", COLOR_PANEL)])

style.configure("TSpinbox", arrowsize=14, padding=4,
                 fieldbackground=COLOR_PANEL, foreground=COLOR_TEXT,
                 bordercolor=COLOR_BORDER, lightcolor=COLOR_PANEL, darkcolor=COLOR_BORDER)

style.configure("Indigo.Horizontal.TProgressbar", troughcolor=COLOR_BORDER,
                 background=COLOR_ACCENT, borderwidth=0, thickness=10)

# Spalten des Root-Grids gleichmäßig verteilen
for i in range(3):
    root.columnconfigure(i, weight=1)

# Bildvorschau-Zeile bekommt den gesamten uebrigen Platz -> Vorschau so gross wie moeglich
root.rowconfigure(4, weight=1)

# ---- Titelzeile ------------------------------------------------------------
lbl_title = ttk.Label(root, text="ELA-Tool", style="Title.TLabel")
lbl_title.grid(column=0, row=0, columnspan=3, padx=24, pady=(20, 0), sticky="w")

lbl_subtitle = ttk.Label(
    root,
    text="Bild auswählen, Quality (Q) und Multiplier (M) einstellen und mit \"ELA berechnen\" auswerten.",
    style="Sub.TLabel"
)
lbl_subtitle.grid(column=0, row=1, columnspan=3, padx=24, pady=(4, 8), sticky="w")

# ---- Info-Box: Kurzerklärung was ELA macht (Breite passt sich dem Fenster an) ----
info_card = RoundedFrame(root, bg_outer=COLOR_BG, fill=COLOR_INFO_BG,
                          border=COLOR_INFO_BORDER, radius=16, shadow=False)
info_card.grid(column=0, row=2, columnspan=3, padx=24, pady=(0, 10), sticky="ew")

lbl_info_head = tk.Label(info_card.body, text="ℹ  Was macht dieses Tool?",
                          bg=COLOR_INFO_BG, fg=COLOR_INFO_TEXT, font=FONT_HEAD, anchor="w")
lbl_info_head.pack(fill="x", pady=(0, 4))

lbl_info_text = tk.Label(
    info_card.body,
    text=("Error Level Analysis (ELA) hilft dabei, mögliche Bildmanipulationen sichtbar zu machen. "
          "Das Bild wird erneut mit der eingestellten JPEG-Qualität (Q) komprimiert und mit dem "
          "Original verglichen. Bereiche mit auffälligen Kompressionsunterschieden werden über den "
          "Multiplikator (M) verstärkt dargestellt – so lassen sich nachträglich bearbeitete Stellen "
          "leichter erkennen."),
    bg=COLOR_INFO_BG, fg=COLOR_INFO_TEXT, font=FONT_TEXT,
    justify="left", wraplength=980
)
lbl_info_text.pack(fill="x")

# Karte auf die tatsaechlich benoetigte Hoehe des Inhalts bringen (siehe
# RoundedFrame.fit_height_to_content: ein Canvas waechst sonst NICHT mit
# seinem Inhalt mit, sondern bliebe auf Tk's recht grosser Standardgroesse)
info_card.fit_height_to_content()

# Wraplength der Info-Box dynamisch an die tatsächliche Breite der Karte anpassen
info_card.body.bind("<Configure>", update_info_wrap)

# ---- Dateiauswahl -----------------------------------------------------------
file_frame = ttk.Frame(root)
file_frame.grid(column=0, row=3, columnspan=3, sticky="ew", padx=24, pady=(0, 8))
file_frame.columnconfigure(0, weight=1)

entry_var = tk.StringVar(value="Keine Datei ausgewählt")

file_pill = RoundedFrame(file_frame, bg_outer=COLOR_BG, fill=COLOR_PANEL,
                          border=COLOR_BORDER, radius=14, shadow=False, height=46)
file_pill.grid(column=0, row=0, sticky="ew")

lbl_dataname = tk.Label(file_pill.body, textvariable=entry_var, anchor="w",
                         bg=COLOR_PANEL, fg=COLOR_TEXT, font=FONT_TEXT)
lbl_dataname.pack(fill="both", expand=True)

btn_chosedata = RoundedButton(file_frame, text="Durchsuchen …", command=datei_auswaehlen,
                               bg_outer=COLOR_BG, fill=COLOR_SECONDARY, hover=COLOR_SECONDARY_HOVER,
                               active=COLOR_SECONDARY_ACTIVE, fg=COLOR_SECONDARY_TEXT,
                               padx=18, pady=9)
btn_chosedata.grid(column=1, row=0, padx=(12, 0))

# ---- Bildvergleich (Original / ELA-Vorschau) -- so gross wie moeglich -------
images_frame = ttk.Frame(root)
images_frame.grid(column=0, row=4, columnspan=3, sticky="nsew", padx=24)
images_frame.columnconfigure(0, weight=1)
images_frame.columnconfigure(1, weight=1)
images_frame.rowconfigure(0, weight=1)

old_card = RoundedFrame(images_frame, bg_outer=COLOR_BG, fill=COLOR_PANEL,
                         border=COLOR_BORDER, radius=20, shadow=True)
old_card.grid(column=0, row=0, sticky="nsew", padx=(0, 10), pady=4)
old_card.body.columnconfigure(0, weight=1)
old_card.body.rowconfigure(1, weight=1)

tk.Label(old_card.body, text="Original", bg=COLOR_PANEL, fg=COLOR_TEXT, font=FONT_HEAD, anchor="w").grid(
    column=0, row=0, sticky="w", pady=(0, 8))
lbl_pic_old = tk.Label(old_card.body, bg=COLOR_PANEL, text="Kein Bild geladen",
                        fg=COLOR_SUBTEXT, font=FONT_TEXT, anchor="center")
lbl_pic_old.grid(column=0, row=1, sticky="nsew")

new_card = RoundedFrame(images_frame, bg_outer=COLOR_BG, fill=COLOR_PANEL,
                         border=COLOR_BORDER, radius=20, shadow=True)
new_card.grid(column=1, row=0, sticky="nsew", padx=(10, 0), pady=4)
new_card.body.columnconfigure(0, weight=1)
new_card.body.rowconfigure(1, weight=1)

tk.Label(new_card.body, text="ELA-Vorschau", bg=COLOR_PANEL, fg=COLOR_TEXT, font=FONT_HEAD, anchor="w").grid(
    column=0, row=0, sticky="w", pady=(0, 8))
lbl_pic_new = tk.Label(new_card.body, bg=COLOR_PANEL, text="Noch nicht berechnet",
                        fg=COLOR_SUBTEXT, font=FONT_TEXT, anchor="center")
lbl_pic_new.grid(column=0, row=1, sticky="nsew")

# Vorschaubilder bei jeder Groessenaenderung des Fensters neu einpassen
images_frame.bind("<Configure>", schedule_preview_update)

# ---- Parameter-Karte (Quality / Multiplier / PSNR) -------------------------
para_card = RoundedFrame(root, bg_outer=COLOR_BG, fill=COLOR_PANEL,
                          border=COLOR_BORDER, radius=20, shadow=True)
para_card.grid(column=0, row=5, columnspan=3, sticky="ew", padx=24, pady=10)
for c in range(3):
    para_card.body.columnconfigure(c, weight=1)

# Parameter 1: Quality (Q) — nur Ganzzahlen, per Regler oder Eingabefeld
quality_var = tk.IntVar(value=75)  # Default bei 75

tk.Label(para_card.body, text="Quality (Q)", bg=COLOR_PANEL, fg=COLOR_TEXT, font=FONT_HEAD).grid(
    column=0, row=0, sticky="w", pady=(0, 4))

scale_quality = ttk.Scale(
    para_card.body,
    from_=1, to=100,
    orient="horizontal",
    length=260,
    variable=quality_var,
    command=lambda v: on_scale_move(v, quality_var)
)
scale_quality.grid(column=0, row=1, sticky="w")

spin_quality = ttk.Spinbox(
    para_card.body, from_=1, to=100, width=6, textvariable=quality_var,
    command=lambda: on_spinbox_commit(quality_var)
)
spin_quality.grid(column=1, row=1, sticky="w", padx=(10, 0))
spin_quality.bind("<Return>", lambda e: on_spinbox_commit(quality_var))
spin_quality.bind("<FocusOut>", lambda e: on_spinbox_commit(quality_var))

# Parameter 2: Multiplier (M) — nur Ganzzahlen, per Regler oder Eingabefeld
multiplier_var = tk.IntVar(value=30)  # Default bei 30

tk.Label(para_card.body, text="Multiplier (M)", bg=COLOR_PANEL, fg=COLOR_TEXT, font=FONT_HEAD).grid(
    column=0, row=2, sticky="w", pady=(18, 4))

scale_multiplier = ttk.Scale(
    para_card.body,
    from_=1, to=100,
    orient="horizontal",
    length=260,
    variable=multiplier_var,
    command=lambda v: on_scale_move(v, multiplier_var)
)
scale_multiplier.grid(column=0, row=3, sticky="w")

spin_multiplier = ttk.Spinbox(
    para_card.body, from_=1, to=100, width=6, textvariable=multiplier_var,
    command=lambda: on_spinbox_commit(multiplier_var)
)
spin_multiplier.grid(column=1, row=3, sticky="w", padx=(10, 0))
spin_multiplier.bind("<Return>", lambda e: on_spinbox_commit(multiplier_var))
spin_multiplier.bind("<FocusOut>", lambda e: on_spinbox_commit(multiplier_var))

# PSNR-Anzeige
tk.Label(para_card.body, text="PSNR", bg=COLOR_PANEL, fg=COLOR_TEXT, font=FONT_HEAD).grid(
    column=2, row=0, sticky="w", pady=(0, 4))
lbl_psnr_out = tk.Label(para_card.body, text="-", bg=COLOR_PANEL, fg=COLOR_TEXT, font=FONT_TEXT)
lbl_psnr_out.grid(column=2, row=1, sticky="w")

# Karte auf die tatsaechlich benoetigte Hoehe des Inhalts bringen (siehe oben)
para_card.fit_height_to_content()

# ---- Fortschrittsbalken: laeuft waehrend die ELA-Berechnung im Hintergrund ----
# arbeitet (unbestimmter Modus, da generate_ela() selbst keine Zwischenschritte
# nach aussen meldet und ela_tool_gui.py bewusst die einzige geaenderte Datei
# bleiben soll)
progress_bar = ttk.Progressbar(root, orient="horizontal", mode="indeterminate",
                                style="Indigo.Horizontal.TProgressbar")
progress_bar.grid(column=0, row=6, columnspan=3, sticky="ew", padx=24, pady=(0, 6))

# ---- Aktionsbuttons: Start (ELA berechnen) & Speichern ---------------------
action_frame = ttk.Frame(root)
action_frame.grid(column=0, row=7, columnspan=3, sticky="ew", padx=24, pady=(0, 6))
action_frame.columnconfigure(0, weight=1)

btn_start = RoundedButton(action_frame, text="▶  ELA berechnen", command=berechne_ela,
                           bg_outer=COLOR_BG, fill=COLOR_ACCENT, hover=COLOR_ACCENT_HOVER,
                           active=COLOR_ACCENT_ACTIVE, fg="white", padx=26, pady=12)
btn_start.grid(column=1, row=0, padx=(0, 12))

btn_save = RoundedButton(action_frame, text="ELA-Bild speichern", command=save_pic,
                          bg_outer=COLOR_BG, fill=COLOR_SECONDARY, hover=COLOR_SECONDARY_HOVER,
                          active=COLOR_SECONDARY_ACTIVE, fg=COLOR_SECONDARY_TEXT,
                          padx=22, pady=12)
btn_save.grid(column=2, row=0)

# ---- Statuszeile -------------------------------------------------------------
status_var = tk.StringVar(value="Bereit. Bitte ein Bild auswählen.")
lbl_status = ttk.Label(root, textvariable=status_var, style="Sub.TLabel")
lbl_status.grid(column=0, row=8, columnspan=3, sticky="w", padx=24, pady=(4, 10))


# starten Event-Loop
root.mainloop()
