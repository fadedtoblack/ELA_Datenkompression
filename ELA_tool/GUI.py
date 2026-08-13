import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import os
from PIL import Image, ImageTk

from ela import generate_ela




# FUNKTIONEN

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

        # Anzeige des alten Bildes
        img_old = resize_image(pic_old.copy(), 250)
        tk_img_old = ImageTk.PhotoImage(img_old)

        lbl_pic_old.config(image=tk_img_old)
        lbl_pic_old.image = tk_img_old

        # Anzeige des neuen Bildes
        img_new = resize_image(pic_new.copy(), 250)
        tk_img_new = ImageTk.PhotoImage(img_new)

        lbl_pic_new.config(image=tk_img_new)
        lbl_pic_new.image = tk_img_new

        lbl_psnr_out.config(text="-") # PSNR-Anzeige zuruecksetzen bei neuem Bild

        # erste ELA_Vorschau mit aktuellem Slidewerten berechnen
        update_image()


# Resize der Bilder 
def resize_image(img, target_width):
    ratio = target_width / img.width
    height = int(img.height * ratio)
    return img.resize((target_width, height))       


# Update der Schieberegler anzeigen 
def update_label(value, label):
    label.config(text=f"{float(value):.2f}")


# Bild live bearbeiten -> ELA-Vorschau berechnen 
def update_image(*args):
    global pic_new

    if dateipfad_global is None:
        return
    
    # Ausgang: immer Original 
    img = pic_old.copy()

    # Werte aus dem Slider holen 
    quality_val = quality_val.get()
    multiplier_val = multiplier_val.get()

    # ELA-Bild berechnen OHNE speichern nach Aenderung der Sliderwerte
    ela_array = generate_ela(
        dateipfad_global,
        quality=quality_val,
        multiplier=multiplier_val,
        # output_dir="ELA_tool/output",
        save_intermediates=False
    )

    # numpy_Array -> PIL-Image,  fuer die die Anzeige und seperates Speichern
    pic_new = Image.fromarray(ela_array)
    
    #Anzeige skalieren
    display_img = resize_image(pic_new.copy(), 250)
    tk_img = ImageTk.PhotoImage(display_img)
    lbl_pic_new.config(image=tk_img)
    lbl_pic_new.image = tk_img


# Berechnung des ELA-Bildes beim loslassen des Sliders
def on_slider_release(event):
    update_image()


# Bild Speichern 
def save_pic():
    global pic_new

    if dateipfad_global is None:
        print("Kein Bild vorhanden!")
        return

    quality_val = quality_val.get()
    multiplier_val = multiplier_val.get()

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
        print("ELA_Bild gepseichert im Ordner:", ausgabe_ordner)
    except Exception as e:
        print("Fehler beim Speichern des ELA-Bildes:", e)



# Erstellen GUI Fenster 
root = tk.Tk()
root.title("ELA - Tool")
root.geometry("800x400")
root.minsize(width=800, height=500)       
root.maxsize(width=800, height=500)

# WIDGETS

# globale Zustandsvariablen 
pic_old = None
pic_new = None
dateipfad_global = None

# Einstellung Spaltengröße 
for i in range(3):
    root.columnconfigure(i, weight=1)

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

quality_var = tk.DoubleVar(value=75.0) # Default bei 75.00

scale_quality = ttk.Scale(para_frame, 
                        from_=1, to=100, 
                        orient="horizontal",
                        length=150, 
                        variable=quality_var, 
                        command=lambda v: update_label(v, lbl_quality_out)
                        )
scale_quality.grid(column=0, row=1)

lbl_quality_out =tk.Label(para_frame, anchor="center", text="75.00") # Default bei 75.00
lbl_quality_out.grid(column=0, row=2)

#Platzhalter
ph =tk.Label(para_frame)
ph.grid(column=0, row=3)

# Parameter 2: Multiplier (M)
lbl_multiplier = tk.Label(para_frame, anchor="center", text="Multiplier (M)")
lbl_multiplier.grid(column=0, row=4)

multiplier_var = tk.DoubleVar(value=30.0) # Default bei 30.00

scale_multiplier = ttk.Scale(para_frame,
                        from_=1, to=100, 
                        orient="horizontal", 
                        length=150, 
                        variable=multiplier_var, 
                        command=lambda v: update_label(v, lbl_multiplier_out))
scale_multiplier.grid(column=0, row=5)
scale_multiplier.bind("<ButtonRelease-1>", on_slider_release)  # Event-Handler für Slider loslassen

lbl_multiplier_out =tk.Label(para_frame, anchor="center", text="30.00") # Default bei 30.00
lbl_multiplier_out.grid(column=0, row=6)

# PSNR - Anzeige
lbl_psnr =tk.Label(para_frame, anchor="center", text="PSNR:")
lbl_psnr.grid(column=0, row=7)

lbl_psnr_out =tk.Label(para_frame, anchor="center", text="-")
lbl_psnr_out.grid(column=0, row=8)

# Speicherbutton 
btn_save = tk.Button(root, text="ELA-Bild speichern", command=save_pic, anchor="center")
btn_save.grid(column=1, row=3)


# starten Event-Loop
root.mainloop()