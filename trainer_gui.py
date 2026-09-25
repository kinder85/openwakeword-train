#!/usr/bin/env python3
"""Interface d'entraînement de modèles openWakeWord custom.

Workflow :
  1. saisir une phrase → générer les échantillons TTS (voix par langue :
     fr/en, sélecteur dedié)
  2. ÉCOUTER chaque échantillon (▶) et vérifier la prononciation
     (transcription vosk affichée en contrôle, modèle par langue)
  3. lancer l'entraînement → .onnx + JSON métadonnées dans openwakeword/
     → auto-découvert par run.py/gui.py au prochain lancement
  4. (optionnel) convertir un .onnx existant en .tflite (int8 + float32)

Dépôt autonome (plus aucune dépendance à l'écosystème hapcvoice) et
bilingue fr/en : la langue d'interface (affichage) et la langue
d'entraînement (contenus cible) sont deux réglages indépendants.

Usage : python trainer_gui.py
"""

import os
import queue
import threading
import tkinter as tk
from tkinter import ttk

import numpy as np
import sounddevice as sd

import trainer as tr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# même palette que gui.py
BG = "#12151a"
PANEL = "#1a1f26"
FG = "#e6e9ef"
FG_DIM = "#8b93a1"
COL_OK = "#2ecc71"
COL_WAKE = "#ff9f43"
COL_LISTEN = "#38bdf8"

# ------------------------------------------------------------------ i18n UI
# Toutes les chaînes visibles de l'interface, par langue. LANG_UI = langue
# d'affichage ; LANG_TRAIN = langue des contenus d'entraînement.
UI_STRINGS = {
    "fr": {
        "win_title": "Entraînement wake word",
        "title": "ENTRAÎNEMENT WAKE WORD",
        "subtitle": "phrase cible → échantillons TTS → écoute → entraînement → .onnx",
        "target": "Phrase cible",
        "model_name": "nom du modèle",
        "voices": "Voix (Ctrl+clic)",
        "samples_per_voice": "échantillons\npar voix",
        "lang_label": "langue UI",
        "train_lang": "langue entraînement",
        "gen": "1)  Générer les échantillons",
        "samples": "2)  Échantillons générés — écoute et prononciation",
        "play": "▶  Écouter la sélection",
        "transcript": "transcription vosk de la sélection :",
        "threshold": "3)  Seuil",
        "negatives": "phrases négatives",
        "tflite_opt": "produire les .tflite (HA/ESPHome/ESP32)",
        "train_btn": "Entraîner et exporter",
        "ready_tip": "Prêt. Astuce : 6-10 voix × 6 échantillons donnent un bon modèle.",
        "voices_loaded": "✓ {n} voix {lang} chargées",
        "voices_unavailable": "✗ liste des voix indisponible (connexion ?) : {err}",
        "gens_done": "✓ {n} échantillons générés — écoute-les (▶) avant d'entraîner !",
        "gen_progress": "génération {d}/{t}…",
        "gen_failed": "✗ génération : {kind}: {err}",
        "train_failed": "✗ entraînement : {kind}: {err}",
        "gen_negatives": "Génération des phrases négatives…",
        "no_samples": "⚠ génère d'abord les échantillons (étape 1)",
        "need_phrase_voice": "⚠ choisis une phrase et au moins une voix",
        "exported": "{v} Modèle « {name} » exporté dans openwakeword/ — auto-découvert au prochain lancement de la GUI.",
        "gap": "  écart de séparation {gap:.2f} ({qual}) — teste-le au micro en mode test.",
        "gap_good": "bon",
        "gap_weak": "faible : ajoute des échantillons",
        "no_real": "⚠ enregistrements réels du micro absents — lance « python first_run_train.py » : ils éliminent les faux positifs en usage réel.",
        "no_vosk_model": "⚠ modèle vosk absent ({lang}) — lance first_run_train.py",
        "transcript_err": "erreur : {err}",
        "conv_title": "4)  Conversion .onnx → .tflite (post-création)",
        "conv_onnx": "fichier .onnx",
        "conv_calib": "calibration .npy (optionnel)",
        "conv_btn": "Convertir en .tflite",
        "conv_browse": "Parcourir…",
        "conv_no_onnx": "⚠ choisis un fichier .onnx à convertir",
        "conv_ok": "✓ Conversion terminée — .tflite dans {dir}",
        "conv_failed": "✗ conversion : {kind}: {err}",
        "conv_missing_deps": "✗ onnx2tf absent — installe `pip install onnx2tf[tensorflow] tf-keras`",
    },
    "en": {
        "win_title": "Wake word training",
        "title": "WAKE WORD TRAINING",
        "subtitle": "target phrase → TTS samples → listen → train → .onnx",
        "target": "Target phrase",
        "model_name": "model name",
        "voices": "Voices (Ctrl+click)",
        "samples_per_voice": "samples\nper voice",
        "lang_label": "UI language",
        "train_lang": "training language",
        "gen": "1)  Generate samples",
        "samples": "2)  Generated samples — listen and pronunciation",
        "play": "▶  Play selection",
        "transcript": "vosk transcription of selection:",
        "threshold": "3)  Threshold",
        "negatives": "negative phrases",
        "tflite_opt": "produce .tflite (HA/ESPHome/ESP32)",
        "train_btn": "Train and export",
        "ready_tip": "Ready. Tip: 6-10 voices × 6 samples give a good model.",
        "voices_loaded": "✓ {n} {lang} voices loaded",
        "voices_unavailable": "✗ voice list unavailable (connection ?) : {err}",
        "gens_done": "✓ {n} samples generated — listen (▶) before training!",
        "gen_progress": "generation {d}/{t}…",
        "gen_failed": "✗ generation: {kind}: {err}",
        "train_failed": "✗ training: {kind}: {err}",
        "gen_negatives": "Generating negative phrases…",
        "no_samples": "⚠ generate samples first (step 1)",
        "need_phrase_voice": "⚠ pick a phrase and at least one voice",
        "exported": "{v} Model « {name} » exported into openwakeword/ — auto-discovered on next GUI launch.",
        "gap": "  separation gap {gap:.2f} ({qual}) — test it with the microphone in test mode.",
        "gap_good": "good",
        "gap_weak": "weak: add more samples",
        "no_real": "⚠ real microphone recordings missing — run « python first_run_train.py »: they eliminate false positives in real use.",
        "no_vosk_model": "⚠ vosk model missing ({lang}) — run first_run_train.py",
        "transcript_err": "error: {err}",
        "conv_title": "4)  .onnx → .tflite conversion (post-training)",
        "conv_onnx": ".onnx file",
        "conv_calib": "calibration .npy (optional)",
        "conv_btn": "Convert to .tflite",
        "conv_browse": "Browse…",
        "conv_no_onnx": "⚠ pick a .onnx file to convert",
        "conv_ok": "✓ Conversion done — .tflite in {dir}",
        "conv_failed": "✗ conversion: {kind}: {err}",
        "conv_missing_deps": "✗ onnx2tf missing — install `pip install onnx2tf[tensorflow] tf-keras`",
    },
}


class TrainerApp:
    def __init__(self, root):
        self.root = root
        self.ui_lang = "fr"      # langue d'affichage de l'interface
        self.voices = []          # [(short_name, gender)]
        self.samples = []         # chemins des positifs générés
        self.negatives = []       # chemins des négatifs
        self.lang_var = tk.StringVar(value="fr")  # langue d'entraînement
        self.ui_q = queue.Queue()
        self._play_after = None
        self._load_settings()

        self._setup_style()
        self._build_ui()
        if not os.path.exists(tr.REAL_SPEECH_WAV) or not os.path.exists(tr.REAL_AMBIENT_WAV):
            self._log_k("no_real")
        threading.Thread(target=self._load_voices_bg, daemon=True).start()

    # ---------------- réglages persistés (langue UI, langue entraînement) ----------------
    def _settings_path(self):
        return os.path.join(SCRIPT_DIR, "trainer_settings.json")

    def _load_settings(self):
        import json
        try:
            with open(self._settings_path(), encoding="utf-8") as f:
                s = json.load(f)
            self.ui_lang = s.get("ui_lang", "fr")
            self._lang_train_saved = s.get("train_lang", "fr")
        except Exception:
            self._lang_train_saved = "fr"
        if self.ui_lang not in UI_STRINGS:
            self.ui_lang = "fr"

    def _save_settings(self):
        import json
        try:
            with open(self._settings_path(), "w", encoding="utf-8") as f:
                json.dump({"ui_lang": self.ui_lang, "train_lang": self.lang_var.get()},
                          f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # ---------------- i18n helpers ----------------
    def _tt(self, key, **kw):
        """Traduit une clé d'interface dans la langue d'affichage courante."""
        templ = UI_STRINGS[self.ui_lang].get(key, UI_STRINGS["fr"].get(key, key))
        try:
            return templ.format(**kw) if kw else templ
        except (KeyError, IndexError, ValueError):
            return templ

    # ---------------- UI ----------------
    def _setup_style(self):
        """Recolore les widgets ttk (Combobox, Progressbar) pour matcher le thème
        sombre custom — sinon ils gardent le style clair par défaut et jurent."""
        try:
            style = ttk.Style(self.root)
            style.theme_use("clam")
            style.configure(".", background=BG, foreground=FG,
                            fieldbackground=BG, bordercolor=PANEL,
                            lightcolor=PANEL, darkcolor=PANEL,
                            troughcolor=PANEL, arrowcolor=FG,
                            selectbackground=COL_LISTEN, selectforeground="#0b0f14")
            style.configure("TCombobox", fieldbackground=BG, background=PANEL,
                            foreground=FG, arrowcolor=FG, bordercolor=PANEL,
                            darkcolor=PANEL, lightcolor=PANEL)
            style.map("TCombobox",
                      fieldbackground=[("readonly", BG)],
                      foreground=[("readonly", FG)],
                      selectbackground=[("readonly", PANEL)])
            style.configure("TSpinbox", fieldbackground=BG, background=PANEL,
                            foreground=FG, arrowcolor=FG, bordercolor=PANEL,
                            darkcolor=PANEL, lightcolor=PANEL)
            style.configure("TProgressbar", background=COL_LISTEN,
                            troughcolor=PANEL, borderwidth=0,
                            lightcolor=COL_LISTEN, darkcolor=COL_LISTEN)
        except Exception:
            pass

    def _build_ui(self):
        pad = {"padx": 12, "pady": 4}

        tk.Label(self.root, text=self._tt("title"), font=("Segoe UI", 16, "bold"),
                 fg=FG, bg=BG).pack(pady=(10, 0))
        tk.Label(self.root, text=self._tt("subtitle"),
                 font=("Segoe UI", 9), fg=FG_DIM, bg=BG).pack()

        # --- étape 1 : phrase + génération ---
        p1 = tk.Frame(self.root, bg=PANEL)
        p1.pack(fill="x", padx=10, pady=6)
        row = tk.Frame(p1, bg=PANEL)
        row.pack(fill="x", padx=10, pady=(8, 2))
        tk.Label(row, text=self._tt("target"), font=("Segoe UI", 10, "bold"),
                 fg=FG, bg=PANEL).pack(side="left")
        self.phrase_var = tk.StringVar(value="yo Ève")
        tk.Entry(row, textvariable=self.phrase_var, width=24, bg=BG, fg=FG,
                 insertbackground=FG, relief="flat").pack(side="left", padx=8)
        tk.Label(row, text=self._tt("model_name"), fg=FG_DIM, bg=PANEL).pack(side="left", padx=(8, 0))
        self.name_var = tk.StringVar(value="")
        tk.Entry(row, textvariable=self.name_var, width=14, bg=BG, fg=FG,
                 insertbackground=FG, relief="flat").pack(side="left", padx=6)

        row2 = tk.Frame(p1, bg=PANEL)
        row2.pack(fill="x", padx=10, pady=(2, 4))
        tk.Label(row2, text=self._tt("voices"), fg=FG_DIM, bg=PANEL).pack(side="left")
        self.voice_box = tk.Listbox(row2, height=6, selectmode="multiple", bg=BG,
                                    fg=FG, relief="flat", highlightthickness=0,
                                    exportselection=False)
        self.voice_box.pack(side="left", padx=8, pady=4)
        tk.Label(row2, text=self._tt("samples_per_voice"), fg=FG_DIM, bg=PANEL,
                 justify="center").pack(side="left")
        self.n_var = tk.IntVar(value=6)
        tk.Spinbox(row2, from_=2, to=20, textvariable=self.n_var, width=5,
                   bg=BG, fg=FG, insertbackground=FG, relief="flat",
                   buttonbackground=PANEL).pack(side="left", padx=6)

        # langue d'entraînement (contenus) et langue d'interface (affichage)
        tk.Label(row2, text=self._tt("train_lang"), fg=FG_DIM, bg=PANEL,
                 font=("Segoe UI", 8)).pack(side="left", padx=(8, 2))
        self.lang_box = ttk.Combobox(row2, textvariable=self.lang_var, width=4,
                                     values=["fr", "en", "de", "es"], state="readonly")
        self.lang_box.pack(side="left")
        self.lang_box.bind("<<ComboboxSelected>>", lambda e: self._on_lang_change())
        tk.Label(row2, text=self._tt("lang_label"), fg=FG_DIM, bg=PANEL,
                 font=("Segoe UI", 8)).pack(side="left", padx=(8, 2))
        self._ui_lang_tk = tk.StringVar(value=self.ui_lang)
        self.ui_lang_box = ttk.Combobox(row2, textvariable=self._ui_lang_tk,
                                        width=4, values=["fr", "en"], state="readonly")
        self.ui_lang_box.pack(side="left")
        self.ui_lang_box.bind("<<ComboboxSelected>>", lambda e: self._on_ui_lang_change())

        self.gen_btn = tk.Button(row2, text=self._tt("gen"), command=self.generate,
                                 bg=COL_LISTEN, fg="#0b0f14",
                                 font=("Segoe UI", 10, "bold"), relief="flat", padx=10)
        self.gen_btn.pack(side="left", padx=8)

        # --- étape 2 : écoute des échantillons ---
        p2 = tk.Frame(self.root, bg=PANEL)
        p2.pack(fill="both", expand=True, padx=10, pady=6)
        tk.Label(p2, text=self._tt("samples"), font=("Segoe UI", 10, "bold"),
                 fg=FG, bg=PANEL).pack(anchor="w", padx=10, pady=(8, 2))
        self.sample_list = tk.Listbox(p2, bg=BG, fg=FG, relief="flat", highlightthickness=0)
        self.sample_list.pack(fill="both", expand=True, padx=10, pady=(0, 4))
        row3 = tk.Frame(p2, bg=PANEL)
        row3.pack(fill="x", padx=10, pady=(0, 10))
        tk.Button(row3, text=self._tt("play"), command=self.play_selected,
                  bg=COL_OK, fg="#0b0f0d", font=("Segoe UI", 10, "bold"),
                  relief="flat", padx=10).pack(side="left")
        tk.Label(row3, text=self._tt("transcript"), fg=FG_DIM,
                 bg=PANEL).pack(side="left", padx=(12, 4))
        self.transcript_var = tk.StringVar(value="")
        tk.Label(row3, textvariable=self.transcript_var, fg=COL_LISTEN,
                 bg=PANEL, font=("Consolas", 10)).pack(side="left")
        self.sample_list.bind("<<ListboxSelect>>", lambda e: self._transcribe_selected())

        # --- étape 3 : entraînement ---
        p3 = tk.Frame(self.root, bg=PANEL)
        p3.pack(fill="x", padx=10, pady=6)
        row4 = tk.Frame(p3, bg=PANEL)
        row4.pack(fill="x", padx=10, pady=8)
        tk.Label(row4, text=self._tt("threshold"), fg=FG_DIM, bg=PANEL).pack(side="left")
        self.thresh_var = tk.DoubleVar(value=0.5)
        tk.Spinbox(row4, from_=0.1, to=0.9, increment=0.05, textvariable=self.thresh_var,
                   width=5, bg=BG, fg=FG, insertbackground=FG, relief="flat",
                   buttonbackground=PANEL).pack(side="left", padx=6)
        tk.Label(row4, text=self._tt("negatives"), fg=FG_DIM, bg=PANEL).pack(side="left", padx=(12, 0))
        self.nneg_var = tk.IntVar(value=24)
        tk.Spinbox(row4, from_=10, to=60, textvariable=self.nneg_var, width=5,
                   bg=BG, fg=FG, insertbackground=FG, relief="flat",
                   buttonbackground=PANEL).pack(side="left", padx=6)
        self.tflite_var = tk.BooleanVar(value=True)
        tk.Checkbutton(row4, text=self._tt("tflite_opt"), variable=self.tflite_var,
                       bg=PANEL, fg=FG_DIM, selectcolor=PANEL, activebackground=PANEL,
                       activeforeground=FG, font=("Segoe UI", 8)).pack(side="left", padx=(12, 0))
        self.train_btn = tk.Button(row4, text=self._tt("train_btn"), command=self.train,
                                   bg=COL_WAKE, fg="#161006",
                                   font=("Segoe UI", 10, "bold"), relief="flat", padx=10)
        self.train_btn.pack(side="right")

        # --- étape 4 : conversion .onnx → .tflite (post-création) ---
        p4 = tk.Frame(self.root, bg=PANEL)
        p4.pack(fill="x", padx=10, pady=6)
        tk.Label(p4, text=self._tt("conv_title"), font=("Segoe UI", 10, "bold"),
                 fg=FG, bg=PANEL).pack(anchor="w", padx=10, pady=(8, 2))
        row5 = tk.Frame(p4, bg=PANEL)
        row5.pack(fill="x", padx=10, pady=(0, 8))
        tk.Label(row5, text=self._tt("conv_onnx"), fg=FG_DIM, bg=PANEL).pack(side="left")
        self.conv_onnx_var = tk.StringVar(value="")
        tk.Entry(row5, textvariable=self.conv_onnx_var, width=34, bg=BG, fg=FG,
                 insertbackground=FG, relief="flat").pack(side="left", padx=6)
        tk.Button(row5, text=self._tt("conv_browse"), command=self._browse_onnx,
                  bg=PANEL, fg=FG, relief="flat", padx=8).pack(side="left")
        tk.Label(row5, text=self._tt("conv_calib"), fg=FG_DIM, bg=PANEL).pack(side="left", padx=(12, 0))
        self.conv_calib_var = tk.StringVar(value="")
        tk.Entry(row5, textvariable=self.conv_calib_var, width=24, bg=BG, fg=FG,
                 insertbackground=FG, relief="flat").pack(side="left", padx=6)
        tk.Button(row5, text=self._tt("conv_browse"), command=self._browse_calib,
                  bg=PANEL, fg=FG, relief="flat", padx=8).pack(side="left")
        self.conv_btn = tk.Button(row5, text=self._tt("conv_btn"), command=self.convert_onnx,
                                  bg=COL_LISTEN, fg="#0b0f14",
                                  font=("Segoe UI", 10, "bold"), relief="flat", padx=10)
        self.conv_btn.pack(side="left", padx=8)

        # --- progression (opérations longues) ---
        self.progress_frame = tk.Frame(self.root, bg=PANEL)
        self.progress_var = tk.DoubleVar(value=0)
        self.progress = ttk.Progressbar(self.progress_frame, variable=self.progress_var,
                                        mode="determinate", maximum=100, length=360)
        self.progress.pack(side="left", fill="x", expand=True, padx=6)
        self.progress_label = tk.Label(self.progress_frame, text="", fg=FG_DIM,
                                       bg=PANEL, width=28, anchor="w")
        self.progress_label.pack(side="left", padx=4)

        # --- journal ---
        self.log = tk.Text(self.root, height=8, bg=PANEL, fg=FG, relief="flat",
                           font=("Consolas", 9), state="disabled")
        self.log.tag_config("ok", foreground=COL_OK)
        self.log.tag_config("warn", foreground=COL_WAKE)
        self.log.tag_config("err", foreground="#e74c3c")
        self.log.tag_config("default", foreground=FG)
        self.log.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self._log_k("ready_tip")

        self._apply_ui_lang()          # applique langue d'affichage aux libellés dynamiques
        self.lang_var.set(self._lang_train_saved)  # restaure langue d'entraînement
        self.root.after(80, self._drain)

    def _apply_ui_lang(self):
        # les libellés créés dans _build_ui sont déjà traduits via _tt() au moment
        # de la création ; pour un changement à chaud on force un rebuild.
        self.root.title(self._tt("win_title"))

    # ---------------- progression ----------------
    def _progress_start(self, mode="determinate", maximum=100, label=""):
        self._progress_mode = mode
        self.progress.config(mode=mode, maximum=maximum)
        self.progress_label.config(text=label)
        self.progress_frame.pack(fill="x", padx=10, pady=2, before=self.log)
        if mode == "indeterminate":
            self.progress.start(12)

    def _progress_update(self, value):
        self.progress_var.set(value)
        self.progress_label.config(text="%d%%" % int(value))

    def _progress_stop(self):
        if getattr(self, "_progress_mode", "determinate") == "indeterminate":
            self.progress.stop()
        self.progress_frame.pack_forget()
        self.progress_var.set(0)
        self.progress_label.config(text="")

    def _on_ui_lang_change(self):
        self.ui_lang = self.ui_lang_box.get()
        self._ui_lang_tk.set(self.ui_lang)
        self._save_settings()
        # rebuild léger : on détruit et reconstruit l'UI pour re-rendre tous les libellés
        for child in self.root.winfo_children():
            child.destroy()
        self._build_ui()

    def _on_lang_change(self):
        # langue d'entraînement changée → on recharge les voix adaptées
        self.voices = []
        self.sample_list.delete(0, "end")
        self.negatives = []
        self._save_settings()
        threading.Thread(target=self._load_voices_bg, daemon=True).start()

    def _log_k(self, key, **kw):
        """Log d'une clé i18n, traduite au moment de l'affichage (thread-safe)."""
        s = self._tt(key, **kw)
        tag = ("ok" if s.startswith("✓")
               else "err" if s.startswith("✗")
               else "warn" if s.startswith("⚠")
               else "default")
        self._log(s, tag)

    def _log(self, line, tag="default"):
        self.log.config(state="normal")
        self.log.insert("end", line + "\n", tag)
        self.log.see("end")
        self.log.config(state="disabled")

    # ---------------- voix ----------------
    def _load_voices_bg(self):
        try:
            lang = self.lang_var.get()
            voices = tr.list_voices(lang)
        except Exception as e:
            self.ui_q.put(("log_k", "voices_unavailable", {"err": e}))
            return
        self.ui_q.put(("voices", voices))

    def _apply_voices(self, voices):
        self.voices = voices
        self.voice_box.delete(0, "end")
        if not voices:
            self._log(self._tt("voices_unavailable", err="liste vide"), "warn")
            return
        for name, gender in voices:
            tag = "F" if gender == "Female" else "M"
            self.voice_box.insert("end", "%s  [%s]" % (name, tag))
        self.voice_box.selection_set(0)
        self._log_k("voices_loaded", n=len(voices), lang=self.lang_var.get())

    def _selected_voices(self):
        return [self.voices[i][0] for i in self.voice_box.curselection()]

    # ---------------- génération ----------------
    def _model_name(self):
        name = self.name_var.get().strip()
        if name:
            return tr.sanitize_name(name)
        return tr.sanitize_name(self.phrase_var.get().strip().lower())

    def generate(self):
        phrase = self.phrase_var.get().strip()
        voices = self._selected_voices()
        if not phrase or not voices:
            self._log_k("need_phrase_voice")
            return
        self.gen_btn.config(state="disabled")
        n = int(self.n_var.get())
        run_dir = os.path.join("training_runs", self._model_name())
        self.samples = []
        self._progress_start("determinate", maximum=100, label="")

        def worker():
            try:
                paths = tr.generate_samples(
                    phrase, voices, n, os.path.join(run_dir, "positives"),
                    progress_cb=lambda d, t, p: self.ui_q.put(("gen_progress", d, t)))
                self.samples = paths
                self.ui_q.put(("gen_done", len(paths)))
            except Exception as e:
                self.ui_q.put(("log_k", "gen_failed", {"kind": type(e).__name__, "err": e}))
                self.ui_q.put(("gen_done", 0))
        threading.Thread(target=worker, daemon=True).start()

    def _after_generate(self, count):
        self.sample_list.delete(0, "end")
        for p in self.samples:
            self.sample_list.insert("end", os.path.basename(p))
        self.gen_btn.config(state="normal")
        self._progress_stop()
        if count:
            self._log_k("gens_done", n=count)

    # ---------------- écoute / transcription ----------------
    def play_selected(self):
        sel = self.sample_list.curselection()
        if not sel or sel[0] >= len(self.samples):
            return
        audio = tr.load_wav_16k(self.samples[sel[0]])
        sd.stop()
        sd.play((np.clip(audio, -1, 1) * 32767).astype(np.int16), 16000)

    def _transcribe_selected(self):
        sel = self.sample_list.curselection()
        if not sel or sel[0] >= len(self.samples):
            return
        path = self.samples[sel[0]]

        def worker():
            try:
                import vosk_helper
                lang = self.lang_var.get()
                text = vosk_helper.transcribe(path, "model", lang)
                if text is None:
                    text = self._tt("no_vosk_model", lang=lang)
                self.ui_q.put(("transcript", text))
            except Exception as e:
                self.ui_q.put(("transcript", self._tt("transcript_err", err=e)))
        threading.Thread(target=worker, daemon=True).start()

    # ---------------- entraînement ----------------
    def train(self):
        if not self.samples:
            self._log_k("no_samples")
            return
        self.train_btn.config(state="disabled")
        phrase = self.phrase_var.get().strip()
        name = self._model_name()
        nneg = int(self.nneg_var.get())
        threshold = float(self.thresh_var.get())
        run_dir = os.path.join("training_runs", name)
        self._progress_start("indeterminate", label=self._tt("gen_negatives"))

        def log_cb(msg):
            self.ui_q.put(("log", msg))

        def worker():
            try:
                if not self.negatives:
                    self.ui_q.put(("log_k", "gen_negatives"))
                    self.negatives = tr.generate_negatives(
                        nneg, os.path.join(run_dir, "negatives"),
                        lang=self.lang_var.get())
                res = tr.train_model(phrase, self.samples, self.negatives, name,
                                     threshold=threshold, log_cb=log_cb,
                                     with_tflite=self.tflite_var.get())
                self.ui_q.put(("trained", res))
            except Exception as e:
                self.ui_q.put(("log_k", "train_failed", {"kind": type(e).__name__, "err": e}))
                self.ui_q.put(("trained", None))
        threading.Thread(target=worker, daemon=True).start()

    def _after_train(self, res):
        self.train_btn.config(state="normal")
        self._progress_stop()
        if not res:
            return
        gap = res["pos_score_mean"] - res["neg_score_mean"]
        verdict = "✓" if gap > 0.6 else "⚠"
        self._log_k("exported", v=verdict, name=self._model_name())
        self._log_k("gap", gap=gap,
                    qual=self._tt("gap_good") if gap > 0.6 else self._tt("gap_weak"))

    # ---------------- conversion .onnx → .tflite ----------------
    def _browse_onnx(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title=self._tt("conv_onnx"),
            filetypes=[("ONNX model", "*.onnx"), ("All files", "*.*")])
        if path:
            self.conv_onnx_var.set(path)

    def _browse_calib(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title=self._tt("conv_calib"),
            filetypes=[("NumPy array", "*.npy"), ("All files", "*.*")])
        if path:
            self.conv_calib_var.set(path)

    def convert_onnx(self):
        onnx_path = self.conv_onnx_var.get().strip()
        if not onnx_path or not os.path.exists(onnx_path):
            self._log_k("conv_no_onnx")
            return
        self.conv_btn.config(state="disabled")
        calib = self.conv_calib_var.get().strip() or None
        self._progress_start("indeterminate", label=self._tt("conv_btn"))

        def worker():
            try:
                import onnx_to_tflite
                out_dir = os.path.dirname(os.path.abspath(onnx_path))
                ok = onnx_to_tflite.convert(onnx_path, out_dir, calib_path=calib)
                self.ui_q.put(("conv_done", ok, out_dir))
            except ImportError:
                self.ui_q.put(("log_k", "conv_missing_deps"))
                self.ui_q.put(("conv_done", False, None))
            except Exception as e:
                self.ui_q.put(("log_k", "conv_failed", {"kind": type(e).__name__, "err": e}))
                self.ui_q.put(("conv_done", False, None))
        threading.Thread(target=worker, daemon=True).start()

    def _after_convert(self, ok, out_dir):
        self.conv_btn.config(state="normal")
        self._progress_stop()
        if ok:
            self._log_k("conv_ok", dir=out_dir)

    # ---------------- boucle UI ----------------
    def _drain(self):
        try:
            while True:
                item = self.ui_q.get_nowait()
                kind = item[0]
                if kind == "log":
                    self._log(item[1])
                elif kind == "log_k":
                    key = item[1]
                    kw = item[2] if len(item) > 2 else {}
                    self._log_k(key, **kw)
                elif kind == "voices":
                    self._apply_voices(item[1])
                elif kind == "gen_progress":
                    d, t = item[1], item[2]
                    self._progress_update(100 * d / t if t else 0)
                    if d % 5 == 0 or d == t:
                        self._log_k("gen_progress", d=d, t=t)
                elif kind == "gen_done":
                    self._after_generate(item[1])
                elif kind == "transcript":
                    self.transcript_var.set(item[1])
                elif kind == "trained":
                    self._after_train(item[1])
                elif kind == "conv_done":
                    self._after_convert(item[1], item[2])
        except queue.Empty:
            pass
        self.root.after(80, self._drain)


if __name__ == "__main__":
    root = tk.Tk()
    TrainerApp(root)
    root.mainloop()
