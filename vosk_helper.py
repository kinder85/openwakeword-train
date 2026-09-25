#!/usr/bin/env python3
"""Helper vosk local — remplace la dépendance à hapcvoice (run.find_vosk_model).

Deux fonctions :
- find_vosk_model(base, locale) : renvoie le chemin du modèle vosk valide pour
  la locale (logique reprise de hapcvoice run.py:133-145), sinon None.
- transcribe(path_16k_wav, model_dir, locale) : transcrit un WAV 16 kHz mono
  via vosk et renvoie le texte (logique déplacée de trainer_gui.py).

Vosks par langue dans model/<locale>/ (cf. first_run.ensure_models).
"""

import json
import os

import numpy as np
import soundfile as sf
import vosk


def find_vosk_model(base="model", locale="fr"):
    """Renvoie le chemin du modèle vosk pour la locale, sinon None."""
    def is_vosk_model(path):
        return os.path.isdir(path) and (os.path.isdir(os.path.join(path, 'am'))
                                        or os.path.isdir(os.path.join(path, 'conf')))

    target = os.path.join(base, locale)
    if is_vosk_model(target):
        return target
    if is_vosk_model(base):
        return base
    if os.path.isdir(target):
        # si la locale contient un seul sous-dossier modèle (zip extrait)
        for entry in sorted(os.listdir(target)):
            sub = os.path.join(target, entry)
            if is_vosk_model(sub):
                return sub
    if os.path.isdir(base):
        # fallback : n'importe quel modèle valide trouvé dans base
        for entry in sorted(os.listdir(base)):
            sub = os.path.join(base, entry)
            if is_vosk_model(sub):
                return sub
    return None


def transcribe(path_16k_wav, model_dir="model", locale="fr", sr=16000):
    """Transcrit un WAV 16 kHz (mono) via vosk et renvoie le texte.

    Renvoie le texte reconnu (str), ou "" si aucun, ou None si le modèle vosk
    de la locale est introuvable.
    """
    model_path = find_vosk_model(model_dir, locale)
    if model_path is None:
        return None
    audio, src = sf.read(path_16k_wav, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if src != sr:
        n_out = int(round(len(mono) * sr / src))
        mono = np.interp(np.linspace(0, len(mono) - 1, n_out), np.arange(len(mono)), mono)

    vm = vosk.Model(model_path)
    rec = vosk.KaldiRecognizer(vm, sr)
    rec.AcceptWaveform((np.clip(mono, -1, 1) * 32767).astype(np.int16).tobytes())
    return json.loads(rec.FinalResult()).get("text", "")


if __name__ == "__main__":
    import sys
    loc = sys.argv[2] if len(sys.argv) > 2 else "fr"
    print("modèle vosk (%s) : %s" % (loc, find_vosk_model("model", loc)))
