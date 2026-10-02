#!/usr/bin/env python3
"""Capture l'ambiance réelle avec la TV allumée pour l'entraînement.

Pourquoi : le modèle est entraîné sur du bruit synthétique (blanc/rose/50 Hz)
à SNR élevé. Au runtime, dès qu'il y a de la VRAIE parole (TV allumée), le
réseau n'a jamais appris à y entendre le wake word → faux négatifs.

Ce script enregistre ~120-180 s d'ambiance TV (dialogue + musique, sans que
tu parles) et l'écrit dans training_runs/_real_negatives/real_120s.wav — le
chemin exact que trainer.train_model() lit comme :
  - contexte des positifs (le wake word est posé PAR-DESSUS cette ambiance)
  - négatifs réels (découpés en clips de 2,4 s)

Il purge aussi le cache clips/ pour forcer le re-découpage du nouveau fichier.

Usage :
  python capture_ambiance_tv.py            # 120 s par défaut
  python capture_ambiance_tv.py --dur 180  # durée personnalisée
"""

import argparse
import os
import shutil
import sys
import time

import numpy as np

# même mécanique d'enregistrement que first_run_train.py
def record(seconds, sr=48000):
    import sounddevice as sd
    import soundfile as sf
    rec = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="int16")
    sd.wait()
    path = os.path.join("training_runs", "_rec_tmp.wav")
    sf.write(path, rec, sr, subtype="PCM_16")
    data, _ = sf.read(path, dtype="float32")
    return path, data


def rms_profile(data, sr, window_s=1.0):
    n = int(round(sr * window_s))
    a = data
    if len(a) < n:
        return np.array([np.sqrt((a ** 2).mean()) if len(a) else 0.0])
    return np.array([np.sqrt((a[i:i + n] ** 2).mean()) for i in range(0, len(a) - n + 1, n)])


def countdown(seconds, label, log_cb=None):
    msg = "%s dans %d s" % (label, seconds)
    if log_cb:
        log_cb(msg)
    else:
        print(msg, flush=True)
    for i in range(seconds, 0, -1):
        if log_cb:
            log_cb("  %d…" % i)
        else:
            print("  %d…" % i, flush=True)
        time.sleep(1)


def capture(dur=120, log_cb=None):
    """Enregistre l'ambiance TV et l'écrit dans real_120s.wav (contexte des
    positifs + négatifs réels). Purge le cache clips/. Renvoie (rms_median, rms_max)."""
    real_dir = os.path.join("training_runs", "_real_negatives")
    os.makedirs(real_dir, exist_ok=True)
    out = os.path.join(real_dir, "real_120s.wav")

    if log_cb:
        log_cb("● ENREGISTREMENT — ambiance TV (%d s)…" % dur)
    else:
        print("● ENREGISTREMENT — ambiance TV (%d s)…" % dur, flush=True)
    path, data = record(dur)
    rms = rms_profile(data, 48000)
    med, mx = float(np.median(rms)), float(rms.max())
    if log_cb:
        log_cb("… terminé — RMS médian %.4f, max %.4f" % (med, mx))
    else:
        print("… terminé — RMS médian %.4f, max %.4f" % (med, mx))
    if med < 0.002:
        log_cb and log_cb("⚠ très peu de signal — vérifie que la TV est allumée et le micro actif")
    if med > 0.05:
        log_cb and log_cb("⚠ niveau très élevé — risque de clipping, baisse le volume TV")

    os.replace(path, out)
    clips_dir = os.path.join(real_dir, "clips")
    if os.path.isdir(clips_dir):
        shutil.rmtree(clips_dir)
    return med, mx


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dur", type=int, default=120,
                    help="durée d'enregistrement en secondes (défaut 120)")
    args = ap.parse_args()

    print("=" * 70)
    print("CAPTURE AMBIANCE TV — %d s" % args.dur)
    print("=" * 70)
    print("Allume la TV (dialogue + musique), reste SILENCIEUX.")
    print("Le wake word sera entraîné à être entendu PAR-DESSUS ce bruit.")
    print("")
    try:
        input("Appuie sur Entrée quand la TV est allumée et que tu es prêt(e)… ")
    except EOFError:
        print("(entrée non disponible — départ immédiat)")
    countdown(5, "départ")
    med, mx = capture(args.dur)
    print("✓ Ambiance TV enregistrée : training_runs/_real_negatives/real_120s.wav")
    print("  (cache clips/ purgé — le prochain entraînement re-découpera le nouveau fichier)")
    print("")
    print("Au prochain entraînement, trainer.py chargera cette ambiance comme")
    print("contexte des positifs + négatifs réels. Bonne détection TV ! 😉")


if __name__ == "__main__":
    sys.exit(main())
