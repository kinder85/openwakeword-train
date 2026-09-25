#!/usr/bin/env python3
"""Premier lancement : télécharge les modèles absents.

- modèle Vosk (par langue, ex. vosk-model-small-fr-0.22) dans model/<locale>/
- modèles openWakeWord officiels dans openwakeword/

Appelé automatiquement par trainer_gui.py et first_run_train.py : si les
modèles sont déjà présents pour la locale demandée, il ne fait rien (aucun
re-téléchargement inutile).

Réplique locale autonome de hapcvoice/first_run.py (plus aucune dépendance
à l'écosystème hapcvoice) — chaque langue vit dans son sous-dossier.
"""

import os
import zipfile
import urllib.request

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Modèle Vosk par langue (nom de dossier + URL du zip).
# Références : https://alphacephei.com/vosk/models
VOSK_MODELS = {
    "fr": ("vosk-model-small-fr-0.22",
           "https://alphacephei.com/vosk/models/vosk-model-small-fr-0.22.zip"),
    "en": ("vosk-model-small-en-us-0.15",
           "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"),
    "de": ("vosk-model-small-de-0.15",
           "https://alphacephei.com/vosk/models/vosk-model-small-de-0.15.zip"),
    "es": ("vosk-model-small-es-0.42",
           "https://alphacephei.com/vosk/models/vosk-model-small-es-0.42.zip"),
}


def model_dir_for(models_dir, locale):
    """Chemin du dossier du modèle vosk pour la locale (model/<locale>/)."""
    return os.path.join(models_dir, locale)


def has_vosk_model(models_dir="model", locale="fr"):
    """True si un modèle vosk valide (avec am/ ou conf/) existe pour locale."""
    base = model_dir_for(models_dir, locale)

    def is_vosk(p):
        return os.path.isdir(p) and (os.path.isdir(os.path.join(p, 'am'))
                                     or os.path.isdir(os.path.join(p, 'conf')))

    # soit le dossier est lui-même le modèle, soit il contient un modèle
    if is_vosk(base):
        return True
    if os.path.isdir(base):
        return any(is_vosk(os.path.join(base, e)) for e in os.listdir(base))
    return False


def has_wakeword(wakeword_dir="openwakeword"):
    """True si au moins un modèle wake word (.onnx, hors modèles de features) existe."""
    if not os.path.isdir(wakeword_dir):
        return False
    return any(f.endswith(".onnx")
               and not any(x in f for x in ("embedding", "melspectrogram", "silero"))
               for f in os.listdir(wakeword_dir))


def download_vosk(models_dir="model", locale="fr"):
    if locale not in VOSK_MODELS:
        print("[first_run] locale inconnue : %s (disponibles : %s)"
              % (locale, ", ".join(sorted(VOSK_MODELS))))
        return False
    folder, url = VOSK_MODELS[locale]
    dest = model_dir_for(models_dir, locale)
    os.makedirs(dest, exist_ok=True)
    zpath = os.path.join(dest, folder + ".zip")
    print("[first_run] modèle vosk (%s) absent → téléchargement de %s (~40 Mo)"
          % (locale, url))
    try:
        urllib.request.urlretrieve(url, zpath)
        with zipfile.ZipFile(zpath) as z:
            # anti zip-slip : on refuse tout membre qui sortirait de dest
            dest_abs = os.path.abspath(dest)
            for m in z.infolist():
                target = os.path.abspath(os.path.join(dest, m.filename))
                if ".." in m.filename.split("/") or os.path.commonpath([dest_abs, target]) != dest_abs:
                    raise ValueError("membre de zip non sûr : %r" % m.filename)
            z.extractall(dest)
        os.remove(zpath)
        return has_vosk_model(models_dir, locale)
    except Exception as e:
        print("[first_run] échec du téléchargement vosk (%s) : %s" % (locale, e))
        return False


def download_wakewords(wakeword_dir="openwakeword"):
    os.makedirs(wakeword_dir, exist_ok=True)
    print("[first_run] modèles openwakeword absents → téléchargement des modèles officiels")
    try:
        from openwakeword.utils import download_models
        download_models(target_directory=wakeword_dir)
        return has_wakeword(wakeword_dir)
    except Exception as e:
        print("[first_run] échec du téléchargement openwakeword : %s" % e)
        return False


def ensure_models(models_dir="model", wakeword_dir="openwakeword", locale="fr"):
    """Complète ce qui manque pour la locale ; True si les deux familles sont prêtes."""
    ok = True
    if not has_vosk_model(models_dir, locale):
        ok = download_vosk(models_dir, locale) and ok
    if not has_wakeword(wakeword_dir):
        ok = download_wakewords(wakeword_dir) and ok
    if ok:
        print("[first_run] modèles prêts (locale=%s)" % locale)
    return ok


if __name__ == "__main__":
    import sys
    loc = sys.argv[1] if len(sys.argv) > 1 else "fr"
    sys.exit(0 if ensure_models(locale=loc) else 1)
