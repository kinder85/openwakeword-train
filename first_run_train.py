#!/usr/bin/env python3
"""First run dédié à l'entraînement : prépare TOUT le nécessaire.

1. Paquets Python d'entraînement (requirements-train.txt) si absents.
2. Modèles vosk + openWakeWord si absents (first_run.ensure_models).
3. Enregistrements réels du micro — LA matière qui élimine les faux
   positifs en usage réel :
     - 90 s : lis le texte affiché à voix haute (parole réelle)
     - 120 s : reste silencieux (ambiance de la pièce)
   Ces enregistrements servent de négatifs réels ET de fond sonore
   contextuel pour les positifs (cf. trainer.py). Ignorés si déjà présents.
4. Découpage des enregistrements en clips négatifs (mis en cache).

Usage : python first_run_train.py   (répondre aux invites, micro branché)
"""

import os
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

TEXTE_A_LIRE = {
    "fr": """Le wake word est la porte d'entrée de l'assistant vocal. Il écoute en
continu, à faible coût, et ne réveille le système complet que lorsque la
phrase d'activation est reconnue. Un bon modèle doit réagir à la voix de
son propriétaire tout en ignorant la télévision, les conversations, la
musique et les bruits de la maison. Pour cela, il a besoin d'exemples
variés : plusieurs voix, plusieurs débits, plusieurs pièces, et surtout
beaucoup d'audio qui ne contient PAS la phrase d'activation. C'est
exactement ce que fait cet enregistrement : chaque phrase lue ici, chaque
respiration, chaque bruit de la pièce devient un exemple négatif qui
apprend au modèle à ne pas déclencher pour rien. Parle d'une voix
naturelle, ni trop rapide ni trop lente, comme si tu t'adressais à
quelqu'un dans la cuisine.

L'assistant peut faire bien des choses : régler la lumière, lancer une
musique, rappeler un rendez-vous, donner la météo du matin. Mais il ne doit
réagir que lorsqu'on s'adresse vraiment à lui. Imagine que tu es en train de
lire un livre, que la radio joue, que quelqu'un parle dans la pièce d'à
côté : le modèle doit rester silencieux, patient, sans se déclencher. C'est
tout l'art du wake word : être à l'écoute sans être envahissant. Continue à
lire d'une voix posée, en variant le rythme, comme si tu racontais une
histoire à quelqu'un qui t'écoute avec attention.""",
    "en": """The wake word is the gateway to the voice assistant. It listens
continuously, at a low cost, and only wakes up the full system when the
activation phrase is recognized. A good model must react to its owner's
voice while ignoring the television, conversations, music and household
noises. To achieve this, it needs varied examples: several voices, several
speaking rates, several rooms, and above all lots of audio that does NOT
contain the activation phrase. That is exactly what this recording is for:
every sentence read here, every breath, every noise in the room becomes a
negative example that teaches the model not to trigger for nothing. Speak
in a natural voice, neither too fast nor too slow, as if you were talking
to someone in the kitchen.

The assistant can do many things: dim the lights, play some music, remind
you of an appointment, give you the morning weather. But it should only
react when you are truly speaking to it. Imagine you are reading a book,
the radio is playing, someone is talking in the next room: the model must
stay quiet, patient, without triggering. That is the whole art of the wake
word — being attentive without being intrusive. Keep reading in a calm
voice, varying the pace, as if you were telling a story to someone who is
listening carefully.""",
    "de": """Das Wakeword ist der Eingang zum Sprachassistenten. Er hört dauerhaft
zu, kostet wenig, und weckt das komplette System erst, wenn der
Aktivierungssatz erkannt wird. Ein gutes Modell muss auf die Stimme seines
Besitzers reagieren und dabei Fernseher, Gespräche, Musik und Geräusche im
Haus ignorieren. Dafür braucht es vielfältige Beispiele: mehrere Stimmen,
mehrere Geschwindigkeiten, mehrere Räume und vor allem viel Audio, das den
Aktivierungssatz NICHT enthält. Genau dafür ist diese Aufnahme da: Jeder
Satz, den du hier liest, jeder Atemzug, jedes Geräusch im Raum wird zu einem
negativen Beispiel, das dem Modell beibringt, nicht grundlos auszulösen.
Sprich mit natürlicher Stimme, weder zu schnell noch zu langsam, als würdest
du mit jemandem in der Küche sprechen.

Der Assistent kann viele Dinge tun: das Licht dimmen, Musik abspielen, an
einen Termin erinnern oder das morgendliche Wetter ansagen. Aber er darf nur
reagieren, wenn man wirklich mit ihm spricht. Stell dir vor, du liest ein
Buch, das Radio läuft, jemand redet im Nebenzimmer: Das Modell muss still
und geduldig bleiben, ohne auszulösen. Das ist die ganze Kunst des
Wakewords – aufmerksam zu sein, ohne aufdringlich zu sein. Lies weiter mit
ruhiger Stimme, wechsle das Tempo, als würdest du jemandem eine Geschichte
erzählen, die aufmerksam zuhört.""",
    "es": """La palabra de activación es la puerta de entrada del asistente de voz.
Escucha de forma continua, con poco coste, y solo despierta el sistema
completo cuando reconoce la frase de activación. Un buen modelo debe reaccionar
a la voz de su dueño e ignorar la televisión, las conversaciones, la música
y los ruidos de la casa. Para ello necesita ejemplos variados: varias voces,
varios ritmos, varias habitaciones y, sobre todo, mucho audio que NO
contenga la frase de activación. De eso sirve esta grabación: cada frase que
lees aquí, cada respiración, cada ruido de la habitación se convierte en un
ejemplo negativo que enseña al modelo a no activarse por nada. Habla con voz
natural, ni demasiado rápido ni demasiado lento, como si le hablaras a
alguien en la cocina.

El asistente puede hacer muchas cosas: atenuar la luz, poner música, recordar
una cita o dar el tiempo de la mañana. Pero solo debe reaccionar cuando de
verdad le hablas. Imagina que estás leyendo un libro, que suena la radio, que
alguien habla en la habitación de al lado: el modelo debe permanecer en
silencio, paciente, sin activarse. Ese es todo el arte de la palabra de
activación: estar atento sin ser invasivo. Sigue leyendo con voz tranquila,
variando el ritmo, como si contaras una historia a alguien que escucha con
atención.""",
}

# ---------------------------------------------------------------- paquets
def ensure_packages():
    """Installe les paquets d'entraînement manquants (pip --user)."""
    needed = [("edge_tts", "edge-tts"), ("audiomentations", "audiomentations"), ("onnx", "onnx")]
    missing = [pkg for mod, pkg in needed if not _importable(mod)]
    if not missing:
        print("[1/4] Paquets d'entraînement : déjà présents")
        return True
    print("[1/4] Installation des paquets manquants : %s" % ", ".join(missing))
    import subprocess
    try:
        subprocess.check_call(_pip_cmd() + missing)
        return all(_importable(mod) for mod, _ in needed)
    except subprocess.CalledProcessError as e:
        print("✗ échec pip : %s — installe manuellement : %s" % (e, " ".join(_pip_cmd()) + " -r requirements-train.txt"))
        return False


def _importable(mod):
    try:
        __import__(mod)
        return True
    except ImportError:
        return False


def _pip_cmd():
    """Commande pip pour l'interpréteur courant.

    - Dans un venv (sys.prefix != sys.base_prefix) : installe DANS le venv
      (sans --user, qui viserait le site-user système).
    - Hors venv (python système, ex. Debian PEP 668 "externally managed") :
      --user est la bonne méthode (le site système n'est pas inscriptible).
    """
    if sys.prefix != sys.base_prefix:
        return [sys.executable, "-m", "pip", "install"]
    return [sys.executable, "-m", "pip", "install", "--user"]


# ---------------------------------------------------------------- deps tflite
TFLITE_DEPS = [("tensorflow", "tensorflow"), ("onnx2tf", "onnx2tf"),
               ("tf_keras", "tf-keras")]


def ensure_tflite_deps(force=False):
    """Installe les dépendances .tflite (tensorflow + onnx2tf + tf-keras) si
    l'utilisateur vise Home Assistant / ESPHome / ESP32.

    Pose une question interactive (sauf `force=True` ou si déjà présentes).
    Renvoie True si les deps tflite sont disponibles.
    """
    missing = [pkg for mod, pkg in TFLITE_DEPS if not _importable(mod)]
    if not missing:
        print("[1b] Dépendances .tflite (tensorflow + onnx2tf) : déjà présentes")
        return True

    print("\n" + "=" * 70)
    print("FORMAT .tflite (optionnel) — Home Assistant / ESPHome / ESP32")
    print("=" * 70)
    print("Le .tflite permet d'utiliser le wake word sur Home Assistant,")
    print("ESPHome ou un ESP32. Il nécessite des dépendances lourdes")
    print("(tensorflow + onnx2tf, ~600 Mo). Sans elles, seul le .onnx est")
    print("produit (suffisant pour le runtime openwakeword/hapcvoice).")
    if not force:
        try:
            rep = input("Veux-tu installer le support .tflite ? [o/N] ").strip().lower()
        except EOFError:
            rep = ""
        if rep not in ("o", "oui", "y", "yes"):
            print("→ support .tflite ignoré (seul le .onnx sera produit).")
            return False

    print("[1b] Installation des dépendances .tflite : %s" % ", ".join(missing))
    import subprocess
    try:
        subprocess.check_call(_pip_cmd() + missing)
        return all(_importable(mod) for mod, _ in TFLITE_DEPS)
    except subprocess.CalledProcessError as e:
        print("✗ échec pip (tflite) : %s — installe manuellement : "
              "%s \"onnx2tf[tensorflow]\" tf-keras" % (e, " ".join(_pip_cmd())))
        return False


# ---------------------------------------------------------------- enregistrements
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


def countdown(seconds, label):
    print("%s dans %d s" % (label, seconds), flush=True)
    for i in range(seconds, 0, -1):
        print("  %d…" % i, flush=True)
        time.sleep(1)


def record_speech(lang="fr"):
    print("\n" + "=" * 70)
    print("ENREGISTREMENT 1 — PAROLE RÉELLE (90 s) [lang=%s]" % lang)
    print("=" * 70)
    print("Lis ce texte À VOIX HAUTE, d'une voix naturelle (recommence le")
    print("paragraphe si tu finis avant la fin de l'enregistrement) :\n")
    print(TEXTE_A_LIRE[lang] + "\n")
    input("Appuie sur Entrée quand tu es prêt(e)… ")
    countdown(8, "départ")
    print("● PARLE — lis le texte (90 s)…", flush=True)
    path, data = record(90)
    rms = rms_profile(data, 48000)
    spoken = int((rms > 0.008).sum())
    print("… terminé — %d s de parole détectées sur 90" % spoken)
    if spoken < 30:
        print("⚠ très peu de parole détectée — vérifie le micro et relance le script")
        return False
    os.replace(path, os.path.join("training_runs", "_real_negatives", "real_speech_90s.wav"))
    return True


def record_ambient(lang="fr"):
    print("\n" + "=" * 70)
    print("ENREGISTREMENT 2 — AMBIANCE (120 s) [lang=%s]" % lang)
    print("=" * 70)
    input("Reste SILENCIEUX pendant 2 minutes (aucune parole). Entrée pour démarrer… ")
    countdown(5, "départ")
    print("● SILENCE — ambiance de la pièce (120 s)…", flush=True)
    path, data = record(120)
    rms = rms_profile(data, 48000)
    print("… terminé — RMS médian %.4f, max %.4f" % (np.median(rms), rms.max()))
    if np.median(rms) > 0.01:
        print("⚠ environnement très bruyant — les clips resteront utilisables mais")
        print("  idéalement, refais l'enregistrement dans une pièce plus calme")
    os.replace(path, os.path.join("training_runs", "_real_negatives", "real_120s.wav"))
    return True


def ensure_real_audio(lang="fr"):
    os.makedirs(os.path.join("training_runs", "_real_negatives"), exist_ok=True)
    have_speech = os.path.exists(os.path.join("training_runs", "_real_negatives", "real_speech_90s.wav"))
    have_ambient = os.path.exists(os.path.join("training_runs", "_real_negatives", "real_120s.wav"))
    if have_speech and have_ambient:
        print("[3/4] Enregistrements réels du micro : déjà présents")
        return True
    print("[3/4] Enregistrements réels du micro : manquants — nécessaires pour")
    print("      éliminer les faux positifs (négatifs réels + contexte des positifs).")
    if not have_speech:
        if not record_speech(lang):
            return False
    if not have_ambient:
        if not record_ambient(lang):
            return False
    return True


# ---------------------------------------------------------------- main
def main(argv=None):
    import argparse
    import first_run
    parser = argparse.ArgumentParser(
        description="First run : préparation de l'entraînement de wake words custom "
                    "(paquets, modèles vosk + openWakeWord, enregistrements réels)")
    parser.add_argument("--lang", default="fr", choices=sorted(first_run.VOSK_MODELS),
                        help="langue d'entraînement (modèle vosk + texte à lire)")
    tflite_grp = parser.add_mutually_exclusive_group()
    tflite_grp.add_argument("--tflite", action="store_true",
                            help="installer les dépendances .tflite (HA/ESPHome/ESP32) sans demander")
    tflite_grp.add_argument("--no-tflite", action="store_true",
                            help="ne pas installer les dépendances .tflite")
    args = parser.parse_args(argv)
    lang = args.lang
    print("=" * 70)
    print("FIRST RUN — préparation de l'entraînement de wake words custom (lang=%s)" % lang)
    print("=" * 70)

    if not ensure_packages():
        sys.exit(1)
    if args.no_tflite:
        print("[1b] support .tflite ignoré (--no-tflite)")
    elif not ensure_tflite_deps(force=args.tflite):
        # échec d'installation : on continue sans tflite (le .onnx suffit)
        pass
    if not first_run.ensure_models(locale=lang):
        sys.exit(1)
    print("[2/4] Modèles vosk + openWakeWord : prêts (lang=%s)" % lang)

    if not ensure_real_audio(lang):
        sys.exit(1)
    print("[3/4] Enregistrements réels : prêts (lang=%s)" % lang)

    import trainer
    clips = trainer.real_negative_paths()
    if not clips:
        print("[4/4] ✗ aucun clip réel produit — vérifie training_runs/_real_negatives/")
        sys.exit(1)
    print("[4/4] Clips négatifs réels : %d (training_runs/_real_negatives/clips/)" % len(clips))

    print("\n✓ Prêt. Lance python trainer_gui.py — les négatifs réels seront")
    print("  inclus automatiquement dans chaque entraînement.")


if __name__ == "__main__":
    main()
