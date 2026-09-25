#!/usr/bin/env python3
"""Entraînement de modèles openWakeWord custom — pipeline complet local.

1. Génération des échantillons positifs par TTS (edge-tts, voix françaises
   multiples, prosodies variées) — à ÉCOUTER avant entraînement.
2. Négatifs : phrases françaises aléatoires (autres que la cible) + bruits.
3. Augmentation (audiomentations) : bruit, pitch, vitesse, distorsion.
4. Features : embeddings openWakeWord (melspectrogram → embedding, 96 dims).
5. Entraînement d'un classifieur numpy répliquant exactement l'architecture
   des modèles openWakeWord : FC 1536→32 + LayerNorm + ReLU, FC 32→32 +
   LayerNorm + ReLU, FC 32→1 + Sigmoid (entrée [1, 16, 96]).
6. Export .onnx + métadonnées JSON → auto-découvert par run.py/gui.py.

Utilisable headless (voir __main__) ou via trainer_gui.py.
"""

import asyncio
import glob
import json
import os
import random
import re
import wave

import numpy as np
import soundfile as sf
from audiomentations import Compose, AddGaussianNoise, PitchShift, TimeStretch, ClippingDistortion, AddBackgroundNoise

# ---------------------------------------------------------------- constantes
SR = 16000
WINDOW_SAMPLES = 20480          # 1,28 s = 16 trames de 80 ms (entrée du modèle)
EMBED_DIM = 96
N_FRAMES = 16

# Caractères autorisés dans un nom de modèle/fichier (anti-traversée de chemin)
_NAME_SAFE = re.compile(r"[^A-Za-z0-9_\-]")


def sanitize_name(name, fallback="wakeword"):
    """Réduit un nom de modèle à des caractères sûrs pour un chemin de fichier.

    Remplace tout caractère hors [A-Za-z0-9_-] par '_', retire les '_' de bord,
    et garantit un nom non vide (ni '', ni '.', ni '..') — évite toute
    traversée de chemin via un nom de phrase ou de modèle fourni par l'UI/CLI.
    """
    clean = _NAME_SAFE.sub("_", str(name)).strip("_")
    return clean or fallback

# Phrases négatives (tout sauf la cible) — vie quotidienne, varied speakers.
# Structurées par langue ; on ne doit PAS y inclure la phrase cible.
NEGATIVE_SENTENCES = {
    "fr": [
        "Il fait beau aujourd'hui.",
        "Quelle heure est-il à Tokyo ?",
        "J'ai besoin de faire les courses.",
        "Le train part dans dix minutes.",
        "Rappelle-moi d'acheter du pain.",
        "La réunion est reportée à demain matin.",
        "Ce film était vraiment passionnant.",
        "Attention, la route est glissante.",
        "Je vais sortir le chien avant de partir.",
        "Peux-tu baisser le volume de la télévision ?",
        "Les enfants jouent dans le jardin.",
        "Le facteur est passé ce matin.",
        "Il faut arroser les plantes ce week-end.",
        "J'ai oublié mes clés sur la table.",
        "Le dîner sera prêt vers vingt heures.",
        "Pense à fermer les volets le soir.",
        "La météo annonce de la pluie demain.",
        "Ce restaurant a d'excellents avis.",
        "Mon ordinateur portable est en charge.",
        "Le colis arrive en fin de journée.",
        "On pourrait partir en balade dimanche.",
        "La machine à laver a terminé son cycle.",
        "N'oublie pas de sortir la poubelle.",
        "Le match commence dans une heure.",
        "Cette recette demande trois œufs.",
        "Le bus est en retard ce soir.",
        "Je lis un roman passionnant en ce moment.",
        "Les vacances approchent à grands pas.",
        "Il faudra réparer la porte du garage.",
        "Le chauffage semble ne plus fonctionner.",
        "Allume la télévision s'il te plaît.",
        "Éteins la lumière du couloir.",
        "Monte le son un petit peu.",
        "Rappelle-moi demain à midi.",
        "Mets de la musique douce.",
        "Ferme les rideaux du salon.",
        "Ouvre la fenêtre de la cuisine.",
        "Donne-moi la météo de ce week-end.",
        "Lance une minuterie de dix minutes.",
        "Régale-toi avec ce gâteau.",
        "Passe-moi le sel s'il te plaît.",
        "Je voudrais un café serré.",
        "Le chat dort sur le canapé.",
        "Ton téléphone sonne depuis tout à l'heure.",
        "Il y a du courrier pour toi.",
        "On se retrouve vers quinze heures ?",
        "Ce dossier est vraiment urgent.",
        "Arrête de faire du bruit.",
        "Baisse un peu le chauffage.",
        "Où ai-je mis mes lunettes ?",
    ],
    "en": [
        "It's a beautiful day today.",
        "What time is it in Tokyo?",
        "I need to go grocery shopping.",
        "The train leaves in ten minutes.",
        "Remind me to buy some bread.",
        "The meeting is postponed until tomorrow morning.",
        "That movie was really exciting.",
        "Careful, the road is slippery.",
        "I'll walk the dog before leaving.",
        "Can you turn down the volume?",
        "The kids are playing in the garden.",
        "The mailman came by this morning.",
        "We need to water the plants this weekend.",
        "I left my keys on the table.",
        "Dinner will be ready around eight o'clock.",
        "Remember to close the shutters tonight.",
        "The forecast calls for rain tomorrow.",
        "That restaurant has excellent reviews.",
        "My laptop is charging.",
        "The package arrives at the end of the day.",
        "We could go for a walk on Sunday.",
        "The washing machine has finished its cycle.",
        "Don't forget to take out the trash.",
        "The game starts in an hour.",
        "This recipe needs three eggs.",
        "The bus is running late tonight.",
        "I'm reading an exciting novel right now.",
        "The holidays are approaching fast.",
        "We'll need to fix the garage door.",
        "The heater doesn't seem to work anymore.",
        "Turn on the television please.",
        "Turn off the hallway light.",
        "Turn up the volume a little bit.",
        "Remind me tomorrow at noon.",
        "Play some soft music.",
        "Close the living room curtains.",
        "Open the kitchen window.",
        "Give me this weekend's weather.",
        "Start a ten minute timer.",
        "Enjoy this cake.",
        "Pass me the salt please.",
        "I'd like a strong coffee.",
        "The cat is sleeping on the sofa.",
        "Your phone has been ringing for a while.",
        "There's some mail for you.",
        "Shall we meet around three o'clock?",
        "This file is really urgent.",
        "Stop making so much noise.",
        "Turn the heating down a bit.",
        "Where did I put my glasses?",
    ],
    "de": [
        "Heute ist ein schöner Tag.",
        "Wie spät ist es in Tokio?",
        "Ich muss einkaufen gehen.",
        "Der Zug fährt in zehn Minuten ab.",
        "Erinnere mich daran, Brot zu kaufen.",
        "Die Besprechung ist auf morgen früh verschoben.",
        "Der Film war wirklich spannend.",
        "Vorsicht, die Straße ist glatt.",
        "Ich gehe vor dem Weggehen mit dem Hund raus.",
        "Kannst du die Lautstärke leiser stellen?",
        "Die Kinder spielen im Garten.",
        "Der Briefträger ist heute Morgen vorbeigekommen.",
        "Wir müssen am Wochenende die Pflanzen gießen.",
        "Ich habe meine Schlüssel auf dem Tisch gelassen.",
        "Das Abendessen ist gegen acht Uhr fertig.",
        "Denk daran, heute Abend die Rollläden zu schließen.",
        "Die Wettervorhersage kündigt Regen für morgen an.",
        "Dieses Restaurant hat ausgezeichnete Bewertungen.",
        "Mein Laptop lädt gerade.",
        "Das Paket kommt im Laufe des Tages an.",
        "Wir könnten am Sonntag einen Spaziergang machen.",
        "Die Waschmaschine hat ihren Zyklus beendet.",
        "Vergiss nicht, den Müll rauszubringen.",
        "Das Spiel beginnt in einer Stunde.",
        "Dieses Rezept braucht drei Eier.",
        "Der Bus hat heute Abend Verspätung.",
        "Ich lese gerade einen spannenden Roman.",
        "Die Ferien rücken schnell näher.",
        "Wir müssen die Garagentür reparieren.",
        "Die Heizung scheint nicht mehr zu funktionieren.",
        "Schalte bitte den Fernseher ein.",
        "Mach das Flurlicht aus.",
        "Mach die Lautstärke etwas lauter.",
        "Erinnere mich morgen um zwölf Uhr.",
        "Spiel leise Musik.",
        "Zieh die Vorhänge im Wohnzimmer zu.",
        "Öffne das Küchenfenster.",
        "Sag mir das Wetter für dieses Wochenende.",
        "Stelle einen Timer auf zehn Minuten.",
        "Genieße diesen Kuchen.",
        "Gib mir bitte das Salz.",
        "Ich hätte gerne einen starken Kaffee.",
        "Die Katze schläft auf dem Sofa.",
        "Dein Telefon klingelt schon eine Weile.",
        "Da ist Post für dich.",
        "Treffen wir uns gegen drei Uhr?",
        "Diese Datei ist wirklich dringend.",
        "Hör auf, so viel Lärm zu machen.",
        "Dreh die Heizung etwas runter.",
        "Wo habe ich meine Brille hingelegt?",
    ],
    "es": [
        "Hoy hace un día precioso.",
        "¿Qué hora es en Tokio?",
        "Tengo que ir a hacer la compra.",
        "El tren sale en diez minutos.",
        "Recuérdame comprar pan.",
        "La reunión se ha pospuesto hasta mañana por la mañana.",
        "Esa película fue muy emocionante.",
        "Cuidado, la carretera está resbaladiza.",
        "Voy a pasear al perro antes de irme.",
        "¿Puedes bajar el volumen?",
        "Los niños juegan en el jardín.",
        "El cartero ha pasado esta mañana.",
        "Tenemos que regar las plantas este fin de semana.",
        "He dejado las llaves sobre la mesa.",
        "La cena estará lista sobre las ocho.",
        "Acuérdate de cerrar las persianas esta noche.",
        "El pronóstico anuncia lluvia para mañana.",
        "Ese restaurante tiene reseñas excelentes.",
        "Mi portátil se está cargando.",
        "El paquete llega a lo largo del día.",
        "Podríamos dar un paseo el domingo.",
        "La lavadora ha terminado su ciclo.",
        "No olvides sacar la basura.",
        "El partido empieza dentro de una hora.",
        "Esta receta necesita tres huevos.",
        "El autobús lleva retraso esta noche.",
        "Ahora mismo estoy leyendo una novela apasionante.",
        "Las vacaciones se acercan rápidamente.",
        "Tendremos que arreglar la puerta del garaje.",
        "La calefacción parece que ya no funciona.",
        "Enciende la televisión por favor.",
        "Apaga la luz del pasillo.",
        "Sube un poco el volumen.",
        "Recuérdame mañana a mediodía.",
        "Pon música suave.",
        "Cierra las cortinas del salón.",
        "Abre la ventana de la cocina.",
        "Dime el tiempo de este fin de semana.",
        "Pon un temporizador de diez minutos.",
        "Disfruta de esta tarta.",
        "Pásame la sal por favor.",
        "Quisiera un café fuerte.",
        "El gato duerme en el sofá.",
        "Tu teléfono lleva un rato sonando.",
        "Hay correo para ti.",
        "¿Quedamos sobre las tres?",
        "Este archivo es muy urgente.",
        "Deja de hacer tanto ruido.",
        "Baja un poco la calefacción.",
        "¿Dónde he puesto mis gafas?",
    ],
}

# Variantes de prosodie edge-tts (débit, ton) pour la diversité
PROSODY_VARIANTS = [
    {"rate": "+0%", "pitch": "+0Hz"},
    {"rate": "-5%", "pitch": "+0Hz"},
    {"rate": "+5%", "pitch": "+0Hz"},
    {"rate": "+0%", "pitch": "-10Hz"},
    {"rate": "+0%", "pitch": "+10Hz"},
    {"rate": "-10%", "pitch": "+5Hz"},
    {"rate": "+10%", "pitch": "-5Hz"},
]


# ---------------------------------------------------------------- TTS (edge-tts)
async def _list_voices(lang="fr"):
    import edge_tts
    voices = await edge_tts.list_voices()
    return [v for v in voices if v["Locale"].startswith(lang)]


def list_voices(lang="fr"):
    """Voix edge-tts pour une langue (ex. fr, en) — [(ShortName, Gender)]."""
    voices = asyncio.run(_list_voices(lang))
    return [(v["ShortName"], v["Gender"]) for v in voices]


def list_french_voices():
    """Alias rétro-compat de list_voices('fr') — à ne pas utiliser pour les
    chemins multilingues (toujours passer par list_voices(lang))."""
    return list_voices("fr")


async def _tts_one(text, voice, out_mp3, rate, pitch):
    import edge_tts
    com = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
    await com.save(out_mp3)


def _mp3_to_16k(mp3_path):
    """Décode un mp3 (soundfile/libsndfile) et renvoie float32 mono 16 kHz."""
    data, sr = sf.read(mp3_path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != SR:
        n_out = int(round(len(mono) * SR / sr))
        x_in = np.arange(len(mono), dtype=np.float64)
        mono = np.interp(np.linspace(0, len(mono) - 1, n_out), x_in, mono).astype(np.float32)
    return mono


def _trim_silence(audio, thresh=0.01, pad_ms=120):
    """Coupe les silences de début/fin (seuil d'énergie) avec une marge."""
    if len(audio) == 0:
        return audio
    win = int(SR * 0.02)
    energy = np.convolve(np.abs(audio), np.ones(win) / win, mode="same")
    idx = np.where(energy > thresh)[0]
    if len(idx) == 0:
        return audio
    pad = int(SR * pad_ms / 1000)
    start = max(0, int(idx[0]) - pad)
    end = min(len(audio), int(idx[-1]) + pad)
    return audio[start:end]


def _write_wav(path, audio):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())


def generate_samples(phrase, voices, n_per_voice, out_dir, progress_cb=None):
    """Génère les échantillons positifs (phrase × voix × prosodies) en WAV 16 kHz.

    Renvoie la liste des chemins. `out_dir` est créé si besoin.
    """
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    total = len(voices) * n_per_voice
    done = 0
    variants = list(PROSODY_VARIANTS)
    for voice in voices:
        for i in range(n_per_voice):
            pros = variants[i % len(variants)]
            text = phrase if i < len(variants) else phrase.capitalize()
            mp3 = os.path.join(out_dir, "tmp.mp3")
            asyncio.run(_tts_one(text, voice, mp3, pros["rate"], pros["pitch"]))
            audio = _trim_silence(_mp3_to_16k(mp3))
            os.remove(mp3)
            out = os.path.join(out_dir, "pos_%03d.wav" % done)
            _write_wav(out, audio)
            paths.append(out)
            done += 1
            if progress_cb:
                progress_cb(done, total, out)
    return paths


def generate_negatives(n, out_dir, lang="fr", voices=None, progress_cb=None):
    """Génère des clips négatifs (phrases non cibles de la langue) en WAV 16 kHz."""
    os.makedirs(out_dir, exist_ok=True)
    if lang not in NEGATIVE_SENTENCES:
        raise ValueError("langue inconnue : %s (disponibles : %s)"
                         % (lang, ", ".join(sorted(NEGATIVE_SENTENCES))))
    if not voices:
        voices = [v for v, _ in list_voices(lang)]
    rng = random.Random(42)
    sentences = [s for s in rng.sample(NEGATIVE_SENTENCES[lang],
                                       min(n, len(NEGATIVE_SENTENCES[lang])))]
    while len(sentences) < n:
        sentences.append(rng.choice(NEGATIVE_SENTENCES[lang]))
    paths = []
    variants = list(PROSODY_VARIANTS)
    for i, sentence in enumerate(sentences[:n]):
        mp3 = os.path.join(out_dir, "tmp_neg.mp3")
        pros = variants[i % len(variants)]
        asyncio.run(_tts_one(sentence, voices[i % len(voices)], mp3, pros["rate"], pros["pitch"]))
        audio = _trim_silence(_mp3_to_16k(mp3))
        os.remove(mp3)
        out = os.path.join(out_dir, "neg_%03d.wav" % i)
        _write_wav(out, audio)
        paths.append(out)
        if progress_cb:
            progress_cb(i + 1, n, out)
    return paths


# ---------------------------------------------------------------- augmentation
def _make_noise_files(noise_dir):
    """Quelques fichiers de bruit de fond synthétiques pour AddBackgroundNoise."""
    os.makedirs(noise_dir, exist_ok=True)
    t = np.arange(SR * 5) / SR
    rng = np.random.default_rng(0)
    _write_wav(os.path.join(noise_dir, "white.wav"), 0.3 * rng.standard_normal(SR * 5).astype(np.float32))
    pink = np.cumsum(rng.standard_normal(SR * 5)).astype(np.float32)
    _write_wav(os.path.join(noise_dir, "pink.wav"), 0.02 * pink / max(1e-9, np.max(np.abs(pink))))
    hum = (0.2 * np.sin(2 * np.pi * 50 * t) + 0.1 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
    _write_wav(os.path.join(noise_dir, "hum.wav"), hum)


def make_augmenter(noise_dir):
    return Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.6),
        AddBackgroundNoise(noise_dir, min_snr_db=10, max_snr_db=30, p=0.5),
        PitchShift(min_semitones=-3, max_semitones=3, p=0.4),
        TimeStretch(min_rate=0.9, max_rate=1.1, p=0.4),
        ClippingDistortion(min_percentile_threshold=0, max_percentile_threshold=10, p=0.2),
    ], p=1.0)


def load_wav_16k(path):
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != SR:
        n_out = int(round(len(mono) * SR / sr))
        x_in = np.arange(len(mono), dtype=np.float64)
        mono = np.interp(np.linspace(0, len(mono) - 1, n_out), x_in, mono).astype(np.float32)
    return mono


def _fixed_length(audio, length=WINDOW_SAMPLES, rng=None):
    """Pad/coupe un clip vers une longueur fixe (padding centré aléatoire)."""
    rng = rng or np.random.default_rng()
    if len(audio) >= length:
        start = rng.integers(0, len(audio) - length + 1)
        return audio[start:start + length]
    pad = length - len(audio)
    left = int(rng.integers(0, pad + 1)) if pad > int(SR * 0.3) else pad // 2
    return np.concatenate([np.zeros(left, dtype=np.float32), audio,
                           np.zeros(pad - left, dtype=np.float32)])


# trames « à froid » de l'extraction batch à exclure : au runtime, le mode
# streaming fournit toujours des trames avec du contexte audio réel — les
# premières trames d'un clip batch n'ont pas d'équivalent (mesuré : offset 9)
WARMUP_FRAMES = 9

# fond sonore réel (ambiance micro) pour contextualiser les clips — cf.
# set_context_audio() ; fallback : silence numérique (jamais utilisé en
# direct pour les positifs si l'enregistrement réel existe)
_CTX_AUDIO = None


def set_context_audio(path):
    """Charge l'ambiance réelle du micro servant de contexte d'entraînement."""
    global _CTX_AUDIO
    import soundfile as sf
    if not path or not os.path.exists(path):
        _CTX_AUDIO = None
        return False
    d, sr = sf.read(path, dtype="float32", always_2d=True)
    d = d.mean(axis=1)
    if sr != SR:
        n_out = int(round(len(d) * SR / sr))
        d = np.interp(np.linspace(0, len(d) - 1, n_out), np.arange(len(d)), d)
    _CTX_AUDIO = d.astype(np.float32)
    return True


def ambient_clip(n_samples, rng=None):
    """Segment aléatoire du fond sonore réel (ou silence si absent)."""
    rng = rng or np.random.default_rng()
    if _CTX_AUDIO is None or len(_CTX_AUDIO) <= n_samples:
        return np.zeros(n_samples, dtype=np.float32)
    st = int(rng.integers(0, len(_CTX_AUDIO) - n_samples))
    return _CTX_AUDIO[st:st + n_samples]


def _phrase_centered(audio, margin_s=0.5, min_total_s=2.6):
    """Étend un clip (phrase déjà débarrassée de ses silences) avec la phrase
    centrée, sur une durée suffisante pour ≥16 trames et plusieurs fenêtres
    couvrant la phrase. Le padding est ensuite toujours bruité par
    l'augmentation — jamais du silence numérique."""
    need = max(int(SR * min_total_s), len(audio) + int(SR * margin_s))
    pad = need - len(audio)
    left = pad // 2
    return np.concatenate([np.zeros(left, dtype=np.float32), audio,
                           np.zeros(pad - left, dtype=np.float32)])


def make_silence_windows(n, seed=0):
    """Fenêtres d'embeddings issues d'audio SANS parole (silence, souffle,
    ronflette, bruit faible) — négatifs explicites contre les faux positifs
    sur le silence. Les trames de warm-up batch sont exclues (alignement
    streaming)."""
    rng = np.random.default_rng(seed)
    dur = int(SR * 4.0)
    t = np.arange(dur) / SR
    kinds = [
        np.zeros(dur, dtype=np.float32),
        (0.002 * rng.standard_normal(dur)).astype(np.float32),
        (0.01 * rng.standard_normal(dur)).astype(np.float32),
        (0.03 * np.sin(2 * np.pi * 50 * t)).astype(np.float32),
        (0.005 * np.sin(2 * np.pi * 100 * t) + 0.002 * rng.standard_normal(dur)).astype(np.float32),
    ]
    X = []
    per = max(1, n // len(kinds))
    for kind in kinds:
        for _ in range(per):
            start = rng.integers(0, len(kind) - int(SR * 3.5))
            clip = kind[start:start + int(SR * 3.5)]
            e = compute_embeddings(clip[None, :])[0]
            for s in range(WARMUP_FRAMES, len(e) - N_FRAMES + 1, 2):
                X.append(e[s:s + N_FRAMES])
    X = np.stack(X).astype(np.float32)
    order = rng.permutation(len(X))
    return X[order][:n]


def build_windows(pos_paths, neg_paths, augmenter=None, seed=0, stride=2,
                  n_silence=None):
    """Construit les fenêtres d'embeddings + labels.

    Alignement streaming : les 9 premières trames batch (warm-up, sans
    contexte audio réel) sont exclues de toutes les fenêtres.

    Positifs : phrase placée dans un fond sonore réel (context_audio —
    ambiance du micro) pour reproduire les conditions du runtime ;
    augmentation TOUJOURS appliquée ; seules les fenêtres couvrant la
    phrase sont retenues. Négatifs : fenêtres glissantes sur des clips de
    parole non cible + fenêtres de silence/bruit seul (n_silence).
    """
    rng = np.random.default_rng(seed)
    ambient = lambda n: ambient_clip(n, rng)

    def windows_from(e, lo, hi):
        """Fenêtres de 16 trames dont le centre est dans [lo, hi] (frames)."""
        F = len(e)
        out = []
        if F < N_FRAMES + WARMUP_FRAMES:
            return out
        centers = [s for s in range(WARMUP_FRAMES, F - N_FRAMES + 1)
                   if lo <= s + N_FRAMES // 2 <= hi]
        if not centers:
            centers = [max(WARMUP_FRAMES, min(int(lo) - N_FRAMES // 2, F - N_FRAMES))]
        step = max(1, len(centers) // 5)
        return out + centers[::step][:5]

    X, y = [], []
    for p in pos_paths:
        a = load_wav_16k(p)
        pre = ambient(int(SR * 1.5))
        post = ambient(int(SR * 2.0))
        clip = np.concatenate([pre, a, post])
        if augmenter is not None:
            clip = augmenter(clip, sample_rate=SR)
        clip = clip + rng.normal(0, 0.001, len(clip)).astype(np.float32)
        e = compute_embeddings(clip[None, :])[0]
        F = len(e)
        center = (len(pre) + len(a) / 2) / len(clip) * (F - 1)
        half_span = max(6, int(len(a) / len(clip) * (F - 1) / 2) + 4)
        for s in windows_from(e, center - half_span, center + half_span):
            X.append(e[s:s + N_FRAMES])
            y.append(1)
    for p in neg_paths:
        a = load_wav_16k(p)
        if len(a) < int(SR * 4.0):
            a = np.concatenate([ambient(int(SR * 1.5)), a, ambient(int(SR * 1.5))])
        if augmenter is not None:
            a = augmenter(a, sample_rate=SR)
        e = compute_embeddings(a[None, :])[0]
        for s in range(WARMUP_FRAMES, len(e) - N_FRAMES + 1, stride):
            X.append(e[s:s + N_FRAMES])
            y.append(0)
    n_silence = n_silence if n_silence is not None else len(pos_paths) * 2
    if n_silence > 0:
        for w in make_silence_windows(n_silence, seed=seed):
            X.append(w)
            y.append(0)
    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int32)
    order = rng.permutation(len(y))
    return X[order], y[order]


# ---------------------------------------------------------------- features
_featurizer = None

def get_featurizer():
    global _featurizer
    if _featurizer is None:
        from openwakeword.utils import AudioFeatures
        # signature d'API variable selon la version d'openwakeword : les
        # versions récentes (≥0.4.0) ont supprimé `inference_framework` (onnx
        # par défaut, chemin vers les modèles resources du package)
        try:
            _featurizer = AudioFeatures(inference_framework="onnx")
        except TypeError:
            _featurizer = AudioFeatures()
    return _featurizer


def compute_embeddings(clips):
    """(N, L) float [-1,1] → (N, F, 96) embeddings openWakeWord."""
    fe = get_featurizer()
    pcm = (np.clip(clips, -1.0, 1.0) * 32767).astype(np.int16)
    if pcm.ndim == 1:
        pcm = pcm[None, :]
    # évite le défaut d'arrondi (±1 trame) de _get_melspectrogram_batch sur
    # certaines longueurs : on ajoute une marge de 320 échantillons
    pad = 320 - (pcm.shape[1] % 320)
    pcm = np.concatenate([pcm, np.zeros((pcm.shape[0], pad), dtype=np.int16)], axis=1)
    return fe.embed_clips(pcm).astype(np.float32)


# ---------------------------------------------------------------- modèle numpy
class WakeClassifier:
    """Réplique exacte de l'architecture onnx des modèles openWakeWord."""

    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = (rng.standard_normal((32, 1536)) * np.sqrt(2 / 1536)).astype(np.float32)
        self.b1 = np.zeros(32, dtype=np.float32)
        self.g1 = np.ones(32, dtype=np.float32)
        self.beta1 = np.zeros(32, dtype=np.float32)
        self.W2 = (rng.standard_normal((32, 32)) * np.sqrt(2 / 32)).astype(np.float32)
        self.b2 = np.zeros(32, dtype=np.float32)
        self.g2 = np.ones(32, dtype=np.float32)
        self.beta2 = np.zeros(32, dtype=np.float32)
        self.W3 = np.zeros((1, 32), dtype=np.float32)
        self.b3 = np.zeros(1, dtype=np.float32)

    # ---- forward
    def _layer_norm(self, x, g, beta, eps=1e-5):
        mu = x.mean(axis=1, keepdims=True)
        var = ((x - mu) ** 2).mean(axis=1, keepdims=True)
        return (x - mu) / np.sqrt(var + eps) * g + beta

    def forward(self, X):
        h = X.reshape(len(X), -1)                      # (N, 1536)
        z1 = h @ self.W1.T + self.b1
        h1 = np.maximum(self._layer_norm(z1, self.g1, self.beta1), 0)
        z2 = h1 @ self.W2.T + self.b2
        h2 = np.maximum(self._layer_norm(z2, self.g2, self.beta2), 0)
        logits = (h2 @ self.W3.T + self.b3).ravel()
        return logits, (h, z1, h1, z2, h2)

    def predict_proba(self, X):
        logits, _ = self.forward(X)
        return 1.0 / (1.0 + np.exp(-logits))

    # ---- entraînement (Adam, BCE)
    def fit(self, X, y, epochs=40, batch_size=128, lr=1e-3, wd=1e-4,
            val_split=0.15, progress_cb=None, seed=0):
        rng = np.random.default_rng(seed)
        y = y.astype(np.float32)
        idx_val = rng.choice(len(y), size=max(8, int(len(y) * val_split)), replace=False)
        mask_val = np.zeros(len(y), dtype=bool)
        mask_val[idx_val] = True
        Xtr, ytr = X[~mask_val], y[~mask_val]
        Xva, yva = X[mask_val], y[mask_val]

        params = [self.W1, self.b1, self.g1, self.beta1, self.W2, self.b2,
                  self.g2, self.beta2, self.W3, self.b3]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        t = 0

        def _layer_norm_bwd(dout, z, g):
            mu = z.mean(axis=1, keepdims=True)
            var = ((z - mu) ** 2).mean(axis=1, keepdims=True)
            xhat = (z - mu) / np.sqrt(var + 1e-5)
            N = z.shape[1]
            dg = (dout * xhat).sum(axis=0)
            dbeta = dout.sum(axis=0)
            dxhat = dout * g
            dz = (1.0 / np.sqrt(var + 1e-5)) * (
                dxhat - dxhat.mean(axis=1, keepdims=True)
                - xhat * (dxhat * xhat).mean(axis=1, keepdims=True))
            return dz, dg, dbeta

        for epoch in range(epochs):
            order = rng.permutation(len(ytr))
            total_loss = 0.0
            for start in range(0, len(ytr), batch_size):
                idx = order[start:start + batch_size]
                xb, yb = Xtr[idx], ytr[idx]
                logits, cache = self.forward(xb)
                h, z1, h1, z2, h2 = cache
                p = 1.0 / (1.0 + np.exp(-logits))
                eps = 1e-7
                loss = -(yb * np.log(p + eps) + (1 - yb) * np.log(1 - p + eps)).mean()
                total_loss += loss * len(yb)

                dlogits = (p - yb) / len(yb)                     # (B,)
                dW3 = dlogits[None, :] @ h2                      # (1, B) @ (B, 32)
                db3 = dlogits.sum(keepdims=True)
                dh2 = dlogits[:, None] * self.W3                 # (B, 32)
                dz2 = dh2 * (z2 > 0)
                dz2, dg2, dbeta2 = _layer_norm_bwd(dz2, z2, self.g2)
                dW2 = dz2.T @ h1
                db2 = dz2.sum(axis=0)
                dh1 = dz2 @ self.W2
                dz1 = dh1 * (z1 > 0)
                dz1, dg1, dbeta1 = _layer_norm_bwd(dz1, z1, self.g1)
                dW1 = dz1.T @ h
                db1 = dz1.sum(axis=0)

                grads = [dW1, db1, dg1, dbeta1, dW2, db2, dg2, dbeta2, dW3, db3]
                t += 1
                for pi, (p_, g_) in enumerate(zip(params, grads)):
                    if pi in (0, 4, 8):                          # poids : weight decay
                        g_ = g_ + wd * p_
                    m[pi] = 0.9 * m[pi] + 0.1 * g_
                    v[pi] = 0.999 * v[pi] + 0.001 * g_ ** 2
                    mh = m[pi] / (1 - 0.9 ** t)
                    vh = v[pi] / (1 - 0.999 ** t)
                    p_ -= lr * mh / (np.sqrt(vh) + 1e-8)

            if progress_cb:
                pv = self.predict_proba(Xva)
                info = {
                    "epoch": epoch + 1, "loss": total_loss / len(ytr),
                    "val_pos": float(pv[yva == 1].mean()) if (yva == 1).any() else 0.0,
                    "val_neg": float(pv[yva == 0].mean()) if (yva == 0).any() else 0.0,
                }
                progress_cb(info)

    # ---- export onnx
    def export_onnx(self, out_path, model_name):
        import onnx
        from onnx import TensorProto, helper, numpy_helper

        def init(name, arr):
            return numpy_helper.from_array(np.ascontiguousarray(arr, dtype=np.float32), name)

        def layernorm(prefix, x):
            mu_out = prefix + "/mu"
            mu = helper.make_node("ReduceMean", [x], [mu_out], axes=[1])
            sub = helper.make_node("Sub", [x, mu_out], [prefix + "/sub"])
            pow_c = helper.make_node("Constant", [], [prefix + "/c2"],
                                     value=helper.make_tensor(prefix + "/c2t", TensorProto.FLOAT, [1], [2.0]))
            pw = helper.make_node("Pow", [prefix + "/sub", prefix + "/c2"], [prefix + "/pow"])
            var = helper.make_node("ReduceMean", [prefix + "/pow"], [prefix + "/var"], axes=[1])
            c_e = helper.make_node("Constant", [], [prefix + "/ce"],
                                   value=helper.make_tensor(prefix + "/cet", TensorProto.FLOAT, [1], [1e-5]))
            ad = helper.make_node("Add", [prefix + "/var", prefix + "/ce"], [prefix + "/adde"])
            sq = helper.make_node("Sqrt", [prefix + "/adde"], [prefix + "/sqrt"])
            dv = helper.make_node("Div", [prefix + "/sub", prefix + "/sqrt"], [prefix + "/div"])
            mul = helper.make_node("Mul", [prefix + "/div", prefix + "/g"], [prefix + "/mul"])
            add = helper.make_node("Add", [prefix + "/mul", prefix + "/b"], [prefix + "/out"])
            return [mu, sub, pow_c, pw, var, c_e, ad, sq, dv, mul, add], prefix + "/out"

        inits = [
            init("layer1.weight", self.W1), init("layer1.bias", self.b1),
            init("/layernorm1/g", self.g1), init("/layernorm1/b", self.beta1),
            init("blocks.0.fcn_layer.weight", self.W2), init("blocks.0.fcn_layer.bias", self.b2),
            init("/blocks.0/layer_norm/g", self.g2), init("/blocks.0/layer_norm/b", self.beta2),
            init("last_layer.weight", self.W3), init("last_layer.bias", self.b3),
        ]
        nodes = [helper.make_node("Flatten", ["input"], ["/flatten/Flatten_output_0"])]
        nodes.append(helper.make_node("Gemm", ["/flatten/Flatten_output_0", "layer1.weight", "layer1.bias"],
                                      ["/layer1/out"], transB=1))
        ln1_nodes, ln1_out = layernorm("/layernorm1", "/layer1/out")
        nodes += ln1_nodes
        nodes.append(helper.make_node("Relu", [ln1_out], ["/relu1"]))
        nodes.append(helper.make_node("Gemm", ["/relu1", "blocks.0.fcn_layer.weight", "blocks.0.fcn_layer.bias"],
                                      ["/blocks.0/out"], transB=1))
        ln2_nodes, ln2_out = layernorm("/blocks.0/layer_norm", "/blocks.0/out")
        nodes += ln2_nodes
        nodes.append(helper.make_node("Relu", [ln2_out], ["/blocks.0/relu"]))
        nodes.append(helper.make_node("Gemm", ["/blocks.0/relu", "last_layer.weight", "last_layer.bias"],
                                      ["/logits"], transB=1))
        nodes.append(helper.make_node("Sigmoid", ["/logits"], [model_name]))

        graph = helper.make_graph(
            nodes, "wakeword",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, N_FRAMES, EMBED_DIM])],
            [helper.make_tensor_value_info(model_name, TensorProto.FLOAT, [1, 1])],
            initializer=inits)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 9   # compatibilité avec l'onnxruntime installé (max 11)
        onnx.checker.check_model(model)
        onnx.save(model, out_path)


# ---------------------------------------------------------------- export tflite
def export_tflite(onnx_path, calibration_X, out_path, model_name,
                  quant_type="per-tensor", log_cb=None):
    """Convertit le .onnx exporté en .tflite (int8 + float32).

    Destiné à Home Assistant / ESPHome / ESP32 (leur runtime tflite), pas au
    runtime openwakeword (qui reste en onnx). Utilise onnx2tf (backend
    tf_converter) avec calibration sur les embeddings d'entraînement.

    - `calibration_X` : (N, 16, 96) float32 — embeddings utilisés pour la
      quantisation int8 (le plus représentatif du runtime).
    - `keep_shape_absolutely_input_names=['input']` : préserve le layout
      [1,16,96] (sans ça onnx2tf transpose en [1,96,16] et casse la compat).
    - Deux fichiers produits dans le dossier de `out_path` :
        * `<name>.tflite`          — int8 full (ESP32/ESPHome, léger) ;
        * `<name>_float32.tflite`  — float32 (Home Assistant, fidélité max).

    Nécessite tensorflow + onnx2tf + tf-keras (deps optionnelles, cf.
    requirements-train.txt). Renvoie True si au moins le .tflite int8 a été
    écrit.
    """
    def log(msg):
        if log_cb:
            log_cb(msg)

    try:
        import onnx2tf
    except ImportError:
        log("⚠ export .tflite ignoré : onnx2tf absent — installe "
            "`pip install onnx2tf[tensorflow] tf-keras` pour le format tflite.")
        return False

    import tempfile
    import shutil
    import numpy as np

    calib = np.asarray(calibration_X, dtype=np.float32)
    if calib.ndim == 3:
        # calibration sur TOUT le batch (la plage int8 doit couvrir la variété
        # des embeddings) — la limite batch=1 ne concerne que le modèle de sortie
        calib = calib[:64]

    with tempfile.TemporaryDirectory() as tmp:
        calib_path = os.path.join(tmp, "calib.npy")
        np.save(calib_path, calib)
        out_folder = os.path.join(tmp, "conv_out")
        os.makedirs(out_folder, exist_ok=True)

        log("Conversion .tflite (onnx2tf, calibration sur %d embeddings)…"
            % len(np.asarray(calibration_X)))
        onnx2tf.convert(
            input_onnx_file_path=onnx_path,
            output_folder_path=out_folder,
            output_integer_quantized_tflite=True,
            quant_type=quant_type,
            custom_input_op_name_np_data_path=[["input", calib_path, [0.0], [1.0]]],
            tflite_backend="tf_converter",
            keep_shape_absolutely_input_names=["input"],
            non_verbose=True,
            verbosity="info",
        )

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        ok = False

        # int8 full : <stem>_full_integer_quant.tflite → <name>.tflite
        int8_candidates = glob.glob(os.path.join(out_folder, "*_full_integer_quant.tflite"))
        if int8_candidates:
            shutil.copyfile(int8_candidates[0], out_path)
            log("Export .tflite int8 : " + out_path)
            ok = True
        else:
            log("⚠ export .tflite : aucun fichier full_integer_quant produit.")

        # float32 : <stem>_float32.tflite → <name>_float32.tflite
        f32_candidates = glob.glob(os.path.join(out_folder, "*_float32.tflite"))
        if f32_candidates:
            f32_path = os.path.splitext(out_path)[0] + "_float32.tflite"
            shutil.copyfile(f32_candidates[0], f32_path)
            log("Export .tflite float32 : " + f32_path)
        else:
            log("⚠ export .tflite : aucun fichier float32 produit.")

        return ok


# ---------------------------------------------------------------- pipeline
# ---- négatifs réels (enregistrements du micro, cf. first_run_train.py) ----
REAL_AUDIO_DIR = os.path.join("training_runs", "_real_negatives")
REAL_SPEECH_WAV = os.path.join(REAL_AUDIO_DIR, "real_speech_90s.wav")
REAL_AMBIENT_WAV = os.path.join(REAL_AUDIO_DIR, "real_120s.wav")


def real_negative_paths():
    """Clips négatifs issus des enregistrements réels du micro (parole lue +
    ambiance), découpage consécutif complet — aucun angle mort.

    Retourne [] si les enregistrements sont absents (cf. first_run_train.py
    pour les créer). Le découpage est mis en cache dans clips/.
    """
    clips_dir = os.path.join(REAL_AUDIO_DIR, "clips")
    files = sorted(glob.glob(os.path.join(clips_dir, "real_*.wav")))
    if files:
        return files
    sources = [p for p in (REAL_SPEECH_WAV, REAL_AMBIENT_WAV) if os.path.exists(p)]
    if not sources:
        return []
    os.makedirs(clips_dir, exist_ok=True)
    k = 0
    for path in sources:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        mono = data.mean(axis=1)
        n = int(round(len(mono) * SR / sr))
        a = np.interp(np.linspace(0, len(mono) - 1, n), np.arange(len(mono)), mono).astype(np.float32)
        seg = int(SR * 2.4)
        for st in range(0, len(a) - seg + 1, seg):
            _write_wav(os.path.join(clips_dir, "real_%03d.wav" % k), a[st:st + seg])
            k += 1
    return sorted(glob.glob(os.path.join(clips_dir, "real_*.wav")))


def train_model(phrase, pos_paths, neg_paths, out_name, out_dir="openwakeword",
                threshold=0.5, tokens=None, epochs=100, seed=0, progress_cb=None,
                log_cb=None, with_tflite=True):
    """Pipeline complet : fenêtres → embeddings → entraînement → export .onnx + JSON.

    `with_tflite=False` : ne produit que le .onnx (pas de .tflite int8/float32),
    utile si les deps tensorflow/onnx2tf ne sont pas installées.

    Renvoie le dict de validation (scores positifs/négatifs sur données tenues à l'écart).
    """
    def log(msg):
        if log_cb:
            log_cb(msg)

    noise_dir = os.path.join("training_runs", "_noise")
    _make_noise_files(noise_dir)
    augmenter = make_augmenter(noise_dir)
    set_context_audio(REAL_AMBIENT_WAV)
    real = [p for p in real_negative_paths()
            if os.path.abspath(p) not in {os.path.abspath(q) for q in neg_paths}]
    if real:
        log("Négatifs réels du micro inclus : %d clips" % len(real))
        neg_paths = list(neg_paths) + real

    log("Découpe en fenêtres d'embeddings + augmentation…")
    X, y = build_windows(pos_paths, neg_paths, augmenter, seed=seed)
    log("Jeu de données : %d fenêtres (%d positives, %d négatives)" % (len(y), int(y.sum()), int((y == 0).sum())))

    model = WakeClassifier(seed=seed)

    def cb(info):
        log("epoch %2d/%d — loss %.4f — score val: pos %.3f / neg %.3f"
            % (info["epoch"], epochs, info["loss"], info["val_pos"], info["val_neg"]))

    log("Entraînement (%d epochs)…" % epochs)
    model.fit(X, y, epochs=epochs, progress_cb=cb, seed=seed)

    onnx_path = os.path.join(out_dir, out_name + ".onnx")
    log("Export : " + onnx_path)
    model.export_onnx(onnx_path, out_name)

    # export .tflite int8/float32 (optionnel, pour HA/ESPHome/ESP32) — même pesée
    tflite_path = os.path.join(out_dir, out_name + ".tflite")
    if with_tflite:
        export_tflite(onnx_path, X, tflite_path, out_name, log_cb=log)
    else:
        log("Export .tflite ignoré (option désactivée) — seul le .onnx est produit.")

    # calibration .npy — embeddings d'entraînement (N,16,96) pour la quantisation
    # int8 : réutilisable par onnx_to_tflite.py --calib pour une conversion
    # post-création plus fidèle que la calibration synthétique.
    try:
        _calib = np.asarray(X[:64], dtype=np.float32)  # même slice que export_tflite
        _calib_path = os.path.join(out_dir, out_name + ".calib.npy")
        np.save(_calib_path, _calib)
        log("Calibration enregistrée : " + _calib_path)
    except Exception as e:
        log("⚠ calibration .npy non enregistrée : %s" % e)

    meta = {
        # libellé distinct par modèle (une même phrase entraînée deux fois
        # ne doit pas donner deux entrées identiques dans la GUI)
        "label": out_name.replace("_", " "),
        "phrase": phrase,
        "threshold": threshold,
    }
    if tokens is not None:
        meta["tokens"] = tokens
    with open(os.path.join(out_dir, out_name + ".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # validation avec le modèle exporté, via onnxruntime (chemin d'inférence réel)
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(seed)
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    n_check = min(30, len(idx_pos), len(idx_neg))
    pos_scores, neg_scores = [], []
    for i in rng.choice(idx_pos, n_check, replace=False):
        out = sess.run(None, {"input": X[i][None]})[0]
        pos_scores.append(float(out.ravel()[0]))
    for i in rng.choice(idx_neg, n_check, replace=False):
        out = sess.run(None, {"input": X[i][None]})[0]
        neg_scores.append(float(out.ravel()[0]))
    # test critique : le silence ne doit JAMAIS déclencher
    silence = make_silence_windows(min(30, len(idx_pos)), seed=seed + 1)
    silence_scores = [float(sess.run(None, {"input": w[None]})[0].ravel()[0])
                      for w in silence]
    results = {
            "onnx": onnx_path,
            "tflite": tflite_path if os.path.exists(tflite_path) else None,
            "json": os.path.join(out_dir, out_name + ".json"),
            "pos_score_mean": float(np.mean(pos_scores)),
            "neg_score_mean": float(np.mean(neg_scores)),
            "silence_score_mean": float(np.mean(silence_scores)),
            "silence_score_max": float(np.max(silence_scores)),
            "n_windows": int(len(y)),
        }
    log("Validation batch : positifs %.3f / négatifs %.3f / SILENCE %.3f (max %.3f)"
        % (results["pos_score_mean"], results["neg_score_mean"],
           results["silence_score_mean"], results["silence_score_max"]))

    # --- validation STREAMING (chemin d'inférence réel du runtime) ---
    from openwakeword.model import Model
    import inspect
    # Détection de l'API par inspection de la signature (robuste aux changements
    # de nom, sans warning de dépréciation) :
    #   récente (≥0.4.0) : wakeword_model_paths, framework onnx par défaut
    #   ancienne         : wakeword_models + inference_framework="onnx"
    if "wakeword_model_paths" in inspect.signature(Model.__init__).parameters:
        owwm = Model(wakeword_model_paths=[onnx_path])
    else:
        owwm = Model(wakeword_models=[onnx_path], inference_framework="onnx")

    def stream_scores(audio16):
        owwm.reset()
        return [max(owwm.predict(audio16[i:i + 1280]).values())
                for i in range(0, len(audio16) - 1279, 1280)]

    rng = np.random.default_rng(seed + 2)
    pos_stream = []
    for p in rng.choice(pos_paths, min(5, len(pos_paths)), replace=False):
        a = load_wav_16k(p)
        clip = np.concatenate([ambient_clip(int(SR * 1.5), rng), a, ambient_clip(int(SR * 2.0), rng)])
        a16 = (np.clip(clip, -1, 1) * 32767).astype(np.int16)
        pos_stream.append(max(stream_scores(a16)))
    results["stream_pos_max"] = float(np.max(pos_stream))
    results["stream_pos_mean"] = float(np.mean(pos_stream))
    log("Validation streaming : « %s » max %.3f (moyenne %.3f sur %d échantillons)"
        % (phrase, results["stream_pos_max"], results["stream_pos_mean"], len(pos_stream)))
    if results["stream_pos_max"] < threshold:
        log("⚠ le wake word n'atteint pas le seuil en streaming — vérifier l'alignement")
    log("Modèle prêt — auto-découvert au prochain lancement de la GUI.")
    return results


if __name__ == "__main__":
    import sys
    lang = "fr"
    args = list(sys.argv[1:])
    if "--lang" in args:
        i = args.index("--lang")
        lang = args[i + 1] if i + 1 < len(args) else "fr"
        del args[i:i + 2]
    phrase = " ".join(args) or ("yo Ève" if lang == "fr" else "hey there")
    voices = [v for v, _ in list_voices(lang)][:8]
    print("voix (%s) :" % lang, voices)
    stem = sanitize_name(phrase)
    run_dir = os.path.join("training_runs", stem)
    print("génération des positifs…")
    pos = generate_samples(phrase, voices, 8, os.path.join(run_dir, "positives"),
                           progress_cb=lambda d, t, p: print("  %d/%d" % (d, t)))
    print("génération des négatifs… (lang=%s)" % lang)
    neg = generate_negatives(30, os.path.join(run_dir, "negatives"), lang)
    res = train_model(phrase, pos, neg, stem, log_cb=print)
    print(res)
