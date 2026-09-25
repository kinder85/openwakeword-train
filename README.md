# openwakeword-train

Entraînement de **wake words custom openWakeWord**, pipeline local complet,
**autonome** (aucune dépendance à l'écosystème hapcvoice) et **multilingue**
(français + anglais minimum, extensible).

## Qu'est-ce que ça fait

1. **Positifs** : génération TTS (edge-tts) de la phrase cible, plusieurs voix
   de la langue choisie × variantes de prosodie.
2. **Négatifs** : phrases de la vie quotidienne dans la même langue + bruits +
   (si faits) des **négatifs réels** enregistrés au micro.
3. **Augmentation** (audiomentations) : bruit, pitch, vitesse, distorsion.
4. **Features** : embeddings openWakeWord (16 trames × 96 dims).
5. **Entraînement** d'un classifieur numpy répliquant exactement l'architecture
   openWakeWord (1536→32→32→1, LayerNorm+ReLU, Adam, BCE).
6. **Export** `.onnx` + JSON métadonnées dans `openwakeword/`, et **`.tflite` int8**
   (optionnel, pour Home Assistant / ESPHome / ESP32).

Le modèle exporté est **auto-découvert** par le runtime hapcvoice (`run.py` /
`gui.py`) si celui-ci est présent dans le même dossier. L'entraînement lui-même
n'en a pas besoin.

## Installation (autonome)

Dans un **environnement virtuel** (recommandé — évite de polluer le Python
système, et les dépendances `.tflite` sont lourdes) :

```bash
git clone https://github.com/kinder85/openwakeword-train.git
cd openwakeword-train
python3 -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
python -m pip install -r requirements-train.txt
```

Sans venv (installation système) :

```bash
git clone https://github.com/kinder85/openwakeword-train.git
cd openwakeword-train
python -m pip install -r requirements-train.txt
```

**Activation du venv** : `source .venv/bin/activate` n'est valable que dans la
session de terminal courante → à refaire à chaque nouvelle fenêtre si tu
veux que `python` pointe vers le venv. Alternative sans activer : appelle
directement l'interpréteur du venv, p. ex. `./.venv/bin/python trainer_gui.py`
(équivalent, et indépendant de la session).

Au premier lancement, `first_run_train.py` télécharge les modèles manquants
(vosk par langue + openWakeWord officiels).

## Utilisation

### Préparation initiale (une fois par langue)

```bash
python first_run_train.py --lang fr     # ou en, de, es
```

Installe les paquets, télécharge les modèles (vosk de la langue + openWakeWord),
puis demande **deux enregistrements réels au micro** :
- 90 s où tu lis un texte affiché (parole réelle) ;
- 120 s de silence (ambiance de la pièce).

Ces enregistrements deviennent les **négatifs réels** et le fond sonore de
contexte des positifs — c'est ce qui élimine les faux positifs en usage réel.
Ils ne sont demandés qu'une fois et réutilisés par tous les entraînements.

### Interface graphique

```bash
python trainer_gui.py
```

Trois étapes :
1. saisir la **phrase cible**, choisir la **langue** (fr/en), les voix TTS et le
   nombre d'échantillons → génération ;
2. **écouter** chaque échantillon (▶) et vérifier la prononciation — la
   transcription vosk s'affiche en contrôle (modèle de la langue choisie) ;
3. **entraîner et exporter** → `.onnx` + JSON dans `openwakeword/` (et, si la
   case **« produire les .tflite »** est cochée, les `.tflite` int8 + float32).

Une **4e étape** permet de convertir un `.onnx` existant en `.tflite`
(int8 + float32) sans relancer l'entraînement : choisis le fichier `.onnx`
(et éventuellement un `.npy` d'embeddings en calibration), puis « Convertir en
.tflite ».

Deux réglages de langue **indépendants** :
- **langue d'interface** (affichage de l'UI : fr/en) ;
- **langue d'entraînement** (contenus cible : négatifs, voix, texte à lire,
  transcription vosk).

Une case à cocher **« produire les .tflite »** (étape 3) active/désactive la
génération des `.tflite` (int8 + float32) — décoche-la si les dépendances
tensorflow/onnx2tf ne sont pas installées (le `.onnx` seul suffit au runtime
openwakeword/hapcvoice).

Les réglages sont mémorisés dans `trainer_settings.json` (local, ignoré par git).

### Ligne de commande (headless)

```bash
python trainer.py "hey there" --lang en     # entraîne un wake word anglais
```

### Conversion .onnx → .tflite (post-création)

Si tu as déjà un `.onnx` (entraîné plus tôt, ou reçu) et que tu veux les
`.tflite` sans relancer l'entraînement :

```bash
python onnx_to_tflite.py openwakeword/mon_wake.onnx
```

Produit `<mon_wake>.tflite` (int8) + `<mon_wake>_float32.tflite` dans le
dossier du `.onnx` (ou `--out-dir`). Pour une meilleure fidélité int8, fournis
les embeddings d'entraînement en calibration :

```bash
python onnx_to_tflite.py mon_wake.onnx --calib embeddings.npy
```

Sans `--calib`, une calibration synthétique aléatoire est utilisée (moins
fidèle). Depuis l'entraînement, le fichier `<mon_wake>.calib.npy` est
**détecté automatiquement** s'il est présent à côté du `.onnx`. Nécessite les
deps `.tflite` (tensorflow + onnx2tf).

## Modèles vosk par langue

| Langue | Modèle | Taille |
|---|---|---|
| fr | `vosk-model-small-fr-0.22` | ~40 Mo |
| en | `vosk-model-small-en-us-0.15` | ~40 Mo |
| de | `vosk-model-small-de-0.15` | ~45 Mo |
| es | `vosk-model-small-es-0.42` | ~39 Mo |

Téléchargés automatiquement dans `model/<locale>/` par `first_run_train.py`.

## Sortie

Chaque entraînement produit dans `openwakeword/` :
- `<nom>.onnx` — le modèle (runtime openwakeword / hapcvoice) ;
- `<nom>.tflite` — le modèle quantisé **int8** (ESP32 / ESPHome, léger) ;
- `<nom>_float32.tflite` — le modèle **float32** (Home Assistant, fidélité max) ;
- `<nom>.calib.npy` — les embeddings d'entraînement (N,16,96) servant de
  **calibration int8** (réutilisés automatiquement par `onnx_to_tflite.py`) ;
- `<nom>.json` — métadonnées (`label`, `phrase`, `threshold`).

## Format `.tflite` (Home Assistant / ESPHome / ESP32)

Le `.tflite` est généré automatiquement en fin d'entraînement, en **deux
variantes** via `onnx2tf` (calibration sur les embeddings d'entraînement,
layout `[1, 16, 96]`) :
- **int8 full** (`<nom>.tflite`) — entrée/sortie int8, ~3.7x plus petit,
  pour ESP32 / ESPHome (microcontrôleur à ressources limitées) ;
- **float32** (`<nom>_float32.tflite`) — fidélité maximale, pour Home
  Assistant (RPi / PC).

Le runtime openwakeword/hapcvoice, lui, reste en `.onnx`.

**Dépendances optionnelles** (lourdes, ~600 Mo) — nécessaires uniquement pour
le format `.tflite` :

```bash
python -m pip install "onnx2tf[tensorflow]" tf-keras
```

Sans ces paquets, l'entraînement produit le `.onnx` seul et ignore le `.tflite`
(un message le signale dans le journal).

## Structure

```
trainer.py            pipeline complet (génération → embeddings → entraînement → onnx)
trainer_gui.py        interface graphique bilingue (fr/en)
onnx_to_tflite.py     conversion .onnx → .tflite (int8 + float32) post-création
first_run.py          téléchargement des modèles absents (vosk par langue + openWakeWord)
first_run_train.py    first run : paquets + modèles + enregistrements réels + découpage
vosk_helper.py        lookup du modèle vosk par langue + transcription
requirements-train.txt  toutes les dépendances (installation seule)
model/                modèles vosk par langue (ignoré par git, téléchargés)
openwakeword/         modèles wake word exportés (.onnx + .tflite int8/float32 + JSON)
training_runs/        données d'entraînement locales (ignoré par git)
```

## Notes

- **Auto-découverte**: le `.onnx` produit est ramassé par `run.py`/`gui.py` du
  runtime hapcvoice **si présent dans le même dossier**. Ce dépôt est un simple
  outil d'entraînement ; il ne contient pas le runtime.
- **Réseau requis** : au moment de la génération TTS (edge-tts) et au 1er
  téléchargement des modèles vosk. L'entraînement lui-même tourne en local.
- Ajouter une langue = ajouter une entrée dans `NEGATIVE_SENTENCES`
  (`trainer.py`), `TEXTE_A_LIRE` (`first_run_train.py`), `VOSK_MODELS`
  (`first_run.py`) et éventuellement `UI_STRINGS` (`trainer_gui.py`).
