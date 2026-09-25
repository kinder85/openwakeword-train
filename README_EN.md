# openwakeword-train

Training of **custom openWakeWord wake words**, complete local pipeline,
**self-contained** (no dependency on the hapcvoice ecosystem) and
**multilingual** (French + English at least, extensible).

## What it does

1. **Positives**: TTS generation (edge-tts) of the target phrase, several
   voices of the chosen language × prosody variants.
2. **Negatives**: everyday sentences in the same language + noises + (if
   done) **real negatives** recorded with the microphone.
3. **Augmentation** (audiomentations): noise, pitch, speed, clipping.
4. **Features**: openWakeWord embeddings (16 frames × 96 dims).
5. **Training** of a numpy classifier replicating exactly the openWakeWord
   architecture (1536→32→32→1, LayerNorm+ReLU, Adam, BCE).
6. **Export** `.onnx` + JSON metadata into `openwakeword/`, and **`.tflite` int8**
   (optional, for Home Assistant / ESPHome / ESP32).

The exported model is **auto-discovered** by the hapcvoice runtime (`run.py` /
`gui.py`) if present in the same folder. Training itself does not need it.

## Installation (self-contained)

In a **virtual environment** (recommended — keeps the system Python clean,
and the `.tflite` dependencies are heavy):

```bash
git clone https://github.com/kinder85/openwakeword-train.git
cd openwakeword-train
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements-train.txt
```

Without a venv (system install):

```bash
git clone https://github.com/kinder85/openwakeword-train.git
cd openwakeword-train
python -m pip install -r requirements-train.txt
```

**Venv activation**: `source .venv/bin/activate` only lasts for the current
terminal session → repeat on every new window if you want `python` to point
at the venv. Alternative without activating: call the venv interpreter
directly, e.g. `./.venv/bin/python trainer_gui.py` (equivalent and session-
independent).

On first launch, `first_run_train.py` downloads the missing models (vosk per
language + official openWakeWord).

## Usage

### Initial setup (once per language)

```bash
python first_run_train.py --lang fr     # or en, de, es
```

Installs the packages, downloads the models (vosk of the language +
openWakeWord), then asks for **two real microphone recordings**:
- 90 s where you read an on-screen text (real speech);
- 120 s of silence (room ambience).

These recordings become the **real negatives** and the background-noise
context for the positives — this is what removes false positives in real use.
They are requested only once and reused by every training run.

### Graphical interface

```bash
python trainer_gui.py
```

Three steps:
1. enter the **target phrase**, choose the **language** (fr/en), the TTS
   voices and the number of samples → generation;
2. **listen** to each sample (▶) and check the pronunciation — the vosk
   transcription is shown as a check (model of the chosen language);
3. **train and export** → `.onnx` + JSON in `openwakeword/` (and, if the
   **« produce .tflite »** box is checked, the int8 + float32 `.tflite`).

A **4th step** lets you convert an existing `.onnx` to `.tflite`
(int8 + float32) without re-running training: pick the `.onnx` file (and
optionally a `.npy` of embeddings as calibration), then « Convert to
.tflite ».

Two **independent** language settings:
- **interface language** (UI display: fr/en);
- **training language** (target content: negatives, voices, reading text,
  vosk transcription).

A **"produce .tflite"** checkbox (step 3) toggles the `.tflite` generation
(int8 + float32) — uncheck it if the tensorflow/onnx2tf dependencies are not
installed (the `.onnx` alone is enough for the openwakeword/hapcvoice runtime).

Settings are remembered in `trainer_settings.json` (local, git-ignored).

### Command line (headless)

```bash
python trainer.py "hey there" --lang en     # train an English wake word
```

### .onnx → .tflite conversion (post-training)

If you already have a `.onnx` (trained earlier, or received) and want the
`.tflite` files without re-running training:

```bash
python onnx_to_tflite.py openwakeword/my_wake.onnx
```

Produces `<my_wake>.tflite` (int8) + `<my_wake>_float32.tflite` in the
`.onnx` folder (or `--out-dir`). For better int8 fidelity, provide the
training embeddings as calibration:

```bash
python onnx_to_tflite.py my_wake.onnx --calib embeddings.npy
```

Without `--calib`, a random synthetic calibration is used (less faithful).
Since training, the `<my_wake>.calib.npy` file is **auto-detected** if present
next to the `.onnx`. Requires the `.tflite` deps (tensorflow + onnx2tf).

## vosk models by language

| Language | Model | Size |
|---|---|---|
| fr | `vosk-model-small-fr-0.22` | ~40 MB |
| en | `vosk-model-small-en-us-0.15` | ~40 MB |
| de | `vosk-model-small-de-0.15` | ~45 MB |
| es | `vosk-model-small-es-0.42` | ~39 MB |

Auto-downloaded into `model/<locale>/` by `first_run_train.py`.

## Output

Each training produces in `openwakeword/`:
- `<name>.onnx` — the model (openwakeword / hapcvoice runtime);
- `<name>.tflite` — the **int8** quantized model (ESP32 / ESPHome, light);
- `<name>_float32.tflite` — the **float32** model (Home Assistant, max fidelity);
- `<name>.calib.npy` — the training embeddings (N,16,96) used as **int8
  calibration** (auto-reused by `onnx_to_tflite.py`);
- `<name>.json` — metadata (`label`, `phrase`, `threshold`).

## `.tflite` format (Home Assistant / ESPHome / ESP32)

The `.tflite` is generated automatically at the end of training, in **two
variants** via `onnx2tf` (calibration on the training embeddings, layout
`[1, 16, 96]`):
- **full int8** (`<name>.tflite`) — int8 input/output, ~3.7x smaller, for
  ESP32 / ESPHome (resource-limited microcontroller);
- **float32** (`<name>_float32.tflite`) — maximum fidelity, for Home
  Assistant (RPi / PC).

The openwakeword/hapcvoice runtime stays on `.onnx`.

**Optional dependencies** (heavy, ~600 MB) — only needed for the `.tflite`
format:

```bash
python -m pip install "onnx2tf[tensorflow]" tf-keras
```

Without these packages, training produces the `.onnx` alone and skips the
`.tflite` (a message is logged).

## Structure

```
trainer.py            full pipeline (generation → embeddings → training → onnx)
trainer_gui.py        bilingual graphical interface (fr/en)
onnx_to_tflite.py     .onnx → .tflite conversion (int8 + float32) post-training
first_run.py          download of missing models (vosk per language + openWakeWord)
first_run_train.py    first run: packages + models + real recordings + split
vosk_helper.py        vosk model lookup per language + transcription
requirements-train.txt  all dependencies (single install)
model/                vosk models per language (git-ignored, downloaded)
openwakeword/         exported wake word models (.onnx + .tflite int8/float32 + JSON)
training_runs/        local training data (git-ignored)
```

## Notes

- **Auto-discovery**: the produced `.onnx` is picked up by `run.py`/`gui.py`
  of the hapcvoice runtime **if present in the same folder**. This repository
  is a simple training tool; it does not contain the runtime.
- **Network required**: when generating TTS (edge-tts) and on the first vosk
  model download. Training itself runs locally.
- Adding a language = add an entry in `NEGATIVE_SENTENCES`
  (`trainer.py`), `TEXTE_A_LIRE` (`first_run_train.py`), `VOSK_MODELS`
  (`first_run.py`) and optionally `UI_STRINGS` (`trainer_gui.py`).
