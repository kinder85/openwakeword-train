#!/usr/bin/env python3
"""Convertit un .onnx de wake word en .tflite (int8 + float32) — post-création.

Usage :
    python onnx_to_tflite.py openwakeword/mon_wake.onnx
    python onnx_to_tflite.py mon_wake.onnx --calib embeddings.npy
    python onnx_to_tflite.py mon_wake.onnx --out-dir openwakeword

Produit dans le dossier de sortie (défaut : dossier du .onnx) :
    <nom>.tflite          — int8 full (ESP32 / ESPHome, léger)
    <nom>_float32.tflite  — float32 (Home Assistant, fidélité max)

Calibration int8 :
    - Si `--calib` pointe vers un .npy d'embeddings d'entraînement (N,16,96),
      il est utilisé (le plus représentatif du runtime).
    - Sinon, une calibration synthétique aléatoire en [0,1] est générée
      (moins fidèle — préfère fournir les embeddings réels si tu les as).

Nécessite tensorflow + onnx2tf + tf-keras (deps optionnelles, cf.
requirements-train.txt). Le .onnx seul suffit au runtime openwakeword/hapcvoice.
"""

import argparse
import glob
import os
import shutil
import sys
import tempfile

import numpy as np


def log(msg):
    print(msg, flush=True)


def detect_input(onnx_path):
    """Nom + shape de l'entrée du modèle onnx (défaut : input, [1,16,96])."""
    import onnx
    model = onnx.load(onnx_path)
    if not model.graph.input:
        raise SystemExit("✗ modèle onnx sans entrée.")
    inp = model.graph.input[0]
    name = inp.name
    shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    return name, shape


def make_calibration(calib_path, input_shape):
    """Calibration (N, ...) float32 en [0,1] — embeddings réels si fournis,
    sinon synthétiques aléatoires (N=64)."""
    if calib_path:
        if not os.path.exists(calib_path):
            raise SystemExit("✗ fichier de calibration introuvable : %s" % calib_path)
        calib = np.load(calib_path)
        calib = np.asarray(calib, dtype=np.float32)
        if calib.ndim == 3:
            calib = calib[:64]  # calibration sur tout le batch (plage int8)
        log("Calibration : %s (%d échantillons)" % (calib_path, len(calib)))
        return calib
    # synthétique : shape [1,16,96] → [64,16,96], valeurs en [0,1]
    n = 64
    shape = [n] + [s if s and s > 1 else 1 for s in input_shape[1:]]
    calib = np.random.default_rng(0).random(shape, dtype=np.float32)
    log("Calibration : synthétique aléatoire en [0,1] (%s) — fournis --calib "
        "pour une meilleure fidélité int8." % (shape,))
    return calib


def convert(onnx_path, out_dir, calib_path=None, quant_type="per-tensor"):
    input_name, input_shape = detect_input(onnx_path)
    log("Entrée onnx : %s %s" % (input_name, input_shape))

    try:
        import onnx2tf
    except ImportError:
        log("✗ onnx2tf absent — installe `pip install onnx2tf[tensorflow] tf-keras` "
            "pour le format tflite.")
        return False

    calib = make_calibration(calib_path, input_shape)

    with tempfile.TemporaryDirectory() as tmp:
        calib_file = os.path.join(tmp, "calib.npy")
        np.save(calib_file, calib)
        out_folder = os.path.join(tmp, "conv_out")
        os.makedirs(out_folder, exist_ok=True)

        log("Conversion .tflite (onnx2tf, calibration sur %d embeddings)…" % len(calib))
        onnx2tf.convert(
            input_onnx_file_path=onnx_path,
            output_folder_path=out_folder,
            output_integer_quantized_tflite=True,
            quant_type=quant_type,
            custom_input_op_name_np_data_path=[[input_name, calib_file, [0.0], [1.0]]],
            tflite_backend="tf_converter",
            keep_shape_absolutely_input_names=[input_name],
            non_verbose=True,
            verbosity="info",
        )

        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(onnx_path))[0]
        ok = False

        # int8 full : <stem>_full_integer_quant.tflite → <stem>.tflite
        int8_candidates = glob.glob(os.path.join(out_folder, "*_full_integer_quant.tflite"))
        if int8_candidates:
            out_int8 = os.path.join(out_dir, stem + ".tflite")
            shutil.copyfile(int8_candidates[0], out_int8)
            log("Export .tflite int8 : " + out_int8)
            ok = True
        else:
            log("⚠ aucun fichier full_integer_quant produit.")

        # float32 : <stem>_float32.tflite → <stem>_float32.tflite
        f32_candidates = glob.glob(os.path.join(out_folder, "*_float32.tflite"))
        if f32_candidates:
            out_f32 = os.path.join(out_dir, stem + "_float32.tflite")
            shutil.copyfile(f32_candidates[0], out_f32)
            log("Export .tflite float32 : " + out_f32)
        else:
            log("⚠ aucun fichier float32 produit.")

        return ok


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Convertit un .onnx de wake word en .tflite (int8 + float32).")
    parser.add_argument("onnx", help="chemin vers le modèle .onnx à convertir")
    parser.add_argument("--calib", default=None,
                        help=".npy d'embeddings d'entraînement (N,16,96) pour la "
                             "calibration int8 (sinon synthétique)")
    parser.add_argument("--out-dir", default=None,
                        help="dossier de sortie (défaut : dossier du .onnx)")
    args = parser.parse_args(argv)

    if not os.path.exists(args.onnx):
        raise SystemExit("✗ fichier onnx introuvable : %s" % args.onnx)
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.onnx))

    # auto-détection d'une calibration sœur <stem>.calib.npy (produite par
    # l'entraînement) — sinon calibration synthétique
    calib = args.calib
    if not calib:
        _sibling = os.path.join(out_dir,
                                os.path.splitext(os.path.basename(args.onnx))[0] + ".calib.npy")
        if os.path.exists(_sibling):
            calib = _sibling
            log("Calibration auto-détectée : " + _sibling)

    ok = convert(args.onnx, out_dir, calib_path=calib)
    if not ok:
        sys.exit(1)
    log("✓ Conversion terminée — .tflite dans %s" % out_dir)


if __name__ == "__main__":
    main()
