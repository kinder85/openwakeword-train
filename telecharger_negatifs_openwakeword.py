#!/usr/bin/env python3
"""Télécharge les négatifs génériques openWakeWord (ACAV100M) — re-téléchargement manuel.

Extrait du dataset public `davidscripka/openwakeword_features` :
  - un SOUS-ENSEMBLE d'ACAV100M — features déjà calculées par l'embedding Google,
    ~2000 h d'audio réel varié (parole multilingue, bruit, musique) ;
  - le jeu de validation faux positifs (~11 h d'audio réel).

Le fichier ACAV100M complet pèse 17,28 Go. On ne prend que la TÊTE via une requête
HTTP Range (le tableau est en ordre C) puis on réécrit l'en-tête .npy pour la forme
tronquée : ~614 Mo seulement pour 200 000 fenêtres.

Ces fichiers sont réutilisés AUTOMATIQUEMENT par chaque entraînement
(cf. trainer.py : OWW_ACAV_NPY / OWW_VALIDATION_NPY, dossier OWW_FEATURES_DIR).

Normalement `first_run_train.py` (étape [4/5]) les propose déjà. Ce script sert à
les (re)télécharger indépendamment — par exemple après un `rm -rf` du dossier, ou
pour choisir une autre quantité.

Usage :
    python telecharger_negatifs_openwakeword.py                  # 200k fenêtres
    python telecharger_negatifs_openwakeword.py --rows 100000    # plus léger
    python telecharger_negatifs_openwakeword.py --dest /chemin   # autre dossier
    python telecharger_negatifs_openwakeword.py --force          # re-télécharge
    python telecharger_negatifs_openwakeword.py --val-only       # jeu 11 h seul

Le dossier de destination suit la variable d'environnement OWW_FEATURES_DIR
(défaut : ./openwakeword_features, à côté de ce script).
"""

import argparse
import os
import struct
import sys
import urllib.request

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET = ("https://huggingface.co/datasets/davidscripka/"
           "openwakeword_features/resolve/main")
URL_ACAV = DATASET + "/openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
URL_VALIDATION = DATASET + "/validation_set_features.npy"

ROWS_DEFAUT = 200000
ROWS_MAX = 5625000          # taille réelle du fichier (au-delà : inutile)
FRAMES, EMBED_DIM = 16, 96
ITEM_SIZE = 2               # float16
BLOC = 1024 * 1024          # 1 Mio par lecture réseau


def _dest_defaut():
    """Dossier des features : OWW_FEATURES_DIR (env) sinon ./openwakeword_features."""
    return os.path.join(SCRIPT_DIR,
                        os.environ.get("OWW_FEATURES_DIR", "openwakeword_features"))


def _entete_npy(url):
    """Offset des données d'un .npy distant, lu par une requête Range de 256 octets."""
    req = urllib.request.Request(url, headers={"Range": "bytes=0-255"})
    with urllib.request.urlopen(req, timeout=60) as r:
        head = r.read(256)
    if head[:6] != b"\x93NUMPY":
        raise RuntimeError("réponse inattendue : ce n'est pas un fichier .npy")
    maj, mino = head[6], head[7]
    if (maj, mino) == (1, 0):
        hlen = struct.unpack("<H", head[8:10])[0]
        return 10 + hlen
    hlen = struct.unpack("<I", head[8:12])[0]
    return 12 + hlen


def _telecharger(url, chemin, headers=None, label=""):
    """Télécharge en flux vers `chemin`, avec affichage de progression.

    Renvoie le nombre d'octets écrits. Échoue si le serveur ignore la requête
    Range (on éviterait sinon de tirer les 17 Go par mégarde).
    """
    requete_range = headers and "Range" in headers
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=300) as r, open(chemin, "wb") as f:
        if requete_range and getattr(r, "status", 206) == 200:
            f.close()
            os.remove(chemin)
            raise RuntimeError("le serveur a ignoré la requête Range "
                               "(téléchargement complet refusé)")
        total = int(r.headers.get("Content-Length") or 0)
        fait = 0
        while True:
            bloc = r.read(BLOC)
            if not bloc:
                break
            f.write(bloc)
            fait += len(bloc)
            if total:
                print("\r  %-12s %6.1f%%  (%6.0f / %6.0f Mo)" % (
                    label, 100.0 * fait / total, fait / 1e6, total / 1e6),
                    end="", flush=True)
    print()
    return fait


def telecharger_acav(dest, rows, force=False):
    """Récupère `rows` fenêtres d'ACAV100M → <dest>/acav100m_200k.npy."""
    import numpy as np
    rows = max(1, min(int(rows), ROWS_MAX))
    chemin = os.path.join(dest, "acav100m_200k.npy")
    if os.path.exists(chemin) and not force:
        print("  ACAV100M : déjà présent (%s)" % chemin)
        return chemin

    off = _entete_npy(URL_ACAV)
    taille = off + rows * FRAMES * EMBED_DIM * ITEM_SIZE
    print("  ACAV100M : %d fenêtres (~%.0f h) — ~%.0f Mo à télécharger"
          % (rows, rows * FRAMES / 12.5 / 3600.0, taille / 1e6))
    tmp = chemin + ".part"
    _telecharger(URL_ACAV, tmp, {"Range": "bytes=0-%d" % (taille - 1)}, "ACAV100M")

    with open(tmp, "rb") as f:
        f.seek(off)
        buf = f.read(rows * FRAMES * EMBED_DIM * ITEM_SIZE)
    if len(buf) != rows * FRAMES * EMBED_DIM * ITEM_SIZE:
        os.remove(tmp)
        raise RuntimeError("téléchargement incomplet (%d octets utiles reçus)" % len(buf))
    arret = np.frombuffer(buf, dtype="<f2").reshape(rows, FRAMES, EMBED_DIM)
    # réécrit un .npy VALIDE pour la forme tronquée (l'original annonce 5 625 000 lignes)
    with open(chemin, "wb") as g:
        np.lib.format.write_array(g, arret, allow_pickle=False)
    os.remove(tmp)
    print("  ✓ écrit %s (%d, %d, %d) float16" % (chemin, rows, FRAMES, EMBED_DIM))
    return chemin


def telecharger_validation(dest, force=False):
    """Récupère le jeu de validation faux positifs (~11 h, flux plat)."""
    chemin = os.path.join(dest, "validation_set_features.npy")
    if os.path.exists(chemin) and not force:
        print("  Jeu de validation : déjà présent (%s)" % chemin)
        return chemin
    print("  Jeu de validation faux positifs (~11 h) — ~185 Mo")
    _telecharger(URL_VALIDATION, chemin, None, "validation")
    print("  ✓ écrit %s" % chemin)
    return chemin


def verifier(dest):
    """Relit les deux fichiers et affiche forme/dtype. Renvoie True si sains."""
    try:
        import numpy as np
    except ImportError:
        print("⚠ numpy absent — vérification sautée (pip install numpy)")
        return True
    ok = True
    for nom in ("acav100m_200k.npy", "validation_set_features.npy"):
        chemin = os.path.join(dest, nom)
        if not os.path.exists(chemin):
            print("  ✗ manquant : %s" % chemin)
            ok = False
            continue
        try:
            a = np.load(chemin, mmap_mode="r")
        except Exception as e:
            print("  ✗ illisible : %s (%s)" % (chemin, e))
            ok = False
            continue
        print("  %-32s %-18s %s  (%.0f Mo)" % (nom, str(a.shape), a.dtype,
                                              os.path.getsize(chemin) / 1e6))
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Télécharge les négatifs génériques openWakeWord (ACAV100M) "
                    "pour l'entraînement des wake words.")
    ap.add_argument("--dest", default=_dest_defaut(),
                    help="dossier de destination (défaut : OWW_FEATURES_DIR ou "
                         "./openwakeword_features)")
    ap.add_argument("--rows", type=int, default=ROWS_DEFAUT,
                    help="nombre de fenêtres ACAV100M (défaut %d ; max %d)"
                         % (ROWS_DEFAUT, ROWS_MAX))
    ap.add_argument("--force", action="store_true",
                    help="re-télécharger même si les fichiers sont déjà présents")
    ap.add_argument("--val-only", action="store_true",
                    help="ne télécharger que le jeu de validation faux positifs")
    args = ap.parse_args(argv)

    os.makedirs(args.dest, exist_ok=True)
    print("=" * 70)
    print("Négatifs génériques openWakeWord")
    print("  source  : davidscripka/openwakeword_features")
    print("  dossier : %s" % args.dest)
    print("=" * 70)

    try:
        if not args.val_only:
            telecharger_acav(args.dest, args.rows, force=args.force)
        telecharger_validation(args.dest, force=args.force)
    except Exception as e:
        print("\n✗ échec : %s" % e)
        print("  (réessaie plus tard ; l'entraînement reste possible sans, mais avec "
              "plus de faux positifs)")
        return 1

    print("\nVérification :")
    ok = verifier(args.dest)
    print("\n✓ Terminé%s — ces fichiers seront utilisés automatiquement par chaque "
          "entraînement." % ("" if ok else " (avec des manques signalés ci-dessus)"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
