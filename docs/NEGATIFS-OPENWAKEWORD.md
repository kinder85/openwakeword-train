# Négatifs génériques openWakeWord (ACAV100M)

Doc d'exploitation. Ajouté le 13/10/2026, après recherche sur la méthode
officielle d'openWakeWord.

## 1. Le problème que ça règle

Un modèle de wake word se juge sur deux choses : **rappeler** la phrase cible, et
**ne pas** se déclencher sur tout le reste. Un entraînement qui ne dispose que de
quelques dizaines/centaines de négatifs (phrases TTS + sons réels) n'apprend pas la
bonne frontière : il confond des sons proches (« blablabla ») avec la phrase et
déclenche sur à peu près toute parole.

La méthode officielle utilise **des milliers d'heures** d'audio réel en négatif. On
ne peut pas collecter ça localement — mais l'auteur d'openWakeWord a **publié le
résultat déjà converti** en features, au format exact d'entrée du modèle.

**C'est le levier n°1 contre les faux positifs.**

## 2. Ce qu'on télécharge

Source : dataset HuggingFace [`davidscripka/openwakeword_features`](https://huggingface.co/datasets/davidscripka/openwakeword_features).

| Fichier local | Contenu | Format | Taille |
|---|---|---|---|
| `acav100m_200k.npy` | 200 000 fenêtres d'ACAV100M (~71 h d'audio réel) | `(200000, 16, 96)` float16 | ~614 Mo |
| `validation_set_features.npy` | Jeu de validation faux positifs (~11 h) | `(481345, 96)` float32 — **flux plat** | ~185 Mo |

Le fichier ACAV100M **complet** pèse 17,28 Go (5 625 000 fenêtres, ~2000 h). On n'en
prend qu'une **tête** : le tableau est en ordre C → requête **HTTP Range** sur le
début du fichier, puis réécriture de l'en-tête `.npy` pour la forme tronquée.

> « 16bit » dans le nom amont = **float16**, pas int16 : aucune mise à l'échelle
> n'est nécessaire (plage ≈ −45 … +55, identique aux embeddings du projet).

## 3. Où ils sont stockés

Dossier défini par la variable d'environnement **`OWW_FEATURES_DIR`**
(défaut : `./openwakeword_features`, à côté du dépôt).

```
openwakeword_features/
├── acav100m_200k.npy              614 Mo
└── validation_set_features.npy    185 Mo
```

`trainer.py` les lit via :

```python
OWW_FEATURES_DIR   = os.environ.get("OWW_FEATURES_DIR", "openwakeword_features")
OWW_ACAV_NPY       = .../acav100m_200k.npy
OWW_VALIDATION_NPY = .../validation_set_features.npy
```

## 4. Comment les récupérer

### a) Au first run (recommandé, une fois)

```bash
python first_run_train.py --lang fr
```

L'étape `[4/5]` propose le téléchargement :

| Option | Effet |
|---|---|
| *(question `[O/n]`)* | télécharge 200 000 fenêtres + le jeu de validation |
| `--oww` | télécharge sans demander |
| `--no-oww` | refuse |
| `--oww-rows 100000` | ajuste la quantité |

### b) Script autonome (re-téléchargement manuel)

```bash
python telecharger_negatifs_openwakeword.py
```

| Option | Effet |
|---|---|
| *(aucune)* | 200 000 fenêtres + validation, dans `OWW_FEATURES_DIR` |
| `--rows 100000` | échantillon plus léger (~307 Mo) |
| `--val-only` | seulement le jeu de validation |
| `--dest /chemin` | autre dossier |
| `--force` | re-télécharge même si présent |

Utile après un `rm -rf` du dossier, ou pour choisir une autre quantité. Le script
est **idempotent** (ne refait rien si les fichiers sont là), affiche la progression,
**relit** les fichiers en fin de course, et **refuse** le téléchargement complet si
le serveur ignorait la requête Range.

### c) Depuis la GUI (`trainer_gui.py`)

Un panneau **« Négatifs génériques openWakeWord »** (juste sous l'étape 3) affiche
l'état (`présents / incomplets / absents`), permet de **vérifier**, et de
**télécharger** (avec choix du nombre de fenêtres) en tâche de fond.

## 5. Comment ils sont utilisés à l'entraînement

Automatique :

- `charger_features_oww()` — lecture **mmap**, tirage de `n` fenêtres, cast float32.
- `build_windows(oww_neg_features=...)` — bloc concaténé d'un coup.
- **Validation non biaisée** : ces fenêtres portent un id de groupe
  `≥ GROUP_BASE_OWW` et restent **toujours en entraînement** (sinon la validation
  serait noyée dessous et l'early stopping optimiserait à l'envers).
- `train_model(n_neg_oww=...)` — `None` = tout, `0` = désactive.
- **Validation faux positifs** sur le jeu ~11 h : score max/moyen + ordre de grandeur
  en faux positifs/heure, et **seuil suggéré**.

### Résultats mesurés (200 000 négatifs, seuil 0,5)

| Indicateur | Avant | Après |
|---|---|---|
| Négatifs streaming (max) | 0,551 | **0,048** (0/8 clips) |
| Silence (max) | 0,009 | **0,009** |
| Positifs ≥ 0,7 | — | **72/72** (max 0,978) |
| Pic sur le jeu 11 h | — | 0,813 → **seuil ~0,85** conseillé |

## 6. Régler le seuil

Un **pic résiduel ~0,81** subsiste sur le jeu de validation (parole réelle difficile)
alors que la cible monte à **0,978** : un seuil de détection autour de **0,85**
sépare proprement les deux.

## 7. Dépannage

| Symptôme | Cause / remède |
|---|---|
| `ModuleNotFoundError: numpy` | utiliser le venv du projet (`.venv/bin/python …`) |
| `le serveur a ignoré la requête Range` | CDN/proxy atypique → réessayer |
| `téléchargement incomplet` | réseau coupé → relancer (idempotent) |
| Entraînement lent | réduire l'échantillon (`--rows`, ou `n_neg_oww`) |
| « Negatifs generiques openWakeWord absents » dans le log | dossier supprimé → relancer le script ou le first run |

## 8. Références

- Dataset : <https://huggingface.co/datasets/davidscripka/openwakeword_features>
- Amont : <https://github.com/dscripka/openWakeWord>
