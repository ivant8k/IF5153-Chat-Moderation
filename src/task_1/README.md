# Task 1: Lexical Normalization

Chat mentah -> teks normal per slot, dengan delimiter ` | `. Jumlah slot output sama dengan input, sehingga tiap kata normal bisa dipetakan kembali ke kata asli untuk masking.

```
raw:   u r 1d1ot lol
src:   u | r | 1d1ot | lol
norm:  you | are | idiot | lol
```

## Alur

| Tahap | Folder | Isi |
| --- | --- | --- |
| Preprocessing | `prepare/` | MultiLexNorm, CONDA, NoisyHate -> format slot, label normalisasi CONDA, augmentasi penyamaran. Detail di [HOWTO](prepare/HOWTO.md) |
| Model | `normalizer/` | train, predict, evaluate, baseline, inferensi (`normalize.py`) |
| Notebook | `notebooks/` | EDA, runner train/predict/eval per tahap (dibuat `make_notebooks.py`), ringkasan |

## Eksperimen

Seleksi bertahap, 3 seed per konfigurasi, skor utama macro-ERR di set dev:

| Tahap | Yang dibandingkan | Pemenang |
| --- | --- | --- |
| A | t5-small, flan-t5-small, byt5-small, zero-shot flan | byt5-small |
| B | campuran data latih dan upsampling | MultiLexNorm + CONDA + augmentasi, 1x |
| C | format delimiter vs polos + DTW | delimiter |
| D | learning rate | 1e-3 |
| E | transfer konfigurasi dari proxy flan-t5-small ke byt5-small | config dasar |
| F | greedy vs beam, whitelist, repair slot | greedy, tanpa whitelist dan repair |

Baseline: LAI (salin input), MFR (kamus pasangan tersering), RULE (aturan).
Hasil, uji statistik, dan error analysis di `notebooks/summary.ipynb`.

## Menjalankan

Dari root repo, setelah data siap ([data/README.md](../../data/README.md)).

```bash
uv run python src/task_1/normalizer/train.py --run a_byt5-small_s2026
uv run python src/task_1/normalizer/predict.py --run a_byt5-small_s2026 --split test
uv run python src/task_1/normalizer/evaluate.py --stage final
```

Run id dan konfigurasinya ada di `normalizer/runs.py`. Untuk cloud, pakai notebook di `notebooks/train/`.

## Model Final

`a_byt5-small_s2026`, checkpoint di `outputs/task_1/runs/a_byt5-small_s2026/checkpoints/best/` (tidak di git).

```python
from normalize import Normalizer   # dari src/task_1/normalizer
Normalizer("a_byt5-small_s2026").normalize_text("u | r | 1d1ot | lol")
```

## Notebook

| Notebook | Isi |
| --- | --- |
| `notebooks/eda.ipynb` | EDA data Task 1 |
| `notebooks/summary.ipynb` | eksperimen, hasil, error analysis |
