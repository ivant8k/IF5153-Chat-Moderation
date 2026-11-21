# Task 2: Toxicity Classification

Gerbang semantik pipeline. Input: teks hasil Task 1 (delimiter `|` boleh disertakan). Output: sinyal trigger untuk Task 3 + warning terstruktur.

```
normalized text -> DistilRoBERTa (7 output) -> trigger? --ya--> Task 3 (span + teks normalized ber-delimiter) -> smart masking
                                             \-> warnings (kategori Jigsaw) -> pesan ke pemain
```

## Desain

- **Model**: `distilroberta-base`, mean pooling, head linear 7 output sigmoid: `gate` (toksik apa pun) + 6 kategori Jigsaw (`toxic, severe_toxic, obscene, threat, insult, identity_hate`).
- **Data fusion + masked loss**: label berbeda antar dataset ditangani lewat mask pada loss.

  | Sumber | gate | kategori |
  | --- | --- | --- |
  | Jigsaw | any(6 label) | 6 label asli |
  | CONDA intent E/I (toksik) | 1 | di-mask (tidak ada label kategori) |
  | CONDA intent A/O (bukan toksik) | 0 | semua 0 |

- **Teks**: lowercase semua (output Task 1 selalu lowercase). CONDA dilatih dalam dua varian: `norm` (output normalisasi Task 1) dan `raw`.
- **Split CONDA** mengikuti Task 1 (match-level 80/10/10, seed 2026). Baris csv test CONDA tidak berlabel dan dibuang. Satu contoh = satu baris CONDA (chat dalam baris digabung dengan spasi).
- **Ambang** (`thresholds.json` per run), dipilih di dev CONDA (teks `norm`):
  - `balanced`: F1 gate maksimum (mode utama).
  - `strict`: ambang terendah dengan precision >= 0.95 (selalu >= balanced).
  - Kategori: ambang F1 maksimum per label di dev Jigsaw.
- **Warning**: hanya saat `trigger`. Kategori spesifik di atas ambang, urut probabilitas. Jika gate menyala tapi tidak ada kategori yang lolos: fallback `["toxic"]`.

## Kontrak output (`predict.py`)

```python
from predict import ToxicityClassifier
clf = ToxicityClassifier("e2_mix", mode="balanced")   # run id di outputs/task_2/runs/
clf.classify("u | r | trash")   # delimiter dihapus hanya untuk klasifikasi; boleh list
```

```json
{"text": "u | r | trash", "p_toxic": 0.93, "trigger": true,
 "labels": {"toxic": 0.9, "severe_toxic": 0.02, "obscene": 0.1, "threat": 0.01, "insult": 0.88, "identity_hate": 0.01},
 "warnings": ["insult"], "message": "Warning: your message contains insults.",
 "mode": "balanced", "threshold": 0.4}
```

Pipeline: jika `trigger` false, lewati Task 3. Jika true, Task 3 menerima teks normalized ber-delimiter dari Task 1 bersama `trigger`; gunakan field `text` dari hasil classifier yang mempertahankan teks input persis.

## Arm eksperimen

| Run | Isi |
| --- | --- |
| `b0_tfidf` | TF-IDF + Logistic Regression, data sama dengan E2 |
| `z_toxicbert` | zero-shot `unitary/toxic-bert` (label sama dengan Jigsaw) |
| `e1_jigsaw` | DistilRoBERTa, Jigsaw saja |
| `e2_mix` | DistilRoBERTa, Jigsaw + CONDA, masked loss (kandidat utama) |
| `e3_seq` | `e1_jigsaw` lalu fine-tune CONDA saja (LR kecil) |

## Set evaluasi

| Set | Isi |
| --- | --- |
| `conda_bench:{norm,raw,t1pred}` | bench Task 1; `t1pred` = prediksi ByT5 final Task 1 (end-to-end) |
| `conda_gold:{gold,t1pred}` | baris CONDA yang normalisasinya diperiksa manual; bandingkan gold dengan hasil ByT5 |
| `conda_dev:*` | dipakai kalibrasi ambang |
| `jigsaw_dev`, `jigsaw_test` | kategori (mean AUC, macro AP, macro F1) |
| `noisyhate:{perturbed,t1pred,clean}` | ketahanan terhadap obfuscation; semua diasumsikan toksik, jadi flag rate = recall |

`conda_gold` adalah sampel stratifikasi untuk menilai efek normalisasi Task 1 pada baris yang sama,
bukan sampel acak representatif; jadikan analisis diagnostik, bukan estimasi performa populasi.

Metrik gate: AP, ROC-AUC, ECE, precision/recall/F1 di kedua ambang, FPR pada intent A dan O, recall pada intent E dan I.

## Menjalankan

Dari root repo. Data mentah di `data/raw/{jigsaw,conda}` (git-ignored; lihat README root untuk CONDA, Jigsaw dari Kaggle).

```bash
python src/task_2/prepare.py                    # -> data/processed/task_2
python src/task_2/baseline.py --run b0_tfidf    # CPU
python src/task_2/evaluate.py --run b0_tfidf
python src/task_2/train.py --run e2_mix --sources jigsaw,conda   # GPU
python src/task_2/evaluate.py --run e2_mix
python src/task_2/summary.py                    # tabel banding semua run
```

`prepare.py` memerlukan keluaran prediksi Task 1 final (`a_byt5-small_s42`) untuk CONDA bench,
gold CONDA, dan NoisyHate. Jika file atau ID prediksi tidak ada, proses berhenti dengan error
agar teks gold/silver tidak salah dilaporkan sebagai prediksi Task 1.

Uji cepat di CPU: `train.py --run smoke --sources jigsaw,conda --smoke` lalu `evaluate.py --run smoke --limit 200`.

GPU: gunakan `src/task_2/notebooks/colab_runner.ipynb` (dibuat oleh `make_colab_notebook.py`).

Dependensi: `torch`, `transformers`, `scikit-learn`, `pandas`, `numpy`, `scipy`.
