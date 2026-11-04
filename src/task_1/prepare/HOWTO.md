# Preprocessing Task 1

Python 3.12, library standar. Jalankan dari root repo.

## Alur

| Tahap | Skrip | Fungsi | Output |
| --- | --- | --- | --- |
| 1 | `s1_prepare.py` | raw -> format slot, split CONDA per match 80/10/10 | `mln/`, `conda/lines.tsv`, `noisyhate/` |
| 2 | `s2_candidates.py` | source label per tipe, todo LLM, review MLN | `labeling/`, `type_labels/review_mln.tsv` |
| 3 | manual | label LLM + QC human | `type_labels/`, `slot_labels/` |
| 4 | `s4_build.py` | gabung label + validasi | `conda/{train,dev,bench}.tsv`, `resources/`, `reports/` |
| 5 | `s5_gold_sample.py` | sampel gold dari bench, sekali | `line_labels/gold_conda.tsv`, `line_labels/gold_strata.json` |
| 6 | `s6_augment.py` | salinan dengan satu kata disamarkan | `augmented/` |

## Run Ulang

```bash
python src/task_1/prepare/s1_prepare.py
python src/task_1/prepare/s2_candidates.py
python src/task_1/prepare/s4_build.py
python src/task_1/prepare/s6_augment.py
```

## Struktur

```
data/annotations/task_1/
  guideline.md, labeling_log.md
  type_labels/   review_mln.tsv, llm_types.tsv
  slot_labels/   llm_slots.tsv, context_types.txt
  line_labels/   line_overrides.tsv, gold_conda.tsv, gold_strata.json
  resources/     obfuscation_lexicon.tsv, whitelist.txt (opsional)

data/processed/task_1/
  mln/           train, dev, test
  conda/         lines, train, dev, bench
  noisyhate/     test
  augmented/     train, conda_dev, conda_bench
  labeling/      types, todo_llm_types, todo_llm_slots
  resources/     whitelist.txt
  reports/       build_stats.json, aug_stats.json, failures.tsv
```

## Source Label

Kolom `source` di `labeling/types.tsv`:

| Source | Label |
| --- | --- |
| `context` | per kemunculan, `slot_labels/llm_slots.tsv` |
| `protected` | tetap |
| `mln_changed` | `type_labels/review_mln.tsv` |
| `mln_same`, `symbol_laugh`, `english` | tetap |
| `llm_type` | `type_labels/llm_types.tsv` |
| `llm_slot` | `slot_labels/llm_slots.tsv` |

Prioritas: gold_conda > line_overrides > llm_slots > label per tipe > label otomatis

## QC

- Edit di VSCode, bukan Excel.
- Ubah `final` (atau `gold`, `tgt`), isi `checked` = 1.
- Tanpa `|` dan tab. Di `gold_conda` dan `line_overrides`, jumlah `|` sama dengan `src`.
- Setelah QC: run ulang tahap 4 dan 6.

## Guideline Berubah

Naikkan versi `guideline.md`, catat di `labeling_log.md`, label ulang yang terdampak, run ulang tahap 4 dan 6.
