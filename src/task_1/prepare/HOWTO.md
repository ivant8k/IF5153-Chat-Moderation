# Preprocessing Task 1

`data/raw` + `data/annotations/task_1` -> skrip -> `data/processed/task_1`

Python 3.12, hanya library standar. Jalankan dari root repo.

## Skrip

```bash
python src/task_1/prepare/s1_prepare.py      # raw -> mln_*.tsv, conda_lines.tsv, noisyhate_test.tsv
python src/task_1/prepare/s2_candidates.py   # corong lapis 1-5 -> types.tsv, todo_llm_*.tsv, review_mln.tsv
```

Hasil selalu sama tiap dijalankan ulang. Skrip tidak pernah menimpa isian di `data/annotations/`.

## Langkah Manual

| Langkah | Oleh | File di `data/annotations/task_1/` |
| --- | --- | --- |
| Label lapis 6 (per tipe) | LLM | `llm_types.tsv` |
| Label lapis 7 (per slot) | LLM | `llm_slots.tsv` |
| Label bentuk tersamar lexicon | LLM | `obfuscation_lexicon.tsv` |
| QC | user | `review_mln.tsv`, `llm_types.tsv`, `obfuscation_lexicon.tsv` |

Acuan semua label: `guideline.md`. Riwayat batch: `labeling_log.md`.

## Cara QC

1. Buka file di VSCode, jangan Excel (teks berawalan `=` akan rusak).
2. Baris salah -> ubah kolom `final`.
3. Isi `checked` dengan `1`. Kolom `note` opsional.
4. Jumlah kata boleh berubah, tapi jangan menulis `|` atau tab.

## Aturan Berubah

Naikkan versi di `guideline.md`, catat di `labeling_log.md`, lalu label yang terdampak dikerjakan ulang.

## Belum Ada

`s4_build.py` (gabung label + validasi), `s5_gold_sample.py`, `s6_augment.py`.
