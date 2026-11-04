# Labeling Log Task 1

Catatan setiap batch pelabelan LLM. Input diambil dari `data/processed/task_1/todo_llm_*.tsv`, output ditulis ke folder ini.

| Batch | Tanggal | Guideline | Lapis | Input | Jumlah | Output | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L6-001 | 2026-11-03 | v1 | 6 | todo_llm_types.tsv | 344 tipe: 120 diubah, 224 dibiarkan | llm_types.tsv | menunggu QC user |
| L6-002 | 2026-11-04 | v2 | 6 | todo_llm_types.tsv | 1 tipe baru (`n i k i t a`) dari aturan gabung huruf | llm_types.tsv | menunggu QC user |
| OBF-001 | 2026-11-04 | v2 | - | lexicon T CONDA, entri berisi angka/simbol | 50 bentuk tersamar | obfuscation_lexicon.tsv | menunggu QC user |
| L7-00 s/d L7-11 | 2026-11-04 | v2 | 7 | todo_llm_slots.tsv | 8760 slot (7036 tipe): 3574 slot diubah | llm_slots.tsv | dinilai lewat gold |

## Catatan

- L6-001 dicek ulang terhadap v2: penyamaran yang ada (`w0w`, `t0`, `n1`, `fkn`) sudah dinormalkan, tidak ada perubahan.
- Lapis 7 dilabel per tipe dengan melihat sampai 2 baris konteks, lalu label diterapkan ke semua slot tipe itu. Alasan: tipe lapis 7 muncul paling banyak 4 kali, konteks hampir tidak berbeda. Kolom `batch` di `llm_slots.tsv` menunjuk batch tipe.
- Lapis 7 tidak di-QC per baris. Kualitasnya diukur dari kecocokan silver dengan gold (`gold_conda.tsv`).
- Keterbatasan: penyamaran yang terpecah jadi 2 token (`f uck`) tidak bisa dipulihkan karena aturan gabung huruf butuh minimal 3 huruf tunggal.
- Kasus ragu dibiarkan sesuai guideline, contoh `eniggerma`, `cumbacket`, `mitches`.
