# Labeling Log Task 1

| Batch | Tanggal | Guideline | Jenis | Input | Jumlah | Output | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TYPE-001 | 2026-11-03 | v1 | per tipe | todo_llm_types | 344 tipe, 120 diubah | type_labels/llm_types.tsv | QC human |
| TYPE-002 | 2026-11-04 | v2 | per tipe | todo_llm_types | 1 tipe (`n i k i t a`) | type_labels/llm_types.tsv | QC human |
| OBF-001 | 2026-11-04 | v2 | lexicon | lexicon T CONDA, entri berangka/simbol | 50 bentuk | resources/obfuscation_lexicon.tsv | QC human |
| SLOT-00 s/d SLOT-11 | 2026-11-04 | v2 | per kemunculan | todo_llm_slots, source llm_slot | 8760 slot (7036 tipe), 3574 diubah | slot_labels/llm_slots.tsv | dinilai lewat gold |
| CTX-001 | 2026-11-04 | v2 | per kemunculan | todo_llm_slots, source context (`d`) | 79 slot, 6 jadi `the` | slot_labels/llm_slots.tsv | dinilai lewat gold |
| OVR-001 | 2026-11-04 | v2 | per baris | reports/failures.tsv | 10 baris, salah positif lexicon T | line_labels/line_overrides.tsv | QC human |

## Catatan

- TYPE-001 dicek ulang dengan v2.
- SLOT: label per tipe, maks 2 baris konteks, diterapkan ke semua kemunculan tipe itu.
- CTX: tipe di `context_types.txt`, label per kemunculan.
- `llm_slots.tsv` tanpa QC per baris.
- `f uck` (2 token) tidak tergabung jadi satu slot.
