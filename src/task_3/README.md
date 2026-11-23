# Task 3: Toxic Span Extraction dan Censorship

Teks normal dari Task 1 -> tag BIO per kata -> masking kata asli pemain. Hanya dijalankan jika Task 2 memberi trigger.

```
raw:    u r 1d1ot lol
norm:   you | are | idiot | lol
tag:    O     O     B-TOXIC O
hasil:  u r ***** lol
```

## Desain

- Data latih: SemEval 2021 Task 5 Toxic Spans (`data/raw/semeval2021/`), offset karakter -> BIO per subword.
- Model: `bert-base-uncased` + CRF (`BertCRF`). Pembanding: `BertLinear` dan lexicon baseline.
- Post-process: tag per subword -> per kata utuh.
- Smart masking (`censor.py:align_and_mask`): tag per kata normal dipetakan ke slot raw lewat tokenisasi Task 1. Slot yang punya tag toksik dimasking seluruhnya. Jika jumlah slot tidak cocok, masking memakai slot raw (`alignment_valid` false).

## Menjalankan

Dari root repo.

```bash
uv run python src/task_3/train.py --model crf          # crf, linear, lexicon, all
uv run python src/task_3/evaluate.py                   # tsd_trial dan tsd_test
uv run python src/task_3/predict.py --raw-text "u r 1d1ot lol" --delim-norm "you | are | idiot | lol"
```

Checkpoint di `outputs/task_3/runs/baseline/bert_crf.pt` (tidak di git), hasil evaluasi di `outputs/task_3/results/`.

## API

```python
from predict import ToxicSpanCensor   # dari src/task_3
censor = ToxicSpanCensor("outputs/task_3/runs/baseline/bert_crf.pt")
censor.censor("u r 1d1ot lol", "you | are | idiot | lol", is_toxic_trigger=True)
# censored_text, masked_slot_indices, predicted_bio_tags, alignment_valid
```

## Notebook

`notebook/`: EDA, data prep, baseline, post-process.
