# Pipeline

Gabungan Task 1 -> Task 2 -> Task 3. Kata yang tidak disensor tetap dalam ejaan asli pemain.

```
chat mentah -> T1 normalisasi per slot -> T2 p_toxic + warning -> T3 tag BIO per kata (jika trigger) -> masking slot asli
```

## Model

Bobot tidak disimpan di git. Taruh di lokasi berikut:

```
outputs/task_1/runs/a_byt5-small_s2026/checkpoints/best/   ByT5 Task 1
outputs/task_2/runs/e2_mix/                                 DistilRoBERTa Task 2 (isi task2-e2-mix-v1.0.zip)
outputs/task_3/runs/baseline/bert_crf.pt                    BERT-CRF Task 3
models/pretrained/bert-base-uncased/                        config + tokenizer BERT
```

Config dan tokenizer BERT (bobot encoder diambil dari `bert_crf.pt`):

```bash
uv run hf download google-bert/bert-base-uncased config.json tokenizer.json tokenizer_config.json vocab.txt \
  --local-dir models/pretrained/bert-base-uncased
```

Run yang dipakai diatur di `settings.py`.

## Demo

```bash
uv run python src/pipeline/demo.py                                  # interaktif
uv run python src/pipeline/demo.py --text "u r 1d1ot lol" "gg wp"   # sekali jalan
uv run python src/pipeline/demo.py --mode strict --device cpu
```

Perintah mode interaktif: `/balanced`, `/strict`, `/detail`, `/quit`.

## Evaluasi

Dijalankan dari `evaluation/evaluation.ipynb`: warning, sensor, kata tersamar, perbandingan varian (ablation), sumber kesalahan, dan waktu proses di CONDA bench, gold CONDA, dan NoisyHate. Flag `RUN_MODELS` menjalankan ulang semua model (sekitar 4 menit di GPU). Hasil di `outputs/pipeline/`.

## Struktur

| File | Isi |
| --- | --- |
| `settings.py` | path dan run final tiap task |
| `loader.py` | import modul task yang bernama sama tanpa bentrok |
| `moderator.py` | kelas `Moderator`: chat -> hasil per tahap |
| `demo.py` | demo CLI |
| `evaluation/sets.py` | data uji dan label |
| `evaluation/scores.py` | metrik, bootstrap SE, uji beda |
| `evaluation/runner.py` | prediksi per varian, metrik, waktu proses |
| `evaluation/analysis.py` | atribusi kesalahan per tahap |
