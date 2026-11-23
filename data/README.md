# Data

```
data/
  raw/           dataset asli, tidak disimpan di git
  processed/     hasil preprocessing per task
  annotations/   label manual Task 1 (tidak bisa dibuat ulang)
```

## Unduh Data Raw

Jalankan dari root repo.

```bash
mkdir -p data/raw

# MultiLexNorm++ (en), normdict, Wortschatz
uv run hf download hadung1802/mlnorm-resources --repo-type dataset --local-dir tmp_mlnorm \
  multilexnorm++/en/train.norm multilexnorm++/en/dev.norm multilexnorm++/en/test.norm \
  normdict/en.normdict.json wortschatz/en/eng_news_2025_1M-words.txt README.md
mkdir -p data/raw/multilexnorm
mv tmp_mlnorm/multilexnorm++/en tmp_mlnorm/normdict tmp_mlnorm/wortschatz tmp_mlnorm/README.md data/raw/multilexnorm/
rm -rf tmp_mlnorm

# CONDA
git clone --depth 1 https://github.com/usydnlp/CONDA.git data/raw/conda
rm -rf data/raw/conda/.git

# NoisyHate
uv run hf download NoisyHate/Noisy_Hate_Data --repo-type dataset --local-dir data/raw/noisyhate \
  "toxic detection model test set with perturbations.csv" README.md
rm -rf data/raw/noisyhate/.cache
```

SemEval 2021 Toxic Spans (`data/raw/semeval2021/`, dari [toxic_spans](https://github.com/ipavlopoulos/toxic_spans)) sudah ikut di git.

Jigsaw (hanya untuk melatih ulang Task 2): unduh `train.csv`, `test.csv`, `test_labels.csv` dari kompetisi Kaggle `jigsaw-toxic-comment-classification-challenge`, taruh di `data/raw/jigsaw/`.

Struktur hasil:

```
data/raw/
  multilexnorm/   README.md, en/{train,dev,test}.norm, normdict/en.normdict.json, wortschatz/en/eng_news_2025_1M-words.txt
  conda/          README.md, data/CONDA_{train,valid,test}.csv, resources/lexicon_refined_1209.csv
  noisyhate/      README.md, toxic detection model test set with perturbations.csv
  semeval2021/    README.md, data/tsd_{train,trial,test}.csv, baselines/, evaluation/
  jigsaw/         train.csv, test.csv, test_labels.csv
```

Lisensi dan sitasi mengikuti README di tiap folder sumber.

## Cek Isi

`md5sum <file>`:

```
fbf24e9d18e49c8932b5f9f8bcdde30d  data/raw/multilexnorm/en/train.norm
fac54838a5e0f66aaeff7e5730145ac5  data/raw/multilexnorm/en/dev.norm
5338484b3b089e66509fc08c6065a894  data/raw/multilexnorm/en/test.norm
565a53bb90a8af378c240c9d7a136189  data/raw/multilexnorm/normdict/en.normdict.json
d722ecfd2a37f815effcce6f4a4a9a70  data/raw/multilexnorm/wortschatz/en/eng_news_2025_1M-words.txt
1d3ef6527d17598c5afc69bfea20088e  data/raw/conda/data/CONDA_train.csv
b1033be386cae2450d136f270222a487  data/raw/conda/data/CONDA_valid.csv
bd4fbfa98daa5f376943ff441742576c  data/raw/conda/data/CONDA_test.csv
b58189efd9b0a344cd5e10637df8b372  data/raw/conda/resources/lexicon_refined_1209.csv
1ea561a8e5b139b0bea743aa25b2c8b7  data/raw/noisyhate/toxic detection model test set with perturbations.csv
5973ec67ffa7cba223bcfb618a4a136d  data/raw/semeval2021/data/tsd_train.csv
1de9c0631c1aba2b3f25ae97ad9d7e2c  data/raw/semeval2021/data/tsd_trial.csv
e055fb79689e84c21529ae9c518481fe  data/raw/semeval2021/data/tsd_test.csv
```

## Simpan di Luar Repo

Data asli bisa disimpan di luar repo, `data/raw/<dataset>` cukup symlink. Git tidak menulis menembus symlink, jadi pull, merge, atau checkout ke commit lama tidak menghapus data asli.

```bash
RAW=~/datasets/IF5153-Chat-Moderation/raw
mkdir -p $RAW
for d in conda multilexnorm noisyhate; do
  mv data/raw/$d $RAW/$d
  ln -s $RAW/$d data/raw/$d
done
```

Clone lama yang masih punya `data/raw/{conda,multilexnorm,noisyhate}` sebagai folder biasa: jalankan perintah di atas sebelum `git pull`. Jika pull dilakukan duluan, git menghapus folder itu dan data perlu diunduh ulang.

## Processed dan Annotations

| Folder | Dibuat oleh | Detail |
| --- | --- | --- |
| `processed/task_1/` | `src/task_1/prepare/` | [HOWTO Task 1](../src/task_1/prepare/HOWTO.md) |
| `annotations/task_1/` | label LLM + QC manual | [HOWTO Task 1](../src/task_1/prepare/HOWTO.md) |
| `processed/task_2/` | `src/task_2/prepare.py` | [README Task 2](../src/task_2/README.md) |

Task 3 membaca SemEval langsung dari `raw/semeval2021/`.
