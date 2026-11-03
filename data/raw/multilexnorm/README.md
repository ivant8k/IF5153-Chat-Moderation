# mlnorm Resources

Resource files for the [mlnorm](https://github.com/mlnorm/mlnorm)
multilingual lexical normalization toolkit. Not included in the PyPI
package due to size.

## Contents

| Directory | Size | Description |
|-----------|------|-------------|
| [`multilexnorm++/`](multilexnorm++/) | ~13 MB | MultiLexNorm 2026 benchmark data in `.norm` format (17 languages) |
| [`normdict/`](normdict/) | ~276 MB | Unified normalization dictionaries (17 languages) |
| [`dict/`](dict/) | ~77 MB | LLM-generated word definition files (JSONL) |
| [`hunspell/`](hunspell/) | ~42 MB | Hunspell spell-checker dictionaries (14 languages) |
| [`wortschatz/`](wortschatz/) | ~1.9 GB | Corpus word frequency lists from the Leipzig Corpora Collection (15 languages) |

## Download

```bash
# Download everything
hf download hadung1802/mlnorm-resources --repo-type dataset --local-dir resources/

# Download a single resource directory
hf download hadung1802/mlnorm-resources normdict/ --repo-type dataset --local-dir resources/
```

---

## multilexnorm++

Pre-tokenized `.norm` files for 17 language varieties from the
MultiLexNorm 2026 shared task on multilingual lexical normalization
of social media text.

**Original source (CSV format)**:
[weerayut/multilexnorm2026-full-pub](https://huggingface.co/datasets/weerayut/multilexnorm2026-full-pub)

**License**: Refer to the original dataset repository for licensing
terms.

### Format

Tab-separated, two columns per line. Sentences are separated by blank
lines.

```
u	you
wanna	want to
go	go
?	?

im	i'm
gr8	great
```

- **Column 1**: Raw (original) token as it appears in the source text.
- **Column 2**: Normalized (standard) form.
- A normalized form containing a space (e.g. `wanna` → `want to`)
  encodes a **split** (one token normalizes to multiple words).
- An empty normalized form encodes a **merge tail** (this token is
  absorbed into the preceding token's normalization).

### Directory structure

```
multilexnorm++/{lang}/train.norm
multilexnorm++/{lang}/test.norm
multilexnorm++/{lang}/dev.norm      (where available)
```

### Languages and splits

| Language | Code | Splits |
|----------|------|--------|
| Danish | `da` | train, test |
| German | `de` | train, dev, test |
| English | `en` | train, dev, test |
| Spanish | `es` | train, test |
| Croatian | `hr` | train, dev, test |
| Indonesian | `id` | train, dev, test |
| Indonesian-English | `iden` | train, dev, test |
| Italian | `it` | train, test |
| Japanese | `ja` | train, dev, test |
| Korean | `ko` | train, dev, test |
| Dutch | `nl` | train, dev, test |
| Slovenian | `sl` | train, dev, test |
| Serbian | `sr` | train, dev, test |
| Thai | `th` | train, dev, test |
| Turkish | `tr` | train, test |
| Turkish-German | `trde` | train, test |
| Vietnamese | `vi` | train, dev, test |

---

## normdict

Pre-built normalization dictionaries for 17 language varieties. Each
file maps raw (non-standard) tokens to their normalized forms, with
observation counts and provenance metadata, alongside a vocabulary of
known standard-form words with corpus frequency scores.

### Format

Each file is a JSON object with three top-level keys:

```json
{
  "_meta": {
    "lang": "en",
    "version": 2,
    "sources": ["multilexnorm", "wortschatz", "hunspell"],
    "entry_count": 19237,
    "vocab_count": 997207
  },
  "entries": {
    "bruh": {"norm": "brother", "count": 49, "total": 49, "tier": 2, "sources": ["multilexnorm"]},
    ...
  },
  "_vocab": {
    "known_words": ["the", "and", ...],
    "word_freqs": {"the": 4812345, ...}
  }
}
```

- **`_meta`**: Language code, format version, source list, and counts.
- **`entries`**: Normalization mappings. Each key is a raw token; the
  value contains the best normalized form, observation counts, a trust
  tier (1 = curated, 2 = benchmark, 3 = mined, 4 = generated), and
  provenance tags.
- **`_vocab`**: Standard-language vocabulary and corpus frequencies.

### Sources

These dictionaries were compiled from three sources:

- **multilexnorm**: Training pairs from the MultiLexNorm 2026 benchmark.
- **wortschatz**: Word frequencies from the Leipzig Corpora Collection.
- **hunspell**: Word stems from Hunspell spell-checker dictionaries.

### License

These files are derived works combining data from the sources listed
above. Refer to each source's license for terms:

- MultiLexNorm: see the [original dataset](https://huggingface.co/datasets/weerayut/multilexnorm2026-full-pub)
- Wortschatz: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (Leipzig Corpora Collection)
- Hunspell: mixed open-source licenses (GPL, LGPL, BSD — varies by language)

### Files

| File | Language |
|------|----------|
| `da.normdict.json` | Danish |
| `de.normdict.json` | German |
| `en.normdict.json` | English |
| `es.normdict.json` | Spanish |
| `hr.normdict.json` | Croatian |
| `id.normdict.json` | Indonesian |
| `iden.normdict.json` | Indonesian-English (code-switching) |
| `it.normdict.json` | Italian |
| `ja.normdict.json` | Japanese |
| `ko.normdict.json` | Korean |
| `nl.normdict.json` | Dutch |
| `sl.normdict.json` | Slovenian |
| `sr.normdict.json` | Serbian |
| `th.normdict.json` | Thai |
| `tr.normdict.json` | Turkish |
| `trde.normdict.json` | Turkish-German (code-switching) |
| `vi.normdict.json` | Vietnamese |

---

## dict

Word definition files generated by a large language model. Each file
contains a list of standard-form words for a language, with
definitions and example sentences.

### Format

Newline-delimited JSON (JSONL). Each line is a word entry:

```json
{"word_id": 101, "word": "va", "meanings": [{"definition": "...", "example": "..."}]}
```

- **`word`**: The standard-form word.
- **`meanings`**: List of definitions with example sentences.

### License

These files were generated by a large language model and are provided
as-is. No third-party data license applies.

### Files

| File | Language |
|------|----------|
| `en.jsonl` | English |
| `vi.jsonl` | Vietnamese |

---

## hunspell

Hunspell `.dic` and `.aff` files for 14 languages. These are standard
open-source spell-checker dictionaries redistributed from their
respective upstream projects.

### Format

Each language directory contains two files:

- **`{lang}.dic`**: Word list with optional affix flags
  (e.g. `hello/MS`).
- **`{lang}.aff`**: Affix rules for word inflection.

### Sources and licenses

Licenses vary by language. Each `.aff` file contains the original
copyright and license header from its upstream project.

| Language | License | Source |
|----------|---------|--------|
| Danish (`da`) | GPL 2.0 / LGPL 2.1 / MPL 1.1 tri-license | Stavekontrolden |
| German (`de`) | GPL 2.0 / GPL 3.0 | [igerman98](https://www.j3e.de/ispell/igerman98/) |
| English (`en`) | BSD | [SCOWL](http://wordlist.aspell.net/) |
| Spanish (`es`) | GPL 3.0 / LGPL 3.0 / MPL 1.1 | [RLA-ES](https://github.com/sbosio/rla-es) |
| Croatian (`hr`) | GPL / LGPL | Community contribution |
| Indonesian (`id`) | LGPL 3.0 | [hunspell-id](https://github.com/pnmtjoelern/hunspell-id) |
| Italian (`it`) | GPL 3.0 | Davide Prina / Andrea Pescetti |
| Korean (`ko`) | GPL 3.0 | [hunspell-dict-ko](https://spellcheck-ko.github.io/) |
| Dutch (`nl`) | BSD / CC BY 3.0 | [OpenTaal](https://github.com/OpenTaal/opentaal-hunspell) |
| Slovenian (`sl`) | GPL / LGPL | Community contribution |
| Serbian (`sr`) | GPL / LGPL | Community contribution |
| Thai (`th`) | GPL 3.0 | Community contribution |
| Turkish (`tr`) | GPL / LGPL | Community contribution |
| Vietnamese (`vi`) | GPL 3.0 | Community contribution |

When redistributing these files, retain the license headers present
in each `.aff` file. Refer to the upstream project for full license
text.

---

## wortschatz

Word frequency data from the Leipzig Corpora Collection (Wortschatz
project) for 15 languages.

**Source**: https://wortschatz.uni-leipzig.de/en/download

**License**: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
(Leipzig Corpora Collection, Universitat Leipzig)

**Attribution**: Dirk Goldhahn, Thomas Eckart, and Uwe Quasthoff.
"Building Large Monolingual Dictionaries at the Leipzig Corpora
Collection: From 100 to 200 Languages." *Proceedings of the 8th
International Conference on Language Resources and Evaluation
(LREC'12)*, 2012.

### Format

Each language directory contains three files from the Wortschatz
download:

- **`*-words.txt`**: Tab-separated, three columns: rank, word,
  frequency count.
- **`*-sentences.txt`**: Tab-separated sentence corpus.
- **`*-co_n.txt`**: Tab-separated word co-occurrence data.

#### words.txt example

```
1	the	4812345
2	of	2301567
3	and	2198432
```

### Directory structure

```
wortschatz/{lang}/{corpus_name}-words.txt
wortschatz/{lang}/{corpus_name}-sentences.txt
wortschatz/{lang}/{corpus_name}-co_n.txt
```

### Languages and corpora

| Directory | Corpus | Language |
|-----------|--------|----------|
| `da/` | `dan_newscrawl_2023_1M` | Danish |
| `de/` | `deu_news_2025_1M` | German |
| `en/` | `eng_news_2025_1M` | English |
| `es/` | `spa_news_2024_1M` | Spanish |
| `hr/` | `hrv_wikipedia_2021_1M` | Croatian |
| `id/` | `ind_news_2024_1M` | Indonesian |
| `it/` | `ita_news_2024_1M` | Italian |
| `ja/` | `jpn_wikipedia_2021_1M` | Japanese |
| `ko/` | `kor_news_2022_1M` | Korean |
| `nl/` | `nld_news_2024_1M` | Dutch |
| `sl/` | `slv_wikipedia_2021_1M` | Slovenian |
| `sr/` | `srp_news_2023_30K` | Serbian |
| `th/` | `tha_community_2021` | Thai |
| `tr/` | `tur_news_2024_1M` | Turkish |
| `vi/` | `vie_news_2022_1M` | Vietnamese |
