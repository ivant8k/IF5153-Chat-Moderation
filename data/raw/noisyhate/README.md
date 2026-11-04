# toxic-detection-testset

## Table of Contents
- [Table of Contents](#table-of-contents)
- [Dataset Description](#dataset-description)
  - [Dataset Summary](#dataset-summary)
  - [Languages](#languages)
- [Dataset Structure](#dataset-structure)
  - [Data Instances](#data-instances)
  - [Data Fields](#data-fields)
  - [Data Splits](#data-splits)
- [Dataset Creation](#dataset-creation)
  - [Curation Rationale](#curation-rationale)
  - [Source Data](#source-data)
  - [Annotations](#annotations)
  - [Personal and Sensitive Information](#personal-and-sensitive-information)
- [Additional Information](#additional-information)
  - [Dataset Curators](#dataset-curators)
  - [Citation Information](#citation-information)

## Dataset Description

### Dataset Summary

This dataset a test set for toxic detection that contains both clean data and it's perturbed version with human-written perturbations online. 
In addition, our dataset can be used to benchmark misspelling correctors as well.


### Languages

English

## Dataset Structure

### Data Instances
```
{
  "clean_version": "this is pretty much exactly how i feel damn",
  "perturbed_version": "this is pretty much exactly how i feel daaammnn",
  "toxicity": 0.7,
  "obscene": 0.7,
  "sexual_explicit": 0,
  "identity_attack": 0,
    ...
  "insult": 0.2,
  "quality_mean": 4
}
	
```

### Data Fields

This dataset is derived from the [Jigsaw data](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/data). Hence, it keeps all the useful metrics and attributes.

**Main**
* clean_version
* perturbed_version


**Metrics**
* toxicity
* severe_toxicity
* obscene
* threat
* insult
* identity_attack
* sexual_explicit
**Identity attributes**
* male
* female
* transgender
* other_gender
* heterosexual
* homosexual_gay_or_lesbian
* bisexual
* other_sexual_orientation
* christian
* jewish
* muslim
* hindu
* buddhist
* atheist
* other_religion
* black
* white
* asian
* latino
* other_race_or_ethnicity
* physical_disability
* intellectual_or_learning_disability
* psychiatric_or_mental_illness
* other_disability
### Data Splits

test: 1339
## Dataset Creation
### Curation Rationale
[More Information Needed]
### Source Data
#### Initial Data Collection and Normalization
Jigsaw is a famous toxic speech classification dataset containing approximately 2 million public comments from the Civil Comments platform. In addition to the toxic score labels for toxicity classification, the Jigsaw dataset also provides several toxicity sub-type dimensions which indicate particular comments' target groups, such as male, female, black, and Asian. Due to these prolific identity annotations and significant data volume, we adopt this dataset as our raw data source. Since the dataset has been used as the standard benchmark dataset for content moderation tasks, this adoption will also help reduce the entry barrier in adopting NoisyHate from the community.

Since the comments from the Jigsaw dataset contain a lot of special characters, emojis, and informal language, data cleaning was necessary to ensure data quality. Following a typical text processing pipeline, we removed duplicated texts, special characters, special punctuation, hyperlinks, and numbers. Since we only focused on English texts, sentences containing non-standard English words were filtered out. 13,1982 comments remained after this cleaning step.

#### Who are the source language producers?
The source data is provided by the Conversation AI team, a research initiative founded by Jigsaw and Google.
### Annotations
#### Annotation process
In the annotation process, we display a guideline to explain the definition of human-generated perturbation and provide examples of both high-quality and low-quality perturbations. This training phase has been suggested to warrant high-quality responses from the human worker, especially for labeling tasks. Each MTurk worker is then presented with a pair of a perturbed sentences and its clean version and is asked to determine the quality of the perturbed one (Guideline and UI can be found in our [paper](#citation-information)).


We recruited five different workers from the North America region through five assignments to assess each pair. A five-second countdown timer was also set for each task to ensure workers spent enough time on it. To ensure the quality of their responses, we designed an attention question that asks them to click on the perturbed word in the given sentences before they provide their quality ratings. Workers who cannot correctly identify the perturbation's location in the given sentence will be blocked for future batches. We aimed to pay the workers at an average rate of \$10 per hour, which is well above the federal minimum wage (\$7.25 per hour). The payment of each task was estimated by the average length of the sentences, which totals around 25 words per pair, and the average reading speed of native speakers is around 228 words per minute. 

#### Who are the annotators?
US Amazon MTurk workers with HIT Approval Rate greater than 98%, and Number of HITs approved greater than 1000.
### Personal and Sensitive Information
N/A
## Additional Information
### Dataset Curators
[More Information Needed]
### Citation Information
paper is coming soon

