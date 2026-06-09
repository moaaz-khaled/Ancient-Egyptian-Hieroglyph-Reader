# 🏺 𓂀 Hieroglyphics Dataset Collection 𓇳

> A comprehensive collection of datasets and resources used for Hieroglyphic NLP, Translation, Sequence Assembly, and Symbol Understanding.

𓂀 𓇳 𓅓 𓈖 𓂋

---

# 📂 Dataset Overview

This directory contains multiple datasets collected, cleaned, and curated for different Hieroglyphic Natural Language Processing tasks, including:

- 🔤 Hieroglyphic Translation
- 🧩 Sequence Assembly
- 🏷️ Intent Classification
- 📜 Symbol Recognition
- 📚 Gardiner Sign References

---

# 📖 Alan Gardiner's List of Hieroglyphic Signs

**File:**

```text
Alan Gardiners List of Hieroglyphic Signs.xlsx
```

## Description

This file contains the famous Gardiner Sign List, one of the most widely used classifications of Ancient Egyptian hieroglyphs.

The dataset provides categorized signs according to Alan Gardiner's standard classification system and serves as a reference resource for symbol identification and analysis.

## Source

Original repository:

https://github.com/omnika-datastore/alan-gardiner-list-of-hieroglyphic-signs

The repository itself was compiled from:

https://www.egyptianhieroglyphs.net/gardiners-sign-list/

## Usage

Used as:

- 📚 Symbol reference database
- 🔍 Hieroglyph lookup
- 🏷️ Sign categorization
- 🧠 Symbol interpretation support

---

# 🌍 dataset_cleaned.csv

## Description

This dataset was created for training machine translation models between cleaned Hieroglyphic transliterations and German text.

The dataset combines samples from multiple sources and applies extensive preprocessing and normalization.

### Columns

| Column | Description |
|----------|-------------|
| raw_transliteration | Original transliteration text |
| clean_transliteration | Cleaned transliteration |
| raw_german | Original German translation |
| clean_german | Cleaned German translation |

Example:

| raw_transliteration | clean_transliteration |
|--------------------|-----------------------|
| nḏ (w)di̯ r =s | nḏ wdi̯ r s |

| raw_german | clean_german |
|------------|--------------|
| (es) werde zerrieben, (es) werde darauf gelegt. | werde zerrieben, werde darauf gelegt. |

## Sources

### Source 1

https://huggingface.co/datasets/phiwi/bbaw_egyptian

### Source 2

https://huggingface.co/datasets/thesaurus-linguae-aegyptiae/tla-Earlier_Egyptian_original-v18-premium

## Processing

Several cleaning and normalization procedures were applied, including:

- Removing formatting artifacts
- Removing unnecessary brackets and symbols
- Standardizing transliteration formatting
- Cleaning German translations
- Normalizing whitespace

## Statistics

- 📊 Approximately 89,000 sentence pairs
- 🌍 Hieroglyphic Transliteration → German

## Usage

Used for training:

- 🤖 NLLB
- 🌐 Machine Translation Models
- 📚 Parallel Corpus Research

---

# 🏷️ Gardiner_Sign_List.csv

## Description

This dataset contains Hieroglyphic signs extracted from JSesh resources.

A total of **7029 hieroglyphic symbols** were collected and organized for symbol-level processing tasks.

## Source

JSesh Project:

https://jsesh.qenherkhopeshef.org/

## Usage

Used for:

- 🔤 Symbol vocabulary creation
- 🏺 Hieroglyph recognition
- 📚 Sign inventory generation
- 🧠 Symbol embedding experiments

---

# 🔠 character_level.csv

## Description

This dataset was generated from the cleaned transliterations found in `dataset_cleaned.csv`.

Each transliteration sequence was decomposed into individual character-level tokens to support sequence assembly tasks.

Example:

Clean Text:

```text
htp di nsw
```

Character Representation:

```text
h t p d i n s w
```

## Columns

| Column | Description |
|---------|-------------|
| characters | Character-level sequence |
| clean_text | Original cleaned transliteration |

## Generation Process

1. Start with cleaned transliterations from `dataset_cleaned.csv`
2. Remove noise and formatting artifacts
3. Split transliterations into character-level sequences
4. Store both representations

## Usage

Used for training:

- 🔄 Sequence Assembly Models
- 🔤 Character-Level Language Models
- 🧩 Sequence Reconstruction Systems

---

# 🎯 hieroglyph_intentions.csv

## Description

This dataset contains manually curated intent labels for Hieroglyphic expressions.

During the project, no suitable public intent-classification dataset for Hieroglyphic language was found.

Therefore, the dataset was fully created and annotated manually.

## Source

✨ Completely handcrafted by the project team.

No public source dataset was used.

## Usage

Used for:

- 🎯 Intent Classification
- 🤖 Hieroglyphic Assistants
- 💬 Semantic Understanding
- 🧠 NLP Experiments

---

# ⚠️ Notes

Some datasets are derived from external academic and public resources.

Please refer to the original sources for licensing and attribution requirements.

---

𓂀 𓇳 𓅓 𓈖 𓂋

Built for Hieroglyphic NLP Research 🏺