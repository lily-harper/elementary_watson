# Elementary Watson

Elementary Watson is a small author-identification project using BPE tokenization
and n-gram language models. It trains a shared Hugging Face BPE tokenizer on the
training texts, builds separate Tolkien and Conan Doyle language models, and
classifies new passages by comparing each model's perplexity.

For a detailed training and prediction walkthrough, see [PIPELINES.md](PIPELINES.md).

## Repo Structure

```text
.
├── pipeline.py              # Final training and artifact generation
├── src/
│   ├── tokenize.py          # BPE tokenizer training
│   ├── make_ngram.py        # N-gram count tables
│   ├── prep.py              # Text cleanup and chapter splitting
│   ├── what_hyperparameters.py
│   └── predict.py           # Prediction CLI
├── data/                    # Source and sample text files
├── outputs/                 # Generated model artifacts and results
├── PIPELINES.md             # Function/module flow diagrams
└── requirements.txt
```

## Setup

Clone the repo, create a virtual environment, and install the dependencies:

```bash
git clone <repo-url>
cd elementary_watson
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing

The final training script expects `outputs/hyperparameter_results.csv` to exist.
With that file present, regenerate the locked tokenizer and author models with:

```bash
python pipeline.py
```

To classify a passage from a file:

```bash
python -m src.predict --file data/sample/tolkien.txt
```

To classify direct text:

```bash
python -m src.predict --text "In a hole in the ground there lived a hobbit."
```
