from collections import Counter
from dataclasses import dataclass
from itertools import product
from math import log
from pathlib import Path

import numpy as np
import pandas as pd

from src.make_ngram import make_ngram_counts
from src.prep import no_gutenberg, randomize_chapters, split_chapters
from src.tokenize import build_tokenizer


def load_books(book_files: dict[str, tuple[str | Path, int]]) -> dict[str, str]:
    """Read each source book and remove Gutenberg wrapping when present."""
    books = {}

    for author, (file, chapter_count) in book_files.items():
        book = Path(file).read_text(encoding="utf-8")
        books[author] = no_gutenberg(book)

    return books


def make_validation_split(
    book_files: dict[str, tuple[str | Path, int]],
    validation_fraction: float = 0.2,
    random_seed: int = 42,
) -> tuple[dict[str, str], dict[str, str]]:
    """Return author-keyed training and validation text from whole chapters."""
    np.random.seed(random_seed)
    training_texts = {}
    validation_texts = {}

    for author, (file, chapter_count) in book_files.items():
        book = Path(file).read_text(encoding="utf-8")
        book = no_gutenberg(book)

        validation_count = max(1, int(chapter_count * validation_fraction))
        validation_chapters = randomize_chapters(
            chapter_count,
            validation_count,
        )

        validation_text, training_text = split_chapters(
            book,
            validation_chapters,
        )
        training_texts[author] = training_text
        validation_texts[author] = validation_text

    return training_texts, validation_texts


def prepare_author_model(text: str, tokenizer, n: int, k: float) -> pd.DataFrame:
    """Turn one text string into a smoothed n-gram count table."""
    token_ids = tokenizer.encode(text).ids
    return make_ngram_counts([token_ids], n=n, k=k)


@dataclass
class NgramEngine:
    tokenizer: object
    n: int
    k: float
    ngram_tables: dict[str, pd.DataFrame]

    def __post_init__(self):
        self.ngram_counts = {}
        self.context_counts = {}
        self.unigram_totals = {}

        for author, table in self.ngram_tables.items():
            counts = Counter(dict(zip(table["ngram_ids"], table["count"])))
            self.ngram_counts[author] = counts
            self.unigram_totals[author] = sum(counts.values())

            contexts = Counter()
            if self.n > 1:
                for ngram, count in counts.items():
                    contexts[ngram[:-1]] += count
            self.context_counts[author] = contexts

    def score(self, text: str, author: str) -> float:
        """Return the average log probability of text under one author model."""
        token_ids = self.tokenizer.encode(text).ids
        vocab_size = self.tokenizer.get_vocab_size()

        if len(token_ids) < self.n:
            return float("-inf")

        log_probability = 0.0
        number_of_ngrams = len(token_ids) - self.n + 1

        for index in range(number_of_ngrams):
            ngram = tuple(token_ids[index:index + self.n])
            count = self.ngram_counts[author].get(ngram, 0)

            if self.n == 1:
                context_count = self.unigram_totals[author]
            else:
                context_count = self.context_counts[author].get(ngram[:-1], 0)

            probability = (count + self.k) / (
                context_count + self.k * vocab_size
            )
            log_probability += log(probability)

        return log_probability / number_of_ngrams

    def predict(self, text: str) -> str:
        """Return the author whose language model scores the text highest."""
        return max(self.ngram_tables, key=lambda author: self.score(text, author))


def fit_engine(
    training_texts: dict[str, str],
    vocab_size: int,
    n: int,
    k: float = 1,
) -> NgramEngine:
    """Fit BPE and one n-gram language model per author from training strings."""
    tokenizer = build_tokenizer(list(training_texts.values()), vocab_size)

    ngram_tables = {}
    for author, text in training_texts.items():
        ngram_tables[author] = prepare_author_model(text, tokenizer, n, k)

    return NgramEngine(tokenizer, n, k, ngram_tables)


def validation_accuracy(engine: NgramEngine, validation_texts: dict[str, str]) -> float:
    correct_predictions = 0

    for actual_author, text in validation_texts.items():
        if engine.predict(text) == actual_author:
            correct_predictions += 1

    return correct_predictions / len(validation_texts)


def find_best_hyperparameters(
    book_files: dict[str, tuple[str | Path, int]],
    vocab_sizes: list[int],
    n_values: list[int],
    k_values: list[float],
    validation_fraction: float = 0.2,
    random_seed: int = 42,
) -> tuple[dict[str, int | float], pd.DataFrame]:
    """Tune BPE vocabulary size, n-gram order, and add-k smoothing on validation."""
    training_texts, validation_texts = make_validation_split(
        book_files,
        validation_fraction,
        random_seed,
    )

    results = []
    for vocab_size, n, k in product(vocab_sizes, n_values, k_values):
        engine = fit_engine(training_texts, vocab_size, n, k)
        accuracy = validation_accuracy(engine, validation_texts)

        results.append({
            "vocab_size": vocab_size,
            "n": n,
            "k": k,
            "validation_accuracy": accuracy,
        })

    results_df = pd.DataFrame(results).sort_values(
        "validation_accuracy",
        ascending=False,
    ).reset_index(drop=True)

    best = results_df.iloc[0]
    best_hyperparameters = {
        "vocab_size": int(best["vocab_size"]),
        "n": int(best["n"]),
        "k": float(best["k"]),
    }

    return best_hyperparameters, results_df
