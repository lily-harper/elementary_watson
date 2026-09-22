from collections import Counter
from dataclasses import dataclass
from itertools import product
from math import exp, log
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
        """Return the average log probability per whitespace-delimited word."""
        token_ids = self.tokenizer.encode(text).ids
        vocab_size = self.tokenizer.get_vocab_size()
        word_count = len(text.split())

        if len(token_ids) < self.n or word_count == 0:
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

        return log_probability / word_count

    def perplexity(self, text: str, author: str) -> float:
        """Return exp of the average negative log probability per word."""
        return exp(-self.score(text, author))

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


def validation_perplexity(engine: NgramEngine, validation_texts: dict[str, str]) -> float:
    """Return corpus-level validation perplexity under the correct author models."""
    total_log_probability = 0.0
    total_word_count = 0

    for actual_author, text in validation_texts.items():
        word_count = len(text.split())
        average_log_probability = engine.score(text, actual_author)

        total_log_probability += average_log_probability * word_count
        total_word_count += word_count

    return exp(-total_log_probability / total_word_count)


def validation_perplexities(
    engine: NgramEngine,
    validation_texts: dict[str, str],
) -> dict[str, float]:
    """Return each author model's perplexity on its own validation text."""
    perplexities = {}

    for author, text in validation_texts.items():
        perplexities[author] = engine.perplexity(text, author)

    return perplexities


def plot_hyperparameter_heatmaps(results_df: pd.DataFrame):
    """Return author-specific validation-perplexity heatmaps for every n."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_values = sorted(results_df["n"].unique())
    model_columns = [
        column
        for column in results_df.columns
        if column.endswith("_validation_perplexity")
        and column != "validation_perplexity"
    ]

    figure, axes = plt.subplots(
        len(n_values),
        len(model_columns),
        figsize=(5 * len(model_columns) + 1, 4 * len(n_values)),
        squeeze=False,
        layout="constrained",
    )
    images = []
    minimum_perplexity = results_df[model_columns].min().min()
    maximum_perplexity = results_df[model_columns].max().max()

    if minimum_perplexity == maximum_perplexity:
        maximum_perplexity += 1

    for row, n in enumerate(n_values):
        n_results = results_df[results_df["n"] == n]

        for column, model_column in enumerate(model_columns):
            axis = axes[row, column]
            heatmap_data = n_results.pivot(
                index="k",
                columns="vocab_size",
                values=model_column,
            ).sort_index().sort_index(axis=1)

            image = axis.imshow(
                heatmap_data.to_numpy(),
                vmin=minimum_perplexity,
                vmax=maximum_perplexity,
                cmap="viridis",
                aspect="auto",
            )
            images.append(image)

            author = model_column.removesuffix("_validation_perplexity").title()
            axis.set_title(f"{author} model, n = {n}")
            axis.set_xlabel("BPE vocabulary size")
            axis.set_ylabel("Smoothing k")
            axis.set_xticks(range(len(heatmap_data.columns)))
            axis.set_xticklabels(heatmap_data.columns)
            axis.set_yticks(range(len(heatmap_data.index)))
            axis.set_yticklabels(heatmap_data.index)

            for heatmap_row in range(len(heatmap_data.index)):
                for heatmap_column in range(len(heatmap_data.columns)):
                    perplexity = heatmap_data.iloc[heatmap_row, heatmap_column]
                    axis.text(
                        heatmap_column,
                        heatmap_row,
                        f"{perplexity:.2f}",
                        ha="center",
                        va="center",
                        color="white",
                    )

    figure.colorbar(
        images[0],
        ax=axes.ravel().tolist(),
        label="Validation perplexity (lower is better)",
    )
    figure.suptitle("Author-model validation perplexity")

    return figure


def find_best_hyperparameters(
    book_files: dict[str, tuple[str | Path, int]],
    vocab_sizes: list[int],
    n_values: list[int],
    k_values: list[float],
    validation_fraction: float = 0.2,
    random_seed: int = 42,
) -> tuple[dict[str, int | float], pd.DataFrame, object]:
    """Tune BPE vocabulary size, n-gram order, and add-k smoothing on validation."""
    training_texts, validation_texts = make_validation_split(
        book_files,
        validation_fraction,
        random_seed,
    )

    results = []
    for vocab_size, n, k in product(vocab_sizes, n_values, k_values):
        engine = fit_engine(training_texts, vocab_size, n, k)
        perplexity = validation_perplexity(engine, validation_texts)
        model_perplexities = validation_perplexities(engine, validation_texts)

        result = {
            "vocab_size": vocab_size,
            "n": n,
            "k": k,
            "validation_perplexity": perplexity,
        }

        for author, model_perplexity in model_perplexities.items():
            result[f"{author}_validation_perplexity"] = model_perplexity

        results.append(result)

    results_df = pd.DataFrame(results).sort_values(
        "validation_perplexity",
        ascending=True,
    ).reset_index(drop=True)

    best = results_df.iloc[0]
    best_hyperparameters = {
        "vocab_size": int(best["vocab_size"]),
        "n": int(best["n"]),
        "k": float(best["k"]),
    }

    heatmap_figure = plot_hyperparameter_heatmaps(results_df)

    return best_hyperparameters, results_df, heatmap_figure
