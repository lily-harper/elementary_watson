from src.prep import no_gutenberg, randomize_chapters, split_chapters
import json
from pathlib import Path
import pandas as pd

from src.what_hyperparamter import (
    find_best_hyperparameters,
    fit_engine,
    load_books,
)
from src.make_ngram import make_ngram_counts

def data_in(file: Path, chapter_count: int, split: float) -> dict[str, str]:
    file = Path(file)

    with open(file, "r", encoding="utf-8") as f:
        book = f.read()

    book = no_gutenberg(book)

    n_val_chaps = int(chapter_count * split)
    validation_chapter_numbers = randomize_chapters(chapter_count, n_val_chaps)
    validation_text, training_text = split_chapters(book, validation_chapter_numbers)

    book_name = file.stem
    return {
        f"{book_name}_val": validation_text,
        f"{book_name}_train": training_text,
    }


BOOK_FILES = {
    "tolkien": ("data/source/hobbit.txt", 19),
    "doyle": ("data/source/lostworld.txt", 16),
}

VOCAB_SIZES = [500, 1000, 1500, 3000]
N_VALUES = [2, 3]
K_VALUES = [0.01, 0.1, 0.5, 1.0]


def run_hyperparameter_search():
    """Run validation tuning and return its best result, table, and heatmap."""
    return find_best_hyperparameters(
        book_files=BOOK_FILES,
        vocab_sizes=VOCAB_SIZES,
        n_values=N_VALUES,
        k_values=K_VALUES,
        validation_fraction=0.2,
        random_seed=42,
    )


def load_best_hyperparameters(results_file: Path) -> dict[str, int | float]:
    """Read the lowest-perplexity configuration from a completed grid search."""
    results_df = pd.read_csv(results_file)
    best = results_df.loc[results_df["validation_perplexity"].idxmin()]

    return {
        "vocab_size": int(best["vocab_size"]),
        "n": int(best["n"]),
        "k": float(best["k"]),
    }


def run_full_training(best_hyperparameters: dict[str, int | float]):
    """Fit the locked tokenizer and author n-gram tables on all source text."""
    full_training_texts = load_books(BOOK_FILES)
    return fit_engine(full_training_texts, **best_hyperparameters)


def make_unigram_tables(final_engine, training_texts: dict[str, str]):
    """Count unigrams for each author with the locked tokenizer."""
    unigram_tables = {}

    for author, text in training_texts.items():
        token_ids = final_engine.tokenizer.encode(text).ids
        unigram_tables[author] = make_ngram_counts(
            [token_ids],
            n=1,
            k=final_engine.k,
        )

    return unigram_tables


if __name__ == "__main__":
    output_directory = Path("outputs")
    output_directory.mkdir(exist_ok=True)
    results_file = output_directory / "hyperparameter_results.csv"

    best_hyperparameters = load_best_hyperparameters(results_file)
    full_training_texts = load_books(BOOK_FILES)
    final_engine = fit_engine(full_training_texts, **best_hyperparameters)
    unigram_tables = make_unigram_tables(final_engine, full_training_texts)

    (output_directory / "locked_hyperparameters.json").write_text(
        json.dumps(best_hyperparameters, indent=2),
        encoding="utf-8",
    )
    final_engine.tokenizer.save(str(output_directory / "final_tokenizer.json"))

    for author, ngram_table in final_engine.ngram_tables.items():
        ngram_table.to_csv(
            output_directory / f"{author}_final_ngram_counts.csv",
            index=False,
        )

    for author, unigram_table in unigram_tables.items():
        unigram_table.to_csv(
            output_directory / f"{author}_final_unigram_counts.csv",
            index=False,
        )

    print("Locked hyperparameters:", best_hyperparameters)
