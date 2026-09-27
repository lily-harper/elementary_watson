import argparse
import ast
import json
from pathlib import Path

import pandas as pd
from tokenizers import Tokenizer

from src.what_hyperparameters import NgramEngine


def load_final_engine(output_directory: str | Path = "outputs") -> NgramEngine:
    """Load the locked tokenizer, hyperparameters, and author n-gram tables."""
    output_directory = Path(output_directory)

    hyperparameters = json.loads(
        (output_directory / "locked_hyperparameters.json").read_text(
            encoding="utf-8",
        )
    )
    tokenizer = Tokenizer.from_file(str(output_directory / "final_tokenizer.json"))

    ngram_tables = {}
    for author in ("tolkien", "doyle"):
        table = pd.read_csv(output_directory / f"{author}_final_ngram_counts.csv")
        table["ngram_ids"] = table["ngram_ids"].map(
            lambda ids: tuple(ast.literal_eval(ids))
        )
        ngram_tables[author] = table

    return NgramEngine(
        tokenizer=tokenizer,
        n=hyperparameters["n"],
        k=hyperparameters["k"],
        ngram_tables=ngram_tables,
    )


def predict_text(text: str, engine: NgramEngine) -> dict:
    """Return the lower-perplexity author and both author-model perplexities."""
    perplexities = {
        author: engine.perplexity(text, author)
        for author in engine.ngram_tables
    }
    most_likely_author = min(perplexities, key=perplexities.get)

    return {
        "most_likely_author": most_likely_author,
        "perplexities": perplexities,
    }


def predict_file(file: str | Path, engine: NgramEngine) -> dict:
    """Read a UTF-8 text file and classify its contents."""
    text = Path(file).read_text(encoding="utf-8")
    return predict_text(text, engine)


def predict_lines(
    file: str | Path,
    output_file: str | Path,
    engine: NgramEngine,
) -> None:
    """Classify each tab-separated ID and passage, preserving input order."""
    lines = Path(file).read_text(encoding="utf-8").splitlines()
    answers = []
    for line_number, line in enumerate(lines, start=1):
        identifier, separator, passage = line.partition("\t")
        if not separator or not identifier or not passage.strip():
            raise ValueError(f"Invalid ID and passage on line {line_number}")
        if identifier.startswith("ITEM-"):
            identifier = "ID-" + identifier[len("ITEM-"):]
        author = predict_text(passage, engine)["most_likely_author"]
        answers.append(f"{identifier}\t{author.title()}")

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(answers) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify text with the locked Tolkien and Doyle language models."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--file", help="Path to a UTF-8 text file.")
    input_group.add_argument("--text", help="Text to classify.")
    input_group.add_argument("--lines", help="Tab-separated ID and passage file.")
    parser.add_argument(
        "--output",
        default="outputs/predictions.txt",
        help="Output path for --lines (default: outputs/predictions.txt).",
    )
    arguments = parser.parse_args()

    final_engine = load_final_engine()

    if arguments.lines:
        predict_lines(arguments.lines, arguments.output, final_engine)
    else:
        if arguments.file:
            result = predict_file(arguments.file, final_engine)
        else:
            result = predict_text(arguments.text, final_engine)

        print("Most likely author:", result["most_likely_author"])
        for author, perplexity in result["perplexities"].items():
            print(f"{author} perplexity: {perplexity:.2f}")
