from src.prep import no_gutenberg, randomize_chapters, split_chapters
from pathlib import Path

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
