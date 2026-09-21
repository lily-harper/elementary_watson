# text normalization, splitting 

from pydoc import text
import re 
import unicodedata
import numpy as np

def no_gutenberg(book: str) -> str:
    gutenberg_start = "*** START OF THE PROJECT GUTENBERG EBOOK "
    gutenberg_end   = "*** END OF THE PROJECT GUTENBERG EBOOK "

    start = book.find(gutenberg_start) 
    end = book.find(gutenberg_end)

    if start == -1:
        return book

    if end == -1:
        raise ValueError("Gutenberg end marker was not found.")

    start_of_text = book.find("\n", start) + 1
    result = book[start_of_text:end]
    return(result)  

def randomize_chapters(total_chapters: int, validation_count: int) -> list[int]:
    chapter_numbers = np.arange(1, total_chapters + 1)
    chap_choices = np.random.choice(
        chapter_numbers,
        size=validation_count,
        replace=False,
    )
    return sorted(chap_choices.tolist())

def split_chapters(book: str, validation_chapter_numbers: list[int]) -> tuple[str, str]:
    """Return the selected chapters first, then all remaining chapters."""

    roman_to_number = {
        "I": 1,
        "II": 2,
        "III": 3,
        "IV": 4,
        "V": 5,
        "VI": 6,
        "VII": 7,
        "VIII": 8,
        "IX": 9,
        "X": 10,
        "XI": 11,
        "XII": 12,
        "XIII": 13,
        "XIV": 14,
        "XV": 15,
        "XVI": 16,
        "XVII": 17,
        "XVIII": 18,
        "XIX": 19,
    }

    # This finds lines like "Chapter IV", "CHAPTER IV", and "Chapter X III".
    chapter_matches = list(re.finditer(
        r"(?im)^\s*chapter\s+([ivxlcdm ]+)\s*$",
        book,
    ))

    validation_chapters = []
    training_chapters = []

    for index, match in enumerate(chapter_matches):
        roman_numeral = match.group(1).replace(" ", "").upper()
        chapter_number = roman_to_number[roman_numeral]

        chapter_start = match.start()
        if index + 1 < len(chapter_matches):
            chapter_end = chapter_matches[index + 1].start()
        else:
            chapter_end = len(book)

        chapter_text = book[chapter_start:chapter_end]

        if chapter_number in validation_chapter_numbers:
            validation_chapters.append(chapter_text)
        else:
            training_chapters.append(chapter_text)

    validation_text = "\n".join(validation_chapters)
    training_text = "\n".join(training_chapters)

    return validation_text, training_text
