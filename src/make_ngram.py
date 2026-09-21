from collections import Counter
import pandas as pd

def grams(corpus,tokenizer, n: int) ->  pd.DataFrame:
    """
    Generate n-grams from a sequence.
    n = 2 for bigrams, 3 for trigrams, etc.
    """
    counts = Counter()
    
    if n==1:
        for sequence in corpus:
            for token_id in sequence:
                counts[token_id] += 1

    elif n==2:
        for sequence in corpus:
            for i in range(len(sequence) - 1):
                bigram = (sequence[i], sequence[i + 1])
                counts[bigram] += 1

    elif n==3 :
        for sequence in corpus:
            for i in range(len(sequence) - 2):
                trigram = (
                    sequence[i],
                    sequence[i + 1],
                    sequence[i + 2]
                )
                counts[trigram] += 1

    rows = []
    for gram, count in counts.items():
        id = (gram, ) if n == 1 else gram 

        rows.append({
            "token_ids": id,
            "token": " ".join(tokenizer.id_to_token(token_id) for token_id in id),
            "count": count
        }) 

    return pd.DataFrame(rows).sort_values("count", ascending=False)

def make_ngram_counts(sequences, n: int, k: float = 1) -> pd.DataFrame:
    """Count token-ID n-grams and add smoothing to each observed n-gram."""

    if n < 1:
        raise ValueError("n must be at least 1")

    if k < 0:
        raise ValueError("k must be zero or greater")

    ngram_counts = Counter()

    for sequence in sequences:
        for i in range(len(sequence) - n + 1):
            ngram = tuple(sequence[i:i + n])
            ngram_counts[ngram] += 1

    rows = []
    for ngram, count in ngram_counts.items():
        rows.append({
            "ngram_ids": ngram,
            "count": count,
            "smoothed_count": count + k,
        })

    return (
        pd.DataFrame(rows, columns=["ngram_ids", "count", "smoothed_count"])
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
