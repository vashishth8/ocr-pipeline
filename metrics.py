#!/usr/bin/env python3

import sys


def levenshtein(a, b):

    previous = list(range(len(b) + 1))

    for i, char_a in enumerate(a, 1):

        current = [i]

        for j, char_b in enumerate(b, 1):

            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            substitution = previous[j - 1] + (
                char_a != char_b
            )

            current.append(
                min(
                    insertion,
                    deletion,
                    substitution
                )
            )

        previous = current

    return previous[-1]


def normalize(text):
    return " ".join(text.split())


def cer(reference, hypothesis):

    if not reference:
        return 0 if not hypothesis else 1

    return (
        levenshtein(reference, hypothesis)
        / len(reference)
    )


def wer(reference, hypothesis):

    reference_words = normalize(
        reference
    ).split()

    hypothesis_words = normalize(
        hypothesis
    ).split()

    if not reference_words:
        return (
            0
            if not hypothesis_words
            else 1
        )

    return (
        levenshtein(
            reference_words,
            hypothesis_words
        )
        / len(reference_words)
    )


if __name__ == "__main__":

    if len(sys.argv) != 3:

        print(
            "Usage: python metrics.py "
            "ground_truth.txt ocr_output.txt"
        )

        sys.exit(1)

    with open(
        sys.argv[1],
        encoding="utf-8"
    ) as f:
        reference = f.read()

    with open(
        sys.argv[2],
        encoding="utf-8"
    ) as f:
        hypothesis = f.read()

    print(
        f"CER: {cer(reference, hypothesis):.4%}"
    )

    print(
        f"WER: {wer(reference, hypothesis):.4%}"
    )
