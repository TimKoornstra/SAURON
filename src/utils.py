# > Imports
# Standard Library
from typing import List

# Third Party
from sentence_transformers import InputExample, SentenceTransformer
import numpy as np


def contrastive_to_binary(examples: List[InputExample])\
        -> List[List[str, str, int]]:
    """
    Convert the contrastive examples to binary examples. This is done by
    creating a positive example for the first two texts and a negative
    example for the first text and each of the remaining texts.

    Parameters
    ----------
    examples : List[InputExample]
        The contrastive examples.

    Returns
    -------
    List[List[str, str, int]]
        The binary examples as a list of lists. Each list contains the
        first text, the second text and the label.
    """
    binary_pairings = []

    for example in examples:

        # Get the positive example
        binary_pairings.append([example.texts[0], example.texts[1], 1])

        # Get the negative examples
        for i in range(2, len(example.texts)):
            binary_pairings.append(
                [example.texts[0], example.texts[i], 0])

    return binary_pairings
