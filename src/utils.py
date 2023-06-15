# > Imports
# Standard Library
import pickle
from typing import Any, List, Tuple

import numpy as np
from sklearn.metrics import roc_curve


def contrastive_to_binary(examples: List[Tuple[Any, Any, Any]])\
        -> List[Tuple[Any, Any, int]]:
    """
    Convert the contrastive examples to binary examples. This is done by
    creating a positive example for the first two texts and a negative
    example for the first text and each of the remaining texts.

    Parameters
    ----------
    examples : List[Tuple[Any, Any, Any]]
        The contrastive examples.

    Returns
    -------
    List[Tuple[Any, Any, int]]
        The binary examples as a list of lists. Each list contains the
        first text, the second text and the label.
    """

    binary_pairings = []

    for example in examples:
        # Get the positive example
        binary_pairings.append((example[0], example[1], 1))

        # Get the negative example
        binary_pairings.append((example[0], example[2], 0))

    return binary_pairings


def get_threshold(data_path: str,
                  model: object) -> float:
    """
    Get the thresholds from the results file.

    Parameters
    ----------
    data_path : str
        The path to the data directory.
    model : object
        The model to use.

    Returns
    -------
    float
        The threshold.
    """
    print("Calculating threshold manually...")

    # Determine the threshold using the validation set and AUC
    # Load the validation data
    with open(f"{data_path}/paired/val-pairings.pkl", "rb") as f:
        val_data = pickle.load(f)

    # Only consider the first 2000 triplets
    val_data = val_data[:2000]

    # Convert to Binary Task
    val_data = contrastive_to_binary(val_data)

    # Get the true labels
    true_labels = [x[2] for x in val_data]

    # Get the first and second texts
    first_texts = [x[0] for x in val_data]
    second_texts = [x[1] for x in val_data]

    # Calculate embeddings in a batch
    first_embeddings = model.model.encode(first_texts)
    second_embeddings = model.model.encode(second_texts)

    # Calculate similarities
    sims = model.similarity(first_embeddings, second_embeddings)

    # Get the threshold
    fpr, tpr, thresholds = roc_curve(true_labels, sims)
    threshold = thresholds[np.argmax(tpr - fpr)]

    return threshold
