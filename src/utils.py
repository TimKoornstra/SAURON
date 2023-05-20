# > Imports
# Standard Library
import pickle
from typing import List, Tuple

# Third Party
from sentence_transformers import InputExample

import numpy as np
from sklearn.metrics import roc_curve


def contrastive_to_binary(examples: List[InputExample])\
        -> List[Tuple[str, str, int]]:
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
    List[Tuple[str, str, int]]
        The binary examples as a list of lists. Each list contains the
        first text, the second text and the label.
    """

    binary_pairings = []

    for example in examples:

        # Get the positive example
        binary_pairings.append((example.texts[0], example.texts[1], 1))

        # Get the negative examples
        for i in range(2, len(example.texts)):
            binary_pairings.append(
                (example.texts[0], example.texts[i], 0))

    return binary_pairings


def get_threshold(data_path: str,
                  model: object,
                  ) -> float:
    """
    Get the thresholds from the results file.

    Parameters
    ----------
    results_path : str
        The path to the results file.

    Returns
    -------
    float
        The threshold.
    """

    # Determine the threshold using the validation set and AUC
    # Load the validation data
    with open(f"{data_path}/paired/val-pairings.pkl", "rb") as f:
        val_data = pickle.load(f)

    print("Calculating threshold manually...")

    # Get the true labels
    # Convert to Binary Task
    val_examples = [InputExample(texts=texts, label=1)
                    for texts in val_data[:10000]]
    val_data = contrastive_to_binary(val_examples)

    # Get the true labels
    true_labels = [x[2] for x in val_data]

    first = [x[0] for x in val_data]
    second = [x[1] for x in val_data]

    # Get the predictions
    sims = model.similarity(first, second)

    # Get the threshold
    fpr, tpr, thresholds = roc_curve(true_labels, sims)

    # Get the threshold
    threshold = thresholds[np.argmax(tpr - fpr)]

    return threshold
