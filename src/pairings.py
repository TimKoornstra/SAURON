# Imports

# Standard Library
from typing import List, Tuple

# Third Party
from sklearn.model_selection import GroupShuffleSplit
import pandas as pd

# Local
from semantics import paraphrase_mining


def split_data(df: pd.DataFrame, train_size: float = 0.8)\
        -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the data into train, validation, and test sets.
    The sets are non-overlapping based on the author_id.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to split.
    train_size : float
        The size of the train set.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        The train, validation, and test sets.
    """

    # Split the data into train and test sets
    train_test_splitter = GroupShuffleSplit(
        n_splits=1, test_size=1-train_size, random_state=42)

    train_idx, test_idx = next(
        train_test_splitter.split(df, groups=df["author_id"]))
    train, test = df.iloc[train_idx], df.iloc[test_idx]

    # Split the test set into validation and test sets (50/50)
    val_test_splitter = GroupShuffleSplit(
        n_splits=1, test_size=0.5, random_state=42)

    val_idx, test_idx = next(val_test_splitter.split(
        test, groups=test["author_id"]))
    val, test = test.iloc[val_idx], test.iloc[test_idx]

    return train, val, test


def create_pairings(df: pd.DataFrame,
                    n_negative_examples: int = 7,
                    semantic_range: Tuple[float, float] = (0.95, 0.99))\
        -> List[Tuple[str, str, int]]:
    """
    Create the pairings of sentences. Each pairing contains two sentences
    and a label: 1 for positive, 0 for negative. The positive examples
    are sentences that are written by the same author. The negative examples
    are sentences that are written by different authors.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to create the pairings from.
    n_negative_examples : int
        The number of negative examples to create for each positive example.
    semantic_range : Tuple[float, float]
        The inclusive range of semantic similarity to use for semantic filtering.

    Returns
    -------
    List[Tuple[str, str, int]]
        The list of pairings.
    """
    # Reset the index, so that the index is the same as the row number
    df = df.reset_index(drop=True)

    # First, we need to calculate the semantic similarity between each
    # pair of sentences.
    paraphrases = paraphrase_mining(df)

    # Next, we only want to keep the sentences that are within the
    # semantic range.
    paraphrases = paraphrases[(paraphrases["similarity"] >= semantic_range[0])
                              & (paraphrases["similarity"] <= semantic_range[1])]

    # Create the list of pairings
    pairings = []

    # Create a set of authors that have all positive examples
    positive_authors = set()

    # Iterate through each row in the DataFrame
    for i, row in df.iterrows():
        # Get the text and author_id
        text = row["text"]
        author_id = row["author_id"]

        # Create the positives examples
        # Check if we have already paired this author with itself
        if author_id not in positive_authors:
            # Find all the sentences that are written by the same author but
            # are not the same sentence

            for _, row2 in df[(df["author_id"] == author_id) &
                              (df["text"] != text)].iterrows():
                # Get the text and author_id
                text2 = row2["text"]

                # Add the pairing to the list
                pairings.append((text, text2, 1))

            # Add the pairing to the paired_with dictionary
            positive_authors.add(author_id)

        # Create the negative examples
        # First, find the sentences that are semantically similar to the
        # current sentence
        similar_sentences = paraphrases[paraphrases["idx_1"] == i]
        negative_examples = set()

        # Iterate through each similar sentence
        # We only want to create n_negative_examples negative examples
        for _, row2 in similar_sentences.iterrows():
            if len(negative_examples) >= n_negative_examples:
                break

            # Get the text and author_id
            text2 = df.iloc[int(row2["idx_2"])]["text"]

            # Check that this is a unique negative example
            if (text, text2, 0) not in negative_examples:
                # Add the pairing to the list
                pairings.append((text, text2, 0))

                # Add the pairing to the negative examples
                negative_examples.add((text, text2, 0))

        # If we don't have enough negative examples, then we need to
        # create some more negative examples by randomly selecting
        # sentences from other authors
        while len(negative_examples) < n_negative_examples:
            # Randomly select a sentence from another author
            text2 = df[df["author_id"] != author_id]["text"].sample().iloc[0]

            # Check that this is a unique negative example
            if (text, text2, 0) not in negative_examples:
                # Add the pairing to the list
                pairings.append((text, text2, 0))

                # Add the pairing to the negative examples
                negative_examples.add((text, text2, 0))

    return pairings
