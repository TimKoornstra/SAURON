# Imports

# Standard Library
from typing import List, Tuple
import time
from collections import defaultdict
import random

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
                    max_negative: int = 7,
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
    s_pairings = []

    # Reset the index, so that the index is the same as the row number
    df = df.reset_index(drop=True)

    # First, we need to calculate the semantic similarity between each
    # pair of sentences.
    paraphrases = paraphrase_mining(df)

    # Next, we only want to keep the sentences that are within the
    # semantic range.
    paraphrases = paraphrases[(paraphrases["similarity"] >= semantic_range[0])
                              & (paraphrases["similarity"] <= semantic_range[1])]

    #################
    start = time.time()

    # Convert all paraphrases to a defaultdict of lists, where the key is the idx_1
    # and the value is a list of idx_2, sorted by similarity
    paraphrases = paraphrases.sort_values("similarity", ascending=False)
    paraphrases = paraphrases.groupby("idx_1")["idx_2"].apply(list).to_dict()
    paraphrases = defaultdict(list, paraphrases)

    # Convert the df to a dictionary of lists, where the key is the author_id
    # and the value is a list of the indices of sentences written by that author
    df["index"] = df.index
    data = df.groupby("author_id")["index"].apply(list).to_dict()

    # Create a lookup table for the indices of the sentences
    lookup = df.set_index("index").to_dict()["text"]

    print("Time to convert: ", time.time() - start)

    # Create the list of pairings
    pairings = []

    data_seen = 0

    # Iterate through each author
    for author_id, sentences in data.items():
        for i in range(len(sentences)):
            # Create the positives examples
            # Check if we have already paired this author with itself
            # Find the unique pairings of sentences
            for j in range(i+1, len(sentences)):
                # Add the pairing to the list
                pairings.append(
                    (lookup[sentences[i]], lookup[sentences[j]], 1))

                # Create max_negative negative examples for each positive example
                negative_examples = 0

                # First, find the sentences that are semantically similar to the
                # current sentence
                # Iterate through each similar sentence
                for k in paraphrases[sentences[i]]:
                    # Add the pairing to the list
                    pairings.append((lookup[sentences[i]], lookup[k], 0))
                    s_pairings.append((lookup[sentences[i]], lookup[k]))

                    # Increment the count of negative negative examples
                    negative_examples += 1

                    # Check if we have enough negative examples
                    if negative_examples >= max_negative:
                        break

                # If we don't have enough negative examples, then we need to
                # create some more negative examples by randomly selecting
                # sentences from other authors
                while negative_examples < max_negative:
                    # Get a random author
                    random_author = random.choice(list(data.keys()))

                    # Make sure that the random author is not the same as the
                    # current author
                    while random_author == author_id:
                        random_author = random.choice(list(data.keys()))

                    # Get a random sentence from the random author
                    random_sentence = random.choice(data[random_author])

                    # Add the pairing to the list
                    pairings.append(
                        (lookup[sentences[i]], lookup[random_sentence], 0))

                    # Increment the count of negative negative examples
                    negative_examples += 1

            # If there are no positive examples, add this as a negative example
            if len(sentences) == 1:
                # Try to find a semantically similar sentence first
                if sentences[i] in paraphrases:
                    # Add the pairing to the list
                    pairings.append(
                        (lookup[sentences[i]],
                         lookup[paraphrases[sentences[i]][0]],
                         0))

                    s_pairings.append(
                        (lookup[sentences[i]],
                         lookup[paraphrases[sentences[i]][0]]))
                else:
                    # Get a random author
                    random_author = random.choice(list(data.keys()))

                    # Make sure that the random author is not the same as the
                    # current author
                    while random_author == author_id:
                        random_author = random.choice(list(data.keys()))

                    # Get a random sentence from the random author
                    random_sentence = random.choice(data[random_author])

                    # Add the pairing to the list
                    pairings.append(
                        (lookup[sentences[i]], lookup[random_sentence], 0))

            data_seen += 1
            print(f" {data_seen}/{len(df)}", end="\r")

    end = time.time()
    print(f"Created {len(pairings)} pairings in {end-start} seconds.")

    print(f"Found {len(s_pairings)} semantic pairings.")

    return pairings, s_pairings
