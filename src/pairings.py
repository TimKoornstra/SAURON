# > Imports

# Standard Library
from collections import defaultdict
import os
import pickle
import random
import time
from typing import List, Tuple

# Third Party
from multiprocess import Pool, cpu_count
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# Local
from semantics import paraphrase_mining

os.environ["TOKENIZERS_PARALLELISM"] = "false"


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
                    semantic_range: Tuple[float, float] = (0.95, 0.99),
                    output_path: str = None,
                    output_name: str = "")\
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
    output_path : str
        The path to save the pairings to.
    output_name : str
        The name of the output file.

    Returns
    -------
    List[Tuple[str, str, int]]
        The list of pairings.
    """
    # Check if the pairings have already been created
    if output_path:
        try:
            print("Loading pairings locally...")
            with open(f"{output_path}/paired/{output_name}-pairings.pkl", "rb") as f:
                pairings = pickle.load(f)
            print("Pairings loaded.")
            return pairings
        except FileNotFoundError:
            print("Pairings not found. Calculating...")

    s_pairings = []
    pairings = []

    # Reset the index, so that the index is the same as the row number
    df = df.reset_index(drop=True)

    # First, we need to calculate the semantic similarity between each
    # pair of sentences.
    if output_path:
        try:
            print("Loading paraphrases locally...")
            with open(f"{output_path}/paraphrases/{output_name}-paraphrases.pkl", "rb") as f:
                paraphrases = pickle.load(f)
            print("Paraphrases loaded.")
        except FileNotFoundError:
            print("Paraphrases not found. Calculating...")
            paraphrases = paraphrase_mining(
                df, output_path=output_path, output_name=output_name)
    else:
        paraphrases = paraphrase_mining(
            df, output_path=output_path, output_name=output_name)

    print(f"Paraphrases before semantic filtering: {paraphrases.shape[0]}")

    # Filter the paraphrases to only include those that are within the
    # semantic range
    paraphrases = paraphrases[(paraphrases["similarity"] >= semantic_range[0]) &
                              (paraphrases["similarity"] <= semantic_range[1])]

    print(f"Paraphrases after semantic filtering: {paraphrases.shape[0]}")

    # Start a timer
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
    data_keys = list(data.keys())

    # Create a lookup table for the indices of the sentences
    lookup = df.set_index("index").to_dict()["text"]

    print("Time to convert: ", time.time() - start)

    # Get the number of cores
    n_cores = cpu_count()

    print(f"Using {n_cores} cores.")

    # Split the data into n_cores chunks and create a list of chunks
    # We need to make sure that the amount of chunks is exactly n_cores
    # so that we can use all the cores
    data_items = list(data.items())
    chunks = [data_items[i::n_cores] for i in range(n_cores)]

    def _create_pairings(authors):
        """
        Create the pairings for a chunk of the data.

        Parameters
        ----------
        authors : List[Tuple[int, List[int]]]
            The chunk of the data containing the author_id and the indices of
            the sentences written by that author.

        Returns
        -------
        Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int]]]
            A tuple containing the list of all pairings and the list of
            semantic pairings.
        """
        # Temp list to store the pairings
        temp_pairings = []
        temp_s_pairings = []

        for author_id, sentences in authors:
            # Iterate through each sentence
            for i in range(len(sentences)):
                # Create the positive examples
                # Find the unique pairings of sentences
                neg_pairings = set()

                for j in range(i+1, len(sentences)):
                    # Add the pairing to the list
                    temp_pairings.append(
                        (lookup[sentences[i]], lookup[sentences[j]], 1))

                    n_neg = 0

                    for k in range(len(neg_pairings), len(paraphrases[sentences[i]])):
                        # Get the index of the paraphrase
                        idx_2 = paraphrases[sentences[i]][k]

                        # If the paraphrase is not by the same author, add it
                        if idx_2 not in sentences:
                            # Add the pairing to the list
                            temp_pairings.append(
                                (lookup[sentences[i]], lookup[idx_2], 0))
                            temp_s_pairings.append(
                                (lookup[sentences[i]], lookup[idx_2]))

                            # Add the pairing to the set of negative pairings
                            neg_pairings.add((sentences[i], idx_2))

                            # Increment the number of negative examples
                            n_neg += 1

                            # If we have enough negative examples, break
                            if n_neg >= max_negative:
                                break

                    # If we do not have enough negative examples, we need to
                    # create some random negative examples
                    while n_neg < max_negative:
                        # Get a random sentence from a random author
                        random_author = random.choice(data_keys)

                        # Make sure that the random author is not the same as
                        # current author
                        while random_author == author_id:
                            random_author = random.choice(data_keys)

                        random_sentence = random.choice(data[random_author])

                        # Add the pairing to the list
                        temp_pairings.append(
                            (lookup[sentences[i]], lookup[random_sentence], 0))

                        # Increment the number of negative examples
                        n_neg += 1

        return (temp_pairings, temp_s_pairings)

    # Create a pool of processes
    with Pool(n_cores) as pool:
        # Use tqdm to show progress
        results = pool.map(_create_pairings, chunks)

    # Flatten the list of lists and separate the pairings and semantic pairings
    for result in results:
        pairings += result[0]
        s_pairings += result[1]

    print(f"Time to create pairings: {time.time() - start}")

    end = time.time()
    print(f"Created {len(pairings)} pairings in {end-start} seconds.")

    print(f"Found {len(s_pairings)} semantic pairings.")

    # Save the pairings
    if output_path:
        with open(f"{output_path}/paired/{output_name}-pairings.pkl", "wb") as f:
            pickle.dump(pairings, f)
        with open(f"{output_path}/paired/{output_name}-s_pairings.pkl", "wb") as f:
            pickle.dump(s_pairings, f)

    return pairings
