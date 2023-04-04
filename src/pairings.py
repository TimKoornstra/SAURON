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


def split_data(df: pd.DataFrame,
               train_size: float = 0.8)\
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
    random.seed(42)

    # Split the data into train and test sets
    train_test_splitter = GroupShuffleSplit(n_splits=1,
                                            test_size=1-train_size,
                                            random_state=42)

    train_idx, test_idx = next(train_test_splitter.split(df,
                                                         groups=df["author_id"]))

    train, test = df.iloc[train_idx], df.iloc[test_idx]

    # Split the test set into validation and test sets (50/50)
    val_test_splitter = GroupShuffleSplit(n_splits=1,
                                          test_size=0.5,
                                          random_state=42)

    val_idx, test_idx = next(val_test_splitter.split(test,
                                                     groups=test["author_id"]))

    val, test = test.iloc[val_idx], test.iloc[test_idx]

    return train, val, test


def create_pairings(df: pd.DataFrame,
                    output_path: str,
                    output_name: str,
                    max_negative: int = 1,
                    semantic_range: Tuple[float, float] = (0.7, 1.0),
                    semantic_proportion: float = 0.5) -> List[List[str]]:
    """
    Create the pairings for the data. The pairings are created by taking
    each sentence and finding the n most similar sentences. The most similar
    sentence is the positive example, and the rest are negative examples.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to create the pairings from.
    n_negative_examples : int
        The number of negative examples to create for each positive example.
    semantic_range : Tuple[float, float]
        The inclusive range of semantic similarity to use for semantic filtering.
    semantic_proportion : float
        The proportion of the pairings to be semantic.
    output_path : str
        The path to save the pairings to.
    output_name : str
        The name of the output file.

    Returns
    -------
    List[List[str]]
        The list of pairings.
    """
    # Check if the pairings have already been created
    try:
        print("Loading pairings locally...")
        with open(f"{output_path}/paired/{output_name}-pairings.pkl", "rb") as f:
            regular_pairings = pickle.load(f)
        print("Pairings loaded.")
        return regular_pairings
    except FileNotFoundError:
        print("Pairings not found. Calculating...")

    # Reset the index, so that the index is the same as the row number
    df = df.reset_index(drop=True)

    # First, we need to calculate the semantic similarity between each
    # pair of sentences.
    try:
        print("Loading paraphrases locally...")
        with open(f"{output_path}/paraphrases/{output_name}-paraphrases.pkl", "rb") as f:
            paraphrases = pickle.load(f)
        print("Paraphrases loaded.")
    except FileNotFoundError:
        print("Paraphrases not found. Calculating...")
        paraphrases = paraphrase_mining(df,
                                        output_path=output_path,
                                        output_name=output_name)

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

    # Reset the index of the df
    df["index"] = df.index

    # Create a dictionary where the key is the index and the value is the conversation_id
    conversation_lookup = df.set_index("index")["conversation_id"].to_dict()

    # Create a lookup table for the indices of the sentences
    lookup = df.set_index("index").to_dict()["text"]

    # Convert the df to a dictionary of lists, where the key is the author_id
    # and the value is a list of the indices of sentences written by that author
    # Also filter out authors with less than 2 sentences
    df = df.groupby("author_id").filter(lambda x: len(x) > 1)
    data = df.groupby("author_id")["index"].apply(list).to_dict()
    data_keys = list(data.keys())

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
        A helper function to create the pairings for a chunk of the data.

        Parameters
        ----------
        authors : List[Tuple[int, List[int]]]
            The chunk of the data containing the author_id and the indices of
            the sentences written by that author.

        Returns
        -------
        Tuple[List[List[str]], List[Tuple[str, str]]]
            A tuple containing the list of all pairings and the list of
            semantic pairings.
        """
        # Temp list to store the pairings
        temp_pairings = []
        temp_s_pairings = []
        temp_conversations = set()

        # Iterate through each author
        for author_id, sentences in authors:

            # Iterate through each sentence
            for i in range(len(sentences)):
                anchor = lookup[sentences[i]]

                # Add the conversation to the set of conversations
                # Take the author_id and the sentence to find the
                # conversation_id in the df
                temp_conversations.add(
                    conversation_lookup[sentences[i]])

                # Count the number of paraphrases used for this anchor
                n_anchor_paraphrases = 0

                # Iterate through the rest of the sentences by the same author.
                # We include the current sentence, so that the model can learn
                # that semantically similar is not necessarily a different
                # author.
                for j in range(i+1, len(sentences)):
                    # Create a predetermined size list of the examples, so that
                    # it contains the anchor, the positive example, and the
                    # negative examples
                    example = [None] * (max_negative + 2)
                    pos = lookup[sentences[j]]

                    # Add the anchor and the positive example to the example
                    example[0] = anchor
                    example[1] = pos

                    # Set the number of negative examples for this positive
                    # example to 0 and a flag to check if all the negative
                    # examples are semantically similar
                    n_neg = 0
                    all_similar = False

                    for k in range(n_anchor_paraphrases, len(paraphrases[sentences[i]])):
                        # Get the index of the paraphrase
                        idx_2 = paraphrases[sentences[i]][k]

                        # Add the pairing to the list
                        example[n_neg + 2] = lookup[idx_2]

                        # Increment the number of negative examples
                        n_neg += 1

                        # Increment the number of paraphrases used for this
                        # anchor
                        n_anchor_paraphrases += 1

                        # If we have enough negative examples, break
                        if n_neg >= max_negative:
                            all_similar = True
                            break

                        # Add the negative example and the positive example to
                        # the set of conversations
                        temp_conversations.add(conversation_lookup[idx_2])

                        temp_conversations.add(
                            conversation_lookup[sentences[j]])

                    # If we do not have enough negative examples, we need to
                    # create some random negative examples
                    while n_neg < max_negative:
                        # Get a random sentence from a random author
                        # Make sure that the random author is not the same as
                        # current author
                        while (random_author := random.choice(data_keys)) == author_id:
                            continue

                        random_sentence = random.choice(data[random_author])

                        # Add the pairing to the list
                        example[n_neg + 2] = lookup[random_sentence]

                        # Increment the number of negative examples
                        n_neg += 1

                    # Add the example to the list of examples
                    if all_similar:
                        temp_s_pairings.append(example)
                    else:
                        temp_pairings.append(example)

        return (temp_pairings, temp_s_pairings, temp_conversations)

    # Create a pool of processes
    with Pool(n_cores) as pool:
        results = pool.map(_create_pairings, chunks)

    semantic_pairings = []
    regular_pairings = []
    conversations = set()

    # Flatten the list of lists and separate the pairings and semantic pairings
    for result in results:
        regular_pairings += result[0]
        semantic_pairings += result[1]
        conversations = conversations.union(result[2])

    # End the timer and print the time it took to create the pairings
    end = time.time()
    print(
        f"Created {len(regular_pairings) + len(semantic_pairings)} pairings in {end-start} seconds.")

    print(f"Found {len(semantic_pairings)} semantic pairings.")

    # Combine the pairings and the semantic pairings such that the semantic
    # pairings are first in the list and the regular pairings are second,
    # following the semantic_proportion constraint
    n_regular = 0  # int(((1 - semantic_proportion) *
    #      len(semantic_pairings)) / semantic_proportion)

    print(
        f"Using {len(semantic_pairings)} semantic pairings and {min(n_regular,len(regular_pairings))} regular pairings from {len(conversations)} conversations.")

    # Combine the pairings and the semantic pairings
    pairings = semantic_pairings + regular_pairings[:n_regular]

    # Shuffle the pairings
    random.seed(42)
    random.shuffle(pairings)

    # Save the pairings
    with open(f"{output_path}/paired/{output_name}-pairings.pkl", "wb") as f:
        pickle.dump(pairings, f)
    with open(f"{output_path}/paired/{output_name}-s_pairings.pkl", "wb") as f:
        pickle.dump(semantic_pairings, f)

    return pairings
