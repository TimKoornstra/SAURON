# > Imports

# Standard Library
from collections import defaultdict
import gc
import os
import pickle
import random
import time
from typing import Tuple

# Third Party
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import tqdm

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


def _create_pairings(args):
    """
    A helper function to create the pairings for a chunk of the data.

    Parameters
    ----------
    args : Tuple[List[Tuple[str, List[str]]], pd.DataFrame, dict, dict, dict]
        A tuple containing the list of authors, the DataFrame, the lookup
        dictionary, the paraphrases dictionary, and the paraphrase scores
        dictionary.

    Returns
    -------
    Tuple[List[Tuple[str, str, str]], set, dict, dict, dict]
        A tuple containing the list of pairings, the set of conversations,
        the dictionary of author counts, the dictionary of anchor counts,
        the dictionary of paraphrase counts, and the dictionary of
        paraphrase info.
    """
    # Create the temporary variables
    temp_s_pairings = []  # List of semantic pairings
    temp_conversations = set()  # Set of conversations
    # Number of utterances per author in the data
    temp_authors = defaultdict(int)
    temp_count_anchor = defaultdict(int)  # Number of times an anchor is used

    conversation_counts = {
        "same_conv": 0,
        "anchorpos_conv": 0,
        "anchorneg_conv": 0,
        "posneg_conv": 0,
        "none_conv": 0,
    }

    temp_paraphrase_info = defaultdict(int)

    # Unpack the arguments
    authors, data, lookup, paraphrases, paraphrase_scores = args

    # Create a defaultdict for counting paraphrase frequencies
    paraphrase_counts = defaultdict(int)

    # Iterate through each author
    for author_id, sentences in authors:

        # Iterate through each sentence
        for i in range(len(sentences)):
            anchor = lookup[sentences[i]]["text"]
            anchor_conv = lookup[sentences[i]]["conversation_id"]

            # Add the conversation to the set of conversations
            # Take the author_id and the sentence to find the
            # conversation_id in the df
            temp_conversations.add(
                lookup[sentences[i]]["conversation_id"])

            # Count the number of paraphrases used for this anchor
            n_anchor_paraphrases = 0

            # Create a set to track chosen (anchor, paraphrase) pairs
            chosen_pairs = set()

            # Iterate through the rest of the sentences by the same author.
            # We include the current sentence, so that the model can learn
            # that semantically similar is not necessarily a different
            # author.
            for j in range(i+1, len(sentences)):
                # Create a predetermined size list of the examples, so that
                # it contains the anchor, the positive example, and the
                # negative examples
                example = [None] * 3
                pos = lookup[sentences[j]]["text"]
                pos_conv = lookup[sentences[j]]["conversation_id"]

                # Add the anchor and the positive example to the example
                example[0] = anchor
                example[1] = pos

                # Consider the first 10 paraphrases for this anchor
                top_paraphrases = [idx for idx in paraphrases[sentences[i]][:10] if (
                    sentences[i], idx) not in chosen_pairs]

                # If there are no unique paraphrases left, skip this
                if len(top_paraphrases) < 1:
                    break

                # Calculate the weights such that the first paraphrase gets the highest weight
                rank_weights = [1 / (1 + np.sqrt(i))
                                for i in range(len(top_paraphrases))]
                weights = [w / sum(rank_weights)
                           for w in rank_weights]  # Normalize weights

                # Sample a negative example with weighted choice
                idx_2 = random.choices(top_paraphrases, weights=weights)[0]

                # Add the pairing to the list
                example[2] = lookup[idx_2]["text"]

                # Update the frequency count
                paraphrase_counts[lookup[idx_2]["text"]] += 1

                # Add this pair to the set of chosen pairs
                chosen_pairs.add((sentences[i], idx_2))

                temp_paraphrase_info[(anchor, example[2],
                                     paraphrase_scores[(sentences[i], idx_2)])] += 1

                # Increment the number of paraphrases used for this
                # anchor
                n_anchor_paraphrases += 1

                # Add the negative example and the positive example to
                # the set of conversations
                temp_conversations.add(lookup[idx_2]["conversation_id"])
                temp_conversations.add(
                    lookup[sentences[j]]["conversation_id"])

                temp_authors[lookup[idx_2]["author_id"]] += 1

                # Collect conversation info
                neg_conv = lookup[idx_2]["conversation_id"]

                # Collect conversation info
                if anchor_conv == pos_conv:
                    if anchor_conv == neg_conv:
                        conversation_counts["same_conv"] += 1
                    else:
                        conversation_counts["anchorpos_conv"] += 1
                elif anchor_conv == neg_conv:
                    conversation_counts["anchorneg_conv"] += 1
                elif pos_conv == neg_conv:
                    conversation_counts["posneg_conv"] += 1
                else:
                    conversation_counts["none_conv"] += 1

                # Add the example to the list of examples
                temp_s_pairings.append(example)
                temp_authors[author_id] += 2
                temp_count_anchor[sentences[i]] += 1

    return (temp_s_pairings,
            temp_conversations,
            temp_authors,
            temp_count_anchor,
            conversation_counts,
            temp_paraphrase_info)


def create_pairings(df: pd.DataFrame,
                    output_path: str,
                    output_name: str):
    """
    Create the pairings for the data. The pairings are created by taking
    each sentence and finding the n most similar sentences. The most similar
    sentence is the positive example, and the rest are negative examples.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to create the pairings from.
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

    print(f"Paraphrases calculated. {len(paraphrases)} paraphrases found.")

    # Start a timer
    start = time.time()

    # Convert all paraphrases to a defaultdict of lists, where the key is the idx_1
    # and the value is a list of idx_2, sorted by similarity
    paraphrases = paraphrases.sort_values("similarity", ascending=False)
    # paraphrase_scores should be a dictionary of lists,  where the key is (idx_1, idx_2)
    # and the value is the similarity score
    paraphrase_scores = paraphrases.set_index(
        ["idx_1", "idx_2"])["similarity"].to_dict()

    paraphrases = paraphrases.groupby("idx_1")["idx_2"].apply(list).to_dict()
    paraphrases = defaultdict(list, paraphrases)

    # Reset the index of the df
    df["index"] = df.index

    lookup = df.set_index(
        "index")[["text", "conversation_id", "author_id"]].to_dict(orient="index")

    # Convert the df to a dictionary of lists, where the key is the author_id
    # and the value is a list of the indices of sentences written by that author
    # Also filter out authors with less than 2 sentences
    df = df.groupby("author_id").filter(lambda x: len(x) > 1)
    data = df.groupby("author_id")["index"].apply(list).to_dict()

    # Delete the df to free up memory
    del df
    gc.collect()

    print("Time to convert: ", time.time() - start)

    # Get the number of cores
    n_cores = cpu_count()

    print(f"Using {n_cores} cores.")

    # Split the data into n_cores chunks and create a list of chunks
    # We need to make sure that the amount of chunks is exactly n_cores
    # so that we can use all the cores
    data_items = list(data.items())
    chunks = (data_items[i::n_cores] for i in range(n_cores))

    chunk_args = ((chunk, {k: data[k] for k, _ in chunk}, lookup,
                   paraphrases,  paraphrase_scores) for chunk in chunks)

    pairings = []
    conversations = set()
    authors = defaultdict(int)
    count_anchor = defaultdict(int)

    conversation_counts = defaultdict(int)

    paraphrase_info = defaultdict(int)

    # Create a pool of processes
    with Pool(n_cores) as pool:
        progress_bar = tqdm.tqdm(total=n_cores,
                                 desc="Processing chunks", unit="chunk")

        # Flatten the list of lists and separate the pairings and semantic pairings
        for result in pool.imap(_create_pairings, chunk_args):
            pairings.extend(result[0])
            conversations.update(result[1])

            for key, value in result[2].items():
                authors[key] += value

            for key, value in result[3].items():
                count_anchor[key] += value

            for key, value in result[4].items():
                conversation_counts[key] += value

            for key, value in result[5].items():
                paraphrase_info[key] += value

            progress_bar.update(1)

        progress_bar.close()

    # End the timer and print the time it took to create the pairings
    end = time.time()
    print(
        f"Found {len(pairings)} semantic pairings in {end - start:.2f} seconds.")

    # Temp variables for statistics
    author_values = authors.values()
    count_anchor_values = count_anchor.values()

    print("==============================")
    print("       Statistics")
    print("==============================")
    print(f"Number of pairings:          {len(pairings):>5}")
    print(f"Number of authors:           {len(authors):>5}")
    print(f"Number of conversations:     {len(conversations):>5}")
    print(f"Maximum utterances per author: {max(author_values):>5}")
    print(f"Minimum utterances per author: {min(author_values):>5}")
    print(
        f"Average utterances per author: {sum(author_values) / len(authors):>5.2f}")
    print(f"Maximum an anchor occurs:    {max(count_anchor_values):>5}")
    print(f"Minimum an anchor occurs:    {min(count_anchor_values):>5}")
    print(
        f"Average an anchor occurs:    {sum(count_anchor_values) / len(count_anchor):>5.2f}")
    print("-----------------------------")
    print("Conversations is the same for")
    print(f"All:        {conversation_counts['same_conv']:>5}")
    print(f"Anchor-pos: {conversation_counts['anchorpos_conv']:>5}")
    print(f"Anchor-neg: {conversation_counts['anchorneg_conv']:>5}")
    print(f"Pos-neg:    {conversation_counts['posneg_conv']:>5}")
    print(f"None:       {conversation_counts['none_conv']:>5}")
    print("-----------------------------")
    print("==============================")

    # Shuffle the pairings
    random.seed(42)
    random.shuffle(pairings)

    # Create a list for all the used paraphrases
    paraphrase_list = []

    # Add the paraphrases to the list
    for key, value in paraphrase_info.items():
        paraphrase_list.append(
            {"anchor": key[0],
             "paraphrase": key[1],
             "score": key[2],
             "occurrences": value})

    # Convert the list into a DataFrame
    paraphrase_df = pd.DataFrame(paraphrase_list)

    # Save the paraphrase_df
    paraphrase_df.to_pickle(
        f"{output_path}/paired/{output_name}-paraphrases.pkl")

    # Save the pairings
    with open(f"{output_path}/paired/{output_name}-pairings.pkl", "wb") as f:
        pickle.dump(pairings, f)

    return pairings
