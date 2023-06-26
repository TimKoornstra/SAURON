# > Imports

# Third Party
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Standard Library
import os
import re


def load_model(cache_folder: str = ".cache/") -> SentenceTransformer:
    """
    Load the paraphrase mining model.

    Parameters
    ----------
    cache_folder : str
        The folder to cache the model in.

    Returns
    -------
    SentenceTransformer
        The model.
    """

    print("Loading paraphrase mining model...")
    model = SentenceTransformer("all-mpnet-base-v2", cache_folder=cache_folder)
    print("Model paraphrase mining loaded.")
    return model


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Remove bad potential paraphrases from the data.

    Parameters
    ----------
    data : pd.DataFrame
        The data to clean.

    Returns
    -------
    pd.DataFrame
        The cleaned data.
    """

    print("Removing bad potential paraphrases...")
    count = 0
    for i, row in data.iterrows():
        if should_remove(row["text"]):
            data.at[i, "text"] = ""
            count += 1
    print(f"Disregarded {count} rows out of {len(data)}")
    print("Removed.")
    return data


def should_remove(text: str) -> bool:
    """
    Check if a text should be removed.

    Parameters
    ----------
    text : str
        The text to check.

    Returns
    -------
    bool
        Whether the text should be removed.
    """

    lines = text.splitlines()
    stripped = text.strip()
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)

    return any(line.startswith(">") for line in lines)\
        or emoji_pattern.sub(r"", stripped).strip() == "" or stripped == "."


def paraphrase_mining(data: pd.DataFrame,
                      output_path: str = None,
                      output_name: str = "",
                      cache_folder: str = ".cache/") -> pd.DataFrame:
    """
    Use paraphrase mining to find sentences with the same meaning.

    Parameters
    ----------
    data : pd.DataFrame
        The data to mine for paraphrases.
    output_path : str, optional
        The path to save the paraphrases to, by default None
    output_name : str, optional
        The name of the output file, by default ""
    cache_folder : str, optional
        The folder to cache the model in, by default ".cache/"

    Returns
    -------
    pd.DataFrame
        A DataFrame with the IDs of the sentences that are paraphrases
        and their similarity score.
    """

    if output_path and os.path.exists(f"{output_path}/paraphrases/{output_name}-paraphrases.pkl"):
        print("Loading paraphrase data...")
        paraphrases = pd.read_pickle(
            f"{output_path}/paraphrases/{output_name}-paraphrases.pkl")
        print("Data loaded.")
        return paraphrases

    model = load_model(cache_folder)
    data = clean_data(data)

    mapping = {i: new_index for new_index, i in enumerate(data.index)}
    texts = [text for text in data["text"].tolist() if text.strip() != ""]
    mapping = {new_index: mapping[old_index] for new_index, old_index in enumerate(
        i for i, text in enumerate(data["text"].tolist()) if text.strip() != "")}

    # Find paraphrases
    print("Finding paraphrases...")
    all_paraphrases = util.paraphrase_mining(
        model,
        texts,
        show_progress_bar=True,
        max_pairs=100000000,
        top_k=100,
        batch_size=512
    )
    print("Paraphrases found.")
    print(len(all_paraphrases))

    # Map the indices in all_paraphrases to their original values
    all_paraphrases_mapped = [(score, mapping[i], mapping[j])
                              for score, i, j in all_paraphrases]

    # Remove paraphrases that are the same sentence or from the same author
    print("Removing duplicates...")
    already_seen = set()
    paraphrases = []
    for score, i, j in all_paraphrases_mapped:
        if data["text"].iloc[i].strip() != data["text"].iloc[j].strip() and\
           data["author_id"].iloc[i] != data["author_id"].iloc[j]:
            # Ensure that we only add the same i, paraphrase once to combat
            # the oversampling
            if (i, data["text"].iloc[j].strip()) not in already_seen:
                paraphrases.append((score, i, j))
                already_seen.add((i, data["text"].iloc[j].strip()))
    print("Duplicates removed.")
    print(len(paraphrases))

    # Convert the paraphrases to a DataFrame
    paraphrases = pd.DataFrame(
        paraphrases,
        columns=["similarity", "idx_1", "idx_2"]
    )

    print(f"Average similarity: {paraphrases['similarity'].mean():.2f}")

    # Save the DataFrame to a pickle file
    if output_path:
        print("Saving paraphrase data...")
        paraphrases.to_pickle(
            f"{output_path}/paraphrases/{output_name}-paraphrases.pkl")
        print("Data saved.")

    # Return the paraphrases
    return paraphrases
