# > Imports

# Third Party
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Standard Library
import re


def generate_embeddings(data: pd.DataFrame,
                        output_path: str = None,
                        cache_folder: str = ".cache/") -> pd.DataFrame:
    """
    Generate the semantic embeddings for the data.

    Parameters
    ----------
    data : pd.DataFrame
        The data to generate the embeddings for.
    output_path : str
        The path to the output directory.
    cache_folder : str
        The path to the cache directory.

    Returns
    -------
    pd.DataFrame
        The data with the embeddings added.
    """

    # Load model
    print("Loading semantic embedding model...")
    model = SentenceTransformer(
        "all-mpnet-base-v2",
        cache_folder=cache_folder)
    print("Model loaded.")

    # Generate embeddings
    print("Generating embeddings...")
    embeddings = model.encode(data["text"].tolist(), show_progress_bar=True)
    print("Embeddings generated.")

    # Add the embeddings to the DataFrame
    data["embeddings"] = embeddings

    if output_path:
        # Save the DataFrame
        print("Saving data...")
        data.to_pickle(f"{output_path}/data/full_dataset.pkl")
        print("Data saved.")

    return data


def paraphrase_mining(data: pd.DataFrame,
                      output_path: str = None,
                      cache_folder: str = ".cache/") -> pd.DataFrame:
    """
    Use paraphrase mining to find sentences with the same meaning.

    Parameters
    ----------
    data : pd.DataFrame
        The data to mine for paraphrases.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the IDs of the sentences that are paraphrases
        and their similarity score.
    """
    # Check if the paraphrase data already exists
    if output_path:
        try:
            print("Loading paraphrase data...")
            paraphrases = pd.read_pickle(
                f"{output_path}/data/paraphrases_{len(data)}.pkl")
            print("Data loaded.")
            return paraphrases
        except FileNotFoundError:
            pass

    # Load model
    print("Loading paraphrase mining model...")
    model = SentenceTransformer(
        "all-mpnet-base-v2",
        cache_folder=cache_folder)
    print("Model paraphrase mining loaded.")

    # We want to remove all lines from the data["text"] that start with
    # ">" because they are replies to other comments.
    print("Removing replies...")

    # Loop through the data["text"] and remove the lines
    for i, row in data.iterrows():
        # Split the text into lines
        lines = row["text"].splitlines()

        # Remove the lines that start with ">"
        lines = [line for line in lines if not line.startswith(">")]

        # Join the lines back together
        data.at[i, "text"] = "\n".join(lines)

    print("Replies removed.")

    # If all characters constist of emojis, replace it with ""
    print("Removing emojis...")
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)

    # See if all characters (except for whitespace) are emojis
    for i, row in data.iterrows():
        if emoji_pattern.sub(r"", row["text"]).strip() == "":
            data.at[i, "text"] = ""

    print("Emojis removed.")

    # Find paraphrases
    print("Finding paraphrases...")
    paraphrases = util.paraphrase_mining(
        model,
        data["text"].tolist(),
        show_progress_bar=True,
        max_pairs=10000000,
        top_k=100,
    )
    print("Paraphrases found.")

    # Remove paraphrases that are the same sentence
    print("Removing duplicates...")
    paraphrases = [
        (score, i, j)
        for score, i, j in paraphrases
        if data["text"].iloc[i].strip() != data["text"].iloc[j].strip()]
    print("Duplicates removed.")

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
            f"{output_path}/data/paraphrases_{len(data)}.pkl")
        print("Data saved.")

    # Return the paraphrases
    return paraphrases
