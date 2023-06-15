# > Imports

# Third Party
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Standard Library
import re


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
                f"{output_path}/paraphrases/{output_name}-paraphrases.pkl")
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

    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)

    # We want to remove all lines from the data["text"] that start with
    # ">" because they are replies to other comments.
    print("Removing bad potential paraphrases...")
    count = 0

    # Loop through the data["text"] and remove the lines
    for i, row in data.iterrows():
        # Split the text into lines
        lines = row["text"].splitlines()
        stripped = row["text"].strip()

        # Check if any line starts with ">"
        if any(line.startswith(">") for line in lines)\
                or emoji_pattern.sub(r"", stripped).strip() == ""\
                or stripped == ".":
            # Remove the entire text
            data.at[i, "text"] = ""
            count += 1
            print(f"Disregarding {count} rows out of {len(data)}", end="\r")

    print("Removed.")

    # Find paraphrases
    print("Finding paraphrases...")
    all_paraphrases = util.paraphrase_mining(
        model,
        data["text"].tolist(),
        show_progress_bar=True,
        max_pairs=100000000,
        top_k=100,
    )
    print("Paraphrases found.")
    print(len(all_paraphrases))

    # Remove paraphrases that are the same sentence or from the same author
    print("Removing duplicates...")
    already_seen = set()
    paraphrases = []
    for score, i, j in all_paraphrases:
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
