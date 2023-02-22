# > Imports

# Third Party
import pandas as pd
from sentence_transformers import SentenceTransformer, util


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
    print("Model semantic embedding loaded.")

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


def paraphrase_mining(data: pd.DataFrame) -> pd.DataFrame:
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

    # Load model
    print("Loading paraphrase mining model...")
    model = SentenceTransformer(
        "all-mpnet-base-v2",
        cache_folder=".cache/")
    print("Model paraphrase mining loaded.")

    # Find paraphrases
    print("Finding paraphrases...")
    paraphrases = util.paraphrase_mining(
        model,
        data["text"].tolist(),
        show_progress_bar=True
    )
    print("Paraphrases found.")

    # Remove paraphrases that are the same sentence
    print("Removing duplicates...")
    paraphrases = [
        (score, i, j)
        for score, i, j in paraphrases
        if data["text"].iloc[i].strip() != data["text"].iloc[j].strip()]
    print("Duplicates removed.")

    # Return a new DataFrame with the paraphrases
    return pd.DataFrame(
        paraphrases,
        columns=["similarity", "idx_1", "idx_2"]
    )
