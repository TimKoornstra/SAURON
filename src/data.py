# > Imports

# Third Party
from convokit import Corpus, download
import pandas as pd


def load_reddit_corpus(output_path: str = ".cache/"):
    """
    Load the Reddit Corpus and return it as a Pandas DataFrame.

    Returns
    -------
    pd.DataFrame : The Reddit Corpus as a Pandas DataFrame.

    """
    # Download the corpus
    corpus = Corpus(
        download("reddit-corpus", data_dir=f"{output_path}/reddit-corpus"))

    # Convert the corpus to a Pandas DataFrame
    corpus = corpus.get_utterances_dataframe()

    # Take only the columns we need
    corpus = corpus[["speaker", "conversation_id",
                     "text", "meta.subreddit"]]

    # Rename the columns
    corpus = corpus.rename(
        columns={
            "id": "utterance_id",
            "speaker": "author_id",
            "meta.subreddit": "subreddit",
        },
    )

    return corpus


def preprocess(df: pd.DataFrame, data_source: str = "reddit"):
    """
    Preprocess the given DataFrame based on the data source.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to preprocess.
    data_source : str
        The data source of the DataFrame.

    Returns
    -------
    pd.DataFrame : The preprocessed DataFrame.

    """
    if data_source == "reddit":
        # Replace mentions with the "[MENTION]" token
        df["text"] = df["text"].str.replace(
            r"\/?u\/[A-Za-z0-9_-]+|\/?r\/[A-Za-z0-9_]+", "[MENTION]",
            regex=True, case=False)

        # Replace all URLs with the "[URL]" token
        df["text"] = df["text"].str.replace(
            r"(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})",
            "[URL]", regex=True, case=False
        )
        # Also do the same for URLs in parentheses
        df["text"] = df["text"].str.replace(
            r"\((https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})",
            "([URL])", regex=True, case=False
        )

        # Remove all invalid utterances
        df = df.drop(df[df["text"].str.strip().str.lower().isin(
            ["[ deleted ]", "[deleted]", "[ removed ]", "[removed]",
             "[mention]", "[url]", ""])].index
        )

        # Remove all utterances from users named "[deleted]"
        df = df.drop(df[df["author_id"].str.lower() == "[deleted]"].index)

    return df


def create_subset(df: pd.DataFrame, n: int, group: str = "subreddit"):
    """
    Create a subset of the given DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to create a subset of.
    n : int
        The number of items in the subset.
    group : str
        The column to group by.

    Returns
    -------
    pd.DataFrame : The subset of the given DataFrame.

    """
    # Create a subset of the DataFrame
    # Count the number of groups
    n_groups = df[group].value_counts().shape[0]

    # Take the first m items from each group such that the total number of items is n
    m = n // n_groups

    # Create the subset
    df2 = df.groupby(group).head(m).copy(deep=True)

    # Create new IDs for the authors and conversations
    df2.loc[:, "author_id"] = df2.groupby("author_id").ngroup()
    df2.loc[:, "conversation_id"] = df2.groupby("conversation_id").ngroup()

    # Reset the index
    df2 = df2.reset_index(drop=True)

    return df2


def load_data(path: str):
    """
    Load the DataFrame from the given path.

    Parameters
    ----------
    path : str
        The path to load the DataFrame from.

    Returns
    -------
    pd.DataFrame : The DataFrame loaded from the given path.

    """
    return pd.read_pickle(path)


def pipeline(data_source: str = "reddit",
             output_path: str = None,
             cache_path: str = ".cache/"):
    """
    Run the preprocessing pipeline on the given DataFrame.

    Parameters
    ----------
    data_source : str
        The data source of the DataFrame.
    save : bool
        Whether to save the DataFrame.

    Returns
    -------
    pd.DataFrame : The preprocessed DataFrame.

    """

    print("Loading data...")
    # If the data source is Reddit, load the Reddit Corpus
    if data_source == "reddit":
        df = load_reddit_corpus(cache_path)
    else:
        raise ValueError(f"Invalid data source: {data_source}")
    print("Data loaded.")

    # Preprocess the DataFrame
    print("Preprocessing data...")
    df = preprocess(df, data_source)
    print("Data preprocessed.")

    # Create a subset of the DataFrame
    print("Creating subset...")
    df = create_subset(df, 1000000)
    print("Subset created.")

    if output_path:
        # Save the DataFrame
        print("Saving data...")
        df.to_pickle(f"{output_path}/data/{data_source}.pkl")
        print("Data saved.")

    return df
