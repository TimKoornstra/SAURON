# > Imports
# Standard Library
import html

# Third Party
from convokit import Corpus, download
import pandas as pd
from transformers import RobertaTokenizer
from tqdm import tqdm


def load_reddit_corpus(cache_path: str = ".cache/",
                       corpus_name: str = "reddit-corpus") -> pd.DataFrame:
    """
    Load the Reddit Corpus and return it as a Pandas DataFrame.

    Parameters
    ----------
    cache_path : str
        The path to the cache directory.
    corpus_name : str
        The name of the corpus.

    Returns
    -------
    pd.DataFrame
        The Reddit Corpus as a Pandas DataFrame.
    """
    # Download the corpus
    print(f"Downloading the {corpus_name} corpus...")
    file = download(corpus_name, data_dir=f"{cache_path}/{corpus_name}")
    print("Done!")

    # Load the corpus
    print("Loading the corpus...")
    corpus = Corpus(filename=file)
    print("Done!")

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


def preprocess(df: pd.DataFrame,
               data_source: str = "reddit") -> pd.DataFrame:
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
    pd.DataFrame
        The preprocessed DataFrame.
    """
    tqdm.pandas()

    if data_source == "reddit":
        # Replace mentions with the "[MENTION]" token
        df["text"] = df["text"].str.replace(
            r"\/?u\/[A-Za-z0-9_-]+|\/?r\/[A-Za-z0-9_]+", "[MENTION]",
            regex=True, case=False)

        # Replace all the URLs in parentheses with the "[URL]" token
        df["text"] = df["text"].str.replace(
            r"\((https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})",
            "([URL])", regex=True, case=False
        )
        # And replace all the URLs not in parentheses with the "[URL]" token
        df["text"] = df["text"].str.replace(
            r"(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})",
            "[URL]", regex=True, case=False
        )

        # Remove all RemindMe! comments
        df = df.drop(df[df["text"].str.lower().str.contains(
            r"!?remindme!?", regex=True, case=False)].index)

        # Remove all invalid utterances
        df = df.drop(df[df["text"].str.strip().str.lower().isin(
            ["[ deleted ]", "[deleted]", "[ removed ]", "[removed]", ""])]
            .index
        )

        # (i.e., utterances that are only mentions, URLs or a combination of
        # mentions and URLs)
        df = df.drop(df[df["text"].str.split().apply(
            lambda x: all(
                word in ["[MENTION]", "[URL]"] for word in x
            )
        )].index)

        # Remove all utterances from users named "[deleted]", "MTGCardFetcher"
        # or "AutoModerator"
        df = df.drop(
            df[df["author_id"].str.strip().str.lower()
               .isin(["[deleted]", "mtgcardfetcher", "automoderator"])].index)

        # Remove all utterances from users that are likely bots
        # (i.e. users that have a username that contains "bot" or all of their
        # utterances contain the word "bot")
        df = df.drop(df[
            (df["author_id"].str.lower().str.contains(
                "bot", regex=False, case=False)) |
            (df["text"].str.lower().str.contains(
                "bot", regex=False, case=False))].index)

    # Remove all utterances from authors that have only one utterance
    df = df.drop(df.groupby("author_id").filter(
        lambda x: x.shape[0] == 1).index)

    # Unescape the HTML entities
    df["text"] = df["text"].apply(html.unescape)

    # Ensure that the texts fit in the maximum length of the RoBERTa model
    tokenizer = RobertaTokenizer.from_pretrained(
        "roberta-base")
    df = df.drop(df[df["text"].progress_apply(
        lambda x: len(tokenizer.encode(x)) > 512)].index)

    print(f"Number of utterances: {df.shape[0]}")

    return df


def create_subset(df: pd.DataFrame,
                  n: int,
                  group: str = "subreddit") -> pd.DataFrame:
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
    pd.DataFrame
        The subset of the given DataFrame.
    """
    # Create a subset of the DataFrame
    # Count the number of groups
    n_groups = df[group].value_counts().shape[0]

    # Take the first m items from each group such that the total number of items is n
    m = n // n_groups

    # Create the subset
    df2 = df.groupby("conversation_id").head(
        10).groupby(group).head(m).copy(deep=True)

    # Create new IDs for the authors and conversations
    df2.loc[:, "author_id"] = df2.groupby("author_id").ngroup()
    df2.loc[:, "conversation_id"] = df2.groupby("conversation_id").ngroup()

    # Reset the index
    df2 = df2.reset_index(drop=True)

    print(
        f"Collected {df2.shape[0]} utterances from {df2['author_id'].nunique()} authors in {df2['conversation_id'].nunique()} conversations.")

    return df2


def load_data(source: str,
              path: str) -> pd.DataFrame:
    """
    Load the DataFrame from the given path.

    Parameters
    ----------
    path : str
        The path to load the DataFrame from.
    name : str
        The name of the DataFrame.

    Returns
    -------
    pd.DataFrame
        The DataFrame loaded from the given path.
    """
    return pd.read_pickle(f"{path}/preprocessed/{source}_data.pkl")


def pipeline(data_source: str = "reddit",
             output_path: str = None,
             cache_path: str = ".cache/") -> pd.DataFrame:
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
    pd.DataFrame
        The preprocessed DataFrame.
    """

    print("Loading data...")
    # If the data source is Reddit, load the Reddit Corpus
    if data_source == "reddit":
        df = load_reddit_corpus(cache_path=cache_path)
    else:
        raise ValueError(f"Invalid data source: {data_source}")
    print("Data loaded.")

    # Preprocess the DataFrame
    print("Preprocessing data...")
    df = preprocess(df=df,
                    data_source=data_source)
    print("Data preprocessed.")

    # Create a subset of the DataFrame
    print("Creating subset...")
    df = create_subset(df=df,
                       n=1000000)
    print("Subset created.")

    if output_path:
        # Save the DataFrame
        print("Saving data...")
        df.to_pickle(f"{output_path}/preprocessed/{data_source}_data.pkl")
        print("Data saved.")

    return df
