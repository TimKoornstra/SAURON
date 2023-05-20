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

    # Print corpus statistics
    print("Corpus statistics:")
    print(f"Number of authors: {len(corpus['author_id'].unique())}")
    print(f"Number of sentences: {len(corpus)}")
    print(
        f"Number of sentences per author: {len(corpus) / len(corpus['author_id'].unique())}")
    print(
        f"Number of conversations: {len(corpus['conversation_id'].unique())}")

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

        df_size = len(df)
        total_authors = df["author_id"].nunique()

        # Remove all RemindMe! comments
        df = df.drop(df[df["text"].str.lower().str.contains(
            r"!?remindme!?", regex=True, case=False)].index)

        print("Removed RemindMe! comments")
        print(
            f"Removed {df_size - len(df)} utterances and {total_authors - df['author_id'].nunique()} authors")
        df_size = len(df)
        total_authors = df["author_id"].nunique()

        # Remove all invalid utterances
        df = df.drop(df[df["text"].str.strip().str.lower().isin(
            ["[ deleted ]", "[deleted]", "[ removed ]", "[removed]", ""])]
            .index
        )

        print("Removed invalid utterances")
        print(
            f"Removed {df_size - len(df)} utterances and {total_authors - df['author_id'].nunique()} authors")
        df_size = len(df)
        total_authors = df["author_id"].nunique()

        # (i.e., utterances that are only mentions, URLs or a combination of
        # mentions and URLs)
        df = df.drop(df[df["text"].str.split().apply(
            lambda x: all(
                word in ["[MENTION]", "[URL]"] for word in x
            )
        )].index)

        print("Removed utterances that are only mentions and/or URLs")
        print(
            f"Removed {df_size - len(df)} utterances and {total_authors - df['author_id'].nunique()} authors")
        df_size = len(df)
        total_authors = df["author_id"].nunique()

        # Remove all utterances from users named "[deleted]", "MTGCardFetcher"
        # or "AutoModerator"
        df = df.drop(
            df[df["author_id"].str.strip().str.lower()
               .isin(["[deleted]", "mtgcardfetcher", "automoderator"])].index)

        print(
            "Removed utterances from users named [deleted], MTGCardFetcher and AutoModerator")
        print(
            f"Removed {df_size - len(df)} utterances and {total_authors - df['author_id'].nunique()} authors")
        df_size = len(df)
        total_authors = df["author_id"].nunique()

        # Remove all utterances from users that are likely bots
        # (i.e. users that have a username that contains "bot" or all of their
        # utterances contain the word "bot")
        df = df.drop(df[
            (df["author_id"].str.lower().str.contains(
                "bot", regex=False, case=False)) |
            (df["text"].str.lower().str.contains(
                "bot", regex=False, case=False))].index)

        print("Removed utterances from users that are likely bots")
        print(
            f"Removed {df_size - len(df)} utterances and {total_authors - df['author_id'].nunique()} authors")

    # Unescape the HTML entities
    df["text"] = df["text"].apply(html.unescape)

    df_size = len(df)
    total_authors = df["author_id"].nunique()

    # Ensure that the texts fit in the maximum length of the RoBERTa model
    tokenizer = RobertaTokenizer.from_pretrained(
        "roberta-base")
    df = df.drop(df[df["text"].progress_apply(
        lambda x: len(tokenizer.encode(x)) > 512)].index)

    print("Removed utterances that are too long")
    print(
        f"Removed {df_size - len(df)} utterances and {total_authors - df['author_id'].nunique()} authors")

    # Take a sample of 10 utterances per author
    df_size = len(df)
    total_authors = df["author_id"].nunique()

    df = df.groupby("author_id").apply(lambda x: x.sample(
        min(len(x), 10), random_state=42)).reset_index(drop=True)

    print("Limited utterances to 10 per author")
    print(
        f"Removed {df_size - len(df)} utterances and {total_authors - df['author_id'].nunique()} authors")

    print(f"Number of utterances: {df.shape[0]}")
    print(f"Number of authors: {df['author_id'].nunique()}")
    print(f"Number of conversations: {df['conversation_id'].nunique()}")
    print(
        f"Number of utterances per author: {df.shape[0] / df['author_id'].nunique()}")

    return df


def anonymize_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Anonymize the given DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to create a subset of.

    Returns
    -------
    pd.DataFrame
        The anonymized DataFrame.
    """
    df2 = df.copy(deep=True)

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
    print("Anonymizing data...")
    df = anonymize_data(df=df)
    print("Data anonymized.")

    if output_path:
        # Save the DataFrame
        print("Saving data...")
        df.to_pickle(f"{output_path}/preprocessed/{data_source}_data.pkl")
        print("Data saved.")

    return df
