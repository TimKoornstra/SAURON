# > Imports
# Standard Library
import html

# Third Party
from convokit import Corpus, download
import dask.dataframe as dd
import pandas as pd
from transformers import RobertaTokenizer


def load_corpus(cache_path: str = ".cache/",
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

    return corpus


def replace_mentions_and_urls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace mentions and URLs in the given DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to replace mentions and URLs in.

    Returns
    -------
    pd.DataFrame
        The DataFrame with mentions and URLs replaced.
    """

    df["text"] = df["text"].replace(
        r"\/?u\/[A-Za-z0-9_-]+|\/?r\/[A-Za-z0-9_]+", "[MENTION]", regex=True)
    df["text"] = df["text"].str.replace(
        r"\((https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})",
        "([URL])", regex=True, case=False
    )
    df["text"] = df["text"].str.replace(
        r"(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})",
        "[URL]", regex=True, case=False
    )
    return df


def remove_invalid_utterances(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove invalid utterances from the given DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to remove invalid utterances from.

    Returns
    -------
    pd.DataFrame
        The DataFrame with invalid utterances removed.
    """

    df = df[~df["text"].str.lower().str.contains("remindme")]
    df = df[~df["text"].str.lower().isin(["[deleted]", "[removed]", ""])]
    df = df[~df["text"].str.split().apply(
        lambda x: set(x).issubset({"[MENTION]", "[URL]"}))]
    df = df[~df["author_id"].str.lower().isin(
        ["[deleted]", "mtgcardfetcher", "automoderator"])]
    df = df[~(df["author_id"].str.lower().str.contains(
        "bot") | df["text"].str.contains("bot"))]
    return df


def remove_long_utterances(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove long utterances from the given DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to remove long utterances from.

    Returns
    -------
    pd.DataFrame
        The DataFrame with long utterances removed.
    """

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    encoded_texts = tokenizer.batch_encode_plus(
        df['text'].tolist(), truncation=False, padding=False)
    long_texts_indices = [i for i, input_ids in enumerate(
        encoded_texts['input_ids']) if len(input_ids) > 512]
    df = df.iloc[[i for i in range(len(df)) if i not in long_texts_indices]]
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the Reddit Corpus.

    Parameters
    ----------
    df : pd.DataFrame
        The Reddit Corpus as a Pandas DataFrame.

    Returns
    -------
    pd.DataFrame
        The preprocessed Reddit Corpus as a Pandas DataFrame.
    """
    # Replace mentions and URLs
    df = replace_mentions_and_urls(df)

    # Remove invalid utterances
    df = remove_invalid_utterances(df)

    # Unescape the HTML entities
    df["text"] = df["text"].apply(html.unescape)

    # Remove utterances that are too long
    df = remove_long_utterances(df)

    ddf = dd.from_pandas(df, npartitions=80)

    # Take a sample of 10 utterances per author using Dask
    def sample(group):
        return group.sample(min(len(group), 10), random_state=42)

    ddf = ddf.groupby("author_id").apply(
        sample, meta=ddf).reset_index(drop=True)

    # Convert the Dask DataFrame back to a pandas DataFrame
    df = ddf.compute()

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

    return df2


def load_data(source: str,
              path: str) -> pd.DataFrame:
    """
    Load the DataFrame from the given path.

    Parameters
    ----------
    source : str
        The data source of the DataFrame.
    path : str
        The path to load the DataFrame from.

    Returns
    -------
    pd.DataFrame
        The DataFrame loaded from the given path.
    """
    return pd.read_pickle(f"{path}/preprocessed/{source}_data.pkl")


def pipeline(output_path: str = None,
             cache_path: str = ".cache/") -> pd.DataFrame:
    """
    Run the preprocessing pipeline on the given DataFrame.

    Parameters
    ----------
    output_path : str, optional
        The path to save the preprocessed DataFrame to, by default None
    cache_path : str, optional
        The path to the cache directory, by default ".cache/"

    Returns
    -------
    pd.DataFrame
        The preprocessed DataFrame.
    """

    print("Loading data...")
    corpus = load_corpus(cache_path=cache_path)
    df = corpus.get_utterances_dataframe()
    print("Data loaded.")

    # Preprocess the DataFrame
    print("Preprocessing data...")
    df = preprocess(df=df)
    print("Data preprocessed.")

    # Create a subset of the DataFrame
    print("Anonymizing data...")
    df = anonymize_data(df=df)
    print("Data anonymized.")

    if output_path:
        # Save the DataFrame
        print("Saving data...")
        df.to_pickle(f"{output_path}/preprocessed/reddit_data.pkl")
        print("Data saved.")

    return df
