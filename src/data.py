# > Imports
# Standard Library
import html

# Third Party
from convokit import Corpus, download
import dask.dataframe as dd
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
    tqdm.pandas()

    def print_stats(df, df_size, total_authors):
        removed_utterances = df_size - len(df)
        removed_authors = total_authors - df['author_id'].nunique()
        print(
            f"Removed {removed_utterances} utterances and {removed_authors} authors")
        return len(df), df['author_id'].nunique()

    # Replace mentions and URLs with the appropriate tokens
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

    df_size, total_authors = len(df), df["author_id"].nunique()

    df = df[~df["text"].str.lower().str.contains("remindme")]
    print("Removed RemindMe! comments")
    df_size, total_authors = print_stats(df, df_size, total_authors)

    # Removed all invalid utterances
    df = df[~df["text"].str.lower().isin(["[deleted]", "[removed]", ""])]
    print("Removed invalid utterances")
    df_size, total_authors = print_stats(df, df_size, total_authors)

    df = df[~df["text"].str.split().apply(
        lambda x: set(x).issubset({"[MENTION]", "[URL]"}))]
    print("Removed utterances that are only mentions and/or URLs")
    df_size, total_authors = print_stats(df, df_size, total_authors)

    df = df[~df["author_id"].str.lower().isin(
        ["[deleted]", "mtgcardfetcher", "automoderator"])]
    print(
        "Removed utterances from users named [deleted], MTGCardFetcher and AutoModerator")
    df_size, total_authors = print_stats(df, df_size, total_authors)

    df = df[~(df["author_id"].str.lower().str.contains(
        "bot") | df["text"].str.contains("bot"))]
    print("Removed utterances from users that are likely bots")
    df_size, total_authors = print_stats(df, df_size, total_authors)

    # Unescape the HTML entities
    df["text"] = df["text"].apply(html.unescape)

    df_size, total_authors = len(df), df["author_id"].nunique()

    # Ensure that the texts fit in the maximum length of the RoBERTa model
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    # Batch encode the text data
    encoded_texts = tokenizer.batch_encode_plus(
        df['text'].tolist(), truncation=False, padding=False)

    # Find the indices of texts that are longer than the max length
    long_texts_indices = [i for i, input_ids in enumerate(
        encoded_texts['input_ids']) if len(input_ids) > 512]
    # Drop the rows with these indices from the dataframe
    df = df.iloc[[i for i in range(len(df)) if i not in long_texts_indices]]

    print("Removed utterances that are too long")
    df_size, total_authors = print_stats(df, df_size, total_authors)

    ddf = dd.from_pandas(df, npartitions=80)

    # Take a sample of 10 utterances per author using Dask
    def sample(group):
        return group.sample(min(len(group), 10), random_state=42)

    ddf = ddf.groupby("author_id").apply(
        sample, meta=ddf).reset_index(drop=True)

    # Convert the Dask DataFrame back to a pandas DataFrame
    df = ddf.compute()

    print("Limited utterances to 10 per author")
    df_size, total_authors = print_stats(df, df_size, total_authors)

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
    df = load_reddit_corpus(cache_path=cache_path)
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
