#!/usr/bin/env python3

# > Imports
# Standard Library
import argparse
import os

# Local
from data import pipeline, load_data
from pairings import split_data, create_pairings
from train import training_pipeline


def add_arguments(parser):
    """
    Add the arguments to the parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to add the arguments to.

    """
    parser.add_argument("--data_source", type=str, default="reddit",
                        help="The data source to use for the model.\
                                (default: reddit)")
    parser.add_argument("-p", "--path", type=str, default="data/",
                        help="The path to the data directory.\
                                (default: 'data/')")
    parser.add_argument("--cache_path", type=str, default=".cache/",
                        help="The path to the cache directory. All temporary\
                                models and data will be saved here.\
                                (default: '.cache/')")


if __name__ == '__main__':
    # Parse the input arguments
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    # If the file exists, load the data from the file
    if os.path.exists(f"{args.path}/preprocessed/{args.data_source}_data.pkl"):
        df = load_data(args.data_source, args.path)
    else:
        # Run the preprocessing pipeline
        df = pipeline(data_source=args.data_source,
                      output_path=args.path,
                      cache_path=args.cache_path)

    # Split the data into train, validation, and test sets
    train, val, test = split_data(df[:500000])

    # Check that the author_id's are non-overlapping
    assert len(set(train["author_id"]).intersection(
        set(val["author_id"]))) == 0

    # Create the pairings
    print("Creating pairings...")

    print("-" * 80)
    print("Train set:")
    train_pairings = create_pairings(
        train,
        semantic_range=(0.7, 1.0),
        max_negative=3,
        output_path=args.path,
        output_name="train")

    print("-" * 80)
    print("Validation set:")
    val_pairings = create_pairings(
        val,
        semantic_range=(0.7, 1.0),
        max_negative=3,
        output_path=args.path,
        output_name="val")

    print("-" * 80)
    print("Test set:")
    test_pairings = create_pairings(
        test,
        semantic_range=(0.7, 1.0),
        max_negative=3,
        output_path=args.path,
        output_name="test")

    print("Pairings created.")

    # Run the training pipeline
    training_pipeline(train_pairings[:16], val_pairings)

    print("Done.")
