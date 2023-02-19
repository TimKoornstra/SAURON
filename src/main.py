#!/usr/bin/env python3

# > Imports
# Standard Library
import argparse
import os

# Local
from data import pipeline, load_data
from semantics import paraphrase_mining


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
    parser.add_argument("-p", "--path", type=str, default="",
                        help="The path to the pre-processed data \
                                pickle file. (default: '')")
    parser.add_argument("--cache_path", type=str, default=".cache/",
                        help="The path to the cache directory. All temporary\
                                models and data will be saved here.\
                                (default: '.cache/')")
    parser.add_argument("--output_path", type=str, default="output/",
                        help="The path to the output directory.\
                                (default: 'output/')")


if __name__ == '__main__':
    # Parse the input arguments
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    # If the file exists, load the data from the file
    if os.path.exists(args.path):
        if args.path.endswith(".pkl"):
            print("Loading data locally...")
            df = load_data(args.path)
            print("Data loaded.")
        else:
            raise ValueError(f"Invalid pickle file: {args.path}")
    else:
        # Run the preprocessing pipeline
        df = pipeline(data_source=args.data_source,
                      output_path=args.output_path,
                      cache_path=args.cache_path)

    # Test the paraphrase mining
    paraphrase_mining(df.iloc[:50000])
