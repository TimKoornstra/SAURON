#!/usr/bin/env python3

# > Imports
# Standard Library
import argparse
import os
import pickle

# Local
from data import pipeline, load_data
from pairings import split_data, create_pairings
from model import StyleEmbeddingModel
from utils import get_threshold

# Constants
TRAIN = "train"
INTERACTIVE = "interactive"
EVALUATE = "evaluate"
VALID_MODES = [TRAIN, INTERACTIVE, EVALUATE]


# Error Handling
class InvalidModeError(Exception):
    pass


class MissingArgumentError(Exception):
    pass


def add_arguments(parser: argparse.ArgumentParser):
    """
    Add the arguments to the parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to add the arguments to.
    """

    parser.add_argument("--data-source", type=str, default="reddit",
                        help="The data source to use for the model.\
                                (default: reddit)")
    parser.add_argument("-p", "--path", type=str, default="data/",
                        help="The path to the data directory.\
                                (default: 'data/')")
    parser.add_argument("--output-path", type=str, default="output/",
                        help="The path to the output directory.\
                                (default: 'output/')")
    parser.add_argument("--cache-path", type=str, default=".cache/",
                        help="The path to the cache directory. All temporary\
                                models and data will be saved here.\
                                (default: '.cache/')")
    parser.add_argument("-b", "--batch-size", type=int, default=8,
                        help="The batch size to use for training.\
                                (default: 8)")
    parser.add_argument("-e", "--epochs", type=int, default=3,
                        help="The number of epochs to train the model for.\
                                (default: 3)")
    parser.add_argument("-m", "--mode", type=str, default="train",
                        help="The mode to run the script in.\
                                (default: 'train')")
    parser.add_argument("--model-path", type=str, default=None,
                        help="The path to the model to use in\
                                interactive mode. (default: None)")


def training_mode(args: argparse.Namespace):
    """
    Run the training mode.

    Parameters
    ----------
    args : argparse.Namespace
        The arguments to use for training.
    """
    # Create necessary directories if they don't exist
    os.makedirs(args.path, exist_ok=True)
    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(args.cache_path, exist_ok=True)

    # Load data if it exists, otherwise run the preprocessing pipeline
    data_file = f"{args.path}/preprocessed/{args.data_source}_data.pkl"
    df = load_data(source=args.data_source, path=args.path)\
        if os.path.exists(data_file)\
        else pipeline(data_source=args.data_source,
                      output_path=args.path,
                      cache_path=args.cache_path)

    # Split the data into train, validation, and test sets
    train, val, test = split_data(df=df)

    # Ensure the author_id's are non-overlapping
    if len(set(train["author_id"]).intersection(set(val["author_id"]))) != 0:
        raise ValueError(
            "Train and validation sets have overlapping author_ids.")

    # Create the pairings
    print("Creating train pairings...")
    train_pairings = create_pairings(
        train, output_path=args.path, output_name="train")
    print("Train pairings created.")

    print("Creating validation pairings...")
    val_pairings = create_pairings(
        val, output_path=args.path, output_name="val")
    print("Validation pairings created.")

    print("Creating test pairings...")
    test_pairings = create_pairings(
        test, output_path=args.path, output_name="test")
    print("Test pairings created.")

    # Create the model
    model = StyleEmbeddingModel(base_model="roberta-base",
                                cache_path=args.cache_path,
                                output_path=args.output_path,
                                name=f"style-allsemv8-\
                                        {args.epochs}-{args.batch_size}")

    # Train the model
    model.train(train_data=train_pairings,
                val_data=val_pairings[:100000],
                batch_size=args.batch_size,
                epochs=args.epochs)

    # Retrieve the optimal cosine threshold for the validation set
    threshold = get_threshold(args.path, model)

    # Evaluate the model
    model.evaluate(test_pairings[:100000],
                   threshold=threshold,
                   stel_dir=f"{args.path}/STEL/")


def interactive_mode(model_path: str, data_path: str):
    """
    Run the interactive mode.

    Parameters
    ----------
    model_path : str
        The path to the model to use.
    data_path : str
        The path to the data directory.
    """

    # Load the model and get the threshold
    model = StyleEmbeddingModel(model_path=model_path)
    threshold = get_threshold(data_path, model)

    while True:
        # Get the input
        text = input("Enter a sentence: ")
        text2 = input("Enter another sentence: ")

        # Get the similarity
        similarity = model.similarity(text, text2)

        # Predict whether the sentences were written by the same author
        prediction = model._predict_cos(text, text2, threshold=threshold)

        # Print the embedding
        print(f"Similarity: {similarity[0]}")
        print(f"Prediction: {prediction[0]}")


def evaluate_mode(model_path: str, data_path: str):
    """
    Run the evaluation mode.

    Parameters
    ----------
    model_path : str
        The path to the model to use.
    data_path : str
        The path to the data directory.
    """

    print("Evaluating model...")
    # Load the model
    print("Loading model...")
    model = StyleEmbeddingModel(model_path=model_path)

    threshold = get_threshold(data_path, model)
    print(f"Threshold: {threshold}")

    print("Model loaded.")

    # Load the data
    print("Loading data...")
    with open(f"{data_path}/paired/test-pairings.pkl", "rb") as f:
        data = pickle.load(f)
    print("Data loaded.")

    # Evaluate the model
    print("Evaluating model...")
    model.evaluate(data[:100000],
                   stel_dir=f"{data_path}/STEL/",
                   threshold=threshold)
    print("Model evaluated.")


def check_mode_and_paths(mode: str,
                         model_path: str,
                         data_path: str):
    """
    Check that the mode and paths are valid.

    Parameters
    ----------
    mode : str
        The mode to run.
    model_path : str
        The path to the model to use.
    data_path : str
        The path to the data directory.
    """

    if mode not in VALID_MODES:
        raise InvalidModeError(
            f"Invalid mode: {mode}. Valid modes are {VALID_MODES}")

    if mode in [INTERACTIVE, EVALUATE] and model_path is None:
        raise MissingArgumentError("No model path given.")

    if mode == EVALUATE and data_path is None:
        raise MissingArgumentError("No data path given.")


# Main function
if __name__ == '__main__':
    # Parse the input arguments
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    check_mode_and_paths(args.mode, args.model_path, args.path)

    if args.mode == TRAIN:
        training_mode(args)

    elif args.mode == INTERACTIVE:
        interactive_mode(args.model_path, args.path)

    elif args.mode == EVALUATE:
        evaluate_mode(args.model_path, args.path)
