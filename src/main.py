#!/usr/bin/env python3

# > Imports
# Standard Library
import argparse
import csv
import os
import pickle
from sklearn.metrics import roc_auc_score

# Local
from data import pipeline, load_data
from pairings import split_data, create_pairings
from model import StyleEmbeddingModel
from utils import get_threshold


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
                        help="The path to the model to use in interactive mode.\
                                (default: None)")


def training_mode(args):
    # If the file exists, load the data from the file
    if os.path.exists(f"{args.path}/preprocessed/{args.data_source}_data.pkl"):
        df = load_data(source=args.data_source,
                       path=args.path)
    else:
        # Run the preprocessing pipeline
        df = pipeline(data_source=args.data_source,
                      output_path=args.path,
                      cache_path=args.cache_path)

    # Print some information about the data
    print("Data information:")
    print(f"Number of authors: {len(df['author_id'].unique())}")
    print(f"Number of sentences: {len(df)}")
    print(
        f"Number of sentences per author: {len(df) / len(df['author_id'].unique())}")
    print(
        f"Max number of sentences per author: {df.groupby('author_id').count()['text'].max()}")
    print(
        f"Number of conversations: {len(df['conversation_id'].unique())}")

    # Split the data into train, validation, and test sets
    train, val, test = split_data(df=df)

    # Check that the author_id's are non-overlapping
    assert len(set(train["author_id"]).intersection(
        set(val["author_id"]))) == 0

    # Create the pairings
    print("Creating pairings...")

    print("-" * 80)
    print("Train set:")
    train_pairings = create_pairings(train,
                                     semantic_range=(0.0, 1.0),
                                     semantic_proportion=0.0000000001,
                                     max_negative=1,
                                     output_path=args.path,
                                     output_name="train")

    print("-" * 80)
    print("Validation set:")
    val_pairings = create_pairings(val,
                                   semantic_range=(0.0, 1.0),
                                   semantic_proportion=1.0,
                                   max_negative=1,
                                   output_path=args.path,
                                   output_name="val")

    print("-" * 80)
    print("Test set:")
    test_pairings = create_pairings(test,
                                    semantic_range=(0.0, 1.0),
                                    semantic_proportion=1.0,
                                    max_negative=1,
                                    output_path=args.path,
                                    output_name="test")

    print("Pairings created.")

    # Create the model
    model = StyleEmbeddingModel(base_model="roberta-base",
                                cache_path=args.cache_path,
                                output_path=args.output_path,
                                name=f"style-nosem-{args.epochs}-{args.batch_size}")

    # Train the model
    print("Training model...")
    model.train(train_data=train_pairings[:724243],
                val_data=val_pairings[:100000],
                batch_size=args.batch_size,
                epochs=args.epochs)
    print("Model trained.")

    # Retrieve the optimal cosine threshold for the validation set
    threshold = get_threshold(
        f"{args.output_path}/{model.name}/eval/binary_classification_evaluation_val_loss_results.csv")

    # Evaluate the model
    print("Evaluating model...")
    model.evaluate(test_pairings[:100000],
                   threshold=threshold,
                   stel_dir=f"{args.path}/STEL/")
    print("Model evaluated.")


def interactive_mode(model_path):
    # Load the model
    model = StyleEmbeddingModel(model_path=model_path)

    threshold = get_threshold(
        f"{model_path}/eval/binary_classification_evaluation_val_loss_results.csv")

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


def evaluate_mode(model_path, data_path):

    print("Evaluating model...")
    # Load the model
    print("Loading model...")
    model = StyleEmbeddingModel(model_path=None)

    try:
        threshold = get_threshold(
            f"{model_path}/eval/binary_classification_evaluation_val_loss_results.csv")
    except FileNotFoundError:
        # Determine the threshold using the validation set and AUC
        # Load the validation data
        with open(f"{data_path}/paired/val-pairings.pkl", "rb") as f:
            val_data = pickle.load(f)

        print("Calculating threshold manually...")

        # Get the true labels
        # Convert to Binary Task
        from utils import contrastive_to_binary
        from sentence_transformers import InputExample
        val_examples = [InputExample(texts=texts, label=1)
                        for texts in val_data[:50000]]
        val_data = contrastive_to_binary(val_examples)

        # Get the true labels
        true_labels = [x[2] for x in val_data]

        first = [x[0] for x in val_data]
        second = [x[1] for x in val_data]

        # Get the predictions
        sims = model.similarity(first, second)

        # Get the threshold
        import numpy as np
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(true_labels, sims)

        # Get the threshold
        threshold = thresholds[np.argmax(tpr - fpr)]

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


if __name__ == '__main__':
    # Parse the input arguments
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    # Check that the mode is valid
    assert args.mode in ["train", "interactive", "evaluate"], "Invalid mode."

    if args.mode == "train":
        training_mode(args)

    elif args.mode == "interactive":
        assert args.model_path is not None, "No model path given."

        interactive_mode(args.model_path)

    elif args.mode == "evaluate":
        assert args.model_path is not None, "No model path given."
        assert args.path is not None, "No data path given."

        evaluate_mode(args.model_path, args.path)
