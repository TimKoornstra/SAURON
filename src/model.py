# > Imports
# Standard Library
import os
import random

# Third Party
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sentence_transformers import SentenceTransformer, losses, InputExample, evaluation
from sentence_transformers.util import cos_sim
import torch
from torch.utils.data import DataLoader

# Local
from utils import contrastive_to_binary
from typing import List, Union


class StyleEmbeddingModel:
    """
    A class to represent a Style Embedding Model.

    Attributes
    ----------
    model : SentenceTransformer
        The SentenceTransformer model.
    output_path : str
        The path to the output directory.
    """

    def __init__(self,
                 base_model: str = "roberta-base",
                 name: str = "style-embedding",
                 model_path: str = None,
                 cache_path: str = ".cache/",
                 output_path: str = "output/") -> None:
        """
        Initialize the StyleEmbeddingModel.

        Parameters
        ----------
        base_model : str
            The base model to use for the SentenceTransformer model.
        name : str
            The name of the model.
        cache_path : str
            The path to the cache directory.
        output_path : str
            The path to the output directory.
        """

        if model_path:
            self.model = SentenceTransformer(model_path,
                                             cache_folder=cache_path)
        else:
            self.model = SentenceTransformer(base_model,
                                             cache_folder=cache_path)

        self.name = name
        self.output_path = output_path

    def train(self,
              train_data: List[List[str]],
              val_data: List[List[str]],
              batch_size: int = 8,
              epochs: int = 3) -> None:
        """
        Train the model.

        Parameters
        ----------
        train_data : List[List[str]]
            The training data in the form of a list of lists of strings, where
            the first string is the anchor sentence, the second string is the
            positive sentence, and the other strings are the negative sentences.
        val_data : List[List[str]]
            The validation data. The format is the same as the training data.
        batch_size : int
            The batch size to use for training.
        epochs : int
            The number of epochs to train the model for.
        """

        # Load the train dataset
        train_examples = [InputExample(texts=texts, label=1)
                          for texts in train_data]

        # Convert to binary classification
        train_data = contrastive_to_binary(train_examples)

        train_examples = [InputExample(texts=[text1, text2], label=label)
                          for text1, text2, label in train_data]

        train_dataloader = DataLoader(
            train_examples, shuffle=True, batch_size=batch_size)
        # train_loss = losses.MultipleNegativesRankingLoss(self.model)
        train_loss = losses.ContrastiveLoss(self.model)

        # Load the validation dataset
        val_examples = [InputExample(texts=texts, label=1)
                        for texts in val_data]
        val_data = contrastive_to_binary(val_examples)

        # Validation split
        val_sen1, val_sen2, val_labels = zip(*val_data)

        evaluator = evaluation.BinaryClassificationEvaluator(
            sentences1=val_sen1,
            sentences2=val_sen2,
            labels=val_labels,
            name="val_loss")

        warmup_steps = int(len(train_dataloader) * epochs * 0.1)

        # Train the model
        self.model.fit(train_objectives=[(train_dataloader, train_loss)],
                       evaluator=evaluator,
                       epochs=epochs,
                       warmup_steps=warmup_steps,
                       output_path=f"{self.output_path}/{self.name}")

    def similarity(self,
                   first: Union[str, List[str]],
                   second: Union[str, List[str]]) -> torch.Tensor:
        """
        Calculate the cosine similarity between two sentences.

        Parameters
        ----------
        first : Union[str, List[str]]
            The first sentence.
        second : Union[str, List[str]]
            The second sentence.

        Returns
        -------
        torch.Tensor
            The cosine similarity between the (lists of) sentences.
        """

        # Get the embeddings for the sentences
        embeddings1 = self.model.encode(first)
        embeddings2 = self.model.encode(second)

        # Calculate the cosine similarity between the embeddings
        if isinstance(first, str):
            cosine_similarities = [cos_sim(embeddings1, embeddings2)]
        else:
            cosine_similarities = [cos_sim(embeddings1[i], embeddings2[i])
                                   for i in range(len(embeddings1))]

        return torch.tensor(cosine_similarities)

    def _predict_cos(self,
                     first: Union[str, List[str]],
                     second: Union[str, List[str]],
                     threshold: int = 0.5) -> List[int]:
        """
        Predict whether two sentences are written by the same author based
        on the cosine similarity between their embeddings.

        Parameters
        ----------
        first : Union[str, List[str]]
            The first sentence.
        second : Union[str, List[str]]
            The second sentence.
        threshold : int
            The threshold to use for the prediction.

        Returns
        -------
        List[int]
            The prediction(s).
        """

        # Get the cosine similarity between the sentences
        cosine_similarities = self.similarity(first, second)

        # Return the predictions
        return (cosine_similarities > threshold).int().tolist()

    def _predict_STEL(self,
                      dir_path: str):
        """
        Use the STEL framework to evaluate the model's ability to separate
        style from content.

        Parameters
        ----------
        dir_path : str
            The path to the directory containing the STEL tasks.
        """
        # Load the task instances
        for file in os.listdir(dir_path):
            if file.endswith(".tsv"):
                task = pd.read_csv(f"{dir_path}/{file}",
                                   sep="\t")

                task_name = task["style type"].iloc[0]

                # Create a list to store the predictions
                predictions = []

                for i, row in task.iterrows():
                    # Calculate the similarities between A1, A2, S1, and S2
                    A1_S1 = self.similarity(row["Anchor 1"], row["Sentence 1"])
                    A1_S2 = self.similarity(row["Anchor 1"], row["Sentence 2"])
                    A2_S1 = self.similarity(row["Anchor 2"], row["Sentence 1"])
                    A2_S2 = self.similarity(row["Anchor 2"], row["Sentence 2"])

                    # Calculate the left and right hand sides of equation (1) from
                    # the paper https://aclanthology.org/2021.emnlp-main.569.pdf
                    lhs = (1 - A1_S1)**2 + (1 - A2_S2)**2
                    rhs = (1 - A1_S2)**2 + (1 - A2_S1)**2

                    # Get the prediction
                    if lhs < rhs:
                        predictions.append(1)
                    if lhs > rhs:
                        predictions.append(2)
                    if lhs == rhs:
                        predictions.append(random.choice([1, 2]))

                # Calculate the accuracy of the model on this task
                accuracy = accuracy_score(
                    task["Correct Alternative"], predictions)

                print(f"Accuracy on {task_name}: {accuracy}")

    def evaluate(self,
                 test_data: List[List[str]],
                 stel_dir: str = None,
                 threshold: int = 0.5) -> None:
        """
        Evaluate the model.

        Parameters
        ----------
        test_data : List[List[str]]
            The test data. The format is the same as the training data.
        threshold : int
            The cosine similarity threshold to use for the prediction.
        """

        # Load the test dataset
        test_examples = [InputExample(texts=texts, label=1)
                         for texts in test_data]
        test_data = contrastive_to_binary(test_examples)

        # Predict the labels for the test data
        first, second, actual = zip(*test_data)

        # Get the predictions
        predicted = self._predict_cos(first, second, threshold)

        # Get the accuracy of the model
        accuracy = accuracy_score(actual, predicted)
        print(f"Accuracy on the AV task: {accuracy}")

        # Get the predictions for the STEL tasks
        if stel_dir is not None:
            self._predict_STEL(stel_dir)
