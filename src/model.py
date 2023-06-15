# > Imports
# Standard Library
import csv
import os
import random
from typing import List, Tuple, Union

# Third Party
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sentence_transformers import SentenceTransformer, losses, InputExample, evaluation
from sentence_transformers.util import cos_sim
import torch
from torch.utils.data import DataLoader
from torch import Tensor

# Local
from utils import contrastive_to_binary


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
        model_path : str
            The path to the model.
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

            # Add custom tokens
            special_tokens_dict = {
                'additional_special_tokens': ['[URL]', '[MENTION]']}
            self.model.tokenizer.add_tokens(
                special_tokens_dict['additional_special_tokens'])

            # Resize the token embeddings
            self.model.auto_model.resize_token_embeddings(
                len(self.model.tokenizer))

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
        print(
            f"Training on {len(train_data)} examples for {epochs} epochs, using a batch size of {batch_size}")

        # Load the train dataset
        train_examples = [InputExample(texts=texts, label=1)
                          for texts in train_data]

        train_dataloader = DataLoader(
            train_examples, shuffle=True, batch_size=batch_size)
        train_loss = losses.MultipleNegativesRankingLoss(self.model)

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
                   first: Union[str, List[str], np.ndarray, Tensor],
                   second: Union[str, List[str], np.ndarray, Tensor]) -> torch.Tensor:
        """
        Calculate the cosine similarity between two sentences.

        Parameters
        ----------
        first : Union[str, List[str], np.ndarray, Tensor]
            The first sentence.
        second : Union[str, List[str], np.ndarray, Tensor]
            The second sentence.

        Returns
        -------
        torch.Tensor
            The cosine similarity between the (lists of) sentences.
        """

        if isinstance(first, str):
            first = self.model.encode(first, device="cuda")
            second = self.model.encode(second, device="cuda")

        if first.ndim == 1:
            cosine_similarities = [cos_sim(first, second)]
        else:
            cosine_similarities = [
                cos_sim(first[i], second[i]) for i in range(len(first))]

        return torch.tensor(cosine_similarities)

    def _predict_cos(self,
                     first: Union[str, List[str], np.ndarray, Tensor],
                     second: Union[str, List[str], np.ndarray, Tensor],
                     threshold: int = 0.5) -> List[int]:
        """
        Predict whether two sentences are written by the same author based
        on the cosine similarity between their embeddings.

        Parameters
        ----------
        first : Union[str, List[str], np.ndarray, Tensor]
            The first sentence.
        second : Union[str, List[str], np.ndarray, Tensor]
            The second sentence.
        threshold : int
            The threshold to use for the prediction.

        Returns
        -------
        List[int]
            The prediction(s).
        """

        if isinstance(first, str):
            first = self.model.encode(first, device="cuda")
            second = self.model.encode(second, device="cuda")

        cosine_similarities = self.similarity(first, second)
        return (cosine_similarities > threshold).int().tolist()

    def _predict_cav(self,
                     anchor: Union[str, List[str], np.ndarray, Tensor],
                     first: Union[str, List[str], np.ndarray, Tensor],
                     second: Union[str, List[str], np.ndarray, Tensor]) -> List[int]:

        assert len(anchor) == len(first) == len(second)

        if isinstance(anchor, str):
            anchor = self.model.encode(anchor, device="cuda")
            first = self.model.encode(first, device="cuda")
            second = self.model.encode(second, device="cuda")

        A1_S1 = self.similarity(anchor, first)
        A1_S2 = self.similarity(anchor, second)
        return (A1_S1 < A1_S2).int().tolist()

    def _STEL_tasks(self,
                    stel_dir: str):

        for file in os.listdir(stel_dir):
            if file.endswith(".tsv"):
                task = pd.read_csv(f"{stel_dir}/{file}", sep="\t")
                task_name = task["style type"].iloc[0]

                anchors1 = self.model.encode(
                    task["Anchor 1"].tolist(), convert_to_tensor=True)
                anchors2 = self.model.encode(
                    task["Anchor 2"].tolist(), convert_to_tensor=True)
                alts1 = self.model.encode(
                    task["Alternative 1.1"].tolist(), convert_to_tensor=True)
                alts2 = self.model.encode(
                    task["Alternative 1.2"].tolist(), convert_to_tensor=True)
                correct_alternative = task["Correct Alternative"].tolist()

                task_data = list(
                    zip(anchors1, anchors2, alts1, alts2, correct_alternative))

                print(f"Evaluating for {task_name}...")
                self._predict_STEL(task_data, task_name)

                stel_oc_task_data = []

                for anchor_1, anchor_2, alt_1, alt_2, correct_alternative in zip(anchors1, anchors2, alts1, alts2, correct_alternative):
                    # If the correct_alternative is 1, anchor_1 is the first in tuple
                    if correct_alternative == 1:
                        stel_oc_task_data.append(
                            (anchor_1, alt_1, anchor_2))  # Alternative 1.1
                        stel_oc_task_data.append(
                            (anchor_2, alt_2, anchor_1))  # Alternative 1.2
                    else:  # If the correct_alternative is 2, anchor_2 is the first in tuple
                        stel_oc_task_data.append(
                            (anchor_2, alt_1, anchor_1))  # Alternative 1.1
                        stel_oc_task_data.append(
                            (anchor_1, alt_2, anchor_2))  # Alternative 1.2

                self._predict_STEL_oc(stel_oc_task_data, task_name)

    def _predict_STEL(self,
                      task_data: List[Tuple[Tensor, Tensor, Tensor, Tensor, int]],
                      task_name: str) -> None:
        """
        Use the STEL framework to evaluate the model's ability to separate
        style from content.

        Parameters
        ----------
        task_data : List[Tuple[Tensor, Tensor, Tensor, Tensor, int]]
            The data for the task.
        task_name : str
            The name of the task.
        """

        # Create a list to store the predictions
        predictions = []

        for A1, A2, S1, S2, _ in task_data:
            # Calculate the left and right hand sides of equation (1) from
            # the paper https://aclanthology.org/2021.emnlp-main.569.pdf
            lhs = (1 - cos_sim(A1, S1))**2 + (1 - cos_sim(A2, S2))**2
            rhs = (1 - cos_sim(A1, S2))**2 + (1 - cos_sim(A2, S1))**2

            # Get the prediction
            if lhs < rhs:
                predictions.append(1)
            elif lhs > rhs:
                predictions.append(2)
            else:
                predictions.append(random.choice([1, 2]))

        # Calculate the accuracy of the model on this task
        accuracy = accuracy_score([data[-1]
                                  for data in task_data], predictions)

        print(f"STEL:     {accuracy}")

    def _predict_STEL_oc(self,
                         task_data: List[Tuple[Tensor, Tensor, Tensor]],
                         task_name: str):

        # Create a list to store the predictions
        predictions = []

        for A1, S, A2 in task_data:
            # Now it is a CAV task
            predictions.append(int(cos_sim(A1, S) < cos_sim(A1, A2)))

        # Calculate the accuracy of the model on this task
        accuracy = accuracy_score([0]*len(task_data), predictions)

        print(f"STEL-o-c: {accuracy}")

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
        stel_dir : str
            The directory containing the STEL tasks.
        threshold : int
            The cosine similarity threshold to use for the prediction.
        """

        # Load the test dataset
        anchor_cav, first_cav, second_cav = zip(*test_data)

        # Compute the embeddings once for all inputs
        anchor_embeddings = self.model.encode(anchor_cav, device="cuda")
        first_embeddings = self.model.encode(first_cav, device="cuda")
        second_embeddings = self.model.encode(second_cav, device="cuda")

        test_data_transformed = list(
            zip(anchor_embeddings, first_embeddings, second_embeddings))

        print("Evaluating on the AV task...")
        test_data_av = contrastive_to_binary(test_data_transformed)

        first_av, second_av, actual_av = zip(*test_data_av)
        predicted_av = self._predict_cos(
            np.array(first_av), np.array(second_av), threshold)

        # Save the predictions to a csv file
        with open(f"predictions-{self.name}.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(["First", "Second", "Actual", "Predicted"])

            for (anchor_text, first_text, second_text), actual, predicted in zip(test_data, actual_av, predicted_av):
                writer.writerow([first_text, second_text, actual, predicted])

        # Get the accuracy of the model
        accuracy_av = accuracy_score(actual_av, predicted_av)
        print(f"Accuracy on the AV task: {accuracy_av}")

        # Get the predictions for the CAV task
        print("Evaluating on the CAV task...")

        predicted_cav = self._predict_cav(
            anchor_embeddings, first_embeddings, second_embeddings)

        actual_cav = [0] * len(anchor_cav)
        accuracy_cav = accuracy_score(actual_cav, predicted_cav)

        print(f"Accuracy on the CAV task: {accuracy_cav}")

        # Get the predictions for the STEL tasks
        if stel_dir is not None:
            print("Performing STEL tasks...")
            self._STEL_tasks(stel_dir)
