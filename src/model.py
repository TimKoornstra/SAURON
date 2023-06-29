# > Imports
# Standard Library
import os
import random
from typing import List, Tuple, Union

# Third Party
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sentence_transformers import SentenceTransformer, losses, InputExample, evaluation, SentencesDataset
from sentence_transformers.util import cos_sim
import torch
from torch import Tensor
from torch.utils.data import DataLoader

# Local
from utils import contrastive_to_binary


class StyleEmbeddingModel:
    """
    A class to represent a Style Embedding Model.

    Attributes
    ----------
    model : SentenceTransformer
        The SentenceTransformer model.
    name : str
        The name of the model.
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
            self.model._first_module().auto_model.resize_token_embeddings(
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
            f"Training on {len(train_data)} examples for {epochs} epochs,\
                    using a batch size of {batch_size}")

        # Load the train dataset
        train_examples = [InputExample(
            texts=[row[0], row[1], row[2]]) for row in train_data]

        train_dataset = SentencesDataset(train_examples, self.model)

        train_dataloader = DataLoader(
            train_dataset, shuffle=True, batch_size=batch_size)

        train_loss = losses.TripletLoss(
            self.model,
            triplet_margin=0.5,
            distance_metric=losses.TripletDistanceMetric.COSINE)

        # Load the validation dataset
        val_anchor, val_pos, val_neg = zip(*val_data)

        evaluator = evaluation.TripletEvaluator(
            sentences1=val_anchor,
            sentences2=val_pos,
            labels=val_neg,
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
            first = self.model.encode(first)
            second = self.model.encode(second)

        if first.ndim == 1:
            cosine_similarities = [cos_sim(first, second)]
        else:
            cosine_similarities = [
                cos_sim(first[i], second[i]) for i in range(len(first))]

        return torch.tensor(cosine_similarities)

    def _predict_cos(self,
                     first: Union[str, List[str], np.ndarray, Tensor],
                     second: Union[str, List[str], np.ndarray, Tensor],
                     threshold: float = 0.5) -> List[int]:
        """
        Predict whether two sentences are written by the same author based
        on the cosine similarity between their embeddings.

        Parameters
        ----------
        first : Union[str, List[str], np.ndarray, Tensor]
            The first sentence.
        second : Union[str, List[str], np.ndarray, Tensor]
            The second sentence.
        threshold : float
            The threshold to use for the prediction.

        Returns
        -------
        List[int]
            The prediction(s).
        """

        if isinstance(first, str):
            first = self.model.encode(first)
            second = self.model.encode(second)

        cosine_similarities = self.similarity(first, second)
        return (cosine_similarities > threshold).int().tolist()

    def _predict_cav(self,
                     anchor: Union[str, List[str], np.ndarray, Tensor],
                     first: Union[str, List[str], np.ndarray, Tensor],
                     second: Union[str, List[str], np.ndarray, Tensor]) -> List[int]:

        assert len(anchor) == len(first) == len(second)

        if isinstance(anchor, str):
            anchor = self.model.encode(anchor)
            first = self.model.encode(first)
            second = self.model.encode(second)

        A1_S1 = self.similarity(anchor, first)
        A1_S2 = self.similarity(anchor, second)
        return (A1_S1 < A1_S2).int().tolist()

    def _STEL_tasks(self,
                    stel_dir: str):
        """
        Run the STEL tasks.

        Parameters
        ----------
        stel_dir : str
            The path to the directory containing the STEL tasks.
        """

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
                correct = task["Correct Alternative"].tolist()

                task_data = list(
                    zip(anchors1, anchors2, alts1, alts2, correct))

                print(f"Evaluating for {task_name}...")
                self._predict_STEL(task_data, task_name)

                # list comprehension
                stel_oc_task_data = [(anchors1[i], alts1[i], anchors2[i]) if correct[i] == 1
                                     else (anchors1[i], alts2[i], anchors2[i]) for i in range(len(anchors1))]

                # extending list comprehension with the second pairs
                stel_oc_task_data.extend([(anchors2[i], alts2[i], anchors1[i]) if correct[i] == 1
                                          else (anchors2[i], alts1[i], anchors1[i]) for i in range(len(anchors2))])

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
        """
        Use the STEL framework to evaluate the model's ability to separate

        Parameters
        ----------
        task_data : List[Tuple[Tensor, Tensor, Tensor]]
            The data for the task.

        task_name : str
            The name of the task.
        """

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
        anchor_embeddings = self.model.encode(anchor_cav)
        first_embeddings = self.model.encode(first_cav)
        second_embeddings = self.model.encode(second_cav)

        test_data_transformed = list(
            zip(anchor_embeddings, first_embeddings, second_embeddings))

        print("Evaluating on the AV task...")
        test_data_av = contrastive_to_binary(test_data_transformed)

        first_av, second_av, actual_av = zip(*test_data_av)
        predicted_av = self._predict_cos(
            np.array(first_av), np.array(second_av), threshold)

        first_av_text, second_av_text, _ = zip(
            *contrastive_to_binary(test_data))

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
