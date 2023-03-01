# > Imports
# Third Party
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sentence_transformers import SentenceTransformer, losses, InputExample, evaluation
from sentence_transformers.util import cos_sim
import torch

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
        cosine_similarities = self.similarity(first, first)

        # Return the predictions
        return (cosine_similarities > threshold).int().tolist()

    def evaluate(self,
                 test_data: List[List[str]]) -> None:
        """
        Evaluate the model.

        Parameters
        ----------
        test_data : List[List[str]]
            The test data. The format is the same as the training data.
        """

        # Load the test dataset
        test_examples = [InputExample(texts=texts, label=1)
                         for texts in test_data]
        test_data = contrastive_to_binary(test_examples)

        # Predict the labels for the test data
        first, second, actual = zip(*test_data)

        # Get the predictions
        predicted = self._predict_cos(first, second)

        # Print the predictions and the ground truth labels
        for prediction, label in zip(predicted, actual):
            print(f"Prediction: {prediction}, Label: {label}")

        # Get the accuracy of the model
        accuracy = accuracy_score(actual, predicted)
        print(f"Accuracy: {accuracy}")

        # Get the F1 score of the model
        f1 = f1_score(actual, predicted)
        print(f"F1: {f1}")
