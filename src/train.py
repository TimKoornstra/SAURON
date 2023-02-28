
from sentence_transformers import SentenceTransformer, losses, InputExample, evaluation
from torch.utils.data import DataLoader

# Get logistic regression model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

import numpy as np


def training(train_data, val_data, test_data, batch_size=8, epochs=3,
             cache_path=".cache/", output_path="output/"):

    # Load the model
    model = SentenceTransformer("roberta-base", cache_folder=cache_path)

    # Load the train dataset
    train_examples = [InputExample(texts=texts, label=1)
                      for texts in train_data]

    train_dataloader = DataLoader(
        train_examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # Load the validation dataset
    val_examples = [InputExample(texts=texts, label=1)
                    for texts in val_data]
    val_data = contrastive_to_binary(val_examples)

    # regression_model = LogisticRegression(
    # solver="lbfgs", max_iter=1000, random_state=42)

    # Validation split
    val_sen1, val_sen2, val_labels = zip(*val_data)

    evaluator = evaluation.BinaryClassificationEvaluator(
        sentences1=val_sen1,
        sentences2=val_sen2,
        labels=val_labels,
        name="val_loss")

    warmup_steps = int(len(train_dataloader) * epochs * 0.1)

    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              show_progress_bar=True,
              output_path=f"{output_path}-model",
              optimizer_params={"lr": 2e-5,
                                "eps": 1e-8},
              warmup_steps=warmup_steps,
              save_best_model=True,
              checkpoint_path=f"{cache_path}/checkpoints",
              evaluator=evaluator,
              )

    # train_binary = contrastive_to_binary(train_examples)

    # Train the logistic regression model
    # train_embeddings = binary_to_embedding(train_binary, model)
    # train_labels = [label for _, _, label in train_binary]

    # regression_model.fit(train_embeddings, train_labels)

    # testing(test_data, model, regression_model)


def testing(test_data, model, regression_model):
    # Load the test dataset
    test_examples = [InputExample(texts=texts, label=1)
                     for texts in test_data]
    test_data = contrastive_to_binary(test_examples)

    # Get the embeddings for the test data
    test_embeddings = binary_to_embedding(test_data, model)

    # Get the labels for the test data
    test_labels = [label for _, _, label in test_data]

    # Get the predictions for the test data
    test_predictions = regression_model.predict(test_embeddings)

    for prediction, label in zip(test_predictions, test_labels):
        print(f"Prediction: {prediction}, Label: {label}")

    # Get the accuracy of the model
    accuracy = accuracy_score(test_labels, test_predictions)
    print(f"Accuracy: {accuracy}")

    # Get the F1 score of the model
    f1 = f1_score(test_labels, test_predictions)
    print(f"F1: {f1}")


def contrastive_to_binary(examples):
    """Convert the contrastive pairings to binary pairings."""
    binary_pairings = []
    for example in examples:

        # Get the positive example
        binary_pairings.append([example.texts[0], example.texts[1], 1])

        # Get the negative examples
        for i in range(2, len(example.texts)):
            binary_pairings.append(
                [example.texts[0], example.texts[i], 0])

    return binary_pairings


def binary_to_embedding(pairings, model):
    """Convert the binary pairings to embeddings."""
    embeddings = []
    for s1, s2, _ in pairings:
        s1_embedding = model.encode(s1)
        s2_embedding = model.encode(s2)

        # Concatenate the embeddings
        embeddings.append(np.concatenate([s1_embedding, s2_embedding]))

    return embeddings
