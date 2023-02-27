
from sentence_transformers import SentenceTransformer, losses, InputExample
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from torch.utils.data import DataLoader


def training(train_data, val_data, test_data, cache_path=".cache/"):

    # Load the model
    model = SentenceTransformer("roberta-base", cache_folder=cache_path)

    # Load the dataset
    train_examples = [InputExample(
        texts=[example[0], example[1]], label=example[2]) for example in train_data]

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=8)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # Evaluate the model
    val = zip(*val_data)

    evaluator = BinaryClassificationEvaluator(*val)

    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              show_progress_bar=True,
              evaluator=evaluator,
              )

    # Evaluate the model

    # Test the model
