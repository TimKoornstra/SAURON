# File to train the model

from transformers import RobertaTokenizer, AutoConfig
from torch.utils.data import DataLoader
from models import RobertaForAuthorshipVerification, AVDataSet
from transformers import get_linear_schedule_with_warmup
import torch
from torch.optim import AdamW
from tqdm.auto import tqdm


def train(model, train_dataloader, optimizer, scheduler, device, num_epochs=5):
    """
    Train the given model.

    Parameters
    ----------
    model
        The model to train.

    train_dataloader
        The DataLoader for the training data.

    optimizer
        The optimizer to use.

    scheduler
        The scheduler to use.

    device
        The device to use.

    num_epochs
        The number of epochs to train for. (default: 5)
    """
    num_training_steps = num_epochs * len(train_dataloader)
    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        model.train()
        running_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            loss = model(input_ids=input_ids,
                         attention_mask=attention_mask,
                         labels=labels)

            loss.backward()
            optimizer.step()
            scheduler.step()
            progress_bar.update(1)

            running_loss += loss.item()

            if step % 100 == 0 and step != 0:
                print('Loss after {} steps: {}'.format(step, running_loss/step))

        print('Loss after epoch {}: {}'.format(
            epoch, running_loss/len(train_dataloader)))
        print()

    return model


def training_pipeline(train_data, val_data):
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    # Defining the dataset
    train_dataset = AVDataSet(train_data, tokenizer, max_len=512)
    val_dataset = AVDataSet(val_data, tokenizer, max_len=512)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Configuring the model
    config = AutoConfig.from_pretrained('roberta-base')

    # Defining the model
    model = RobertaForAuthorshipVerification(config=config)
    model.to(device)

    # Defining the optimizer
    optimizer = AdamW(model.parameters(),
                      lr=1e-5,
                      eps=1e-8)

    # Defining the dataloader
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=4)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=16)

    # Defining the scheduler
    total_steps = len(train_dataloader) * 5
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)

    # Training the model
    model = train(model, train_dataloader, optimizer, scheduler, device)

    return model
