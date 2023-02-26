
# In this file, I will create the models for the Authorship Verification task
# I will finetune a pretrained RoBERTa model for this task
# "roberta-base" is the name of the pretrained model
# I will use the Contrastive Loss function for training as proposed in the paper
# "SimCSE: Simple Contrastive Learning of Sentence Embeddings" by Chen et al.

# Input to the model: Two sentences
# Output of the model: binary classification (0 or 1) for whether the two
# sentences are written by the same author or not

# Contrastive Loss function
# l_i = -log(exp(sim(x_i, x_j)) / (sum_k exp(sim(x_i, x_k)/T)))
# sim(x_i, x_j) = cos(x_i, x_j)
# T is the temperature parameter
# x_i and x_j are the embeddings of the two sentences
# x_k is the embedding of the kth sentence in the batch
# l_i is the loss for the ith sentence in the batch
# Defining the model

import torch
import torch.nn as nn
from transformers import RobertaModel
import os
os.environ["TRANSFORMERS_CACHE"] =\
    "~/Documents/uni/thesis/style-embeddings/.cache/huggingface/transformers"


class RobertaForAuthorshipVerification(nn.Module):
    def __init__(self, model_name, num_labels):
        super(RobertaForAuthorshipVerification, self).__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.roberta(input_ids=input_ids,
                               attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = ContrastiveLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
            return loss
        else:
            return logits

# Defining the Contrastive Loss function


class ContrastiveLoss(nn.Module):
    def __init__(self, T=0.07):
        super(ContrastiveLoss, self).__init__()
        self.T = T

    def forward(self, sim, labels):
        # sim: similarity between the two sentences
        # labels: 0 or 1, whether the two sentences are written by the same author or not
        # sim.shape = (batch_size, 2)
        # labels.shape = (batch_size,)

        # Extracting the similarity between the two sentences
        sim_pos = sim[:, 0]
        sim_neg = sim[:, 1]

        # Computing the loss
        loss = -torch.log(torch.exp(sim_pos/self.T) /
                          (torch.exp(sim_pos/self.T) + torch.exp(sim_neg/self.T)))

        # Computing the mean loss
        loss = loss.mean()

        return loss


class AVDataSet(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence1 = self.data[index][0]
        sentence2 = self.data[index][1]
        label = self.data[index][2]

        encoding = self.tokenizer.encode_plus(sentence1,
                                              sentence2,
                                              add_special_tokens=True,
                                              max_length=self.max_len,
                                              return_token_type_ids=False,
                                              pad_to_max_length=True,
                                              return_attention_mask=True,
                                              return_tensors='pt')

        return {'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)}
