import torch
from torch import nn


class Trainer:
    def __init__(self, num_epochs: int, batch_size: int = 2):
        self.num_epochs = num_epochs
        self.batch_size = batch_size

    def batchify(self, token_seq: list[int]):
        for i in range(0, len(token_seq) - self.batch_size + 1, self.batch_size):
            batch_as_list = token_seq[i : i + self.batch_size]
            X = torch.LongTensor(batch_as_list)[:, :-1]  # last one for target
            Y = torch.LongTensor(batch_as_list)[:, -1]
            yield i, X, Y

    def fit(self, model, training_seq):
        loss_fn = nn.CrossEntropyLoss(reduction="mean")
        optimizer = model.configure_optimizers()

        for epoch in range(self.num_epochs):
            for batch_idx, X, Y in self.batchify(training_seq):
                Y_hat = model(X)
                loss = loss_fn(Y_hat, Y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            print(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {loss.item()}")
