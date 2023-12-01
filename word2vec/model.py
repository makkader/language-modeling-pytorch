import torch
from torch import nn


class Word2Vec(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, padding_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx, max_norm=1
        )
        self.output = nn.Linear(embedding_dim, vocab_size)

    def forward(self, X: torch.Tensor):
        em = self.embedding(X)
        return self.output(em.sum(axis=1))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
