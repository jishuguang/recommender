import torch
from torch import nn


class OneHotLinear(nn.Module):
    """Linear layer for one hot features."""

    def __init__(self, one_hot_dims, output_dim):
        """
        :param one_hot_dims: List[int], dimensions of one-hot categorical features.
        :param output_dim: int, output dimension.
        """
        super().__init__()

        # offsets of the features, note that the offset for the first feature is 0
        self._offsets = torch.tensor([0] + one_hot_dims, dtype=torch.int).cumsum(0)[:-1]

        input_dim = sum(one_hot_dims)
        # embed is used as weight
        self.one_hot_embed = nn.Embedding(input_dim, output_dim)
        self._input_dim = input_dim

    def forward(self, x):
        """
        :param x: Tensor, (n, len(one_hot_dims)).
        :return: Tensor, (n, output_dim).
        """
        x = x + self._offsets.to(x.device)
        x = self.one_hot_embed(x).sum(dim=1)
        return x
