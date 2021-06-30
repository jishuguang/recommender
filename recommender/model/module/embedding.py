import torch
from torch import nn
import pandas as pd

from logging import getLogger


logger = getLogger()


def load_embedding(embed_module, path, requires_grad=False):
    """
    :param embed_module: nn.Embedding module.
    :param path: path to pretrained embedding.
    :param requires_grad: default to False.
    :return: None.
    """

    # load and parse embedding string
    logger.info(f'Loading pretrained embedding: {path}...')
    pretrain_embedding = pd.read_csv(path, index_col=0)
    embed_column = pretrain_embedding.columns[0]
    logger.info(f'Parsing pretrained embedding...')
    pretrain_embedding = pretrain_embedding.apply(
        lambda x: pd.Series([float(n) for n in x[embed_column].split()]), axis=1)

    # transform to tensor, and leave 0 as default embedding
    embedding = torch.tensor(pretrain_embedding.values)
    embedding = torch.cat([embedding.mean(dim=0, keepdim=True), embedding], dim=0)

    # copy
    embed_module.weight.data.copy_(embedding)
    embed_module.weight.requires_grad = requires_grad
    logger.info(f'Embedding dim {embedding.shape}.')


class OneHotEmbedding(nn.Module):
    """Embedding for one-hot features."""

    def __init__(self, one_hot_dims, embed_dims, flat=True):
        """
        :param one_hot_dims: List[int], dimensions of one-hot categorical features.
        :param embed_dims: int or List[int], embedding dimensions for one-hot categorical features.
        :param flat: bool, when embed_dims are all the same, this can be False.
        """
        super().__init__()
        if isinstance(embed_dims, int):
            embed_dims = [embed_dims] * len(one_hot_dims)
        assert len(embed_dims) == len(one_hot_dims)

        if flat:
            assert min(embed_dims) == max(embed_dims)
        self._flat = flat

        for i in range(len(one_hot_dims)):
            self.add_module(f'embed{i}', nn.Embedding(one_hot_dims[i], embed_dims[i]))

    def forward(self, x):
        """
        :param x: Tensor, (n, len(cat_dims)).
        :return: Tensor, when flat is True:  (n, sum(embed_dims));
                         when flat is False: (n, len(cat_dims), embed_dim)
        """
        embedding_list = list()
        for i in range(x.shape[1]):
            embedding_list.append(getattr(self, f'embed{i}')(x[:, i:i+1]))

        if self._flat:
            embedding = torch.cat(embedding_list, dim=2).squeeze()
        else:
            embedding = torch.cat(embedding_list, dim=1)
        return embedding


class ContextEmbedding(nn.Module):
    """Context embedding."""

    def __init__(self, embed_shape, path, output_shape=None, requires_grad=False):
        """
        :param embed_shape: [int, int], shape of the context embedding.
        :param path: path to the embedding.
        :param output_shape: [int, int], [channel, embed_dim].
        :param requires_grad: require grad or not.
        """
        super().__init__()
        self.embed = nn.Embedding(*embed_shape)
        load_embedding(self.embed, path, requires_grad)

        if output_shape is not None:
            channel, embed_dim = output_shape
            self.channel, self.embed_dim = output_shape
            self.linear = nn.Linear(embed_shape[1], channel * embed_dim, bias=True)

    def forward(self, x):
        """
        :param x: Tensor, (n, 1).
        :return: Tensor, (n, c, e).
        """
        embedding = self.embed(x)

        linear_layer = getattr(self, 'linear', None)
        if linear_layer is None:
            return embedding

        return linear_layer(embedding).reshape(x.shape[0], self.channel, self.embed_dim)
