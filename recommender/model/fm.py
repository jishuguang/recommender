import torch
from torch import nn

from utils.log import get_logger
from model.module.linear import OneHotLinear
from model.module.embedding import OneHotEmbedding, ContextEmbedding


logger = get_logger()


class FM(nn.Module):
    """Factorization Machines for multi actions.

    Reference: https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf and
    http://d2l.ai/chapter_recommender-systems/fm.html.
    """

    def __init__(self, cat_dims, embed_dim, output_dim, **kwargs):
        """
        :param cat_dims: List[int], dimensions of categorical features.
        :param embed_dim: int, embedding dimension for all categorical features.
        :param output_dim: int, output dimension.
        """
        super().__init__()

        # linear part
        self.one_hot_linear = OneHotLinear(cat_dims, output_dim)

        # embedding part
        self.one_hot_embed = OneHotEmbedding(cat_dims, embed_dim, flat=False)
        # each output dim has a hadamard weight to project origin one_hot_embed to its own
        self.project = nn.Embedding(output_dim, len(cat_dims))

        # used by other classes
        self.embedding_tensor = None

    def forward(self, data):
        """
        :param data: dict.
        :return: Tensor, (n, output_dim).
        """
        # Tensor(n, len(cat_dims))
        cat = data['cat']

        linear_result = self._linear_result(cat)
        embedding = self._get_embedding(data)  # (n, c, e)
        self.embedding_tensor = embedding
        embed_result = self._embed_result(embedding)
        return linear_result + embed_result

    def _linear_result(self, cat):
        """Calculate the result of linear part."""
        return self.one_hot_linear(cat)

    def _embed_result(self, embedding):
        """Calculate the result of embed part.
        :param embedding: Tensor, (n, c, e).
        """
        # project original embedding
        project_weight = self.project.weight  # (o, c)
        project_embedding = embedding.permute(0, 2, 1).unsqueeze(-1) \
            * project_weight.permute(1, 0)  # (n, e, c, 1) * (c, o) -> (n, e, c, o)
        project_embedding = project_embedding.permute(0, 3, 2, 1)  # (n, o, c, e)
        # interaction
        square_of_sum = torch.sum(project_embedding, dim=2) ** 2
        sum_of_square = torch.sum(project_embedding ** 2, dim=2)
        embed_result = 0.5 * (square_of_sum - sum_of_square).sum(dim=2)
        return embed_result

    def _get_embedding(self, data):
        """Get embedding using data..
        :param data: dict.
        :return: Tensor, (n, c, e)
        """
        # Tensor(n, c)
        cat = data['cat']
        return self.one_hot_embed(cat)

    def get_embedding(self):
        """Get obtained embedding."""
        if self.embedding_tensor is None:
            raise ValueError(f'Embedding of {type(self)} is not initialized.')
        return self.embedding_tensor

    def infer(self, data):
        """Inferring.
        :param data: dict.
        :return: Tensor, (n, len(actions)).
        """
        return self(data).sigmoid()


class FmWithContext(FM):
    """FM with context embedding."""
    # TODO: Context has no benefit to FM in this way. Find a better way to incorporate context information.

    def __init__(self, cat_dims, embed_dim, output_dim, context_embeds, **kwargs):
        """
        :param context_embeds: List[Dict], each dict contain ContextEmbedding configs.
        """
        super().__init__(cat_dims, embed_dim, output_dim)

        project_dim = len(cat_dims)
        for i, context in enumerate(context_embeds):
            self.add_module(f'context{i}', ContextEmbedding(**context))
            if 'output_shape' in context:
                project_dim += context['output_shape'][0]
            else:
                project_dim += 1

        # each output dim has a hadamard weight to project origin one_hot_embed to its own
        self.project = nn.Embedding(output_dim, project_dim)

    def _get_embedding(self, data):
        """Get embedding using data.
        :param data: dict.
        :return: Tensor, (n, c, e)
        """
        embedding_list = [super()._get_embedding(data)]
        context = data['context']
        for i in range(context.shape[1]):
            embedding_list.append(getattr(self, f'context{i}')(context[:, i:i+1]))
        return torch.cat(embedding_list, dim=1)


def build_fm(model_cfg):
    context_embeds = getattr(model_cfg, 'context_embeds', None)
    if context_embeds is not None:
        model = FmWithContext(**model_cfg)
    else:
        model = FM(**model_cfg)
    return model
