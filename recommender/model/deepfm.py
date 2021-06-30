import torch
from torch import nn

from utils.log import get_logger
from .fm import FM
from model.module.embedding import ContextEmbedding

logger = get_logger()


class DeepFm(nn.Module):
    """DeepFm for multi actions.

    Reference: https://www.ijcai.org/Proceedings/2017/0239.pdf.
        http://d2l.ai/chapter_recommender-systems/deepfm.html.
    """

    def __init__(self, cat_dims, embed_dim, output_dim, mlp_dims, dropout, **kwargs):
        super().__init__()

        # FM
        self.fm = FM(cat_dims, embed_dim, output_dim)

        # MLP
        self.mlps = nn.ModuleList()
        mlp_input_dim = len(cat_dims) * embed_dim
        for i, dim in enumerate(mlp_dims):
            self.mlps.append(nn.Sequential(
                nn.Linear(mlp_input_dim, dim, bias=True),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
            mlp_input_dim = dim
        self.mlps.append(nn.Linear(mlp_input_dim, output_dim, bias=True))

        # used by other classes
        self.embedding_tensor = None

    def forward(self, data):
        """
        :param data: Dict{data_key: Tensor}.
        :return: Tensor, (n, output_dim).
        """
        fm_result = self.fm(data)

        embedding = self.fm.get_embedding()  # (n, c, e)
        self.embedding_tensor = embedding
        mlp_result = self._get_initial_mlp_result(embedding, data)  # (n, -1)
        for module in self.mlps:
            mlp_result = module(mlp_result)

        return fm_result + mlp_result

    def _get_initial_mlp_result(self, embedding, data):
        """
        :param embedding: Tensor, (n, c, e).
        :param data: Dict{data_key: Tensor}.
        :return: Tensor, (n, -1).
        """
        return embedding.reshape(embedding.shape[0], -1)

    def infer(self, data):
        """
        :param data: Dict{data_key: Tensor}.
        :return: Tensor, (n, output_dim).
        """
        return self(data).sigmoid()

    def get_embedding(self):
        """Get obtained embedding."""
        if self.embedding_tensor is None:
            raise ValueError(f'Embedding of {type(self)} is not initialized.')
        return self.embedding_tensor


class DeepFmWithContext(DeepFm):
    # TODO: Context has no benefit to DeepFm in this way. Find a better way to incorporate context information.

    def __init__(self, cat_dims, embed_dim, output_dim, mlp_dims, dropout, context_embeds, **kwargs):
        """
        :param context_embeds: List[Dict], each dict contain ContextEmbedding configs.
        """
        super().__init__(cat_dims, embed_dim, output_dim, mlp_dims, dropout)

        mlp_input_dim = len(cat_dims) * embed_dim
        for i, context in enumerate(context_embeds):
            self.add_module(f'context{i}', ContextEmbedding(**context))
            if 'output_shape' in context:
                mlp_input_dim += context['output_shape'][0] * context['output_shape'][1]
            else:
                mlp_input_dim += context['embed_shape'][1]

        self.mlps = nn.ModuleList()
        for i, dim in enumerate(mlp_dims):
            self.mlps.append(nn.Sequential(
                nn.Linear(mlp_input_dim, dim, bias=True),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
            mlp_input_dim = dim
        self.mlps.append(nn.Linear(mlp_input_dim, output_dim, bias=True))

    def _get_initial_mlp_result(self, embedding, data):
        """
        :param embedding: Tensor, (n, c, e).
        :param data: Dict{data_key: Tensor}.
        :return: Tensor, (n, -1).
        """
        embedding_list = [super()._get_initial_mlp_result(embedding, data)]
        context = data['context']
        for i in range(context.shape[1]):
            context_embed = getattr(self, f'context{i}')(context[:, i:i + 1])
            embedding_list.append(context_embed.reshape(context_embed.shape[0], -1))
        return torch.cat(embedding_list, dim=1)


def build_deepfm(model_cfg):
    context_embeds = getattr(model_cfg, 'context_embeds', None)
    if context_embeds is not None:
        model = DeepFmWithContext(**model_cfg)
    else:
        model = DeepFm(**model_cfg)
    return model
