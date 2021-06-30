import torch
from torch import nn

from .deepfm import DeepFm


class XDeepFm(nn.Module):
    """XDeepFm model. (Note, this is slightly different from the model in original paper.)

    Reference: https://www.microsoft.com/en-us/research/uploads/prod/2019/07/ACM_Proceding_p1754-lian.pdf.
    """

    def __init__(self, cat_dims, embed_dim, output_dim,
                 mlp_dims, dropout,
                 cin_dims,
                 **kwargs):
        super().__init__()
        self.deepfm = DeepFm(cat_dims, embed_dim, output_dim, mlp_dims, dropout)

        # cin
        self.cins = nn.ModuleList()
        input_dim = len(cat_dims)
        last_dim = input_dim
        for dim in cin_dims:
            self.cins.append(nn.Sequential(
                nn.Conv3d(1, dim, (1, last_dim, input_dim), bias=False),
                nn.Dropout(dropout)
            ))
            last_dim = dim
        self.linear = nn.Linear(sum(cin_dims), output_dim, bias=True)

    def forward(self, data):
        """
        :param data: Dict{data_key: Tensor}.
        :return: Tensor, (n, output_dim).
        """
        deepfm_result = self.deepfm(data)

        # cin
        embedding = self.deepfm.get_embedding()  # (n, c, e)
        input_feature = embedding.permute(0, 2, 1)  # (n, e, c)
        last_feature = input_feature
        feature_list = list()
        for cin in self.cins:
            feature_3d = input_feature.unsqueeze(-2) * last_feature.unsqueeze(-1)  # (n, e, k, c)
            feature_3d = feature_3d.unsqueeze(1)  # (n, 1, e, k, c)
            feature = cin(feature_3d)  # (n, k_next, e, 1, 1)
            feature = feature.squeeze()  # (n, k_next, e)
            feature = feature.permute(0, 2, 1)  # (n, e, k_next)
            feature_list.append(feature)
            last_feature = feature
        output = torch.cat(feature_list, dim=2).sum(dim=1)  # (n, sum(k))
        cin_result = self.linear(output)

        return deepfm_result + cin_result


def build_xdeepfm(model_cfg):
    context_embeds = getattr(model_cfg, 'context_embeds', None)
    if context_embeds is not None:
        model = XDeepFm(**model_cfg)
    else:
        model = XDeepFm(**model_cfg)
    return model
