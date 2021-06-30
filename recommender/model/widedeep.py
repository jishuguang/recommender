from torch import nn


class WideDeep(nn.Module):
    """Wide and deep model.

    Reference: https://arxiv.org/abs/1606.07792v1.
    """
    # TODO: implement this model.

    def __init__(self):
        super().__init__()

    def forward(self, data):
        """
        :param data: Dict{data_key: Tensor}.
        :return: Tensor, (n, output_dim).
        """
        return data
