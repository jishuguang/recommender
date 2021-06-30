from torch import nn


class BinaryClassInteraction(nn.Module):
    """Multi-class binary classification interaction."""
    # TODO: investigate a better interacting way for multi-task classification.

    def __init__(self, output_dim, *modules):
        """
        :param output_dim: output dimension, i.e. the number of classes.
        :param modules: nn.Module, each module output a Tensor(n, output_dim).
        """
        super().__init__()
        self.linear = nn.Linear(output_dim, output_dim, bias=True)

        self.module_amount = len(modules)
        assert self.module_amount > 0
        for i, module in enumerate(modules):
            self.add_module(f'module{i}', module)

    def forward(self, data):
        """
        :param data: Dict{data_key: Tensor}.
        :return: Tensor, (n, output_dim).
        """
        # Shape: (n, output_dim)
        raw_result = getattr(self, 'module0')(data)
        for i in range(1, self.module_amount):
            raw_result = raw_result + getattr(self, f'module{i}')(data)
        result = self.linear(raw_result)
        return result

    def infer(self, data):
        """
        :param data: Dict{data_key: Tensor}.
        :return: Tensor, (n, output_dim).
        """
        return self(data).sigmoid()
