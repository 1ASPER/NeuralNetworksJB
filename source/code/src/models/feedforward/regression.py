"""A fully connected network that predicts one number."""
import torch


class Regressor(torch.nn.Module):
    """``dim_input -> dim_hidden -> ... -> 1`` with ReLU, ``n_hidden`` hidden layers.

    ``forward`` reads ``batch['data']['features']`` (``[batch, dim_input]``)
    and writes ``batch['signals']['output']`` (``[batch]``).
    """

    def __init__(self, dim_input, dim_hidden=64, n_hidden=2):
        super().__init__()
        layers = []
        for index in range(n_hidden):
            layers += [torch.nn.Linear(dim_input if index == 0 else dim_hidden, dim_hidden),
                       torch.nn.ReLU()]
        layers.append(torch.nn.Linear(dim_hidden, 1))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, batch):
        batch['signals'] = {'output': self.layers(batch['data']['features']).squeeze(-1)}
        return batch
