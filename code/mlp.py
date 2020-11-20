from torch import nn

from types_ import *


class MLP(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List = None, **kwargs) -> None:
        super(MLP, self).__init__()
        self.output_dim = output_dim

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        modules = []
        modules.append(
            nn.Sequential(
                nn.Linear(input_dim, hidden_dims[0], bias=True),
                #nn.BatchNorm1d(hidden_dims[0]),
                nn.ReLU(),
                nn.Dropout(0.1),
            )
        )

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1], bias=True),
                    #nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                )
            )

        self.module = nn.Sequential(*modules)

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1], bias=True),
            #nn.BatchNorm1d(hidden_dims[-1]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dims[-1], output_dim, bias=True),
        )

    def forward(self, input: Tensor) -> Tensor:
        embed = self.module(input)
        output = self.output_layer(embed)

        return output
