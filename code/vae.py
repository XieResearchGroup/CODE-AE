import torch
from torch import nn
from torch.nn import functional as F
from base_ae import BaseAE
from types_ import *
from typing import List

class VAE(BaseAE):

    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: List = None, dop: float = 0.1, noise_flag: bool = True, **kwargs) -> None:
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.dop = dop
        self.noise_flag = noise_flag

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # build encoder
        modules = []

        modules.append(
            nn.Sequential(
                nn.Linear(input_dim, hidden_dims[0], bias=True),
                #nn.BatchNorm1d(hidden_dims[0]),
                nn.ReLU(),
                nn.Dropout(self.dop)
            )
        )

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1], bias=True),
                    #nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.ReLU(),
                    nn.Dropout(self.dop)
                )
            )

        self.embedder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim, bias=True)
        self.encoder = nn.Sequential(
            self.embedder,
            self.fc_mu
        )
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim, bias=True)

        # build decoder
        modules = []

        modules.append(
            nn.Sequential(
                nn.Linear(latent_dim, hidden_dims[-1], bias=True),
                #nn.BatchNorm1d(hidden_dims[-1]),
                nn.ReLU(),
                nn.Dropout(self.dop)
            )
        )

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1], bias=True),
                    #nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.ReLU(),
                    nn.Dropout(self.dop)
                )
            )
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1], bias=True),
            #nn.BatchNorm1d(hidden_dims[-1]),
            nn.ReLU(),
            nn.Dropout(self.dop),
            nn.Linear(hidden_dims[-1], input_dim)
        )

    def encode(self, input: Tensor) -> Tensor:
        if self.noise_flag and self.training:
            embed = self.embedder(input+torch.randn_like(input, requires_grad=False) * 0.1)
        else:
            embed = self.embedder(input)

        mu = self.fc_mu(embed)
        log_var = self.fc_var(embed)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        embed = self.decoder(z)
        outputs = self.final_layer(embed)

        return outputs

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [input, self.decode(z), mu, log_var]

    def loss_function(self, *args, **kwargs) -> dict:
        input = args[0]
        recons = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N'] if 'M_N' in kwargs else 1.0

        recons_loss = F.mse_loss(input, recons)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        loss = recons_loss + kld_weight * kld_loss

        return {'loss': loss, 'recons_loss': recons_loss, 'KLD': kld_loss}

    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
        z = torch.randn(num_samples, self.latent_dim)

        z = z.to(current_device)
        samples = self.decode(z)

        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x)[1]
