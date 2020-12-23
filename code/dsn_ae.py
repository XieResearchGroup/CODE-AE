import torch
from torch import nn
from torch.nn import functional as F
from base_ae import BaseAE
from types_ import *
from typing import List


class DSNAE(BaseAE):

    def __init__(self, shared_encoder, decoder, input_dim: int, latent_dim: int, alpha: float = 1.0,
                 hidden_dims: List = None, dop: float = 0.1, noise_flag: bool = False, norm_flag: bool = False,
                 **kwargs) -> None:
        super(DSNAE, self).__init__()
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.noise_flag = noise_flag
        self.dop = dop
        self.norm_flag = norm_flag

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        self.shared_encoder = shared_encoder
        self.decoder = decoder
        # build encoder
        modules = []

        modules.append(
            nn.Sequential(
                nn.Linear(input_dim, hidden_dims[0], bias=True),
                # nn.BatchNorm1d(hidden_dims[0]),
                nn.ReLU(),
                nn.Dropout(self.dop)
            )
        )

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1], bias=True),
                    # nn.Dropout(0.1),
                    # nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.ReLU(),
                    nn.Dropout(self.dop)
                )
            )
        modules.append(nn.Dropout(self.dop))
        modules.append(nn.Linear(hidden_dims[-1], latent_dim, bias=True))
        # modules.append(nn.LayerNorm(latent_dim, eps=1e-12, elementwise_affine=False))

        self.private_encoder = nn.Sequential(*modules)

        # build decoder
        # modules = []
        #
        # modules.append(
        #     nn.Sequential(
        #         nn.Linear(2 * latent_dim, hidden_dims[-1], bias=True),
        #         # nn.Dropout(0.1),
        #         nn.BatchNorm1d(hidden_dims[-1]),
        #         nn.ReLU()
        #     )
        # )
        #
        # hidden_dims.reverse()
        #
        # for i in range(len(hidden_dims) - 1):
        #     modules.append(
        #         nn.Sequential(
        #             nn.Linear(hidden_dims[i], hidden_dims[i + 1], bias=True),
        #             nn.BatchNorm1d(hidden_dims[i + 1]),
        #             # nn.Dropout(0.1),
        #             nn.ReLU()
        #         )
        #     )
        # self.decoder = nn.Sequential(*modules)

        # self.final_layer = nn.Sequential(
        #     nn.Linear(hidden_dims[-1], hidden_dims[-1], bias=True),
        #     nn.BatchNorm1d(hidden_dims[-1]),
        #     nn.ReLU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(hidden_dims[-1], input_dim)
        # )

    def p_encode(self, input: Tensor) -> Tensor:
        if self.noise_flag and self.training:
            latent_code = self.private_encoder(input + torch.randn_like(input, requires_grad=False) * 0.1)
        else:
            latent_code = self.private_encoder(input)

        if self.norm_flag:
            return F.normalize(latent_code, p=2, dim=1)
        else:
            return latent_code

    def s_encode(self, input: Tensor) -> Tensor:
        if self.noise_flag and self.training:
            latent_code = self.shared_encoder(input + torch.randn_like(input, requires_grad=False) * 0.1)
        else:
            latent_code = self.shared_encoder(input)
        if self.norm_flag:
            return F.normalize(latent_code, p=2, dim=1)
        else:
            return latent_code

    def encode(self, input: Tensor) -> Tensor:
        p_latent_code = self.p_encode(input)
        s_latent_code = self.s_encode(input)

        return torch.cat((p_latent_code, s_latent_code), dim=1)

    def decode(self, z: Tensor) -> Tensor:
        # embed = self.decoder(z)
        # outputs = self.final_layer(embed)
        outputs = self.decoder(z)

        return outputs

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        z = self.encode(input)
        return [input, self.decode(z), z]

    def loss_function(self, *args, **kwargs) -> dict:
        input = args[0]
        recons = args[1]
        z = args[2]

        p_z = z[:, :z.shape[1] // 2]
        s_z = z[:, z.shape[1] // 2:]

        recons_loss = F.mse_loss(input, recons)

        s_l2_norm = torch.norm(s_z, p=2, dim=1, keepdim=True).detach()
        s_l2 = s_z.div(s_l2_norm.expand_as(s_z) + 1e-6)

        p_l2_norm = torch.norm(p_z, p=2, dim=1, keepdim=True).detach()
        p_l2 = p_z.div(p_l2_norm.expand_as(p_z) + 1e-6)

        ortho_loss = torch.mean((s_l2.t().mm(p_l2)).pow(2))
        # ortho_loss = torch.square(torch.norm(torch.matmul(s_z.t(), p_z), p='fro'))
        # ortho_loss = torch.mean(torch.square(torch.diagonal(torch.matmul(p_z, s_z.t()))))
        # if recons_loss > ortho_loss:
        #     loss = recons_loss + self.alpha * 0.1 * ortho_loss
        # else:
        loss = recons_loss + self.alpha * ortho_loss
        return {'loss': loss, 'recons_loss': recons_loss, 'ortho_loss': ortho_loss}

    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(current_device)
        samples = self.decode(z)

        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x)[1]
