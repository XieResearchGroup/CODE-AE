import torch.nn as nn

from types_ import *


class EncoderDecoder(nn.Module):

    def __init__(self, encoder, decoder):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input: Tensor) -> Tensor:
        encoded_input = self.encoder(input)
        output = self.decoder(encoded_input)

        return output

    def encode(self, input: Tensor) -> Tensor:
        return self.encoder(input)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)
