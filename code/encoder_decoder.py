import torch.nn as nn
from types_ import *


class EncoderDecoder(nn.Module):

    def __init__(self, encoder, decoder, normalize_flag=False):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.normalize_flag = normalize_flag


    def forward(self, input: Tensor) -> Tensor:
        encoded_input = self.encode(input)
        if self.normalize_flag:
            encoded_input = nn.functional.normalize(encoded_input, p=2, dim=1)
        output = self.decoder(encoded_input)

        return output

    def encode(self, input: Tensor) -> Tensor:
        return self.encoder(input)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)
