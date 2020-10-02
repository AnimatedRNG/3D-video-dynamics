import torch
import torch.nn as nn
import numpy as np
from torch_utils import positional_encoding
from sirens import SineLayer


class IGR(nn.Module):
    def __init__(
        self,
        input_dims,
        depths,
        skip_in=(4,),
        L=None,
        geometric_init=True,
        radius_init=1,
        beta=100,
    ):
        super().__init__()

        self.input_dims = input_dims
        self.L = L
        self.skip_in = skip_in
        self.layers = nn.ModuleList()

        self.embedding_size = (
            2 * self.L * self.input_dims if self.L is not None else input_dims
        )

        depths = [self.embedding_size] + depths + [1]

        self.num_layers = len(depths)

        for layer_id in range(0, self.num_layers - 1):
            if layer_id + 1 in skip_in:
                out_dim = depths[layer_id + 1] - self.embedding_size
            else:
                out_dim = depths[layer_id + 1]

            linear_layer = nn.Linear(depths[layer_id], out_dim)

            # initialize the layer
            if geometric_init:
                if layer_id == self.num_layers - 2:
                    torch.nn.init.normal_(
                        linear_layer.weight,
                        mean=np.sqrt(np.pi) / np.sqrt(depths[layer_id]),
                        std=1e-5,
                    )
                    torch.nn.init.constant_(linear_layer.bias, -radius_init)
                else:
                    torch.nn.init.constant_(linear_layer.bias, 0.0)
                    torch.nn.init.normal_(
                        linear_layer.weight, 0.0, np.sqrt(2.0) / np.sqrt(out_dim)
                    )

            self.layers.append(linear_layer)

        if beta > 0:
            self.activation = nn.Softplus(beta=beta)
        else:
            self.activation = nn.ReLU()

    def forward(self, inp):
        inp = inp.clone().detach().requires_grad_(True)
        emb = inp

        if self.L is not None:
            emb = positional_encoding(inp, self.L).view(-1, self.embedding_size)
        x = emb

        for layer_id, layer in enumerate(self.layers):
            if layer_id in self.skip_in:
                x = torch.cat([x, emb], -1) / np.sqrt(2)

            x = layer(x)

            if layer_id < self.num_layers - 2:
                x = self.activation(x)

        return x, inp
