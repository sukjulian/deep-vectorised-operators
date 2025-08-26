import torch
import torch_dvf as pydvf
import torch_geometric as pyg


class MLP(torch.nn.Module):
    def __init__(
        self,
        num_input_channels: int,
        num_output_channels: int,
        num_layers: int,
        num_latent_channels: int,
    ):
        super().__init__()

        self.backend = pydvf.nn.mlp.vanilla.MLP(
            num_channels=(
                num_input_channels,
                *[num_latent_channels] * (num_layers - 1),
                num_output_channels,
            ),
            use_norm_in_first=False,
        )

        self.backend = torch.compile(self.backend)

        print(
            f"MLP ({sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)} parameters)"
        )

    def forward(self, data: pyg.data.Data) -> torch.Tensor:
        return self.backend(data.x)
