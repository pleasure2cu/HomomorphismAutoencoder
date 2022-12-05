from typing import List, Tuple

import torch


class SmallDenseEncoder(torch.nn.Module):

    def __init__(self, widths: List[int]):
        super(SmallDenseEncoder, self).__init__()
        self.layers = torch.nn.ModuleList()
        # add flatten layer at the beginning
        self.layers.append(torch.nn.Flatten())
        for i in range(len(widths) - 1):
            self.layers.append(torch.nn.Linear(widths[i], widths[i+1]))
            if i < len(widths) - 2:
                self.layers.append(torch.nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class SmallDenseDecoder(torch.nn.Module):

    def __init__(self, widths: List[int], final_shape: Tuple[int, ...]):
        super(SmallDenseDecoder, self).__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(len(widths) - 1):
            self.layers.append(torch.nn.Linear(widths[i], widths[i+1]))
            if i < len(widths) - 2:
                self.layers.append(torch.nn.ReLU())
        # add reshape layer at the end
        self.layers.append(torch.nn.Unflatten(1, final_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class SmallDensePhi(torch.nn.Module):

    def __init__(self, widths: List[int]):
        super(SmallDensePhi, self).__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(len(widths) - 1):
            self.layers.append(torch.nn.Linear(widths[i], widths[i+1]))
            if i < len(widths) - 2:
                self.layers.append(torch.nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class SmallConvEncoder(torch.nn.Module):

    def __init__(
            self, nbr_channels_start: int, kernel_sizes: List[int], channels: List[int], strides: List[int],
            dense_widths: List[int]
    ):
        super(SmallConvEncoder, self).__init__()
        self.layers = torch.nn.ModuleList()

        # add conv layers
        for i in range(len(channels)):
            self.layers.append(torch.nn.Conv2d(nbr_channels_start, channels[i], kernel_sizes[i], strides[i]))
            self.layers.append(torch.nn.ReLU())
            self.layers.append(torch.nn.BatchNorm2d(channels[i]))
            nbr_channels_start = channels[i]

        # add flatten layer
        self.layers.append(torch.nn.Flatten())

        # add dense layers
        for i in range(len(dense_widths) - 1):
            self.layers.append(torch.nn.Linear(dense_widths[i], dense_widths[i+1]))
            if i < len(dense_widths) - 2:
                self.layers.append(torch.nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class SmallConvDecoder(torch.nn.Module):

    def __init__(
            self, kernel_sizes: List[int], channels: List[int], strides: List[int],
            dense_widths: List[int], shape_after_dense: Tuple[int, int, int], remove_channel_at_end: bool
    ):
        super(SmallConvDecoder, self).__init__()
        self.layers = torch.nn.ModuleList()

        # add dense layers
        for i in range(len(dense_widths) - 1):
            self.layers.append(torch.nn.Linear(dense_widths[i], dense_widths[i+1]))
            self.layers.append(torch.nn.ReLU())

        # add reshape layer
        self.layers.append(torch.nn.Unflatten(1, shape_after_dense))

        # add conv layers
        nbr_channels = shape_after_dense[0]
        for i in range(len(channels)):
            self.layers.append(torch.nn.ConvTranspose2d(nbr_channels, channels[i], kernel_sizes[i], strides[i]))
            if i < len(channels) - 1:
                self.layers.append(torch.nn.ReLU())
                self.layers.append(torch.nn.BatchNorm2d(channels[i]))
            nbr_channels = channels[i]

        if remove_channel_at_end:
            self.layers.append(torch.nn.Flatten(1, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
