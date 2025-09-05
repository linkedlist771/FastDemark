import torch
from torch import nn
import numpy as np


class DepthwiseSeperableConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, **kwargs):
        super(DepthwiseSeperableConv2d, self).__init__()

        self.depthwise = nn.Conv2d(
            input_channels, input_channels, groups=input_channels, **kwargs
        )
        self.pointwise = nn.Conv2d(input_channels, output_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)

        return x


class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=False):
        super(Conv2dBlock, self).__init__()

        self.model = nn.Sequential(
            nn.ReflectionPad2d(int((kernel_size - 1) / 2)),
            DepthwiseSeperableConv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=0,
                bias=bias,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.model(x)


class Concat(nn.Module):
    def __init__(self, dim, *args):
        super(Concat, self).__init__()
        self.dim = dim

        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, input):
        inputs = []
        for module in self._modules.values():
            inputs.append(module(input))

        inputs_shapes2 = [x.shape[2] for x in inputs]
        inputs_shapes3 = [x.shape[3] for x in inputs]

        if np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and np.all(
            np.array(inputs_shapes3) == min(inputs_shapes3)
        ):
            inputs_ = inputs
        else:
            target_shape2 = min(inputs_shapes2)
            target_shape3 = min(inputs_shapes3)

            inputs_ = []
            for inp in inputs:
                diff2 = (inp.size(2) - target_shape2) // 2
                diff3 = (inp.size(3) - target_shape3) // 2
                inputs_.append(
                    inp[
                        :,
                        :,
                        diff2 : diff2 + target_shape2,
                        diff3 : diff3 + target_shape3,
                    ]
                )

        return torch.cat(inputs_, dim=self.dim)

    def __len__(self):
        return len(self._modules)


class SkipEncoderDecoder(nn.Module):
    def __init__(
        self,
        input_depth,
        num_channels_down=[128] * 5,
        num_channels_up=[128] * 5,
        num_channels_skip=[128] * 5,
    ):
        super(SkipEncoderDecoder, self).__init__()

        self.model = nn.Sequential()
        model_tmp = self.model

        for i in range(len(num_channels_down)):
            deeper = nn.Sequential()
            skip = nn.Sequential()

            if num_channels_skip[i] != 0:
                model_tmp.add_module(str(len(model_tmp) + 1), Concat(1, skip, deeper))
            else:
                model_tmp.add_module(str(len(model_tmp) + 1), deeper)

            model_tmp.add_module(
                str(len(model_tmp) + 1),
                nn.BatchNorm2d(
                    num_channels_skip[i]
                    + (
                        num_channels_up[i + 1]
                        if i < (len(num_channels_down) - 1)
                        else num_channels_down[i]
                    )
                ),
            )

            if num_channels_skip[i] != 0:
                skip.add_module(
                    str(len(skip) + 1),
                    Conv2dBlock(input_depth, num_channels_skip[i], 1, bias=False),
                )

            deeper.add_module(
                str(len(deeper) + 1),
                Conv2dBlock(input_depth, num_channels_down[i], 3, 2, bias=False),
            )
            deeper.add_module(
                str(len(deeper) + 1),
                Conv2dBlock(num_channels_down[i], num_channels_down[i], 3, bias=False),
            )

            deeper_main = nn.Sequential()

            if i == len(num_channels_down) - 1:
                k = num_channels_down[i]
            else:
                deeper.add_module(str(len(deeper) + 1), deeper_main)
                k = num_channels_up[i + 1]

            deeper.add_module(
                str(len(deeper) + 1), nn.Upsample(scale_factor=2, mode="nearest")
            )

            model_tmp.add_module(
                str(len(model_tmp) + 1),
                Conv2dBlock(
                    num_channels_skip[i] + k, num_channels_up[i], 3, 1, bias=False
                ),
            )
            model_tmp.add_module(
                str(len(model_tmp) + 1),
                Conv2dBlock(num_channels_up[i], num_channels_up[i], 1, bias=False),
            )

            input_depth = num_channels_down[i]
            model_tmp = deeper_main

        self.model.add_module(
            str(len(self.model) + 1), nn.Conv2d(num_channels_up[0], 3, 1, bias=True)
        )
        self.model.add_module(str(len(self.model) + 1), nn.Sigmoid())

    def forward(self, x):
        return self.model(x)


def input_noise(INPUT_DEPTH, spatial_size, scale=1.0 / 10):
    shape = [1, INPUT_DEPTH, spatial_size[0], spatial_size[1]]
    return torch.rand(*shape) * scale
