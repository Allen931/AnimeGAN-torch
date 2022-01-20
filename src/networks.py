from torch import nn
import torch.nn.functional as F

from utils import initialize_weights


class ConvNormLReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, pad_mode="reflect", groups=1, alpha=0.2,
                 bias=False):
        pad_layer = {
            "zeros": nn.ZeroPad2d,
            "same": nn.ReplicationPad2d,
            "reflect": nn.ReflectionPad2d,
        }
        if pad_mode not in pad_layer:
            raise NotImplementedError

        super(ConvNormLReLU, self).__init__(
            pad_layer[pad_mode](padding),
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=0,
                      groups=groups, bias=bias),
            nn.GroupNorm(num_groups=1, num_channels=out_ch, affine=True),
            nn.LeakyReLU(alpha, inplace=True)
        )


class Conv(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=4, stride=2, padding=0, pad_mode='zeros', sn=False, bias=False):
        pad_layer = {
            "zeros": nn.ZeroPad2d,
            "same": nn.ReplicationPad2d,
            "reflect": nn.ReflectionPad2d,
        }
        if pad_mode not in pad_layer:
            raise NotImplementedError

        if (kernel_size - stride) % 2 == 0:
            pad_top = padding
            pad_bottom = padding
            pad_left = padding
            pad_right = padding

        else:
            pad_top = padding
            pad_bottom = kernel_size - stride - pad_top
            pad_left = padding
            pad_right = kernel_size - stride - pad_left

        if sn:
            super(Conv, self).__init__(
                pad_layer[pad_mode]((pad_top, pad_bottom, pad_right, pad_left)),
                nn.utils.spectral_norm(
                    nn.Conv2d(in_channels=in_ch,
                              out_channels=out_ch,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=0,
                              padding_mode=pad_mode,
                              bias=bias)
                )
            )
        else:
            super(Conv, self).__init__(
                pad_layer[pad_mode]((pad_top, pad_bottom, pad_right, pad_left)),
                nn.Conv2d(in_channels=in_ch,
                          out_channels=out_ch,
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=0,
                          padding_mode=pad_mode,
                          bias=bias)
            )


class InvertedResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, expansion_ratio=2):
        super(InvertedResBlock, self).__init__()

        self.use_res_connect = in_ch == out_ch
        bottleneck = int(round(in_ch * expansion_ratio))
        layers = []
        if expansion_ratio != 1:
            layers.append(ConvNormLReLU(in_ch, bottleneck, kernel_size=1, padding=0))

        # dw
        layers.append(ConvNormLReLU(bottleneck, bottleneck, groups=bottleneck, bias=True))
        # pw
        layers.append(nn.Conv2d(bottleneck, out_ch, kernel_size=1, padding=0, bias=False))
        layers.append(nn.GroupNorm(num_groups=1, num_channels=out_ch, affine=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        out = self.layers(input)
        if self.use_res_connect:
            out = input + out
        return out


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.block_a = nn.Sequential(
            ConvNormLReLU(3, 32, kernel_size=7, padding=3),
            ConvNormLReLU(32, 64, stride=2, padding=(0, 1, 0, 1)),
            ConvNormLReLU(64, 64)
        )

        self.block_b = nn.Sequential(
            ConvNormLReLU(64, 128, stride=2, padding=(0, 1, 0, 1)),
            ConvNormLReLU(128, 128)
        )

        self.block_c = nn.Sequential(
            ConvNormLReLU(128, 128),
            InvertedResBlock(128, 256, 2),
            InvertedResBlock(256, 256, 2),
            InvertedResBlock(256, 256, 2),
            InvertedResBlock(256, 256, 2),
            # InvertedResBlock(256, 256, 2),
            ConvNormLReLU(256, 128),
        )

        self.block_d = nn.Sequential(
            ConvNormLReLU(128, 128),
            ConvNormLReLU(128, 128)
        )

        # self.block_c_1 = nn.Sequential(
        #     ConvNormLReLU(128, 128),
        #     InvertedResBlock(128, 256, 2),
        #     InvertedResBlock(256, 256, 2),
        #     InvertedResBlock(256, 256, 2),
        #     InvertedResBlock(256, 256, 2),
        #     InvertedResBlock(256, 256, 2),
        #     ConvNormLReLU(256, 128),
        # )
        #
        # self.block_d_1 = nn.Sequential(
        #     ConvNormLReLU(128, 128),
        #     ConvNormLReLU(128, 128)
        # )

        # self.block_c_2 = nn.Sequential(
        #     ConvNormLReLU(128, 128),
        #     InvertedResBlock(128, 256, 2),
        #     InvertedResBlock(256, 256, 2),
        #     ConvNormLReLU(256, 128),
        # )
        #
        # self.block_d_2 = nn.Sequential(
        #     ConvNormLReLU(128, 128),
        #     ConvNormLReLU(128, 128)
        # )
        #
        self.block_e = nn.Sequential(
            ConvNormLReLU(128, 64),
            ConvNormLReLU(64, 64),
            ConvNormLReLU(64, 32, kernel_size=7, padding=3)
        )

        self.out_layer = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh()
        )

        initialize_weights(self)

    def forward(self, input, align_corners=True):
        out = self.block_a(input)
        half_size = out.size()[-2:]
        out = self.block_b(out)
        out_1 = self.block_c(out)

        if align_corners:
            out_1 = F.interpolate(out_1, half_size, mode="bilinear", align_corners=True)
        else:
            out_1 = F.interpolate(out_1, scale_factor=2, mode="bilinear", align_corners=False)
        out_1 = self.block_d(out_1)

        if align_corners:
            out_1 = F.interpolate(out_1, input.size()[-2:], mode="bilinear", align_corners=True)
        else:
            out_1 = F.interpolate(out_1, scale_factor=2, mode="bilinear", align_corners=False)

        # out_2 = self.block_c_2(out)

        # if align_corners:
        #     out_2 = F.interpolate(out_2, half_size, mode="bilinear", align_corners=True)
        # else:
        #     out_2 = F.interpolate(out_2, scale_factor=2, mode="bilinear", align_corners=False)
        # out_2 = self.block_d_2(out_2)
        #
        # if align_corners:
        #     out_2 = F.interpolate(out_2, input.size()[-2:], mode="bilinear", align_corners=True)
        # else:
        #     out_2 = F.interpolate(out_2, scale_factor=2, mode="bilinear", align_corners=False)
        # out = self.block_e(out_1 + out_2)

        out = self.block_e(out_1)
        out = self.out_layer(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, ch, n_layers, sn):
        super().__init__()
        channels = ch // 2
        self.n_layers = n_layers
        self.conv_0 = nn.Sequential(
            Conv(3, channels, kernel_size=3, stride=1, padding=1, sn=sn),
            nn.LeakyReLU(0.2, inplace=True)
        )

        in_ch = channels

        for i in range(1, n_layers):
            setattr(self, "conv_s2_%d" % i, nn.Sequential(
                Conv(in_ch, channels * 2, kernel_size=3, stride=2, padding=1, sn=sn),
                nn.LeakyReLU(0.2, inplace=True)
            ))

            setattr(self, "conv_s1_%d" % i, nn.Sequential(
                Conv(channels * 2, channels * 4, kernel_size=3, stride=1, padding=1, sn=sn),
                nn.GroupNorm(num_groups=1, num_channels=channels * 4, affine=True),
                nn.LeakyReLU(0.2, inplace=True)
            ))

            in_ch = channels * 4
            channels *= 2

        channels *= 2
        setattr(self, "last_conv", nn.Sequential(
            Conv(channels, channels, kernel_size=3, stride=1, padding=1, sn=sn),
            nn.GroupNorm(num_groups=1, num_channels=channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        ))

        setattr(self, "D_logit", Conv(channels, 1, kernel_size=3, stride=1, padding=1, sn=sn))

        initialize_weights(self)

    def forward(self, input):
        out = self.conv_0(input)

        for i in range(1, self.n_layers):
            out = getattr(self, "conv_s2_%d" % i)(out)
            out = getattr(self, "conv_s1_%d" % i)(out)

        out = getattr(self, "last_conv")(out)
        out = getattr(self, "D_logit")(out)

        return out
