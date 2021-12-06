# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

import os

def SyncBatchNorm(*args, **kwargs):
    """In cpu environment nn.SyncBatchNorm does not have kernel so use nn.BatchNorm2D instead"""
    if paddle.get_device() == 'cpu' or os.environ.get('PADDLESEG_EXPORT_STAGE'):
        return nn.BatchNorm2D(*args, **kwargs)
    elif paddle.distributed.ParallelEnv().nranks == 1:
        return nn.BatchNorm2D(*args, **kwargs)
    else:
        return nn.SyncBatchNorm(*args, **kwargs)


# @manager.MODELS.add_component
class ESPNetV1(nn.Layer):
    def __init__(self, num_classes, in_channels=3, level2_depth=2, level3_depth=3, pretrained=None):
        super().__init__()
        self.encoder = ESPNetEncoder(num_classes, in_channels, level2_depth, level3_depth)

        self.level3_up = nn.Conv2DTranspose(num_classes, num_classes, 2, stride=2, padding=0, output_padding=0, bias_attr=False)
        self.br3 = SyncBatchNorm(num_classes)
        self.level2_proj = Conv(in_channels + 128, num_classes, 1, 1)
        self.combine_l2_l3 = nn.Sequential(
            BNPReLU(2 * num_classes),
            DilatedResidualBlock(2 * num_classes, num_classes, residual=False),
        )
        self.level2_up = nn.Sequential(
            nn.Conv2DTranspose(num_classes, num_classes, 2, stride=2, padding=0, output_padding=0, bias_attr=False),
            BNPReLU(num_classes),
        ) 
        self.out_proj = ConvBNPReLU(16 + in_channels + num_classes, num_classes, 3, 1)
        self.out_up = nn.Conv2DTranspose(num_classes, num_classes, 2, stride=2, padding=0, output_padding=0, bias_attr=False)
    
    def forward(self, x):
        feat1, feat2, feat3 = self.encoder(x) # shape [N, 19, H, W]  [N, 131, H, W]  [N, C, H, W] 

        feat3 = self.level3_up(self.br3(feat3))
        feat2 = self.level2_proj(feat2)
        merge_l2_l3 = self.combine_l2_l3(paddle.concat([feat2, feat3], axis=1))

        up2 = self.level2_up(merge_l2_l3)
        out = self.out_proj(paddle.concat([up2, feat1], axis=1))
        out = self.out_up(out)
        return [out]


class ConvBNPReLU(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2D(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias_attr=False)
        self.bn = SyncBatchNorm(out_channels)
        self.act = nn.PReLU(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class BNPReLU(nn.Layer):
    def __init__(self, channels):
        super().__init__()
        self.bn = SyncBatchNorm(channels)
        self.act = nn.PReLU(channels)

    def forward(self, x):
        x = self.bn(x)
        x = self.act(x)
        return x


class ConvBN(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2D(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias_attr=False)
        self.bn = SyncBatchNorm(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Conv(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2D(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias_attr=False)

    def forward(self, x):
        x = self.conv(x)
        return x


class ConvDilated(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()
        padding = int((kernel_size - 1) / 2) * dilation
        self.conv = nn.Conv2D(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias_attr=False)

    def forward(self, x):
        x = self.conv(x)
        return x


class DownSampler(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        branch_channels = out_channels // 5
        remain_channels = out_channels - branch_channels * 4
        self.conv1 = Conv(in_channels, branch_channels, 3, 2)
        self.d_conv1 = ConvDilated(branch_channels, remain_channels, 3, 1, 1)
        self.d_conv2 = ConvDilated(branch_channels, branch_channels, 3, 1, 2)
        self.d_conv4 = ConvDilated(branch_channels, branch_channels, 3, 1, 4)
        self.d_conv8 = ConvDilated(branch_channels, branch_channels, 3, 1, 8)
        self.d_conv16 = ConvDilated(branch_channels, branch_channels, 3, 1, 16)
        self.bn = SyncBatchNorm(out_channels)
        self.act = nn.PReLU(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        d1 = self.d_conv1(x)
        d2 = self.d_conv2(x)
        d4 = self.d_conv4(x)
        d8 = self.d_conv8(x)
        d16 = self.d_conv16(x)

        feat1 = d2
        feat2 = feat1 + d4
        feat3 = feat2 + d8
        feat4 = feat3 + d16

        feat = paddle.concat([d1, feat1, feat2, feat3, feat4], axis=1)
        out = self.bn(feat)
        out = self.act(out)
        return out
        

class DilatedResidualBlock(nn.Layer):
    def __init__(self, in_channels, out_channels, residual=True):
        super().__init__()
        branch_channels = out_channels // 5
        remain_channels = out_channels - branch_channels * 4
        self.conv1 = Conv(in_channels, branch_channels, 1, 1)
        self.d_conv1 = ConvDilated(branch_channels, remain_channels, 3, 1, 1)
        self.d_conv2 = ConvDilated(branch_channels, branch_channels, 3, 1, 2)
        self.d_conv4 = ConvDilated(branch_channels, branch_channels, 3, 1, 4)
        self.d_conv8 = ConvDilated(branch_channels, branch_channels, 3, 1, 8)
        self.d_conv16 = ConvDilated(branch_channels, branch_channels, 3, 1, 16)
        self.bn = BNPReLU(out_channels)
        self.residual = residual

    def forward(self, x):
        x_proj = self.conv1(x)
        d1 = self.d_conv1(x_proj)
        d2 = self.d_conv2(x_proj)
        d4 = self.d_conv4(x_proj)
        d8 = self.d_conv8(x_proj)
        d16 = self.d_conv16(x_proj)

        feat1 = d2
        feat2 = feat1 + d4
        feat3 = feat2 + d8
        feat4 = feat3 + d16

        feat = paddle.concat([d1, feat1, feat2, feat3, feat4], axis=1)

        if self.residual:
            feat = feat + x
        out = self.bn(feat)
        return out


class PoolDown(nn.Layer):
    def __init__(self, down_sampling_times):
        super().__init__()
        self.pool = nn.Sequential(
            *[
                nn.AvgPool2D(3, stride=2, padding=1) for i in range(down_sampling_times)
            ]
        )

    def forward(self, x):
        x = self.pool(x)
        return x


class ESPNetEncoder(nn.Layer):
    def __init__(self, num_classes, in_channels=3, level2_depth=5, leve3_depth=3):
        super().__init__()
        self.level1 = ConvBNPReLU(in_channels, 16, 3, 2)
        self.sample1 = PoolDown(1)
        self.sample2 = PoolDown(2)

        self.br1 = BNPReLU(in_channels + 16)
        self.level2_0 = DownSampler(in_channels + 16, 64)

        self.level2 = nn.LayerList()
        for i in range(level2_depth):
            self.level2.append(DilatedResidualBlock(64, 64))
        self.br2 = BNPReLU(in_channels + 128)

        self.level3_0 = DownSampler(in_channels + 128, 128)
        self.level3 = nn.LayerList()
        for i in range(0, leve3_depth):
            self.level3.append(DilatedResidualBlock(128, 128))
        self.br3 = BNPReLU(256)

        self.head = Conv(256, num_classes, 1, 1)


    def forward(self, x):
        output0 = self.level1(x)
        input_res1 = self.sample1(x)
        input_res2 = self.sample2(x)
        outputs = []

        output0_cat = self.br1(paddle.concat([output0, input_res1], axis=1))
        outputs.append(output0_cat)
        output1 = self.level2_0(output0_cat)
        feats = [input_res2, output1]

        for i, layer in enumerate(self.level2):
            output1 = layer(output1)
        feats.append(output1)
        output1_cat = self.br2(paddle.concat(feats, axis=1))
        outputs.append(output1_cat)

        output2 = self.level3_0(output1_cat)
        feats = [output2]
        for i, layer in enumerate(self.level3):
            output2 = layer(output2)
        feats.append(output2)
        output2_cat = self.br3(paddle.concat(feats, axis=1))
        out = self.head(output2_cat)
        outputs.append(out)
        return outputs


if __name__ == '__main__':
    model = ESPNetV1(19, 3, 2, 8)
    paddle.summary(model, (4, 3, 256, 256))




















