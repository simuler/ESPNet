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

import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg import utils
from paddleseg.cvlibs import manager, param_init
from paddleseg.models import layers


class BNPReLU(nn.Layer):
    def __init__(self, out_channels, **kwargs):
        super().__init__()
        if 'data_format' in kwargs:
            data_format = kwargs['data_format']
        else:
            data_format = 'NCHW'
        self._batch_norm = layers.SyncBatchNorm(out_channels,
                                                data_format=data_format)
        self._prelu = layers.Activation("prelu")

    def forward(self, x):
        x = self._batch_norm(x)
        x = self._prelu(x)
        return x

class C(nn.Layer):
    def __init__(self, input_channels, out_channels, kSize, stride=1):
        super().__init__()
        padding = int((kSize - 1)/2)
        self.conv = nn.Conv2D(input_channels, out_channels, (kSize, kSize),stride=stride, padding=padding, bias_attr=False)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output


class CDilated(nn.Layer):
    def __init__(self, input_channels, out_channels, kSize, stride=1, d=1):
        super().__init__()
        padding = int((kSize - 1)/2) * d
        self.conv = nn.Conv2D(input_channels, out_channels, (kSize,kSize), stride=stride, padding=padding, bias_attr=False,dilation=d)

    def forward(self,input):
        output = self.conv(input)
        return output


class DownSamplerB(nn.Layer):
    def __init__(self, input_channels, out_channels):
        super().__init__()
        n = int(out_channels/5)
        n1 = out_channels - 4*n
        self.c1 = C(input_channels, n, 3, 2)
        self.d1 = CDilated(n, n1, 3, 1, 1)
        self.d2 = CDilated(n, n, 3, 1, 2)
        self.d4 = CDilated(n, n, 3, 1, 4)
        self.d8 = CDilated(n, n, 3, 1, 8)
        self.d16 = CDilated(n, n, 3, 1, 16)
        self.bn = nn.BatchNorm2D(out_channels, epsilon=1e-3)
        self._prelu = layers.Activation("prelu")

    def forward(self, input):
        output1 = self.c1(input)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)

        add1 = d2
        # add2 = add1 + d4
        # add3 = add2 + d8
        # add4 = add3 + d16 elementwise_add(X, Y, axis=0)
        add2 = paddle.add(add1, d4)
        add3 = paddle.add(add2, d8)
        add4 = paddle.add(add3, d16)

        combine = paddle.concat([d1, add1, add2, add3, add4],axis=1)
        output = self.bn(combine)
        output = self._prelu(output)
        return output


class DilatedParllelResidualBlockB(nn.Layer):
    def __init__(self, input_channels, out_channels, add=True):
        super().__init__()
        n = int(out_channels/5)
        n1 = out_channels - 4*n
        self.c1 = C(input_channels, n, 1, 1)
        self.d1 = CDilated(n, n1, 3, 1, 1) # dilation rate of 2^0
        self.d2 = CDilated(n, n, 3, 1, 2) # dilation rate of 2^1
        self.d4 = CDilated(n, n, 3, 1, 4) # dilation rate of 2^2
        self.d8 = CDilated(n, n, 3, 1, 8) # dilation rate of 2^3
        self.d16 = CDilated(n, n, 3, 1, 16) # dilation rate of 2^4
        self.bn = BNPReLU(out_channels)
        self.add = add

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        # reduce
        output1 = self.c1(input)
        # split and transform
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)

        # heirarchical fusion for de-gridding
        add1 = d2
        # add2 = add1 + d4
        # add3 = add2 + d8
        # add4 = add3 + d16
        add2 = paddle.add(add1, d4)
        add3 = paddle.add(add2, d8)
        add4 = paddle.add(add3, d16)

        #merge
        combine = paddle.concat([d1, add1, add2, add3, add4], axis=1)

        # if residual version
        if self.add:
            combine = input + combine
        output = self.bn(combine)
        return output


class InputProjectionA(nn.Layer):
    '''
    This class projects the input image to the same spatial dimensions as the feature map.
    For example, if the input image is 512 x512 x3 and spatial dimensions of feature map size are 56x56xF, then
    this class will generate an output of 56x56x3
    '''
    def __init__(self, samplingTimes):
        '''
        :param samplingTimes: The rate at which you want to down-sample the image
        '''
        super().__init__()
        self.pool = nn.LayerList()
        for i in range(0, samplingTimes):
            #pyramid-based approach for down-sampling
            self.pool.append(nn.AvgPool2D(3, stride=2, padding=1))

    def forward(self, input):
        '''
        :param input: Input RGB Image
        :return: down-sampled image (pyramid-based approach)
        '''
        for pool in self.pool:
            input = pool(input)
        return input


class ESPNet_Encoder(nn.Layer):
    '''
    This class defines the ESPNet-C network in the paper
    '''
    def __init__(self, classes=19, p=5, q=3):
        '''
        :param classes: number of classes in the dataset. Default is 20 for the cityscapes
        :param p: depth multiplier
        :param q: depth multiplier
        '''
        super().__init__()
        self.level1 = layers.ConvBNPReLU(3, 16, 3, 2)
        self.sample1 = InputProjectionA(1)
        self.sample2 = InputProjectionA(2)

        self.b1 = BNPReLU(16 + 3)
        self.level2_0 = DownSamplerB(16 +3, 64)

        self.level2 = nn.LayerList()
        for i in range(0, p):
            self.level2.append(DilatedParllelResidualBlockB(64 , 64))
        self.b2 = BNPReLU(128 + 3)

        self.level3_0 = DownSamplerB(128 + 3, 128)
        self.level3 = nn.LayerList()
        for i in range(0, q):
            self.level3.append(DilatedParllelResidualBlockB(128 , 128))
        self.b3 = BNPReLU(256)

        self.classifier = C(256, classes, 1, 1)

    def forward(self, input):
        '''
        :param input: Receives the input RGB image
        :return: the transformed feature map with spatial dimensions 1/8th of the input image
        '''
        print('encoder',input)
        output0 = self.level1(input)
        inp1 = self.sample1(input)
        inp2 = self.sample2(input)

        output0_cat = self.b1(paddle.concat([output0, inp1], axis=1))
        output1_0 = self.level2_0(output0_cat) # down-sampled
        
        for i, layer in enumerate(self.level2):
            if i==0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)

        output1_cat = self.b2(paddle.concat([output1,  output1_0, inp2], axis=1))

        output2_0 = self.level3_0(output1_cat) # down-sampled
        for i, layer in enumerate(self.level3):
            if i==0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)

        output2_cat = self.b3(paddle.concat([output2_0, output2], axis=1))

        classifier = self.classifier(output2_cat)

        return classifier

@manager.MODELS.add_component
class ESPNet(nn.Layer):
    '''
    This class defines the ESPNet network
    '''

    def __init__(self, num_classes=19, p=2, q=3):
        '''
        :param classes: number of classes in the dataset. Default is 20 for the cityscapes
        :param p: depth multiplier
        :param q: depth multiplier
        :param encoderFile: pretrained encoder weights. Recall that we first trained the ESPNet-C and then attached the
                            RUM-based light weight decoder. See paper for more details.
        '''
        super().__init__()
        classes = num_classes
        self.encoder = ESPNet_Encoder(classes, p, q)
        # print('encoderfile',encoderFile)
        # if encoderFile != None:
        #     self.encoder.set_state_dict(paddle.load(encoderFile))
        #     print('Encoder loaded!')
        # load the encoder modules
        self.modules = list()
        for i, m in enumerate(self.encoder.children()):
            self.modules.append(m)

        # light-weight decoder
        self.level3_C = C(128 + 3, classes, 1, 1)
        self.br = nn.BatchNorm2D(classes, epsilon=1e-3)
        self.conv = layers.ConvBNPReLU(19 + classes, classes, 3, 1)

        self.up_l3 = nn.Sequential(nn.Conv2DTranspose(classes, classes, 2, stride=2, padding=0, output_padding=0, bias_attr=False))
        self.combine_l2_l3 = nn.Sequential(BNPReLU(2*classes), DilatedParllelResidualBlockB(2*classes , classes, add=False))

        self.up_l2 = nn.Sequential(nn.Conv2DTranspose(classes, classes, 2, stride=2, padding=0, output_padding=0, bias_attr=False), BNPReLU(classes))

        self.classifier = nn.Conv2DTranspose(classes, classes, 2, stride=2, padding=0, output_padding=0, bias_attr=False)

    def forward(self, input):
        '''
        :param input: RGB image
        :return: transformed feature map
        '''
        output0 = self.modules[0](input)
        inp1 = self.modules[1](input)
        inp2 = self.modules[2](input)

        output0_cat = self.modules[3](paddle.concat([output0, inp1], axis=1))
        output1_0 = self.modules[4](output0_cat)  # down-sampled

        for i, layer in enumerate(self.modules[5]):
            if i == 0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)
        output1_cat = self.modules[6](paddle.concat([output1, output1_0, inp2], axis=1))

        output2_0 = self.modules[7](output1_cat)  # down-sampled
        for i, layer in enumerate(self.modules[8]):
            if i == 0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)

        output2_cat = self.modules[9](paddle.concat([output2_0, output2], axis=1)) # concatenate for feature map width expansion

        output2_C = self.up_l3(self.br(self.modules[10](output2_cat))) #RUM

        output1_C = self.level3_C(output1_cat) # project to C-dimensional space

        comb_l2_l3 = self.up_l2(self.combine_l2_l3(paddle.concat([output1_C, output2_C], axis=1))) #RUM

        concat_features = self.conv(paddle.concat([comb_l2_l3, output0_cat], axis=1))

        classifier = self.classifier(concat_features)

        return [classifier]


if __name__ == '__main__':
    import paddle
    import numpy as np

    paddle.enable_static()

    startup_prog = paddle.static.default_startup_program()

    exe = paddle.static.Executor(paddle.CPUPlace())
    exe.run(startup_prog)
    path_prefix = "./output/model"

    [inference_program, feed_target_names, fetch_targets] = (
        paddle.static.load_inference_model(path_prefix, exe))
    print('inference_program:', inference_program)

    tensor_img = np.array(np.random.random((1, 3, 1024, 2048)), dtype=np.float32)
    results = exe.run(inference_program,
                feed={feed_target_names[0]: tensor_img},
                fetch_list=fetch_targets)
