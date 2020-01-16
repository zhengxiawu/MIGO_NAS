# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.
from model.mb_ops import *


class ProxylessNASNets(MyNetwork):

    def __init__(self, n_classes=1000, bn_param=(0.1, 1e-3), dropout_rate=0.1, base_stage_width=None,
                 width_mult_list=1.0, conv_candidates=None, depth_list=4):
        super(ProxylessNASNets, self).__init__()
        self.width_mult_list = int2list(width_mult_list, 1)
        self.depth_list = int2list(depth_list, 1)
        self.base_stage_width = base_stage_width
        self.conv_candidates = [
            '3x3_MBConv3', '3x3_MBConv6',
            '5x5_MBConv3', '5x5_MBConv6',
            '7x7_MBConv3', '7x7_MBConv6',
        ] if conv_candidates is None else conv_candidates

        self.width_mult_list.sort()
        self.ks_list.sort()
        self.expand_ratio_list.sort()
        self.depth_list.sort()

        if base_stage_width == 'google':
            base_stage_width = [32, 16, 24, 32, 64, 96, 160, 320, 1280]
        else:
            # ProxylessNAS Stage Width
            base_stage_width = [32, 16, 24, 40, 80, 96, 192, 320, 1280]

        input_channel = [make_divisible(base_stage_width[0] * width_mult, 8) for width_mult in self.width_mult_list]
        first_block_width = [make_divisible(base_stage_width[1] * width_mult, 8) for width_mult in self.width_mult_list]
        last_channel = [
            make_divisible(base_stage_width[-1] * width_mult, 8) if width_mult > 1.0 else base_stage_width[-1]
            for width_mult in self.width_mult_list
        ]

        # first conv layer

        first_conv = ConvLayer(
            3, max(input_channel), kernel_size=3, stride=2, use_bn=True, act_func='relu6', ops_order='weight_bn_act'
        )

        # first block
        first_block_conv = MBInvertedConvLayer(
            in_channels=max(input_channel), out_channels=max(first_block_width), kernel_size=3, stride=1,
            expand_ratio=1, act_func='relu6',
        )
        first_block = MobileInvertedResidualBlock(first_block_conv, None)
        input_channel = first_block_width

        # inverted residual blocks
        self.block_group_info = []
        blocks = [first_block]
        _block_index = 1

        stride_stages = [2, 2, 2, 1, 2, 1]
        if depth_list is None:
            n_block_list = [2, 3, 4, 3, 3, 1]
            self.depth_list = [4, 4]
            print('Use MobileNetV2 Depth Setting')
        else:
            n_block_list = [max(self.depth_list)] * 5 + [1]

        width_list = []
        for base_width in base_stage_width[2:-1]:
            width = [make_divisible(base_width * width_mult, 8) for width_mult in self.width_mult_list]
            width_list.append(width)

        for width, n_block, s in zip(width_list, n_block_list, stride_stages):
            self.block_group_info.append([_block_index + i for i in range(n_block)])
            _block_index += n_block

            output_channel = width
            for i in range(n_block):
                if i == 0:
                    stride = s
                else:
                    stride = 1

                if stride == 1 and input_channel == width:
                    modified_conv_candidates = conv_candidates + ['Zero']
                else:
                    modified_conv_candidates = conv_candidates + ['3x3_MBConv1']
                self.candidate_ops.append(modified_conv_candidates)
                conv_op = MixedEdge(candidate_ops=build_candidate_ops(
                    modified_conv_candidates, input_channel, width, stride, 'weight_bn_act',
                    act_func='relu6', use_se=False), )

                if stride == 1 and input_channel == output_channel:
                    shortcut = IdentityLayer(input_channel, input_channel)
                else:
                    shortcut = None

                mb_inverted_block = MobileInvertedResidualBlock(conv_op, shortcut)
                blocks.append(mb_inverted_block)
                input_channel = output_channel
        # 1x1_conv before global average pooling
        feature_mix_layer = ConvLayer(
            max(input_channel), max(last_channel), kernel_size=1, use_bn=True, act_func='relu6',
        )
        classifier = LinearLayer(max(last_channel), n_classes, dropout_rate=dropout_rate)


        # set bn param
        self.set_bn_param(momentum=bn_param[0], eps=bn_param[1])

        self.first_conv = first_conv
        self.blocks = blocks
        self.feature_mix_layer = feature_mix_layer
        self.classifier = classifier
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)

    """ MyNetwork required methods """

    @staticmethod
    def name():
        return 'OFAProxylessNASNets'

    def forward(self, x, sample):
        # first conv
        x = self.first_conv(x)
        # first block
        x = self.blocks[0](x)

        assert len(self.blocks) - 1 == len(sample)
        for i in range(len(self.blocks[1:])):
            this_block_conv = self.blocks[i].mobile_inverted_conv
            if isinstance(this_block_conv, MixedEdge):
                this_block_conv.active_index = [sample[i]]
            else:
                raise NotImplementedError
        for block in self.blocks:
            x = block(x)
        x = self.global_avg_pooling(x)
        x = self.feature_mix_layer(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x

    @property
    def module_str(self):
        _str = ''
        for block in self.blocks:
            _str += block.unit_str + '\n'
        return _str

    @property
    def config(self):
        return {
            'name': ProxylessNASNets.__name__,
            'bn': self.get_bn_param(),
            'first_conv': self.first_conv.config,
            'blocks': [
                block.config for block in self.blocks
            ],
            'feature_mix_layer': None if self.feature_mix_layer is None else self.feature_mix_layer.config,
            'classifier': self.classifier.config,
        }

    @staticmethod
    def build_from_config(config):
        raise ValueError('do not support this function')

    def genotype(self, theta):
        genotype = []
        for i in range(theta.shape[0]):
            genotype.append(self.candidate_ops[i][np.argmax(theta[i])])
        return genotype


