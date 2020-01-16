# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.


from model.mb_ops import *


class MobileNetV3(MyNetwork):

    def __init__(self, n_classes=1000, bn_param=(0.1, 1e-5), dropout_rate=0.1, base_stage_width=None,
                 width_mult_list=1.0, conv_candidates=None, depth_list=4):
        super(MobileNetV3, self).__init__()

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

        base_stage_width = [16, 24, 40, 80, 112, 160, 960, 1280]

        final_expand_width = [
            make_divisible(base_stage_width[-2] * max(self.width_mult_list), 8) for _ in self.width_mult_list
        ]
        last_channel = [
            make_divisible(base_stage_width[-1] * max(self.width_mult_list), 8) for _ in self.width_mult_list
        ]

        stride_stages = [1, 2, 2, 2, 1, 2]
        act_stages = ['relu', 'relu', 'relu', 'h_swish', 'h_swish', 'h_swish']
        se_stages = [False, False, True, False, True, True]
        if depth_list is None:
            n_block_list = [1, 2, 3, 4, 2, 3]
            self.depth_list = [4, 4]
            print('Use MobileNetV3 Depth Setting')
        else:
            n_block_list = [1] + [max(self.depth_list)] * 5
        width_list = []
        for base_width in base_stage_width[:-2]:
            width = [make_divisible(base_width * width_mult, 8) for width_mult in self.width_mult_list]
            width_list.append(width)

        input_channel = width_list[0]
        # first conv layer
        first_conv = ConvLayer(3, max(input_channel), kernel_size=3, stride=2, act_func='h_swish')
        first_block_conv = MBInvertedConvLayer(
            in_channels=max(input_channel), out_channels=max(input_channel), kernel_size=3, stride=stride_stages[0],
            expand_ratio=1, act_func=act_stages[0], use_se=se_stages[0],
        )
        first_block = MobileInvertedResidualBlock(first_block_conv, IdentityLayer(input_channel, input_channel))

        # inverted residual blocks
        self.block_group_info = []
        blocks = [first_block]
        _block_index = 1
        feature_dim = input_channel

        for width, n_block, s, act_func, use_se in zip(width_list[1:], n_block_list[1:],
                                                       stride_stages[1:], act_stages[1:], se_stages[1:]):
            self.block_group_info.append([_block_index + i for i in range(n_block)])
            _block_index += n_block

            output_channel = width
            for i in range(n_block):
                if i == 0:
                    stride = s
                else:
                    stride = 1
                    # conv
                if stride == 1 and input_channel == width:
                    modified_conv_candidates = conv_candidates + ['Zero']
                else:
                    modified_conv_candidates = conv_candidates + ['3x3_MBConv1']
                self.candidate_ops.append(modified_conv_candidates)
                conv_op = MixedEdge(candidate_ops=build_candidate_ops(
                    modified_conv_candidates, input_channel, width, stride, 'weight_bn_act',
                act_func=act_func, use_se=use_se), )
                if stride == 1 and feature_dim == output_channel:
                    shortcut = IdentityLayer(feature_dim, feature_dim)
                else:
                    shortcut = None
                blocks.append(MobileInvertedResidualBlock(conv_op, shortcut))
                feature_dim = output_channel
        # final expand layer, feature mix layer & classifier
        final_expand_layer = ConvLayer(max(feature_dim), max(final_expand_width), kernel_size=1, act_func='h_swish')
        feature_mix_layer = ConvLayer(
            max(final_expand_width), max(last_channel), kernel_size=1, bias=False, use_bn=False, act_func='h_swish',
        )
        classifier = LinearLayer(max(last_channel), n_classes, dropout_rate=dropout_rate)

        # set bn param
        self.set_bn_param(momentum=bn_param[0], eps=bn_param[1])

        self.first_conv = first_conv
        self.blocks = blocks
        self.final_expand_layer = final_expand_layer
        self.feature_mix_layer = feature_mix_layer
        self.classifier = classifier
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)

    """ MyNetwork required methods """

    @staticmethod
    def name():
        return 'OFAMobileNetV3'

    def forward(self, x, sample):
        # first conv
        x = self.first_conv(x)
        # first block
        x = self.blocks[0](x)

        assert len(self.blocks)-1 == len(sample)
        for i in range(len(self.blocks[1:])):
            this_block_conv = self.blocks[i].mobile_inverted_conv
            if isinstance(this_block_conv, MixedEdge):
                this_block_conv.active_index = [sample[i]]
            else:
                raise NotImplementedError
        for block in self.blocks:
            x = block(x)
        x = self.final_expand_layer(x)
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
            'name': self.__name__,
            'bn': self.get_bn_param(),
            'first_conv': self.first_conv.config,
            'blocks': [
                block.config for block in self.blocks
            ],
            'final_expand_layer': self.final_expand_layer.config,
            'feature_mix_layer': self.feature_mix_layer.config,
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

