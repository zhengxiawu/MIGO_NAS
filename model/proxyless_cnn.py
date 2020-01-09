# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.

from queue import Queue
import copy

from model.proxyless_ops import *
from utils import LatencyEstimator


class MobileInvertedResidualBlock(MyNetwork):

    def __init__(self, mobile_inverted_conv, shortcut):
        super(MobileInvertedResidualBlock, self).__init__()

        self.mobile_inverted_conv = mobile_inverted_conv
        self.shortcut = shortcut

    def forward(self, x):
        if self.mobile_inverted_conv.is_zero_layer():
            res = x
        elif self.shortcut is None or self.shortcut.is_zero_layer():
            res = self.mobile_inverted_conv(x)
        else:
            conv_x = self.mobile_inverted_conv(x)
            skip_x = self.shortcut(x)
            res = skip_x + conv_x
        return res

    @property
    def module_str(self):
        return '(%s, %s)' % (
            self.mobile_inverted_conv.module_str, self.shortcut.module_str if self.shortcut is not None else None
        )

    @property
    def config(self):
        return {
            'name': MobileInvertedResidualBlock.__name__,
            'mobile_inverted_conv': self.mobile_inverted_conv.config,
            'shortcut': self.shortcut.config if self.shortcut is not None else None,
        }

    @staticmethod
    def build_from_config(config):
        mobile_inverted_conv = set_layer_from_config(config['mobile_inverted_conv'])
        shortcut = set_layer_from_config(config['shortcut'])
        return MobileInvertedResidualBlock(mobile_inverted_conv, shortcut)

    def get_flops(self, x):
        flops1, conv_x = self.mobile_inverted_conv.get_flops(x)
        if self.shortcut:
            flops2, _ = self.shortcut.get_flops(x)
        else:
            flops2 = 0

        return flops1 + flops2, self.forward(x)


class SuperProxylessNASNets(MyNetwork):

    def __init__(self, width_stages, n_cell_stages, conv_candidates, stride_stages,
                 n_classes=1000, width_mult=1, bn_param=(0.1, 1e-3), dropout_rate=0, criterion =None):
        super(SuperProxylessNASNets, self).__init__()
        self._redundant_modules = None
        self._unused_modules = None
        self.criterion = criterion

        input_channel = make_divisible(32 * width_mult, 8)
        first_cell_width = make_divisible(16 * width_mult, 8)
        for i in range(len(width_stages)):
            width_stages[i] = make_divisible(width_stages[i] * width_mult, 8)

        # first conv layer
        first_conv = ConvLayer(
            3, input_channel, kernel_size=3, stride=2, use_bn=True, act_func='relu6', ops_order='weight_bn_act'
        )

        # first block
        first_block_conv = MixedEdge(candidate_ops=build_candidate_ops(
            ['3x3_MBConv1'],
            input_channel, first_cell_width, 1, 'weight_bn_act',
        ), )
        if first_block_conv.n_choices == 1:
            first_block_conv = first_block_conv.candidate_ops[0]
        first_block = MobileInvertedResidualBlock(first_block_conv, None)
        input_channel = first_cell_width

        # blocks
        blocks = nn.ModuleList()
        self.candidate_ops = []
        for width, n_cell, s in zip(width_stages, n_cell_stages, stride_stages):
            for i in range(n_cell):
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
                ), )
                # shortcut
                if stride == 1 and input_channel == width:
                    shortcut = IdentityLayer(input_channel, input_channel)
                else:
                    shortcut = None
                inverted_residual_block = MobileInvertedResidualBlock(conv_op, shortcut)
                blocks.append(inverted_residual_block)
                input_channel = width

        # feature mix layer
        last_channel = make_divisible(1280 * width_mult, 8) if width_mult > 1.0 else 1280
        feature_mix_layer = ConvLayer(
            input_channel, last_channel, kernel_size=1, use_bn=True, act_func='relu6', ops_order='weight_bn_act',
        )

        classifier = LinearLayer(last_channel, n_classes, dropout_rate=dropout_rate)
        self.first_block = first_block
        self.first_conv = first_conv
        self.blocks = blocks
        self.feature_mix_layer = feature_mix_layer
        self.classifier = classifier
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)

        # set bn param
        self.set_bn_param(momentum=bn_param[0], eps=bn_param[1])

    def forward(self, x, sample):
        x = self.first_conv(x)
        x = self.first_block(x)
        assert len(self.blocks) == len(sample)
        for i in range(len(self.blocks)):
            this_block_conv = self.blocks[i].mobile_inverted_conv
            if isinstance(this_block_conv, MixedEdge):
                this_block_conv.active_index = [sample[i]]
            else:
                raise NotImplementedError
        for block in self.blocks:
            x = block(x)
        x = self.feature_mix_layer(x)
        x = self.global_avg_pooling(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x

    @property
    def module_str(self):
        _str = ''
        for block in self.blocks:
            _str += block.unit_str + '\n'
        return _str

    def get_flops(self, x):
        flop, x = self.first_conv.get_flops(x)

        for block in self.blocks:
            delta_flop, x = block.get_flops(x)
            flop += delta_flop

        delta_flop, x = self.feature_mix_layer.get_flops(x)
        flop += delta_flop

        x = self.global_avg_pooling(x)
        x = x.view(x.size(0), -1)  # flatten

        delta_flop, x = self.classifier.get_flops(x)
        flop += delta_flop
        return flop, x

    @property
    def config(self):
        raise ValueError('not needed')

    @staticmethod
    def build_from_config(config):
        raise ValueError('not needed')

    """ architecture parameters related methods """

    @property
    def redundant_modules(self):
        if self._redundant_modules is None:
            module_list = []
            for m in self.modules():
                if m.__str__().startswith('MixedEdge'):
                    module_list.append(m)
            self._redundant_modules = module_list
        return self._redundant_modules


    """ training related methods """

    def unused_modules_off(self):
        self._unused_modules = []
        for m in self.redundant_modules:
            unused = {}
            if MixedEdge.MODE in ['full', 'two', 'full_v2']:
                involved_index = m.active_index + m.inactive_index
            else:
                involved_index = m.active_index
            for i in range(m.n_choices):
                if i not in involved_index:
                    unused[i] = m.candidate_ops[i]
                    m.candidate_ops[i] = None
            self._unused_modules.append(unused)

    def unused_modules_back(self):
        if self._unused_modules is None:
            return
        for m, unused in zip(self.redundant_modules, self._unused_modules):
            for i in unused:
                m.candidate_ops[i] = unused[i]
        self._unused_modules = None

    def set_chosen_op_active(self):
        for m in self.redundant_modules:
            try:
                m.set_chosen_op_active()
            except AttributeError:
                print(type(m), ' do not support `set_chosen_op_active()`')

    def set_active_via_net(self, net):
        assert isinstance(net, SuperProxylessNASNets)
        for self_m, net_m in zip(self.redundant_modules, net.redundant_modules):
            self_m.active_index = copy.deepcopy(net_m.active_index)
            self_m.inactive_index = copy.deepcopy(net_m.inactive_index)

    def expected_latency(self, latency_model: LatencyEstimator):
        expected_latency = 0
        # first conv
        expected_latency += latency_model.predict('Conv', [224, 224, 3], [112, 112, self.first_conv.out_channels])
        # feature mix layer
        expected_latency += latency_model.predict(
            'Conv_1', [7, 7, self.feature_mix_layer.in_channels], [7, 7, self.feature_mix_layer.out_channels]
        )
        # classifier
        expected_latency += latency_model.predict(
            'Logits', [7, 7, self.classifier.in_features], [self.classifier.out_features]  # 1000
        )
        # blocks
        fsize = 112
        for block in self.blocks:
            shortcut = block.shortcut
            if shortcut is None or shortcut.is_zero_layer():
                idskip = 0
            else:
                idskip = 1

            mb_conv = block.mobile_inverted_conv
            if not isinstance(mb_conv, MixedEdge):
                if not mb_conv.is_zero_layer():
                    out_fz = fsize // mb_conv.stride
                    op_latency = latency_model.predict(
                        'expanded_conv', [fsize, fsize, mb_conv.in_channels], [out_fz, out_fz, mb_conv.out_channels],
                        expand=mb_conv.expand_ratio, kernel=mb_conv.kernel_size, stride=mb_conv.stride, idskip=idskip
                    )
                    expected_latency = expected_latency + op_latency
                    fsize = out_fz
                continue

            probs_over_ops = mb_conv.current_prob_over_ops
            out_fsize = fsize
            for i, op in enumerate(mb_conv.candidate_ops):
                if op is None or op.is_zero_layer():
                    continue
                out_fsize = fsize // op.stride
                op_latency = latency_model.predict(
                    'expanded_conv', [fsize, fsize, op.in_channels], [out_fsize, out_fsize, op.out_channels],
                    expand=op.expand_ratio, kernel=op.kernel_size, stride=op.stride, idskip=idskip
                )
                expected_latency = expected_latency + op_latency * probs_over_ops[i]
            fsize = out_fsize
        return expected_latency

    def expected_flops(self, x):
        expected_flops = 0
        # first conv
        flop, x = self.first_conv.get_flops(x)
        expected_flops += flop
        # blocks
        for block in self.blocks:
            mb_conv = block.mobile_inverted_conv
            if not isinstance(mb_conv, MixedEdge):
                delta_flop, x = block.get_flops(x)
                expected_flops = expected_flops + delta_flop
                continue

            if block.shortcut is None:
                shortcut_flop = 0
            else:
                shortcut_flop, _ = block.shortcut.get_flops(x)
            expected_flops = expected_flops + shortcut_flop

            probs_over_ops = mb_conv.current_prob_over_ops
            for i, op in enumerate(mb_conv.candidate_ops):
                if op is None or op.is_zero_layer():
                    continue
                op_flops, _ = op.get_flops(x)
                expected_flops = expected_flops + op_flops * probs_over_ops[i]
            x = block(x)
        # feature mix layer
        delta_flop, x = self.feature_mix_layer.get_flops(x)
        expected_flops = expected_flops + delta_flop
        # classifier
        x = self.global_avg_pooling(x)
        x = x.view(x.size(0), -1)  # flatten
        delta_flop, x = self.classifier.get_flops(x)
        expected_flops = expected_flops + delta_flop
        return expected_flops

    def genotype(self, theta):
        genotype = []
        for i in range(theta.shape[0]):
            genotype.append(self.candidate_ops[i][np.argmax(theta[i])])
        return genotype


