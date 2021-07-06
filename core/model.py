import math
import numpy as np
import torch
from torch.nn.parameter import Parameter
from torch import nn
from torch.nn import functional as F
from torch.utils import model_zoo
from copy import deepcopy


eps = 1e-10


class CNNEncoder(nn.Module):
    def __init__(self, in_c=3):
        super(CNNEncoder, self).__init__()

        self.layer1 = nn.Sequential(
                        nn.Conv2d(in_c, 64, kernel_size=3, padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64, 64, kernel_size=3, padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
                        nn.Conv2d(64, 64, kernel_size=3, padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())
        self.layer4 = nn.Sequential(
                        nn.Conv2d(64, 64, kernel_size=3, padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out  # 64


class MyLinear(nn.Module):

    def __init__(self, in_features, out_features, bias=True, reset_each_iter=False):
        super(MyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.reset_each_iter = reset_each_iter
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        if reset_each_iter:
            assert bias is False

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        if self.reset_each_iter:
            self.reset_parameters()
        return F.linear(input, self.weight, self.bias), self.weight

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, in_c, include_head=False, num_classes=None):
        super(ResNet, self).__init__()

        self.include_head = include_head
        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_c, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.Sigmoid()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        if self.include_head:
            self.fc = nn.Linear(256 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        if self.include_head:
            x = torch.mean(x, dim=(2,3))
            x = x.view(x.size(0), -1)
            x = self.fc(x)

        return x


def feat_extract(pretrained=False, include_head=False, **kwargs):
    """Constructs a ResNet-Mini-Imagenet model"""
    model_urls = {
        'resnet18':     'https://download.pytorch.org/models/resnet18-5c106cde.pth',
        'resnet34':     'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
        'resnet52':     'https://download.pytorch.org/models/resnet50-19c8e357.pth',
        'resnet101':    'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
        'resnet152':    'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    }
    logger = kwargs['opts'].logger
    # resnet"x", x = 1 + sum(layers)x3
    if kwargs['structure'] == 'resnet40':
        model = ResNet(Bottleneck, [3, 4, 6], kwargs['in_c'], include_head, kwargs.get('num_classes'))
    elif kwargs['structure'] == 'resnet19':
        model = ResNet(Bottleneck, [2, 2, 2], kwargs['in_c'], include_head, kwargs.get('num_classes'))
    elif kwargs['structure'] == 'resnet52':
        model = ResNet(Bottleneck, [4, 8, 5], kwargs['in_c'], include_head, kwargs.get('num_classes'))
    elif kwargs['structure'] == 'resnet34':
        model = ResNet(Bottleneck, [3, 4, 4], kwargs['in_c'], include_head, kwargs.get('num_classes'))
    elif kwargs['structure'] == 'shallow':
        model = CNNEncoder(kwargs['in_c'])
    else:
        raise NameError('structure not known {} ...'.format(kwargs['structure']))
    if pretrained:
        logger('Using pre-trained model from pytorch official webiste, {:s}'.format(kwargs['structure']))
        model.load_state_dict(model_zoo.load_url(model_urls[kwargs['structure']]), strict=False)
    return model


# NOTE: this is the core of the Category Traversal Module (CTM)
class CTMNet(nn.Module):
    """repnet => feature concat => layer4 & layer5 & avg pooling => fc => sigmoid"""
    def __init__(self, opts):
        super(CTMNet, self).__init__()

        self.mp_mean = None
        self.delete_mp = None

        self.opts = opts
        if self.opts.fsl.ctm:
            # use forward_CTM method
            self.use_relation_net = self.opts.ctmnet.CE_use_relation
            self.dnet = self.opts.ctmnet.dnet                            # dnet or baseline
            self.dnet_out_c = self.opts.ctmnet.dnet_out_c                # define the reshaper
            try:
                self.baseline_manner = self.opts.ctmnet.baseline_manner
            except:
                self.baseline_manner = ''
            self.dnet_supp_manner = self.opts.ctmnet.dnet_supp_manner

        _logger = opts.logger
        _logger('Building up models ...')
        # feature extractor
        in_c = 1 if opts.dataset.name == 'omniglot' else 3
        self.repnet = feat_extract(
            self.opts.model.resnet_pretrain,
            opts=opts, structure=opts.model.structure, in_c=in_c)

        input_bs = opts.fsl.n_way[0]*opts.fsl.k_shot[0]
        random_input = torch.rand(input_bs, in_c, opts.data.im_size, opts.data.im_size)
        repnet_out = self.repnet(random_input)
        repnet_sz = repnet_out.size()
        assert repnet_sz[2] == repnet_sz[3]
        _logger('\trepnet output sz: {} (assume bs=n_way*k_shot)'.format(repnet_sz))
        self.c = repnet_sz[1]   # supposed to be 64
        self.d = repnet_sz[2]

        if self.opts.fsl.ctm:
            _embedding = repnet_out

            if self.baseline_manner == 'sample_wise_similar':
                assert self.opts.model.structure == 'shallow'
                input_c = _embedding.size(1)
                self.additional_repnet = nn.Sequential(
                    nn.Conv2d(input_c, input_c, kernel_size=3, padding=1),
                    nn.BatchNorm2d(input_c, momentum=1, affine=True),
                    nn.ReLU()
                )

            # RESHAPER
            if not (not self.dnet and self.baseline_manner == 'no_reshaper'):
                assert np.mod(self.dnet_out_c, 4) == 0
                out_size = int(self.dnet_out_c / 4)
                self.inplanes = _embedding.size(1)
                if self.opts.model.structure.startswith('resnet'):
                    self.reshaper = nn.Sequential(
                        self._make_layer(Bottleneck, out_size*2, 3, stride=1),
                        self._make_layer(Bottleneck, out_size, 2, stride=1)
                    )
                else:
                    self.reshaper = self._make_layer(Bottleneck, out_size, 4, stride=1)
                _out_downsample = self.reshaper(_embedding)

            # CONCENTRATOR AND PROJECTOR
            if self.dnet:
                if self.mp_mean:
                    self.inplanes = _embedding.size(1)
                else:
                    # concatenate along the channel for all samples in each class
                    self.inplanes = self.opts.fsl.k_shot[0]*_embedding.size(1)
                if self.opts.model.structure.startswith('resnet'):
                    self.main_component = nn.Sequential(
                        self._make_layer(Bottleneck, out_size*2, 3, stride=1),
                        self._make_layer(Bottleneck, out_size, 2, stride=1)
                    )
                else:
                    self.main_component = self._make_layer(Bottleneck, out_size, 4, stride=1)

                # projector
                if self.delete_mp:
                    assert self.opts.fsl.k_shot[0] == 1
                    del self.main_component
                    # input_c for Projector, no mp
                    self.inplanes = self.opts.fsl.n_way[0]*_embedding.size(1)
                else:
                    # input_c for Projector, has mp
                    self.inplanes = self.opts.fsl.n_way[0]*out_size*4

                if self.opts.model.structure.startswith('resnet'):
                    self.projection = nn.Sequential(
                        self._make_layer(Bottleneck, out_size*2, 3, stride=1),
                        self._make_layer(Bottleneck, out_size, 2, stride=1)
                    )
                else:
                    self.projection = self._make_layer(Bottleneck, out_size, 4, stride=1)

            # RELATION METRIC
            if self.use_relation_net:
                # relation sub_net
                if hasattr(self, 'reshaper'):
                    _input = _out_downsample
                else:
                    _input = _embedding

                if self.opts.model.relation_net == 'res_block':
                    # (256); it is "2" because combining two embedding
                    self.inplanes = 2 * _input.size(1)
                    self.relation1 = self._make_layer(Bottleneck, 32, 2, stride=2)
                    self.relation2 = self._make_layer(Bottleneck, 16, 2, stride=1)

                    _combine = torch.stack([_input, _input], dim=1).view(
                        _input.size(0), -1, _input.size(2), _input.size(3))
                    _out = self.relation2(self.relation1(_combine))
                    self.fc_input_c = _out.size(1)*_out.size(2)*_out.size(3)
                    _half = int(self.fc_input_c/2)
                    self.fc = nn.Sequential(
                        nn.Linear(self.fc_input_c, _half),
                        nn.BatchNorm1d(_half),
                        nn.ReLU(inplace=True),
                        nn.Linear(_half, 1)
                    )
                elif self.opts.model.relation_net == 'simple':
                    input_c = 2 * _input.size(1)
                    self.relation1 = nn.Sequential(
                        nn.Conv2d(input_c, 64, kernel_size=3, padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
                    self.relation2 = nn.Sequential(
                        nn.Conv2d(64, 64, kernel_size=3, padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        # nn.MaxPool2d(2)
                    )

                    _combine = torch.stack([_input, _input], dim=1).view(
                        _input.size(0), -1, _input.size(2), _input.size(3))
                    _out = self.relation2(self.relation1(_combine))
                    self.fc_input_c = _out.size(1) * _out.size(2) * _out.size(3)
                    _half = int(self.fc_input_c / 2)
                    self.fc = nn.Sequential(
                        nn.Linear(self.fc_input_c, _half),
                        nn.ReLU(),
                        nn.Linear(_half, 1),    # except no sigmoid since we use CE
                    )
        # else:
        #     # the original relation network
        #     self.inplanes = 2 * self.c
        #     # the original network in the relation net
        #     # after the relation module (three layers)
        #     self.relation1 = self._make_layer(Bottleneck, 128, 4, stride=2)
        #     self.relation2 = self._make_layer(Bottleneck, 64, 3, stride=2)
        #
        #     if self.CE_loss:
        #         self.fc = nn.Sequential(
        #             nn.Linear(256, 64),
        #             nn.BatchNorm1d(64),
        #             nn.ReLU(inplace=True),
        #             nn.Linear(64, 1)
        #         )
        #     else:
        #         self.fc = nn.Sequential(
        #             nn.Linear(256, 64),
        #             nn.BatchNorm1d(64),
        #             nn.ReLU(inplace=True),
        #             nn.Linear(64, 1),
        #             nn.Sigmoid()  # the only difference
        #         )
        #     combine = torch.stack([repnet_out, repnet_out], dim=1).view(
        #         repnet_out.size(0), -1, repnet_out.size(2), repnet_out.size(3))
        #     out = self.relation2(self.relation1(combine))
        #     _logger('\tafter layer5 sz: {} (assume bs=2)\n'.format(out.size()))
        #     self.pool_size = out.size(2)

        self._initialize_weights()

    def _initialize_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.xavier_normal(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def fac_adjust(self):
        loss_fac = np.max([-2*self.epoch + 10, 1])
        ot_loss_fac = np.min([.2*self.epoch, 1])
        return loss_fac, ot_loss_fac

    def forward_CTM(self, support_x, support_y, query_x, query_y, train=False, optimizer=None):

        target, one_hot, target_support = self.get_target(support_y, query_y)
        batch_sz, support_sz, _d = support_x.size(0), support_x.size(1), support_x.size(3)
        query_sz = query_x.size(1)
        n_way, k_shot = self.opts.fsl.n_way[0], self.opts.fsl.k_shot[0]

        # 1. FEATURE EXTRACTION (FOR NOW DISABLE SWAP)
        # support_sz (25), c (64), d (19), d (19)
        support_xf_ori = self.repnet(support_x.view(batch_sz*support_sz, -1, _d, _d))
        # query_sz (75), c (64), d (19), d (19)
        query_xf_ori = self.repnet(query_x.view(batch_sz*query_sz, -1, _d, _d))

        if self.dnet:
            if not self.delete_mp:
                if not self.mp_mean:
                    support_xf_reshape = support_xf_ori.view(n_way, -1, support_xf_ori.size(2), support_xf_ori.size(3))
                else:
                    support_xf_reshape = support_xf_ori
                mp = self.main_component(support_xf_reshape)                # 5(n_way), 64, 3, 3
                if self.mp_mean:
                    mp = torch.mean(mp.view(n_way, k_shot, mp.size(1), mp.size(2), mp.size(2)), dim=1, keepdim=False)
                _input_P = mp.view(1, -1, mp.size(2), mp.size(3))           # mp -> 1, 5*64, 3, 3
            else:
                _input_P = support_xf_ori.view(1, -1, support_xf_ori.size(2), support_xf_ori.size(3))

            # for P: consider all components
            P = self.projection(_input_P)                                   # 1, 64, 3, 3
            P = F.softmax(P, dim=1)
            if self.dnet_supp_manner == '2' or self.dnet_supp_manner == '3':
                mp_modified = torch.matmul(mp, P)                           # 5, 64, 3, 3

            if self.dnet_supp_manner == '1':
                v = self.reshaper(support_xf_ori)
                v = torch.matmul(v, P)
            elif self.dnet_supp_manner == '2':
                v = self.reshaper(support_xf_ori)                           # 25, 64, 3, 3
                v = v.view(n_way, -1, v.size(1), v.size(2), v.size(3))      # 5, 5(k_shot), 64, 3, 3
                v = torch.matmul(v, mp_modified.unsqueeze(1)).view(support_sz, v.size(2), v.size(3), v.size(3))
            elif self.dnet_supp_manner == '3':
                v = mp_modified

            query = self.reshaper(query_xf_ori)                             # 75, 64, 3, 3
            query = torch.matmul(query, P)
        else:
            # baseline
            if self.baseline_manner == 'no_reshaper':
                v = support_xf_ori
                query = query_xf_ori
            elif self.baseline_manner.startswith('sample_wise'):
                if self.baseline_manner == 'sample_wise_similar':
                    support_xf_ori = self.additional_repnet(support_xf_ori)
                    query_xf_ori = self.additional_repnet(query_xf_ori)
                v = self.reshaper(support_xf_ori)
                query = self.reshaper(query_xf_ori)
            elif self.baseline_manner == 'sum':
                v = self.reshaper(support_xf_ori)
                v = v.view(n_way, -1, v.size(1), v.size(2), v.size(2)).sum(1, keepdim=False)
                query = self.reshaper(query_xf_ori)

        # 2. Standard pipeline
        score = self.get_embedding_score(v, query, n_way, query_sz)

        # 3. Output
        if train:
            # for legacy
            zero = torch.zeros(1).to(self.opts.ctrl.device)
            disc_weights = None
            sinkhorn_loss, loss_discri = zero, zero

            loss = F.cross_entropy(score, target).unsqueeze(0)
            total_loss = loss
            return torch.cat([total_loss, loss, sinkhorn_loss, loss_discri]).unsqueeze(0), disc_weights
        else:
            # TEST
            prediction = score.argmax(dim=-1)
            correct = torch.eq(prediction, target).sum().unsqueeze(0)
            return prediction, correct

    @staticmethod
    def _norm(input):
        return (input - input.min()) / (input.max() - input.min()).clamp(min=eps)

    def get_target(self, support_y, query_y):

        support_y = support_y[0, ::self.opts.fsl.k_shot[0]]
        query_y = query_y[0]

        target = torch.stack([
            torch.nonzero(torch.eq(support_y, entry)) for entry in query_y
        ])
        target = target.view(-1, 1)  # shape: query_size
        one_hot_labels = \
            torch.zeros(target.size(0), self.opts.fsl.n_way[0]).to(self.opts.ctrl.device).scatter_(
                1, target, 1)

        target_support = torch.arange(self.opts.fsl.n_way[0]).unsqueeze(1).expand(
            -1, self.opts.fsl.k_shot[0]).contiguous().view(-1).long().to(self.opts.ctrl.device)

        return target.squeeze(1), one_hot_labels, target_support

    def get_embedding_score(self, support_xf_ori, query_xf_ori, n_way, query_sz):

        # sum up samples with support
        k_shot = int(support_xf_ori.size(0) / n_way)

        if self.use_relation_net:
            ch_sz, spatial_sz = support_xf_ori.size(1), support_xf_ori.size(2)
            # support_xf_ori: 25/5, 256, 5, 5
            # query_xf_ori: 75, 256, 5, 5
            # first expand
            support_xf_ori = support_xf_ori.unsqueeze(0).expand(query_sz, -1, -1, -1, -1).contiguous().view(
                query_sz*n_way*k_shot, ch_sz, spatial_sz, spatial_sz
            )
            query_xf_ori = query_xf_ori.unsqueeze(1).expand(-1, n_way*k_shot, -1, -1, -1).contiguous().view(
                query_sz*n_way*k_shot, ch_sz, spatial_sz, spatial_sz
            )
            embed_combine = torch.stack([support_xf_ori, query_xf_ori], dim=1).view(
                query_sz*n_way*k_shot, -1, spatial_sz, spatial_sz)
            _out = self.relation2(self.relation1(embed_combine))

            _out = _out.view(_out.size(0), -1)
            score = self.fc(_out).view(query_sz, n_way, k_shot)

        else:
            support_xf = support_xf_ori.view(support_xf_ori.size(0), -1)    # size: 25/5 (support_sz/n_way) x feat_dim
            query_xf = query_xf_ori.view(query_xf_ori.size(0), -1)          # size: 75 (query_size) x feat_dim
            feat_dim = support_xf.size(-1)
            support_xf = support_xf.unsqueeze(0).expand(query_sz, -1, -1).contiguous().view(-1, feat_dim)
            query_xf = query_xf.unsqueeze(1).expand(-1, n_way*k_shot, -1).contiguous().view(-1, feat_dim)
            score = -F.pairwise_distance(support_xf, query_xf, p=2)
            score = score.view(query_sz, n_way, k_shot)

        # sum up here
        score = torch.sum(score, dim=2, keepdim=False)
        return score

