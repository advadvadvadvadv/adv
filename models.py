import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as functional, torch.utils.data as torchdata, math

class FirstResBlock(nn.Module):
	def __init__(self, n_filters, batchnorm = True, bias = False, timestep = 1):
		super(FirstResBlock, self).__init__()
		self.timestep = timestep
		self.batchnorm = batchnorm
		self.cv1 = nn.Conv2d(n_filters, n_filters, 3, 1, 1, bias = bias)
		if self.batchnorm:
			self.bn2 = nn.BatchNorm2d(n_filters)
		self.cv2 = nn.Conv2d(n_filters, n_filters, 3, 1, 1, bias = bias)
	def forward(self, x):
		z = self.cv1(x)
		z = functional.relu(self.bn2(z)) if self.batchnorm else functional.relu(x)
		z = self.cv2(z)
		return x + self.timestep * z, z 

class ResBlock(nn.Module):
	def __init__(self, n_filters, batchnorm = True, bias = False, timestep = 1):
		super(ResBlock, self).__init__()
		self.timestep = timestep
		self.batchnorm = batchnorm
		if self.batchnorm :
			self.bn1 = nn.BatchNorm2d(n_filters)
		self.cv1 = nn.Conv2d(n_filters, n_filters, 3, 1, 1, bias = bias)
		if self.batchnorm :
			self.bn2 = nn.BatchNorm2d(n_filters)
		self.cv2 = nn.Conv2d(n_filters, n_filters, 3, 1, 1, bias = bias)
	def forward(self, x):
		z = functional.relu(self.bn1(x)) if self.batchnorm else functional.relu(x)
		z = self.cv1(z) 
		z = functional.relu(self.bn2(z)) if self.batchnorm else functional.relu(x)
		z = self.cv2(z)
		return x + self.timestep * z, z

class ResNetStage(nn.Module):
	def __init__(self, n_blocks, n_filters, first = False, batchnorm = True, bias = False, timestep = 1):
		super(ResNetStage, self).__init__()
		self.blocks = nn.ModuleList([FirstResBlock(n_filters, batchnorm, bias, timestep) if i == 0 and first else ResBlock(n_filters, batchnorm, bias, timestep) for i in range(n_blocks)])
	def forward(self, x):
		rs = []
		for block in self.blocks :
			x, r = block(x)
			rs.append(r)
		return x, rs

class OneRepResNet(nn.Module):
	def __init__(self, datashape, n_classes, n_filters = 32, n_blocks = 9, bias = False, return_residus = True):
		super(OneRepResNet, self).__init__()
		self.return_residus = return_residus
		self.encoder = nn.Sequential(nn.Conv2d(1, n_filters, 3, 1, 1, bias = False), nn.BatchNorm2d(n_filters))
		self.stage = ResNetStage(n_blocks, n_filters, True, bias = bias)
		with torch.no_grad():
			featuresize = self.forward_conv(torch.zeros(*datashape))[0].view(-1).shape[0]
		self.classifier = nn.Sequential(nn.Linear(featuresize, n_classes * 10), nn.BatchNorm1d(n_classes * 10), nn.ReLU(True), nn.Linear(n_classes * 10, n_classes))
	def forward_conv(self, x):
		x = self.encoder(x)
		x, rs = self.stage(x)
		return x, rs
	def forward(self, x):
		x, rs = self.forward_conv(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return (x,rs) if self.return_residus else x

class ResNet110(nn.Module):
	def __init__(self, datashape, n_classes, n_blocks = 18, n_filters = 16, batchnorm = True, bias = False, timestep = 1, return_residus = True):
		super(ResNet110, self).__init__()
		self.return_residus = return_residus
		self.encoder = nn.Sequential(nn.Conv2d(3, n_filters, 3, 1, 1, bias = False), nn.BatchNorm2d(n_filters))
		self.stage1 = ResNetStage(n_blocks, n_filters, True, batchnorm, bias, timestep)
		self.cv1 = nn.Conv2d(n_filters, 2 * n_filters, 1, 2, 0, bias = False)
		self.stage2 = ResNetStage(n_blocks, 2 * n_filters, False, batchnorm, bias, timestep)
		self.cv2 = nn.Conv2d(2 * n_filters, 4 * n_filters, 1, 2, 0, bias = False)
		self.stage3 = ResNetStage(n_blocks, 4 * n_filters, False, batchnorm, bias, timestep)
		self.avgpool = nn.AvgPool2d(8, 8)
		with torch.no_grad():
			featuresize = self.forward_conv(torch.zeros(*datashape))[0].view(-1).shape[0]
		self.classifier = nn.Linear(featuresize, n_classes)
	def forward_conv(self, x):
		rs = dict()
		x = self.encoder(x)
		x, rs[1] = self.stage1(x)
		x = self.cv1(x)
		x, rs[2] = self.stage2(x)
		x = self.cv2(x)
		x, rs[3] = self.stage3(x)
		x = self.avgpool(x)
		return x, [r for i in range(1, 4) for r in rs[i]]
	def forward(self, x):
		x, rs = self.forward_conv(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return (x,rs) if self.return_residus else x
	def feature_list(self, x):
		x_list = []
		x = self.encoder(x)
		x_list.append(x)
		x, _= self.stage1(x)
		x_list.append(x)
		x = self.cv1(x)
		x, _ = self.stage2(x)
		x_list.append(x)
		x = self.cv2(x)
		x, _ = self.stage3(x)
		x_list.append(x)
		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		return self.classifier(x), x_list
	def intermediate_forward(self, x, stage):
		x = self.encoder(x)
		if stage == 1:
			x, _ = self.stage1(x)
		elif stage == 2:
			x, _ = self.stage1(x)
			x = self.cv1(x)
			x, _ = self.stage2(x)
		elif stage == 3:
			x, _ = self.stage1(x)
			x = self.cv1(x)
			x, _ = self.stage2(x)
			x = self.cv2(x)
			x, _ = self.stage3(x)           
		return x

class WideResNet(nn.Module):
	def __init__(self, datashape, n_classes, n_blocks = 4, batchnorm = True, bias = False, timestep = 1, return_residus = True):
		super(WideResNet, self).__init__()
		self.return_residus = return_residus
		self.cv0 = nn.Conv2d(datashape[1], 160, 3, 1, 1, bias = False)
		self.stage1 = ResNetStage(n_blocks, 160, True, batchnorm, bias, timestep)
		self.cv1 = nn.Conv2d(160, 320, 1, 2, 0, bias = False)
		self.stage2 = ResNetStage(n_blocks, 320, False, batchnorm, bias, timestep)
		self.cv2 = nn.Conv2d(320, 640, 1, 2, 0, bias = False)
		self.stage3 = ResNetStage(n_blocks, 640, False, batchnorm, bias, timestep)
		self.bn = nn.BatchNorm2d(640, track_running_stats = True)
		self.avgpool = nn.AvgPool2d(8, 8)
		with torch.no_grad():
			featuresize = self.forward_conv(torch.zeros(*datashape))[0].view(-1).shape[0]
		self.classifier = nn.Linear(featuresize, n_classes)
	def forward_conv(self, x):
		rs = dict()
		x = self.cv0(x)
		x, rs[1] = self.stage1(x)
		x = self.cv1(x)
		x, rs[2] = self.stage2(x)
		x = self.cv2(x)
		x, rs[3] = self.stage3(x)
		x = functional.relu(self.bn(x), inplace = True)
		x = self.avgpool(x)
		return x, [r for i in range(1, 4) for r in rs[i]]
	def forward(self, x):
		x, rs = self.forward_conv(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return (x,rs) if self.return_residus else x
	def feature_list(self, x):
		x_list = []
		x = self.cv0(x)
		x_list.append(x)
		x, _= self.stage1(x)
		x_list.append(x)
		x = self.cv1(x)
		x, _ = self.stage2(x)
		x_list.append(x)
		x = self.cv2(x)
		x, _ = self.stage3(x)
		x_list.append(x)
		x = functional.relu(self.bn(x), inplace = True)
		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		return self.classifier(x), x_list
	def intermediate_forward(self, x, stage):
		x = self.cv0(x)
		if stage == 1:
			x, _ = self.stage1(x)
		elif stage == 2:
			x, _ = self.stage1(x)
			x = self.cv1(x)
			x, _ = self.stage2(x)
		elif stage == 3:
			x, _ = self.stage1(x)
			x = self.cv1(x)
			x, _ = self.stage2(x)
			x = self.cv2(x)
			x, _ = self.stage3(x)             
		return x

class ResNextBlock(nn.Module):
	def __init__(self, in_filters = 256, planes = 64, expansion = 4, cardinality = 32, width = 4, base = 64, stride = 1, downsample = None):
		super(ResNextBlock, self).__init__()
		self.intfilters = cardinality * math.floor(planes * width / base)
		self.outfilters = planes * expansion
		self.cv1 = nn.Conv2d(in_filters, self.intfilters, 1, 1, 0, bias = False)
		self.bn1 = nn.BatchNorm2d(self.intfilters)
		self.cv2 = nn.Conv2d(self.intfilters, self.intfilters, 3, stride, 1, groups = cardinality, bias = False)
		self.bn2 = nn.BatchNorm2d(self.intfilters)
		self.cv3 = nn.Conv2d(self.intfilters, self.outfilters, 1, 1, 0, bias = False)
		self.bn3 = nn.BatchNorm2d(self.outfilters)
		self.downsample = downsample
	def forward(self, x):
		r = functional.relu(self.bn1(self.cv1(x)), inplace = True)
		r = functional.relu(self.bn2(self.cv2(r)), inplace = True)
		r = functional.relu(self.bn3(self.cv3(r)), inplace = True)
		if self.downsample is not None:
			x = self.downsample(x)
		z = functional.relu(x + r, inplace = True)
		return z, z if RES_TYPE == 'outputs' else (z - x if RES_TYPE == 'postrelu' else r)

class ResNextStage(nn.Module):
	def __init__(self, nb, inf = 256, pln = 64, exp = 4, card = 32, width = 4, base = 64, stride = 1):
		super(ResNextStage, self).__init__()
		intf = pln * exp
		ds = nn.Sequential(nn.Conv2d(inf, intf, 1, stride, bias = False), nn.BatchNorm2d(intf)) if stride != 1 or inf != intf else None
		block = lambda i : ResNextBlock(inf, pln, exp, card, width, base, stride, ds) if i == 0 else ResNextBlock(intf, pln, exp, card, width, base, 1)
		self.blocks = nn.ModuleList([block(i) for i in range(nb)])
	def forward(self, x):
		rs = []
		for block in self.blocks :
			x, r = block(x)
			rs.append(r)
		return x, rs

class ResNext50(nn.Module):
	def __init__(self, datashape, n_classes, n_blocks = [3, 4, 6, 3], in_filters = 64, planes = 64, expansion = 4, cardinality = 32, width = 4, base = 64, return_residus = True):
		super(ResNext50, self).__init__()
		intfilters = int(in_filters / 2)
		self.return_residus = return_residus
		self.encoder = nn.Sequential(nn.Conv2d(3, intfilters, 3, 1, 1), nn.BatchNorm2d(intfilters), nn.ReLU(True), nn.Conv2d(intfilters, in_filters, 3, 1, 1), nn.BatchNorm2d(in_filters), nn.ReLU(True))
		self.stage1 = ResNextStage(n_blocks[0], in_filters * 1, planes * 1, expansion, cardinality, width, base, 1)
		self.stage2 = ResNextStage(n_blocks[1], in_filters * 4, planes * 2, expansion, cardinality, width, base, 2)
		self.stage3 = ResNextStage(n_blocks[2], in_filters * 8, planes * 4, expansion, cardinality, width, base, 2)
		self.stage4 = ResNextStage(n_blocks[3], in_filters * 16, planes * 8, expansion, cardinality, width, base, 2)
		self.avgpool = nn.AvgPool2d(7 if datashape[-1] == 224 else 4, 1)
		with torch.no_grad():
			featuresize = self.forward_conv(torch.zeros(*datashape))[0].view(-1).shape[0]
		self.classifier = nn.Linear(featuresize, n_classes)
	def forward_conv(self, x):
		rs = dict()
		x = self.encoder(x)
		x, rs[1] = self.stage1(x)
		x, rs[2] = self.stage2(x)
		x, rs[3] = self.stage3(x)
		x, rs[4] = self.stage4(x)
		x = self.avgpool(x)
		return x, [r for i in range(1, 5) for r in rs[i]]
	def forward(self, x):
		x, rs = self.forward_conv(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return (x,rs) if self.return_residus else x
	def feature_list(self, x):
		x_list = []
		x = self.encoder(x)
		x_list.append(x)
		x, _= self.stage1(x)
		x_list.append(x)
		x, _ = self.stage2(x)
		x_list.append(x)
		x, _ = self.stage3(x)
		x_list.append(x)
		x, _ = self.stage4(x)
		x_list.append(x)
		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		return self.classifier(x), x_list
	def intermediate_forward(self, x, stage):
		x = self.encoder(x)
		if stage == 1:
			x, _ = self.stage1(x)
		elif stage == 2:
			x, _ = self.stage1(x)
			x, _ = self.stage2(x)
		elif stage == 3:
			x, _ = self.stage1(x)
			x, _ = self.stage2(x)
			x, _ = self.stage3(x)
		elif stage == 4:
			x, _ = self.stage1(x)
			x, _ = self.stage2(x)
			x, _ = self.stage3(x)
			x, _ = self.stage4(x)               
		return x


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)

def _RoundChannels(c, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_c = max(min_value, int(c + divisor / 2) // divisor * divisor)
    if new_c < 0.9 * c:
        new_c += divisor
    return new_c

def _RoundRepeats(r):
    return int(math.ceil(r))

def _DropPath(x, drop_prob, training):
    if drop_prob > 0 and training:
        keep_prob = 1 - drop_prob
        if x.is_cuda:
            mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        else:
            mask = Variable(torch.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)

    return x

def _BatchNorm(channels, eps=1e-3, momentum=0.01):
    return nn.BatchNorm2d(channels, eps=eps, momentum=momentum)

def _Conv3x3Bn(in_channels, out_channels, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
        _BatchNorm(out_channels),
        Swish()
    )

def _Conv1x1Bn(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        _BatchNorm(out_channels),
        Swish()
    )

class SqueezeAndExcite(nn.Module):
    def __init__(self, channels, squeeze_channels, se_ratio):
        super(SqueezeAndExcite, self).__init__()

        squeeze_channels = squeeze_channels * se_ratio
        if not squeeze_channels.is_integer():
            raise ValueError('channels must be divisible by 1/ratio')

        squeeze_channels = int(squeeze_channels)
        self.se_reduce = nn.Conv2d(channels, squeeze_channels, 1, 1, 0, bias=True)
        self.non_linear1 = Swish()
        self.se_expand = nn.Conv2d(squeeze_channels, channels, 1, 1, 0, bias=True)
        self.non_linear2 = nn.Sigmoid()

    def forward(self, x):
        y = torch.mean(x, (2, 3), keepdim=True)
        y = self.non_linear1(self.se_reduce(y))
        y = self.non_linear2(self.se_expand(y))
        y = x * y

        return y

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio, drop_path_rate):
        super(MBConvBlock, self).__init__()

        expand = (expand_ratio != 1)
        expand_channels = in_channels * expand_ratio
        se = (se_ratio != 0.0)
        self.residual_connection = (stride == 1 and in_channels == out_channels)
        self.drop_path_rate = drop_path_rate

        convo = []

        if expand:
            # expansion phase
            pw_expansion = nn.Sequential(
                nn.Conv2d(in_channels, expand_channels, 1, 1, 0, bias=False),
                _BatchNorm(expand_channels),
                Swish()
            )
            convo.append(pw_expansion)

        # depthwise convolution phase
        dw = nn.Sequential(
            nn.Conv2d(
                expand_channels,
                expand_channels,
                kernel_size,
                stride,
                kernel_size//2,
                groups=expand_channels,
                bias=False
            ),
            _BatchNorm(expand_channels),
            Swish()
        )
        convo.append(dw)

        if se:
            # squeeze and excite
            squeeze_excite = SqueezeAndExcite(expand_channels, in_channels, se_ratio)
            convo.append(squeeze_excite)

        # projection phase
        pw_projection = nn.Sequential(
            nn.Conv2d(expand_channels, out_channels, 1, 1, 0, bias=False),
            _BatchNorm(out_channels)
        )
        convo.append(pw_projection)

        self.convo = nn.Sequential(*convo)

    def forward(self, x):
        if self.residual_connection:
            r = _DropPath(self.convo(x), self.drop_path_rate, self.training)
            return x + r, r
        else:
            return self.convo(x), None

class MBConvStage(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio, repeats, drop_connect_rate, n_previous_blocks, total_blocks):
        super(MBConvStage, self).__init__()
        drop_rates = [drop_connect_rate * (n_previous_blocks + i) / total_blocks for i in range(repeats)]
        self.blocks = nn.ModuleList([MBConvBlock(in_channels if i == 0 else out_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio, drop_rates[i]) for i in range(repeats)])
    def forward(self, x):
        rs = []
        for block in self.blocks :
            x, r = block(x)
            if r is not None:
                rs.append(r)
        return x, rs

class EfficientNet(nn.Module):
   

    def __init__(self, datashape, n_classes, return_residus = True):
        super(EfficientNet, self).__init__()
        # (in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio, repeats)
        self.config = [
        [32,  16,  3, 1, 1, 0.25, 1],
        [16,  24,  3, 2, 6, 0.25, 2],
        [24,  40,  5, 2, 6, 0.25, 2],
        [40,  80,  3, 2, 6, 0.25, 3],
        [80,  112, 5, 1, 6, 0.25, 3],
        [112, 192, 5, 2, 6, 0.25, 4],
        [192, 320, 3, 1, 6, 0.25, 1]
        ]
        self.return_residus = return_residus
        
        stem_channels = 32
        drop_rate = 0.2


        # stem convolution
        self.stem_conv = _Conv3x3Bn(datashape[1], stem_channels, 2)


        n_previous_blocks = np.cumsum([0] + [conf[6] for conf in self.config])
        total_blocks = n_previous_blocks[-1]
        
        self.stage1 = MBConvStage(*(self.config[0] + [drop_rate, n_previous_blocks[1], total_blocks]))
        self.stage2 = MBConvStage(*(self.config[1] + [drop_rate, n_previous_blocks[1], total_blocks]))
        self.stage3 = MBConvStage(*(self.config[2] + [drop_rate, n_previous_blocks[2], total_blocks]))
        self.stage4 = MBConvStage(*(self.config[3] + [drop_rate, n_previous_blocks[3], total_blocks]))
        self.stage5 = MBConvStage(*(self.config[4] + [drop_rate, n_previous_blocks[4], total_blocks]))
        self.stage6 = MBConvStage(*(self.config[5] + [drop_rate, n_previous_blocks[5], total_blocks]))
        self.stage7 = MBConvStage(*(self.config[6] + [drop_rate, n_previous_blocks[6], total_blocks]))

        self.eval()
        with torch.no_grad():
            feature_size = self.forward_conv(torch.zeros(*datashape))[0].view(-1).shape[0]
            feature_size = 1024
        # last several layers
        self.head_conv = _Conv1x1Bn(self.config[-1][1], feature_size)
        #self.avgpool = nn.AvgPool2d(input_size//32, stride=1)
        self.dropout = nn.Dropout(drop_rate)
        
        self.classifier = nn.Linear(feature_size, n_classes)


    def forward_conv(self, x):
        rs = dict()
        x = self.stem_conv(x)
        x, rs[1] = self.stage1(x)
        x, rs[2] = self.stage2(x)
        x, rs[3] = self.stage3(x)
        x, rs[4] = self.stage4(x)
        x, rs[5] = self.stage5(x)
        x, rs[6] = self.stage6(x)
        x, rs[7] = self.stage7(x)
        return x, [r for i in range(1, 8) for r in rs[i]]

    def forward(self, x):
        x, rs = self.forward_conv(x)
        #x = self.avgpool(x)
        #x = x.view(x.size(0), -1)
        x = self.head_conv(x)
        x = torch.mean(x, (2, 3))
        x = self.dropout(x)
        x = self.classifier(x)

        return (x, rs) if self.return_residus else x

    def feature_list(self, x):
        x_list = []
        x = self.stem_conv(x)
        x_list.append(x)
        x, _= self.stage1(x)
        x_list.append(x)
        x, _ = self.stage2(x)
        x_list.append(x)
        x, _ = self.stage3(x)
        x_list.append(x)
        x, _ = self.stage4(x)
        x_list.append(x)
        x, _ = self.stage5(x)
        x_list.append(x)
        x, _ = self.stage6(x)
        x_list.append(x)
        x, _ = self.stage7(x)
        x_list.append(x)
        x = self.head_conv(x)
        x = torch.mean(x, (2, 3))
        x = self.dropout(x)
        return self.classifier(x), x_list

    def intermediate_forward(self, x, stage):
        x = self.stem_conv(x)
        if stage == 1:
            x, _ = self.stage1(x)
        elif stage == 2:
            x, _ = self.stage1(x)
            x, _ = self.stage2(x)
        elif stage == 3:
            x, _ = self.stage1(x)
            x, _ = self.stage2(x)
            x, _ = self.stage3(x)
        elif stage == 4:
            x, _ = self.stage1(x)
            x, _ = self.stage2(x)
            x, _ = self.stage3(x)
            x, _ = self.stage4(x)    
        elif stage == 5:
            x, _ = self.stage1(x)
            x, _ = self.stage2(x)
            x, _ = self.stage3(x)
            x, _ = self.stage4(x)
            x, _ = self.stage5(x)
        elif stage == 6:
            x, _ = self.stage1(x)
            x, _ = self.stage2(x)
            x, _ = self.stage3(x)
            x, _ = self.stage4(x)
            x, _ = self.stage5(x)
            x, _ = self.stage6(x)
        elif stage == 7:
            x, _ = self.stage1(x)
            x, _ = self.stage2(x)
            x, _ = self.stage3(x)
            x, _ = self.stage4(x)
            x, _ = self.stage5(x)
            x, _ = self.stage6(x)
            x, _ = self.stage7(x)
        return x

# outputs, postrelu, prerelu
RES_TYPE = 'prerelu'

