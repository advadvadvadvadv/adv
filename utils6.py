import torch.nn as nn, os, numpy as np, time
from functools import partial, wraps
import torch.nn.functional as functional
from collections import OrderedDict

stack = lambda d :  {name: np.vstack(inp) for name, inp in d.items()}
get_avg = lambda d, n : [d[i].avg for i in range(n)]
l2norm = lambda x : np.sqrt(np.sum(x ** 2, axis = (1, 2, 3)))

convDiag = lambda x, M : functional.conv2d(x, M, stride = 1, padding = 1, groups = M.shape[0])
convDiagT = lambda x, M : functional.conv_transpose2d(x, M, stride = 1, padding = 1, groups = M.shape[0])

concat = lambda a, b : np.concatenate((a , b)) if a is not None and b is not None else (a if a is not None and b is None else (b if b is not None else None))


def modify_state_dict(state_dict):
	new_state_dict = OrderedDict()
	for k, v in state_dict.items():
		name = k.replace('module.', '')
		new_state_dict[name] = v
	return new_state_dict

def timeit(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print('function {} with args {} and {} took {:.4f} seconds'.format(f.__name__, args[0], args[1], te - ts))
        return result
    return wrap

def topkaccuracy(output, target, topk = (1, )):
	maxk = max(topk)
	num = len(target)
	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))
	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0)
		res.append(correct_k.item() / num)
	return res



def create_classifier(name, n_classes, featureshape, filters = None):
	featuresize = np.prod(featureshape) if (type(featureshape) is list or type(featureshape) is tuple) else featureshape
	if name == '1Lin':
		return nn.Sequential(nn.Linear(featuresize, n_classes))
	if name == '2Lin':
		return nn.Sequential(nn.Linear(featuresize, n_classes * 10), nn.Sigmoid(), nn.Linear(n_classes * 10, n_classes))
	if name == '3Lin':
		return nn.Sequential(nn.Linear(featuresize, n_classes * 10), nn.BatchNorm1d(n_classes * 10), nn.ReLU(True), nn.Linear(n_classes * 10, n_classes))
	

def initialize(name, gain, module):
	if name == 'orthogonal':
		init = partial(nn.init.orthogonal_, gain = gain) 
	elif name == 'normal':
		init = partial(nn.init.normal_, mean = 0, std = gain) 
	elif name == 'kaiming':
		init = partial(nn.init.kaiming_normal_, a = 0, mode = 'fan_out', nonlinearity = 'relu')
	else:
		raise ValueError('Unknown init ' + name)
	if isinstance(module, nn.Conv2d):
		init(module.weight)
		if hasattr(module, 'bias') and module.bias is not None:
			nn.init.constant_(module.bias, 0)
	elif isinstance(module, nn.BatchNorm2d):
		if hasattr(module, 'weight') and module.weight is not None:
			nn.init.constant_(module.weight, 1)
		if hasattr(module, 'bias') and module.bias is not None:
			nn.init.constant_(module.bias, 0)
	elif isinstance(module, nn.Linear):
		init(module.weight)
		if hasattr(module, 'bias') and module.bias is not None:
			nn.init.constant_(module.bias, 0)

def product(iterables):
    if len(iterables) == 0 :
        yield ()
    else :
        it = iterables[0]
        for item in it :
            for items in product(iterables[1: ]) :
                yield (item, ) + items

def make_folder(folder):
	if not os.path.exists(folder):
		os.makedirs(folder)

class AverageMeter(object):
	def __init__(self):
		self.reset()
	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0
	def update(self, val, num):
		self.val = val
		self.sum += val * num
		self.count += num
		self.avg = self.sum / self.count

def update_meters(y, pred, loss, loss_meter, acc_meter, trs = None, trs_meter = None, t = None, time_meter = None):
	num = len(y)
	correct = (pred == y).sum().item()
	accuracy = correct / num
	loss_meter.update(loss, num)
	acc_meter.update(accuracy, num)
	if trs is not None and trs_meter is not None:
		trs_meter.update(trs, num)
	if t is not None and time_meter is not None :
		time_meter.update(t, 1)








