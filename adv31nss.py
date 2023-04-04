import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as functional, torch.utils.data as torchdata
from dataloaders8 import dataloaders
from utils6 import *
from mahalanobis_b import *
from detector_class_b import *
from models import *
from torchsummary import summary
import time, math, numpy as np, argparse, os, collections, sys, inspect, pprint, scipy.stats as st
from functools import partial
from art.attacks.evasion import FastGradientMethod, AutoProjectedGradientDescent, BasicIterativeMethod, CarliniL2Method, DeepFool, BoundaryAttack, HopSkipJump
from art.estimators.classification import PyTorchClassifier
from collections import OrderedDict
from autoattack import AutoAttack
from nss import *


# with maha noise added to val data, noise size changes per attack, only correctly classified with and without noise and successfully attacked are considered

def get_model(datashape, modelname, n_classes, return_residus, folder):
	if modelname == 'resnext50':
		model = ResNext50(datashape, n_classes, return_residus = return_residus)
	elif modelname == 'onerep':
		model =  OneRepResNet(datashape, n_classes, return_residus = return_residus)
	elif modelname == 'resnet110':
		model =  ResNet110(datashape, n_classes, return_residus = return_residus)
	elif modelname == 'wide':
		model =  WideResNet(datashape, n_classes, return_residus = return_residus)
	elif modelname == 'efficientnet':
		model = EfficientNet(datashape, n_classes, return_residus = return_residus)
	else:
		raise NotImplementedError()
	state_dict = torch.load(os.path.join('weights', folder, 'weights.pth'), map_location = torch.device('cpu'))
	state_dict = modify_state_dict(state_dict) if modelname == 'wide' else state_dict
	model.load_state_dict(state_dict)
	model.eval()
	model.to(device)
	return model


def get_attack_names_and_params(attack_names, eps):
	if attack_names in ['fgm', 'pgd', 'bim', 'auto']:
		return [(attack_names, eps)]
	if attack_names in ['df', 'cw2', 'cwinf', 'hsj', 'ba']:
		return [(attack_names, None)]
	if attack_names == 'wb':
		return [('fgm', eps), ('pgd', eps), ('bim', eps), ('auto', eps), ('cw2', None), ('df', None)]
	if attack_names == 'bb':
		return [('hsj', None), ('ba', None)]
	if attack_names == 'wbf':
		return [('fgm', eps), ('pgd', eps), ('bim', eps), ('auto', eps)]
	if attack_names == 'wbs':
		return [('cw2', None), ('df', None)]
	if attack_names == 'all':
		return [('fgm', eps), ('pgd', eps), ('bim', eps), ('auto', eps), ('cw2', None), ('df', None), ('hsj', None), ('ba', None)]
	else:
		raise NotImplementedError()


def get_attack(attack_name, mod, eps = 0.03, batchsize = 32):
	if attack_name == 'pgd':
		return AutoProjectedGradientDescent(estimator = mod, eps = eps, batch_size = batchsize, verbose = False)
	if attack_name == 'fgm':
		return FastGradientMethod(estimator = mod, batch_size = batchsize, eps = eps)
	if attack_name == 'cw2':
		return CarliniL2Method(classifier = mod, batch_size = batchsize, verbose = False)
	if attack_name == 'df':
		return DeepFool(classifier = mod, batch_size = batchsize, verbose = False)
	if attack_name == 'bim':
		return BasicIterativeMethod(estimator = mod, eps = eps, batch_size = batchsize, verbose = False)
	if attack_name == 'hsj':
		return HopSkipJump(classifier = mod, batch_size = batchsize, verbose = False)
	if attack_name == 'ba':
		return BoundaryAttack(estimator = mod, targeted = False, batch_size = batchsize, verbose = False)
	if attack_name == 'auto':
		return AutoAttack(mod, eps = eps, verbose = False)
	if attack_name == None:
		return None
	else:
		raise NotImplementedError()


def get_adver_sample(x, attack, batchsize = 32, y = None, attack_name = None):
	if attack == None:
		return x
	if attack_name == 'auto':
		return attack.run_standard_evaluation(x, y, bs = batchsize)
	else:
		return torch.from_numpy(attack.generate(x = x.detach().cpu().numpy())).to(device, dtype = torch.float)

class Adver :
	def __init__(self, name, eps, mod, batchsize = 32, dataset = None):
		self.name = name
		self.eps = eps
		self.adver = lambda x, y = None : get_adver_sample(x, get_attack(name, mod, eps, batchsize), batchsize, y, name)
		self.noisy = lambda x : add_random_noise(x, *get_random_noise_params(dataset, name)) if dataset is not None else None
	def __str__(self):
		return self.name + ((' ' + str(self.eps)) if self.eps is not None else '')



def get_attacks(attack_names, eps, mode_, classifier, batchsize, dataset, setting = 'seen', train_attack_name = None, train_attack_epsilon = None):
	attack_names_and_epsilons = get_attack_names_and_params(attack_names, eps)
	if setting == 'unseen':
		attack_names_and_epsilons = [i for i in attack_names_and_epsilons if i != (train_attack_name, train_attack_epsilon)]
		train_attack = Adver(train_attack_name, train_attack_epsilon, mode_ if train_attack_name == 'auto' else classifier, batchsize, dataset)
	attacks = [Adver(name, eps, mode_ if name == 'auto' else classifier, batchsize, dataset) for name, eps in attack_names_and_epsilons]
	return attacks, train_attack if setting == 'unseen' else None



	

 



	
def get_model_pred_and_stats(model, x, y):
	out, rs  = model(x)
	_, pred = torch.max(- out.data, 1) if rce else torch.max(out.data, 1)
	correct = (pred == y).item() == 1.0
	x = x.cpu().detach().numpy()
	nss = calculate_brisque_features(x[0].transpose(1, 2, 0), kernel_size = 7, sigma = 7 / 6)
	return pred, correct, nss.tolist()

add_random_noise = lambda x, random_noise_size, min_pixel, max_pixel : torch.clamp(torch.add(x.data, torch.randn(x.size()).cuda(), alpha = random_noise_size), min_pixel, max_pixel)

def get_val_stats(model, loader, attack):

	def add_nss_stats(x, stats):
		stats_ = calculate_brisque_features(x, kernel_size = 7, sigma = 7 / 6)
		stats = stats_ if stats is None else np.vstack((stats, stats_))
		return stats

	t0 = time.time()
	clean_stats, adver_stats = None, None

	for k, (x, y) in enumerate(loader):
		
		try :
			clean_x, y = x.to(device).float(), y.to(device)
			adver_x = attack.adver(clean_x, y) 
			clean_x = clean_x.cpu().detach().numpy()
			adver_x = adver_x.cpu().detach().numpy()
		except (OverflowError, ValueError, TypeError, Exception) as err :
			print('Overflow or value error or type error, skipping train batch', k + 1)
			continue


		for i in range(clean_x.shape[0]):
			try :
				clean_stats = add_nss_stats(clean_x[i].transpose(1, 2, 0), clean_stats)
				adver_stats = add_nss_stats(adver_x[i].transpose(1, 2, 0), adver_stats)
			except (OverflowError, ValueError, TypeError, Exception) as err :
				print('Overflow or value error or type error, skipping train image', (k + 1) * clean_x.shape[0] + i + 1)

	print('val time', time.time() - t0, 'seconds', flush = True)
	return clean_stats, adver_stats


def remove_nan_inf(X):
	X_ = X[~np.isnan(X).any(axis = 1)]
	X_ = X_[~np.isinf(X_).any(axis = 1)]
	return X_


def create_detectors(clean_stats, adver_stats):
	detectors = {}
	clean_stats_, adver_stats_ = remove_nan_inf(clean_stats), remove_nan_inf(adver_stats)
	X = np.concatenate((clean_stats_, adver_stats_))
	Y = np.concatenate((np.full((clean_stats_.shape[0], ), 0), np.full((adver_stats_.shape[0], ), 1)))
	detectors['RF NSS'] = get_trained_detector('RF NSS detector', X, Y, timeit)
	# detectors['LR NSS'] = get_trained_detector('LR NSS detector', X, Y, timeit)
	# detectors['SVC NSS'] = get_trained_detector('SVC NSS detector', X, Y, timeit)
	return list(detectors.values())


def merge_clean_and_noisy_stats(clean_stats, noisy_stats, n_classes):
	clean_and_noisy_transports_stats = {'norms' : np.concatenate((clean_stats['norms'], noisy_stats['norms'])), 'cosines' : np.concatenate((clean_stats['cosines'], noisy_stats['cosines'])),
										'norms class' : [concat(clean_stats['norms class'][j], noisy_stats['norms class'][j]) for j in range(n_classes)], 
									    'cosines class' : [concat(clean_stats['cosines class'][j], noisy_stats['cosines class'][j]) for j in range(n_classes)]
				 	             		}
	clean_and_noisy_maha_stats = {'maha' + magnitude : np.concatenate((clean_stats['maha' + magnitude], noisy_stats['maha' + magnitude])) for magnitude in magnitudes}
	clean_and_noisy_maha_class_stats = {'maha' + mag + ' class' : [concat(clean_stats['maha' + mag + ' class'][j], noisy_stats['maha' + mag + ' class'][j]) for j in range(n_classes)] for mag in magnitudes}
	return {**clean_and_noisy_transports_stats , **clean_and_noisy_maha_stats, **clean_and_noisy_maha_class_stats}
	

def train_detectors(model, attack, loader):
	print('\n------ Training detectors on', attack)
	t0 = time.time()
	clean_stats, adver_stats = get_val_stats(model, loader, attack)
	# clean_and_noisy_stats = merge_clean_and_noisy_stats(clean_stats, noisy_stats, n_classes)
	detectors_no_noise = create_detectors(clean_stats, adver_stats)
	# detectors_with_noise = create_detectors(clean_and_noisy_stats, adver_stats, n_classes, cutoff, 'with noise')
	# detectors = detectors_no_noise + detectors_with_noise
	print('Generating train attacks and training detectors took', time.time() - t0, 'seconds', flush = True)
	return detectors_no_noise

def test_detectors(model, attack, loader, detectors, correct_on_noisy_only):
	print('\n------ Testing detectors on', attack)
	t0 = time.time()
	n_test, n_clean_correct, n_noisy_correct, n_adver_correct = len(loader), 0, 0, 0
	for detector in detectors:
		detector.reset() 
	for k, (x, y) in enumerate(loader):
		try :
			clean_x, y = x.to(device).float(), y.to(device)
			noisy_x = attack.noisy(clean_x)
			adver_x = attack.adver(clean_x, y)
			clean_pred, clean_correct, clean_nss = get_model_pred_and_stats(model, clean_x, y)
			noisy_pred, noisy_correct, noisy_nss = get_model_pred_and_stats(model, noisy_x, y)
			adver_pred, adver_correct, adver_nss = get_model_pred_and_stats(model, adver_x, y)
			n_clean_correct += clean_correct
			n_noisy_correct += noisy_correct
			n_adver_correct += adver_correct
			for detector in detectors:
				clean_detected = detector.detect(nss = clean_nss)
				adver_detected = detector.detect(nss = adver_nss)
				detector.update_counters(clean_correct, clean_detected, adver_correct, adver_detected, noisy_correct if correct_on_noisy_only else None)
		except (OverflowError, ValueError, TypeError, Exception) as err :
			print('Overflow or value error or type error, skipping test image', k)
		if (k + 1) % 200 == 0 :
			print(k + 1, 'test images out of', n_test, 'took', time.time() - t0, 's', flush = True)
	print('Accuracy on clean images', n_clean_correct / n_test, 'Accuracy on attacked images', n_adver_correct / n_test, 'Testing took', time.time() - t0, 'seconds', flush = True)
	for detector in detectors:
		print('\n------', detector.name.ljust(70), ' '.join(stat_name + ' ' + str(round(stat, 3)) for stat_name, stat in detector.stats().items() if stat != -1))
	
def experiment(dataset, modelname, traintype, setting, attack_names, epsilon, train_attack_name, train_attack_epsilon, cutoff, correct_on_noisy_only, batchsize, trainsize, valsize, testsize, seed):
	
	t0 = time.time()
	frame = inspect.currentframe()
	names, _, _, values = inspect.getargvalues(frame)
	print('Experiment from adv31 with parameters:')
	for name in names:
		print('%s = %s' % (name, values[name]))
	if seed is not None:
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
		torch.manual_seed(seed)
		np.random.seed(seed)

	_, _, testloader, _, _, _, _ = dataloaders(dataset, 1, trainsize, valsize, testsize)
	trainloader, valloader, _, datashape, n_classes, _, _ = dataloaders(dataset, batchsize, trainsize, valsize, testsize)
	print(len(trainloader) * batchsize, 'train images', len(valloader) * batchsize, 'val images', len(testloader), 'test images')
	folder = modelname + '-' + dataset + '-' + traintype
	model = get_model(datashape, modelname, n_classes, 1, folder)
	mode_ = get_model(datashape, modelname, n_classes, 0, folder)
	criterion, n_res, n_test = nn.CrossEntropyLoss(), len(model(next(iter(trainloader))[0].to(device))[1]), len(testloader)
	optimizer = optim.SGD(mode_.parameters(), lr = 0.1, momentum = 0.9, weight_decay = 0.0001)
	classifier = PyTorchClassifier(model = mode_, loss = criterion, input_shape = datashape[1:], nb_classes = n_classes, clip_values = (0, 1), optimizer = optimizer)
	attacks, train_attack = get_attacks(attack_names, epsilon, mode_, classifier, batchsize, dataset, setting, train_attack_name, train_attack_epsilon)

	if setting == 'seen':
		for attack in attacks:
			detectors = train_detectors(model, attack, valloader)
			test_detectors(model, attack, testloader, detectors, correct_on_noisy_only)
	elif setting == 'unseen':
		detectors = train_detectors(model, train_attack, valloader)
		for attack in attacks:
			test_detectors(model, attack, testloader, detectors, correct_on_noisy_only)
	else:
		raise NotImplementedError('Setting unknown')

	print('\nTotal time', time.time() - t0, 'seconds')


		
		
	

	
if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("-dat", "--dataset", required = True, choices = ['mnist', 'cifar10', 'cifar100', 'svhn', 'imagenet2012', 'tinyimagenet', 'imagenetdownloader'])
	parser.add_argument("-mod", "--modelname", required = True, choices = ['resnext29', 'resnext50', 'onerep', 'resnet110', 'avgpool', 'wide', 'efficientnet'])
	parser.add_argument("-trt", "--traintype", required = True, choices = ['van', 'rce', 'lap'])
	parser.add_argument("-set", "--setting", required = True, choices = ['seen', 'unseen'])
	parser.add_argument("-att", "--attacknames", required = True, choices = ['fgm', 'pgd', 'bim', 'df', 'cw2', 'auto', 'hsj', 'ba', 'wb', 'bb', 'wbf', 'wbs', 'all'])
	parser.add_argument("-eps", "--epsilon", type = float, default = 0.03)
	parser.add_argument("-tra", "--trainattackname", default = 'fgm', choices = ['fgm', 'pgd', 'bim', 'df', 'cw2', 'cwinf', 'auto', 'hsj', 'ba'])
	parser.add_argument("-tre", "--trainattackepsilon", type = float, default = 0.03)
	parser.add_argument("-cut", "--cutoff", type = int, default = 1)
	parser.add_argument("-cno", "--correctnoisyonly", type = int, default = 0, choices = [0, 1])
	parser.add_argument("-bas", "--batchsize", type = int, default = 128)
	parser.add_argument("-trs", "--trainsize", type = float, default = 1)
	parser.add_argument("-vls", "--valsize", type = float, default = 0.9)
	parser.add_argument("-tss", "--testsize", type = float, default = 0.1)
	parser.add_argument("-see", "--seed", type = int, default = None)
	args = parser.parse_args()

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	magnitudes = ['0', '0.01', '0.001', '0.0014', '0.002', '0.005', '0.0005']
	rce = args.traintype == 'rce'
	parameters = [values for name, values in vars(args).items()]
	experiment(*parameters)