import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as functional, torch.utils.data as torchdata
from dataloaders import dataloaders
from utils6 import *
from mahalanobis_b import *
from detector_class_b import *
from models import *
from torchsummary import summary
import time, math, numpy as np, argparse, os, collections, sys, inspect, pprint, scipy.stats as st
from functools import partial
from art.attacks.evasion import FastGradientMethod, AutoProjectedGradientDescent, BasicIterativeMethod, CarliniL2Method, DeepFool, BoundaryAttack, HopSkipJump, AutoAttack
from art.estimators.classification import PyTorchClassifier
from collections import OrderedDict


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

def get_loaders(ind_dataset, ood1_dataset, ood2_dataset, batchsize, ind_valsize, ind_testsize, ood1_valsize, ood1_testsize, ood2_testsize):
	ind_trainloader, ind_valloader, _, datashape, n_classes, _, _ = dataloaders(ind_dataset, batchsize, 1, ind_valsize, ind_testsize)
	ood1_trainloader, ood1_valloader, _, _, _, _, _ = dataloaders(ood1_dataset, batchsize, 1, ood1_valsize, ood1_testsize)
	_, _, ind_testloader, _, _, _, _ = dataloaders(ind_dataset, 1, 1, ind_valsize, ind_testsize)
	_, _, ood1_testloader, _, _, _, _ = dataloaders(ood1_dataset, 1, 1, ood1_valsize, ood1_testsize)
	_, _, ood2_testloader, _, _, _, _ = dataloaders(ood2_dataset, 1, 1, 0.1, ood2_testsize)
	print('IND:', ind_dataset, len(ind_valloader), 'val batches', len(ind_testloader), 'test batches')
	print('OOD:', ood1_dataset, len(ood1_valloader), 'val batches', len(ood1_testloader), 'test batches')
	print('OOD:', ood2_dataset, len(ood2_testloader), 'test batches')
	return ind_trainloader, ind_valloader, ind_testloader, ood1_valloader, ood1_testloader, ood2_testloader, datashape, n_classes


def get_model_pred_and_stats(model, x, y, nclasses = None, maha_params = None):
	sample_mean, precision, num_output = maha_params if maha_params is not None else (None, None, None)
	out, rs  = model(x)
	_, pred = torch.max(- out.data, 1) if rce else torch.max(out.data, 1)
	correct = (pred == y).item() == 1.0
	norms = [torch.mean(r ** 2, (1,2,3)).cpu().detach().item() for r in rs]
	cosines = [functional.cosine_similarity(torch.ones(r.size()[1:]).flatten().to(device), r[j,:,:,:].flatten(), dim = 0).cpu().detach().item() for r in rs for j in range(r.size()[0])]
	transport = sum(norms)
	M = {None: None}
	for magnitude in magnitudes:
		for layer in range(num_output):
			ngs = get_Mahalanobis_score_adv(model, x, nclasses, sample_mean, precision, layer, float(magnitude)).tolist()
			M[magnitude] = ngs if layer == 0 else M[magnitude] + ngs
	return pred, correct, rs, transport, norms, cosines, M

def get_val_stats(model, ind_loader, ood_loader, n_classes, n_res, maha_params):

	def create_stats_dict():
		transport_stats = {'norms' : [[] for i in range(n_res)], 'cosines' : [[] for i in range(n_res)]}
		transports_class_stats = {'norms class' : [[[] for i in range(n_res)] for j in range(n_classes)], 'cosines class' : [[[] for i in range(n_res)] for j in range(n_classes)]}
		maha_stats = {'maha' + magnitude : None for magnitude in magnitudes}
		maha_class_stats = {'maha' + magnitude + ' class' : [None for j in range(n_classes)] for magnitude in magnitudes}
		return {**transport_stats, **transports_class_stats, **maha_stats, **maha_class_stats}

	def get_model_pred_and_residus(x, y):
		out, rs = model(x)
		_, pred = torch.max(- out.data, 1) if rce else torch.max(out.data, 1)
		correct = pred.eq(y.data).cpu()
		return correct, pred, rs

	def add_batch_norms_cosines(pred, rs, stats):
		for i in range(n_res):
			z = torch.ones(rs[i].size()[1:]).flatten().to(device) # vector of ones
			n = torch.mean(rs[i] ** 2, (1,2,3)).cpu().detach().numpy() # norms of block i
			c = np.array([functional.cosine_similarity(z, rs[i][j,:,:,:].flatten(), dim = 0).cpu().detach().item() for j in range(rs[i].size()[0])]) # cosines of block i
			stats['norms'][i].append(n)
			stats['cosines'][i].append(c)
			for j in range(pred.size()[0]):
				stats['norms class'][pred[j].item()][i].append(n[j: j + 1])
				stats['cosines class'][pred[j].item()][i].append(c[j: j + 1])

	def add_batch_maha(model, x, pred, n_classes, sample_mean, precision, n_stages, stats):
		for mag in magnitudes:
			for layer in range(n_stages):
				ngs = np.asarray(get_Mahalanobis_score_adv(model, x, n_classes, sample_mean, precision, layer, float(mag)), dtype = np.float32)
				M = ngs.reshape((ngs.shape[0], -1)) if layer == 0 else np.concatenate((M, ngs.reshape((ngs.shape[0], -1))), axis = 1)
			stats['maha' + mag] = M if stats['maha' + mag] is None else np.concatenate((stats['maha' + mag], M.reshape((M.shape[0], -1))), axis = 0)
			for j in range(pred.size()[0]):
				stats['maha' + mag + ' class'][pred[j].item()] = M[j: j + 1] if stats['maha' + mag + ' class'][pred[j].item()] is None else np.concatenate((stats['maha' + mag + ' class'][pred[j].item()], M[j: j + 1]), axis = 0)

	def add_batch_stats(x, pred, rs, maha_params, stats):
		add_batch_norms_cosines(pred, rs, stats)
		add_batch_maha(model, x, pred, n_classes, *maha_params, stats)


	def concat_stats(n_classes, stats):
		stats['norms'] = np.transpose(np.vstack([np.concatenate(n) for n in stats['norms']]))  # for each batch sample
		transports = np.sum(stats['norms'], axis = 1) # for each batch sample
		stats['cosines'] = np.transpose(np.vstack([np.concatenate(c) for c in stats['cosines']]))# for each batch sample
		n_not_seen = 0
		for j in range(n_classes):
			not_seen = len(stats['norms class'][j][0]) == 0 
			n_not_seen += not_seen
			stats['norms class'][j] = np.transpose(np.vstack([np.concatenate(n) for n in stats['norms class'][j]])) if not not_seen else None # for each batch sample
			stats['cosines class'][j] = np.transpose(np.vstack([np.concatenate(c) for c in stats['cosines class'][j]])) if not not_seen else None # for each batch sample
		print(n_not_seen, 'classes were not seen')
		stats['transports'] = transports

	t0 = time.time()

	ind_stats, ood_stats = create_stats_dict(), create_stats_dict()

	for (ind_x, ind_y), (ood_x, ood_y) in zip(ind_loader, ood_loader):
		ind_x, ind_y, ood_x, ood_y = ind_x.to(device).float(), ind_y.to(device), ood_x.to(device).float(), ood_y.to(device)
		ind_correct, ind_pred, ind_rs = get_model_pred_and_residus(ind_x, ind_y)
		ood_correct, ood_pred, ood_rs = get_model_pred_and_residus(ood_x, ood_y)
		add_batch_stats(ind_x, ind_pred, ind_rs, maha_params, ind_stats)
		add_batch_stats(ood_x, ood_pred, ood_rs, maha_params, ood_stats)
	
	concat_stats(n_classes, ind_stats)
	concat_stats(n_classes, ood_stats)
	print('Val stats extraction time', time.time() - t0, 'seconds')
	return ind_stats, ood_stats




def create_transport_detectors(ind_norms, ind_cosines, ood_norms, ood_cosines, timeit = False):
	detectors = {}
	X_norms = np.concatenate((ind_norms, ood_norms))
	X_cosines = np.concatenate((ind_cosines, ood_cosines))
	X_norms_and_cosines = np.hstack((X_norms, X_cosines))
	Y = np.concatenate((np.full((ind_norms.shape[0], ), 0), np.full((ood_norms.shape[0], ), 1)))
	detectors['norms'] = get_trained_detector('RF norms detector', X_norms, Y, timeit)
	detectors['cosines'] = get_trained_detector('RF cosines detector', X_cosines, Y, timeit)
	detectors['norms cosines'] = get_trained_detector('RF norms cosines detector', X_norms_and_cosines, Y, timeit)
	detectors['norms cosines ens'] = EnsembleDetector('RF norms cosines ens detector', [detectors['norms'], detectors['cosines'], detectors['norms cosines']])
	detectors['norms cosines vote'] = EnsembleVoteDetector('RF norms cosines vote detector', [detectors['norms'], detectors['cosines'], detectors['norms cosines']])
	return detectors

def create_mahalanobis_detectors_(ind_maha, ood_maha, magnitude = '0', timeit = False):
	detectors = {}
	X_maha = np.concatenate((ind_maha, ood_maha))
	Y = np.concatenate((np.full((ind_maha.shape[0], ), 0), np.full((ood_maha.shape[0], ), 1)))
	#detectors['maha' + magnitude + ' lr'] = get_trained_detector('Mahalanobis' + magnitude + ' LR detector ' + s, X_maha, Y, magnitude)
	detectors['maha' + magnitude + ' rf'] = get_trained_detector('Mahalanobis' + magnitude + ' RF detector', X_maha, Y, timeit, magnitude)
	return detectors

def create_mahalanobis_detectors(ind_stats, ood_stats, timeit = False):
	detectors = {}
	for magnitude in magnitudes:
		detectors.update(create_mahalanobis_detectors_(ind_stats['maha' + magnitude], ood_stats['maha' + magnitude], magnitude, timeit))
	#detectors['maha ens'] = EnsembleDetector('Maha ens detector ' + s, [detectors['maha' + magnitude + ' rf'] for magnitude in magnitudes])
	#detectors['maha vote'] = EnsembleVoteDetector('Maha vote detector ' + s, [detectors['maha' + magnitude + ' rf'] for magnitude in magnitudes])
	return detectors

def create_detectors(ind_stats, ood_stats, n_classes, cutoff):
	detectors = {}
	transport_detectors = create_transport_detectors(ind_stats['norms'], ind_stats['cosines'], ood_stats['norms'], ood_stats['cosines'], 1)
	mahalanobis_detectors = create_mahalanobis_detectors(ind_stats, ood_stats, 1)
	detectors = {**transport_detectors, **mahalanobis_detectors}
	return list(detectors.values())


def get_best_detectors(detectors):
	best_maha_detector_acc, best_tra_detector_acc = 0, 0
	for detector in detectors:
		detector.stats_()
		if 'Maha' in detector.name:
			if detector.acc > best_maha_detector_acc:
				best_maha_detector, best_maha_detector_acc = detector, detector.acc
		else:
			if detector.acc > best_tra_detector_acc:
				best_tra_detector, best_tra_detector_acc = detector, detector.acc
	return best_maha_detector, best_tra_detector



def train_detectors(model, ind_loader, ood_loader, n_classes, n_res, maha_params, cutoff):
	t0 = time.time()
	ind_stats, ood_stats = get_val_stats(model, ind_loader, ood_loader, n_classes, n_res, maha_params)
	detectors = create_detectors(ind_stats, ood_stats, n_classes, cutoff)
	print('Val stats and detector training took', time.time() - t0, 'seconds')
	return detectors

def test_detectors(detectors, model, ind_loader, ood_loader, n_classes, maha_params):
	t0 = time.time()
	for detector in detectors:
		detector.reset() 
	for (ind_x, ind_y), (ood_x, ood_y) in zip(ind_loader, ood_loader):
		ind_x, ind_y, ood_x, ood_y = ind_x.to(device).float(), ind_y.to(device), ood_x.to(device).float(), ood_y.to(device)
		ind_pred, ind_correct, ind_rs, ind_transport, ind_norms, ind_cosines, ind_m = get_model_pred_and_stats(model, ind_x, ind_y, n_classes, maha_params)
		ood_pred, ood_correct, ood_rs, ood_transport, ood_norms, ood_cosines, ood_m = get_model_pred_and_stats(model, ood_x, ood_y, n_classes, maha_params)
		for detector in detectors:
			ind_detected = detector.detect(ind_transport, ind_norms, ind_cosines, ind_pred, ind_m[detector.maha_mag])
			ood_detected = detector.detect(ood_transport, ood_norms, ood_cosines, ood_pred, ood_m[detector.maha_mag])
			detector.update_counters(ind_correct, ind_detected, ood_correct, ood_detected, None)
	print('Testing took', time.time() - t0, 'seconds')
	best_detectors = get_best_detectors(detectors)
	for detector in best_detectors:
		print('\n------', detector.name.ljust(70) , ' '.join(stat_name + ' ' + str(round(stat, 3)) for stat_name, stat in detector.stats().items() if stat != -1))



	
def experiment(ind_dataset, ood1_dataset, ood2_dataset, modelname, traintype, cutoff, batchsize, ind_valsize, ind_testsize, ood1_valsize, ood1_testsize, ood2_testsize, seed):
	
	t0 = time.time()
	frame = inspect.currentframe()
	names, _, _, values = inspect.getargvalues(frame)
	print('Experiment from ood2 with parameters:')
	for name in names:
		print('%s = %s' % (name, values[name]))
	if seed is not None:
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
		torch.manual_seed(seed)
		np.random.seed(seed)

	print('\n------ Detection of ood samples')

	ind_trainloader, ind_valloader, ind_testloader, ood1_valloader, ood1_testloader, ood2_testloader, datashape, n_classes = get_loaders(ind_dataset, ood1_dataset, ood2_dataset, batchsize, 
																																		 ind_valsize, ind_testsize, ood1_valsize, ood1_testsize, ood2_testsize)
	folder = modelname + '-' + ind_dataset + '-' + traintype
	model = get_model(datashape, modelname, n_classes, 1, folder)
	n_res = len(model(next(iter(ind_trainloader))[0].to(device))[1])
	maha_params  = get_maha_params(model, datashape, n_classes, ind_trainloader)
	print('\n------ Training detectors on IND', ind_dataset, 'OOD', ood1_dataset)
	detectors = train_detectors(model, ind_valloader, ood1_valloader, n_classes, n_res, maha_params, cutoff)
	print('\n------ Detection of seen', ood1_dataset, 'ood samples')
	test_detectors(detectors, model, ind_testloader, ood1_testloader, n_classes, maha_params)
	print('\n------ Detection of unseen', ood2_dataset, 'ood samples')
	test_detectors(detectors, model, ind_testloader, ood2_testloader, n_classes, maha_params)
	

	print('\nTotal time', time.time() - t0, 'seconds')


		
		
	

	
if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("-ind", "--inddataset", required = True, choices = ['cifar10', 'cifar100', 'svhn'], nargs = '*')
	parser.add_argument("-ood1", "--ood1dataset", required = True, choices = ['cifar10', 'cifar100', 'svhn'], nargs = '*')
	parser.add_argument("-ood2", "--ood2dataset", required = True, choices = ['cifar10', 'cifar100', 'svhn'], nargs = '*')
	parser.add_argument("-mod", "--modelname", required = True, choices = ['resnext29', 'resnext50', 'onerep', 'resnet110', 'avgpool', 'wide', 'efficientnet'], nargs = '*')
	parser.add_argument("-trt", "--traintype", required = True, choices = ['van', 'rce', 'lap'], nargs = '*')
	parser.add_argument("-cut", "--cutoff", type = int, default = [1], nargs = '*')
	parser.add_argument("-bas", "--batchsize", type = int, default = [128], nargs = '*')
	parser.add_argument("-ivs", "--indvalsize", type = float, default = [0.9], nargs = '*')
	parser.add_argument("-its", "--indtestsize", type = float, default = [0.1], nargs = '*')
	parser.add_argument("-o1vs", "--ood1valsize", type = float, default = [0.9], nargs = '*')
	parser.add_argument("-o1ts", "--ood1testsize", type = float, default = [0.1], nargs = '*')
	parser.add_argument("-o2ts", "--ood2testsize", type = float, default = [0.1], nargs = '*')
	parser.add_argument("-see", "--seed", type = int, default = [None], nargs = '*')
	args = parser.parse_args()

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	magnitudes = ['0', '0.01', '0.001', '0.0014', '0.002', '0.005', '0.0005']
	rce = args.traintype == 'rce'
	parameters = [values[0] for name, values in vars(args).items()]
	experiment(*parameters)
