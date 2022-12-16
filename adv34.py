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
from sklearn import metrics


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
		return [('fgm', eps), ('pgd', eps), ('bim', eps)]
	if attack_names == 'wbs':
		return [('auto', eps), ('cw2', None), ('df', None)]
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
	else:
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

add_random_noise = lambda x, random_noise_size, min_pixel, max_pixel : torch.clamp(torch.add(x.data, torch.randn(x.size()).cuda(), alpha = random_noise_size), min_pixel, max_pixel)

def get_val_stats(model, loader, n_classes, n_res, maha_params, attack, correct_on_noisy_only):

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

	t0, n_train = time.time(), len(loader)

	clean_stats, noisy_stats, adver_stats = create_stats_dict(), create_stats_dict(), create_stats_dict()
	
	for k, (x, y) in enumerate(loader):
		
		try:
		
			clean_x, y = x.to(device).float(), y.to(device)
			noisy_x = attack.noisy(clean_x)
			adver_x = attack.adver(clean_x, y) 

			clean_correct, clean_pred, clean_rs = get_model_pred_and_residus(clean_x, y)
			noisy_correct, noisy_pred, noisy_rs = get_model_pred_and_residus(noisy_x, y)
			adver_correct, adver_pred, adver_rs = get_model_pred_and_residus(adver_x, y)

			if correct_on_noisy_only:
				index = torch.tensor([i for i in range(x.size(0)) if clean_correct[i] == noisy_correct[i] == 1 and adver_correct[i] == 0]).to(device)
				clean_x, clean_pred, clean_rs = torch.index_select(clean_x, 0, index), torch.index_select(clean_pred, 0, index), [torch.index_select(r, 0, index) for r in clean_rs]
				noisy_x, noisy_pred, noisy_rs = torch.index_select(noisy_x, 0, index), torch.index_select(noisy_pred, 0, index), [torch.index_select(r, 0, index) for r in noisy_rs]
				adver_x, adver_pred, adver_rs = torch.index_select(adver_x, 0, index), torch.index_select(adver_pred, 0, index), [torch.index_select(r, 0, index) for r in adver_rs]
				if index.size(0) == 0:
					continue

			add_batch_stats(clean_x, clean_pred, clean_rs, maha_params, clean_stats)
			add_batch_stats(noisy_x, noisy_pred, noisy_rs, maha_params, noisy_stats)
			add_batch_stats(adver_x, adver_pred, adver_rs, maha_params, adver_stats)

		except (OverflowError, ValueError, TypeError) as err :
			print('Overflow or value error or type error, skipping train batch', k)

		if (k + 1) % 20 == 0 :
			print(k + 1, 'train batches out of', n_train, 'took', time.time() - t0, 's') 
			
	concat_stats(n_classes, clean_stats)
	concat_stats(n_classes, noisy_stats)
	concat_stats(n_classes, adver_stats)
	
	print('val time', time.time() - t0, 'seconds')

	return clean_stats, noisy_stats, adver_stats




def create_transport_detectors(clean_norms, clean_cosines, adver_norms, adver_cosines, s = '', timeit = False):
	detectors = {}
	X_norms = np.concatenate((clean_norms, adver_norms))
	X_cosines = np.concatenate((clean_cosines, adver_cosines))
	X_norms_and_cosines = np.hstack((X_norms, X_cosines))
	Y = np.concatenate((np.full((clean_norms.shape[0], ), 0), np.full((adver_norms.shape[0], ), 1)))
	detectors['norms'] = get_trained_detector('RF norms detector ' + s, X_norms, Y, timeit)
	detectors['cosines'] = get_trained_detector('RF cosines detector ' + s, X_cosines, Y, timeit)
	detectors['norms cosines'] = get_trained_detector('RF norms cosines detector ' + s, X_norms_and_cosines, Y, timeit)
	detectors['norms cosines ens'] = EnsembleDetector('RF norms cosines ens detector ' + s, [detectors['norms'], detectors['cosines'], detectors['norms cosines']])
	detectors['norms cosines vote'] = EnsembleVoteDetector('RF norms cosines vote detector ' + s, [detectors['norms'], detectors['cosines'], detectors['norms cosines']])
	P = detectors['norms cosines'].trained_detector.predict_proba(X_norms_and_cosines)[:, 1]
	fpr, tpr, thresholds = metrics.roc_curve(Y, P)
	fpr_at95tpr = fpr[next(x[0] for x in enumerate(tpr) if x[1] > 0.95)]
	auroc = metrics.roc_auc_score(Y, P)
	print('TR detector FPR at 95 TPR:', fpr_at95tpr, 'AUROC:', auroc)
	return detectors

def create_mahalanobis_detectors_(clean_maha, adver_maha, magnitude = '0', s = '', timeit = False):
	detectors = {}
	X_maha = np.concatenate((clean_maha, adver_maha))
	Y = np.concatenate((np.full((clean_maha.shape[0], ), 0), np.full((adver_maha.shape[0], ), 1)))
	#detectors['maha' + magnitude + ' lr'] = get_trained_detector('Mahalanobis' + magnitude + ' LR detector ' + s, X_maha, Y, magnitude)
	detectors['maha' + magnitude + ' rf'] = get_trained_detector('Mahalanobis' + magnitude + ' RF detector ' + s, X_maha, Y, timeit, magnitude)
	if magnitude == '0':
		P = detectors['maha' + '0' + ' rf'].trained_detector.predict_proba(X_maha)[:, 1]
		fpr, tpr, thresholds = metrics.roc_curve(Y, P)
		fpr_at95tpr = fpr[next(x[0] for x in enumerate(tpr) if x[1] > 0.95)]
		auroc = metrics.roc_auc_score(Y, P)
		print('MH detector FPR at 95 TPR:', fpr_at95tpr, 'AUROC:', auroc)
	return detectors

def create_mahalanobis_detectors(clean_stats, adver_stats, s, timeit = False):
	detectors = {}
	for magnitude in magnitudes:
		detectors.update(create_mahalanobis_detectors_(clean_stats['maha' + magnitude], adver_stats['maha' + magnitude], magnitude, s, timeit))
	#detectors['maha ens'] = EnsembleDetector('Maha ens detector ' + s, [detectors['maha' + magnitude + ' rf'] for magnitude in magnitudes])
	#detectors['maha vote'] = EnsembleVoteDetector('Maha vote detector ' + s, [detectors['maha' + magnitude + ' rf'] for magnitude in magnitudes])
	return detectors

def create_detectors(clean_stats, adver_stats, n_classes, cutoff = 1, s = ''):

	detectors = {}
	transport_detectors = create_transport_detectors(clean_stats['norms'], clean_stats['cosines'], adver_stats['norms'], adver_stats['cosines'], s, 0)
	mahalanobis_detectors = create_mahalanobis_detectors(clean_stats, adver_stats, s, 0)
	detectors = {**transport_detectors, **mahalanobis_detectors}
	
	
	class_norms_cosines_detectors, class_norms_cosines_ens_detectors, class_norms_cosines_vote_detectors = [], [], []
	class_maha_detectors = {'maha' + magnitude : [] for magnitude in magnitudes}
	#class_maha_ens_detectors, class_maha_vote_detectors = [], []
	
	t0 = time.time()
	for j in range(n_classes):
		clean_norms_class, clean_cosines_class, adver_norms_class, adver_cosines_class = clean_stats['norms class'][j], clean_stats['cosines class'][j], adver_stats['norms class'][j], adver_stats['cosines class'][j]
		clean_maha_stats_class = {'maha' + magnitude : clean_stats['maha' + magnitude + ' class'][j] for magnitude in magnitudes}
		adver_maha_stats_class = {'maha' + magnitude : adver_stats['maha' + magnitude + ' class'][j] for magnitude in magnitudes}
		if clean_norms_class is not None and adver_norms_class is not None and clean_norms_class.shape[0] + adver_norms_class.shape[0] > cutoff :
			class_transport_detectors_ = create_transport_detectors(clean_norms_class, clean_cosines_class, adver_norms_class, adver_cosines_class, s + ' class ' + str(j), 0)
			class_maha_detectors_ = create_mahalanobis_detectors(clean_maha_stats_class, adver_maha_stats_class, s + ' class ' + str(j), 0)
			class_norms_cosines_detectors.append(class_transport_detectors_['norms cosines'])
			class_norms_cosines_ens_detectors.append(class_transport_detectors_['norms cosines ens'])
			class_norms_cosines_vote_detectors.append(class_transport_detectors_['norms cosines vote'])
			for magnitude in magnitudes:
				class_maha_detectors['maha' + magnitude].append(class_maha_detectors_['maha' + magnitude + ' rf'])
			#class_maha_ens_detectors.append(class_maha_detectors_['maha ens'])
			#class_maha_vote_detectors.append(class_maha_detectors_['maha vote'])
		else:
			class_norms_cosines_detectors.append(detectors['norms cosines'])
			class_norms_cosines_ens_detectors.append(detectors['norms cosines ens'])
			class_norms_cosines_vote_detectors.append(detectors['norms cosines vote'])
			for magnitude in magnitudes:
				class_maha_detectors['maha' + magnitude].append(detectors['maha' + magnitude + ' rf'])
			#class_maha_ens_detectors.append(detectors['maha ens'])
			#class_maha_vote_detectors.append(detectors['maha vote'])
	print('Training class cond detectors took' , time.time() - t0, 'seconds')


	for mag in magnitudes:
		detectors['class cond maha' + mag] = ClassConditionalDetector('Mahalanobis' + mag + ' class cond detector ' + s, class_maha_detectors['maha' + mag], mag)
		detectors['maha' + mag + ' - class cond ens'] = EnsembleDetector('Mahalanobis' + mag + ' - class cond ens detector ' + s, [detectors['maha' + mag + ' rf'], detectors['class cond maha' + mag]], mag)
	detectors['class cond norms cosines'] =  ClassConditionalDetector('RF class cond norms cosines detector ' + s, class_norms_cosines_detectors)
	detectors['class cond norms cosines ens'] = ClassConditionalDetector('RF class cond norms cosines ens detector ' + s, class_norms_cosines_ens_detectors)
	detectors['class cond norms cosines vote'] = ClassConditionalDetector('RF class cond norms cosines vote detector ' + s, class_norms_cosines_vote_detectors)
	detectors['norms cosines - class cond ens'] = EnsembleDetector('RF norms cosines - class cond ens detector ' + s, [detectors['norms cosines'], detectors['class cond norms cosines']])
	detectors['norms cosines ens - class cond ens ens'] = EnsembleDetector('RF norms cosines ens - class cond ens ens detector ' + s, [detectors['norms cosines ens'], detectors['class cond norms cosines ens']])
	detectors['norms cosines vote - class cond vote ens'] = EnsembleDetector('RF norms cosines vote - class cond vote ens detector ' + s, [detectors['norms cosines vote'], detectors['class cond norms cosines vote']])
	
	return list(detectors.values())



def merge_clean_and_noisy_stats(clean_stats, noisy_stats, n_classes):
	clean_and_noisy_transports_stats = {'norms' : np.concatenate((clean_stats['norms'], noisy_stats['norms'])), 'cosines' : np.concatenate((clean_stats['cosines'], noisy_stats['cosines'])),
										'norms class' : [concat(clean_stats['norms class'][j], noisy_stats['norms class'][j]) for j in range(n_classes)], 
									    'cosines class' : [concat(clean_stats['cosines class'][j], noisy_stats['cosines class'][j]) for j in range(n_classes)]
				 	             		}
	clean_and_noisy_maha_stats = {'maha' + magnitude : np.concatenate((clean_stats['maha' + magnitude], noisy_stats['maha' + magnitude])) for magnitude in magnitudes}
	clean_and_noisy_maha_class_stats = {'maha' + mag + ' class' : [concat(clean_stats['maha' + mag + ' class'][j], noisy_stats['maha' + mag + ' class'][j]) for j in range(n_classes)] for mag in magnitudes}
	return {**clean_and_noisy_transports_stats , **clean_and_noisy_maha_stats, **clean_and_noisy_maha_class_stats}
	

def train_detectors(model, attack, loader, n_classes, n_res, maha_params, correct_on_noisy_only, cutoff):
	print('\n------ Training detectors on', attack)
	t0 = time.time()
	clean_stats, noisy_stats, adver_stats = get_val_stats(model, loader, n_classes, n_res, maha_params, attack, correct_on_noisy_only)
	clean_and_noisy_stats = merge_clean_and_noisy_stats(clean_stats, noisy_stats, n_classes)
	detectors_no_noise = create_detectors(clean_stats, adver_stats, n_classes, cutoff, 'no noise')
	detectors_with_noise = create_detectors(clean_and_noisy_stats, adver_stats, n_classes, cutoff, 'with noise')
	detectors = detectors_no_noise + detectors_with_noise
	print('Generating train attacks and training detectors took', time.time() - t0, 'seconds')
	return detectors

def test_detectors(model, attack, loader, n_classes, detectors, maha_params, correct_on_noisy_only):
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
			clean_pred, clean_correct, clean_rs, clean_transport, clean_norms, clean_cosines, clean_m = get_model_pred_and_stats(model, clean_x, y, n_classes, maha_params)
			noisy_pred, noisy_correct, noisy_rs, noisy_transport, noisy_norms, noisy_cosines, noisy_m = get_model_pred_and_stats(model, noisy_x, y, n_classes, maha_params)
			adver_pred, adver_correct, adver_rs, adver_transport, adver_norms, adver_cosines, adver_m = get_model_pred_and_stats(model, adver_x, y, n_classes, maha_params)
			n_clean_correct += clean_correct
			n_noisy_correct += noisy_correct
			n_adver_correct += adver_correct
			for detector in detectors:
				clean_detected = detector.detect(clean_transport, clean_norms, clean_cosines, clean_pred, clean_m[detector.maha_mag])
				adver_detected = detector.detect(adver_transport, adver_norms, adver_cosines, adver_pred, adver_m[detector.maha_mag])
				detector.update_counters(clean_correct, clean_detected, adver_correct, adver_detected, noisy_correct if correct_on_noisy_only else None)
		except (OverflowError, ValueError, TypeError) as err :
			print('Overflow or value error or type error, skipping test image', k)
		if (k + 1) % 100 == 0 :
			print(k + 1, 'test images out of', n_test, 'took', time.time() - t0, 's') 
	print('Accuracy on clean images', n_clean_correct / n_test, 'Accuracy on attacked images', n_adver_correct / n_test, 'Testing took', time.time() - t0, 'seconds')

	for detector in detectors:
		print('\n------', detector.name.ljust(70) , ' '.join(stat_name + ' ' + str(round(stat, 3)) for stat_name, stat in detector.stats().items() if stat != -1))
	
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
	maha_params  = get_maha_params(model, datashape, n_classes, trainloader)
	attacks, train_attack = get_attacks(attack_names, epsilon, mode_, classifier, batchsize, dataset, setting, train_attack_name, train_attack_epsilon)

	if setting == 'seen':
		for attack in attacks:
			detectors = train_detectors(model, attack, valloader, n_classes, n_res, maha_params, correct_on_noisy_only, cutoff)
			test_detectors(model, attack, testloader, n_classes, detectors, maha_params, correct_on_noisy_only)
	elif setting == 'unseen':
		detectors = train_detectors(model, train_attack, valloader, n_classes, n_res, maha_params, correct_on_noisy_only, cutoff)
		for attack in attacks:
			test_detectors(model, attack, testloader, n_classes, detectors, maha_params, correct_on_noisy_only)
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
