import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as functional, torch.utils.data as torchdata
from dataloaders8 import dataloaders
from utils6 import *
from mahalanobis_b import *
from detector_class_b import *
from models_b import *
from torchsummary import summary
import time, math, numpy as np, argparse, os, collections, sys, inspect, pprint, scipy.stats as st
from functools import partial
from art.attacks.evasion import FastGradientMethod, AutoProjectedGradientDescent, BasicIterativeMethod, CarliniL2Method, DeepFool, BoundaryAttack, HopSkipJump, AutoAttack
from art.estimators.classification import PyTorchClassifier, ClassifierMixin, SklearnClassifier, BlackBoxClassifier
from collections import OrderedDict
import torchmin

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

def get_Mahalanobis_score_adv2(model, x, n_classes, sample_mean, precision, layer, magnitude):
    data = Variable(x, requires_grad = True)
    out_features = model.intermediate_forward(data, layer)
    out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
    out_features = torch.mean(out_features, 2)
    gaussian_score = 0
    for i in range(n_classes):
        batch_sample_mean = sample_mean[layer][i]
        zero_f = out_features.data - batch_sample_mean
        term_gau = - 0.5 * torch.mm(torch.mm(zero_f, precision[layer]), zero_f.t()).diag()
        if i == 0:
            gaussian_score = term_gau.view(-1,1)
        else:
            gaussian_score = torch.cat((gaussian_score, term_gau.view(-1, 1)), 1)
    sample_pred = gaussian_score.max(1)[1]
    batch_sample_mean = sample_mean[layer].index_select(0, sample_pred)
    zero_f = out_features - Variable(batch_sample_mean)
    pure_gau = - 0.5 * torch.mm(torch.mm(zero_f, Variable(precision[layer])), zero_f.t()).diag()
    loss = torch.mean(-pure_gau)
    loss.backward()
    gradient =  torch.ge(data.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2
    gradient.index_copy_(1, torch.LongTensor([0]), gradient.index_select(1, torch.LongTensor([0])) / (0.2023))
    gradient.index_copy_(1, torch.LongTensor([1]), gradient.index_select(1, torch.LongTensor([1])) / (0.1994))
    gradient.index_copy_(1, torch.LongTensor([2]), gradient.index_select(1, torch.LongTensor([2])) / (0.2010))
    tempInputs = torch.add(data.data, gradient, alpha = - magnitude)
    with torch.no_grad():
        noise_out_features = model.intermediate_forward(Variable(tempInputs), layer)
        noise_out_features = noise_out_features.view(noise_out_features.size(0), noise_out_features.size(1), -1)
        noise_out_features = torch.mean(noise_out_features, 2)
        noise_gaussian_score = 0
    for i in range(n_classes):
        batch_sample_mean = sample_mean[layer][i]
        zero_f = noise_out_features.data - batch_sample_mean
        term_gau = - 0.5 * torch.mm(torch.mm(zero_f, precision[layer]), zero_f.t()).diag()
        if i == 0:
            noise_gaussian_score = term_gau.view(-1, 1)
        else:
            noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1, 1)), 1)      
    noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim = 1)
    return noise_gaussian_score.cpu().numpy()


def get_attack_names_and_params(attack_names, setting):
	if attack_names == 'fgm':
		if setting == 'seen':
			return [('fgm', 0.03)]
		raise NotImplementedError('Cannnot test on fgm attack for generalization to unseen attack')
	if attack_names == 'pgd':
		return [('pgd', 0.03)]
	if attack_names == 'bim':
		return [('bim', 0.03)]
	if attack_names == 'df':
		return [('df', None)]
	if attack_names == 'cw2':
		return [('cw2', None)]
	if attack_names == 'auto':
		return [('auto', 0.03)]
	if attack_names == 'hsj':
		return [('hsj', None)]
	if attack_names == 'ba':
		return [('ba', None)]
	if attack_names == 'wb':
		a = [('pgd', 0.03), ('bim', 0.03), ('cw2', None), ('df', None)]
		if setting == 'seen':
			a = [('fgm', 0.03)] + a
		return a
	if attack_names == 'bb':
		return [('hsj', None), ('ba', None)]
	if attack_names == 'wbf':
		a = [('pgd', 0.03), ('bim', 0.03)]
		if setting == 'seen':
			a = [('fgm', 0.03)] + a
		return a
	if attack_names == 'wbs':
		return [('df', None), ('cw2', None)]
	if attack_names == 'all':
		a = [('pgd', 0.03), ('bim', 0.03), ('auto', 0.03), ('cw2', None), ('df', None), ('hsj', None), ('ba', None)]
		if setting == 'seen':
			a = [('fgm', 0.03)] + a
		return a
	else:
		raise NotImplementedError()


def get_attack(attack_name, classifier, batchsize, eps = None):
	if attack_name == 'pgd':
		return AutoProjectedGradientDescent(estimator = classifier, eps = eps, batch_size = batchsize, verbose = False)
	if attack_name == 'fgm':
		return FastGradientMethod(estimator = classifier, batch_size = batchsize, eps = eps)
	if attack_name == 'cw2':
		return CarliniL2Method(classifier = classifier, batch_size = batchsize, verbose = False)
	if attack_name == 'df':
		return DeepFool(classifier = classifier, batch_size = batchsize, verbose = False)
	if attack_name == 'bim':
		return BasicIterativeMethod(estimator = classifier, eps = eps, batch_size = batchsize, verbose = False)
	if attack_name == 'hsj':
		return HopSkipJump(classifier = classifier, batch_size = batchsize, verbose = False)
	if attack_name == 'ba':
		return BoundaryAttack(estimator = classifier, targeted = False, batch_size = batchsize, verbose = False)
	if attack_name == 'auto':
		return AutoAttack(estimator = classifier, eps = eps, batch_size = batchsize, targeted = False)
	if attack_name == None:
		return None
	else:
		raise NotImplementedError()

def get_adver_sample(x, attack):
	if attack == None:
		return x
	else:
		return torch.from_numpy(attack.generate(x = x.detach().cpu().numpy())).to(device, dtype = torch.float)

def get_adver_sample2(x, attack):
	if attack == None:
		return x
	else:
		return torch.from_numpy(attack.generate(x = x.detach().cpu().numpy())).cpu()
	
	
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

def get_model_pred_and_stats2(model, x, y, nclasses = None, maha_params = None):
	sample_mean, precision, num_output = maha_params if maha_params is not None else (None, None, None)
	sample_mean = [i.cpu() for i in sample_mean]
	precision = [i.cpu() for i in precision]
	out, rs  = model(x)
	_, pred = torch.max(- out.data, 1) if rce else torch.max(out.data, 1)
	correct = (pred == y).item() == 1.0
	norms = [torch.mean(r ** 2, (1,2,3)).cpu().detach().item() for r in rs]
	cosines = [functional.cosine_similarity(torch.ones(r.size()[1:]).flatten(), r[j,:,:,:].flatten(), dim = 0).cpu().detach().item() for r in rs for j in range(r.size()[0])]
	transport = sum(norms)
	M = {None: None}
	for magnitude in magnitudes:
		for layer in range(num_output):
			ngs = get_Mahalanobis_score_adv2(model, x, nclasses, sample_mean, precision, layer, float(magnitude)).tolist()
			M[magnitude] = ngs if layer == 0 else M[magnitude] + ngs
	return pred, correct, rs, transport, norms, cosines, M

add_random_noise = lambda x, random_noise_size, min_pixel, max_pixel : torch.clamp(torch.add(x.data, torch.randn(x.size()).cuda(), alpha = random_noise_size), min_pixel, max_pixel)

def get_val_stats(model, loader, n_classes, n_res, maha_params, attack, noise_params = None, correct_on_noisy_only = False):

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

	clean_stats, noisy_stats, adver_stats = create_stats_dict(), create_stats_dict(), create_stats_dict()
	
	for k, (x, y) in enumerate(loader):
		
		try:
		
			clean_x, y = x.to(device).float(), y.to(device)
			noisy_x = add_random_noise(clean_x, *noise_params)
			adver_x = get_adver_sample(clean_x, attack) 

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
			
	concat_stats(n_classes, clean_stats)
	concat_stats(n_classes, noisy_stats)
	concat_stats(n_classes, adver_stats)
	
	print('val time', time.time() - t0, 'seconds', flush = True)

	return clean_stats, noisy_stats, adver_stats




def create_transport_detectors(clean_norms, clean_cosines, adver_norms, adver_cosines, s = '', timeit = False):
	detectors = {}
	X_norms = np.concatenate((clean_norms, adver_norms))
	X_cosines = np.concatenate((clean_cosines, adver_cosines))
	X_norms_and_cosines = np.hstack((X_norms, X_cosines))
	Y = np.concatenate((np.full((clean_norms.shape[0], ), 0), np.full((adver_norms.shape[0], ), 1)))
	#detectors['norms'] = get_trained_detector('RF norms detector ' + s, X_norms, Y, timeit)
	#detectors['cosines'] = get_trained_detector('RF cosines detector ' + s, X_cosines, Y, timeit)
	detectors['norms cosines'] = get_trained_detector('RF norms cosines detector ' + s, X_norms_and_cosines, Y, timeit)
	#detectors['norms cosines ens'] = EnsembleDetector('RF norms cosines ens detector ' + s, [detectors['norms'], detectors['cosines'], detectors['norms cosines']])
	#detectors['norms cosines vote'] = EnsembleVoteDetector('RF norms cosines vote detector ' + s, [detectors['norms'], detectors['cosines'], detectors['norms cosines']])
	return detectors

def create_mahalanobis_detectors_(clean_maha, adver_maha, magnitude = '0', s = '', timeit = False):
	detectors = {}
	X_maha = np.concatenate((clean_maha, adver_maha))
	Y = np.concatenate((np.full((clean_maha.shape[0], ), 0), np.full((adver_maha.shape[0], ), 1)))
	#detectors['maha' + magnitude + ' lr'] = get_trained_detector('Mahalanobis' + magnitude + ' LR detector ' + s, X_maha, Y, magnitude)
	detectors['maha' + magnitude + ' rf'] = get_trained_detector('Mahalanobis' + magnitude + ' RF detector ' + s, X_maha, Y, timeit, magnitude)
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
	transport_detectors = create_transport_detectors(clean_stats['norms'], clean_stats['cosines'], adver_stats['norms'], adver_stats['cosines'], s, 1)
	mahalanobis_detectors = create_mahalanobis_detectors(clean_stats, adver_stats, s, 1)
	detectors = {**transport_detectors, **mahalanobis_detectors}
	"""
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
	"""
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

def merge_clean_and_noisy_stats(clean_stats, noisy_stats, n_classes):
	clean_and_noisy_transports_stats = {'norms' : np.concatenate((clean_stats['norms'], noisy_stats['norms'])), 'cosines' : np.concatenate((clean_stats['cosines'], noisy_stats['cosines'])),
										'norms class' : [concat(clean_stats['norms class'][j], noisy_stats['norms class'][j]) for j in range(n_classes)], 
									    'cosines class' : [concat(clean_stats['cosines class'][j], noisy_stats['cosines class'][j]) for j in range(n_classes)]
				 	             		}
	clean_and_noisy_maha_stats = {'maha' + magnitude : np.concatenate((clean_stats['maha' + magnitude], noisy_stats['maha' + magnitude])) for magnitude in magnitudes}
	clean_and_noisy_maha_class_stats = {'maha' + mag + ' class' : [concat(clean_stats['maha' + mag + ' class'][j], noisy_stats['maha' + mag + ' class'][j]) for j in range(n_classes)] for mag in magnitudes}
	return {**clean_and_noisy_transports_stats , **clean_and_noisy_maha_stats, **clean_and_noisy_maha_class_stats}
	

def train_detectors(attack_name, batchsize, eps, dataset, classifier, model, valloader, n_classes, n_res, maha_params, correct_on_noisy_only, cutoff):
	print('\n------ Training detectors on', attack_name, eps if eps is not None else '', flush = True)
	t0 = time.time()
	noise_params = get_random_noise_params(dataset, attack_name) 
	attack = get_attack(attack_name, classifier, batchsize, eps)
	clean_stats, noisy_stats, adver_stats = get_val_stats(model, valloader, n_classes, n_res, maha_params, attack, noise_params, correct_on_noisy_only)
	clean_and_noisy_stats = merge_clean_and_noisy_stats(clean_stats, noisy_stats, n_classes)
	detectors_no_noise = create_detectors(clean_stats, adver_stats, n_classes, cutoff, 'no noise')
	detectors_with_noise = create_detectors(clean_and_noisy_stats, adver_stats, n_classes, cutoff, 'with noise')
	detectors = detectors_no_noise + detectors_with_noise
	print('Generating train attacks and training detectors took', time.time() - t0, 'seconds', flush = True)
	return detectors

def test_detectors(detectors, attack_name, eps, dataset, classifier, model, testloader, n_classes, maha_params, correct_on_noisy_only):
	print('\n------ Testing detectors on', attack_name, eps if eps is not None else '', flush = True)
	t0 = time.time()
	noise_params = get_random_noise_params(dataset, attack_name) 
	attack = get_attack(attack_name, classifier, 1, eps)
	n_test, n_clean_correct, n_noisy_correct, n_adver_correct = len(testloader), 0, 0, 0
	for detector in detectors:
		detector.reset() 
	for j, (x, y) in enumerate(testloader):
		try :
			x, y = x.to(device).float(), y.to(device)
			noisy_x = add_random_noise(x, *noise_params)
			adver_x = get_adver_sample(x, attack)
			clean_pred, clean_correct, clean_rs, clean_transport, clean_norms, clean_cosines, clean_m = get_model_pred_and_stats(model, x, y, n_classes, maha_params)
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
			print('Overflow or value error or type error, skipping test image', j)
	print('Accuracy on clean images', n_clean_correct / n_test, 'Accuracy on attacked images', n_adver_correct / n_test, 'Testing took', time.time() - t0, 'seconds', flush = True)
	best_maha_detector, best_tra_detector = get_best_detectors(detectors)
	for detector in (best_maha_detector, best_tra_detector):
		print('\n------', detector.name.ljust(70) , ' '.join(stat_name + ' ' + str(round(stat, 3)) for stat_name, stat in detector.stats().items() if stat != -1))
	return best_maha_detector, best_tra_detector

def adaptive2(detector, testloader, model, classifier, criterion, maha_params, n_classes, lamda1 = 10, lamda2 = 0.1, max_iter = 100, tol = 0.0001):
	t0 = time.time()
	detector.reset()
	n_images = 0
	n_successfull_simple_attacks = 0
	n_successfull_undetected_simple_attacks = 0
	n_successfull_detected_simple_attacks = 0
	n_successfull_undetected_adaptive_attacks = 0
	n_successfull_small_undetected_adaptive_attacks = 0
	sum_pert = 0
	avg_pert = 0
	attack_name = 'cw2'
	attack = get_attack(attack_name, classifier, 1)
	print('------ Initial attack', attack_name)
	detector_ = SklearnClassifier(model = detector.trained_detector)
	detector_attack = HopSkipJump(detector_, verbose = False) # white box attack for random forest
	mse = nn.MSELoss()
	#model = model.requires_grad_(False).eval()
	#model = model.cpu()
	for j, (x, y) in enumerate(testloader):
		n_images += 1
		print('--- image', j + 1, flush = True)
		x, y = x.to(device).float(), y.to(device)
		adver_x = get_adver_sample(x, attack)
		clean_pred, clean_correct, clean_rs, clean_transport, clean_norms, clean_cosines, clean_m = get_model_pred_and_stats(model, x, y, n_classes, maha_params)
		adver_pred, adver_correct, adver_rs, adver_transport, adver_norms, adver_cosines, adver_m = get_model_pred_and_stats(model, adver_x, y, n_classes, maha_params)
		if clean_correct and not adver_correct:
			n_successfull_simple_attacks += 1
			adver_detected = detector.detect(adver_transport, adver_norms, adver_cosines, adver_pred, adver_m[detector.maha_mag])
			if not adver_detected:
				print('successeful undetected classifier attack', flush = True)
				n_successfull_undetected_simple_attacks += 1
			else:
				print('successeful detected classifier attack, trying adaptive attack', flush = True)
				n_successfull_detected_simple_attacks += 1
				adver_feature_0 = np.array(adver_m[detector.maha_mag]).reshape(1, -1) if detector.type == 'Maha' else np.array(adver_norms + adver_cosines).reshape(1, -1)
				#print('here 1', flush = True)
				adver_feature = detector_attack.generate(x = adver_feature_0)
				l = int(len(adver_feature) / 2)
				adver_feature_detected = detector.detect(adver_transport, adver_feature[0, :l].tolist(), adver_feature[0, l:].tolist(), adver_pred, adver_feature.tolist())
				#print('here 2', flush = True)
				if not adver_feature_detected :
					print('features fool detector, solving inverse problem', flush = True)
					# fun = lambda x0 : - criterion(model(x0)[0], y) + lamda1 * mse(model.norms_cosines(x0), torch.from_numpy(adver_feature)[0]) + lamda2 * torch.sum(model.norms_cosines(x0)[:l] ** 2)
					fun = lambda x0 : - criterion(model(x0)[0], y) + lamda1 * mse(model.norms_cosines(x0), torch.from_numpy(adver_feature)[0].to(device)) + lamda2 * torch.linalg.vector_norm(x0 - x)
					con = lambda x0 : torch.linalg.vector_norm(x0 - x)
					# adver2_x = torchmin.minimize_constr(fun, adver_x, constr = {'fun': con, 'ub': 0.03}, bounds = {'lb': torch.zeros_like(adver_x), 'ub': torch.ones_like(adver_x)}, disp = 0, max_iter = max_iter, tol = tol).x
					adver2_x = torchmin.minimize(fun, adver_x, method = 'bfgs').x
					#print('here 3', flush = True)
					adver2_pred, adver2_correct, adver2_rs, adver2_transport, adver2_norms, adver2_cosines, adver2_m = get_model_pred_and_stats(model, adver2_x, y, n_classes, maha_params)
					adver2_detected = detector.detect(adver2_transport, adver2_norms, adver2_cosines, adver2_pred, adver2_m[detector.maha_mag])
					if not adver2_correct and not adver2_detected:
						n_successfull_undetected_adaptive_attacks += 1
						pert = torch.linalg.vector_norm(adver2_x - x).item()
						print('successeful adaptive attack, perturbation', pert, flush = True)
						sum_pert += pert
						avg_pert = sum_pert / n_successfull_undetected_adaptive_attacks
						if pert < 0.1:
							print('successeful small enough adaptive attack', flush = True)
							n_successfull_small_undetected_adaptive_attacks += 1
					else:
						print('failed adaptive attack, cannot solve inverse problem', flush = True)
				else:
					print('failed adaptive attack, features do not fool detector', flush = True)
		if (j + 1) % 5 == 0:
			m = (n_images, n_successfull_undetected_simple_attacks, n_successfull_detected_simple_attacks, n_successfull_undetected_adaptive_attacks, n_successfull_small_undetected_adaptive_attacks, avg_pert, time.time() - t0)
			#print('im', n_images, 'suc und sim', n_successfull_undetected_simple_attacks, 'suc det sim', n_successfull_detected_simple_attacks, 'suc und ada', n_successfull_undetected_adaptive_attacks, time.time() - t0, 's', flush = True)
			print('im {} suc und sim {} suc det sim {} suc und ada {} suc und small ada {} average perturbation {} time {}'.format(*m))

					 
			
fun2 = lambda x0 : - criterion(model(x_0)[0], y) - lamda * criterion(detector.trained_detector.predict_proba(x_0), detected) # x_0 to norms and cosines






def f_tra(x, model, n_classes, detector): 
	x = torch.from_numpy(x).to(device, dtype = torch.float)
	out, rs = model(x)
	_, pred = torch.max(- out.data, 1) if rce else torch.max(out.data, 1)
	n_res = len(rs)
	norms = [[] for _ in range(n_res)]
	cosines = [[] for _ in range(n_res)]
	for i, r in enumerate(rs):
		z = torch.ones(r.size()[1:]).flatten().to(device) # vector of ones
		n = torch.mean(r ** 2, (1,2,3)).cpu().detach().numpy() # norms of block i
		c = np.array([functional.cosine_similarity(z, r[j,:,:,:].flatten(), dim = 0).cpu().detach().item() for j in range(r.size()[0])]) # cosines of block i
		norms[i].append(n)
		cosines[i].append(c)
	norms = np.transpose(np.vstack([np.concatenate(n) for n in norms]))  # for each batch sample
	cosines = np.transpose(np.vstack([np.concatenate(c) for c in cosines]))# for each batch sample
	pred = pred.detach().cpu().numpy()
	one_hot = np.zeros((norms.shape[0], n_classes + 1))
	for i in range(norms.shape[0]):
		detected = detector.detect(None, norms[i].tolist(), cosines[i].tolist(), pred[i], None)
		one_hot[i, n_classes if detected else pred[i]] = 1
	return one_hot


def f_maha(x, model, n_classes, maha_params, detector): # wrap in art blackbox, target clean wrong class
	x = torch.from_numpy(x).to(device, dtype = torch.float)
	out, rs = model(x)
	_, pred = torch.max(- out.data, 1) if rce else torch.max(out.data, 1)
	n_res = len(rs)
	sample_mean, precision, n_stages = maha_params
	mag = detector.maha_mag
	for layer in range(n_stages):
		ngs = np.asarray(get_Mahalanobis_score_adv(model, x, n_classes, sample_mean, precision, layer, float(mag)), dtype = np.float32)
		M = ngs.reshape((ngs.shape[0], -1)) if layer == 0 else np.concatenate((M, ngs.reshape((ngs.shape[0], -1))), axis = 1)
	pred = pred.detach().cpu().numpy()
	one_hot = np.zeros((M.shape[0], n_classes + 1))
	for i in range(M.shape[0]):
		detected = detector.detect(None, None, None, pred[i], M[i].tolist())
		one_hot[i, n_classes if detected else pred[i]] = 1
	return one_hot





	
def experiment(dataset, modelname, traintype, setting, attack_names, cutoff, correct_on_noisy_only, batchsize, trainsize, valsize, testsize, lamda1, lamda2, max_iter, tol, seed):
	
	t0 = time.time()
	frame = inspect.currentframe()
	names, _, _, values = inspect.getargvalues(frame)
	print('Experiment from adv29adap3 with parameters:')
	for name in names:
		print('%s = %s' % (name, values[name]))
	if seed is not None:
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
		torch.manual_seed(seed)
		np.random.seed(seed)

	_, _, testloader, _, _, _, _ = dataloaders(dataset, 1, trainsize, valsize, testsize)
	trainloader, valloader, _, datashape, n_classes, _, _ = dataloaders(dataset, batchsize, trainsize, valsize, testsize)
	folder = modelname + '-' + dataset + '-' + traintype
	attacks = get_attack_names_and_params(attack_names, setting)
	model = get_model(datashape, modelname, n_classes, 1, folder)
	mode_ = get_model(datashape, modelname, n_classes, 0, folder)
	criterion, n_res, n_test = nn.CrossEntropyLoss(), len(model(next(iter(trainloader))[0].to(device))[1]), len(testloader)
	classifier = PyTorchClassifier(model = mode_, loss = criterion, input_shape = datashape[1:], nb_classes = n_classes, clip_values = (0, 1))
	maha_params  = get_maha_params(model, datashape, n_classes, trainloader)
	model.eval()
	del mode_

	if setting == 'seen':
		for attack_name, eps in attacks:

			detectors = train_detectors(attack_name, batchsize, eps, dataset, classifier, model, valloader, n_classes, n_res, maha_params, correct_on_noisy_only, cutoff)
			best_maha_detector, best_tra_detector = test_detectors(detectors, attack_name, eps, dataset, classifier, model, testloader, n_classes, maha_params, correct_on_noisy_only)
			

			print('\n------ Adaptive attack on classifier + transport detector', flush = True)
			adaptive2(best_tra_detector, testloader, model, classifier, criterion, maha_params, n_classes, lamda1, lamda2, max_iter, tol)
			
			"""
			print('\n------ Adaptive attack on classifier + mahalanobis detector', flush = True)
			adaptive2(best_maha_detector, testloader, model, classifier, criterion, maha_params, n_classes, lamda1, lamda2, max_iter, tol)
			"""
	

	print('\nTotal time', time.time() - t0, 'seconds')


		
		
	

	
if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("-dat", "--dataset", required = True, choices = ['mnist', 'cifar10', 'cifar100', 'svhn', 'imagenet2012', 'tinyimagenet', 'imagenetdownloader'], nargs = '*')
	parser.add_argument("-mod", "--modelname", required = True, choices = ['resnext29', 'resnext50', 'onerep', 'resnet110', 'avgpool', 'wide', 'efficientnet'], nargs = '*')
	parser.add_argument("-trt", "--traintype", required = True, choices = ['van', 'rce', 'lap'], nargs = '*')
	parser.add_argument("-set", "--setting", required = True, choices = ['seen', 'unseen'], nargs = '*')
	parser.add_argument("-att", "--attacknames", required = True, choices = ['fgm', 'pgd', 'bim', 'df', 'cw2', 'auto', 'hsj', 'ba', 'wb', 'bb', 'wbf', 'wbs', 'all'], nargs = '*')
	parser.add_argument("-cut", "--cutoff", type = int, default = [1], nargs = '*')
	parser.add_argument("-cno", "--correctnoisyonly", type = int, default = [0], choices = [0, 1], nargs = '*')
	parser.add_argument("-bas", "--batchsize", type = int, default = [128], nargs = '*')
	parser.add_argument("-trs", "--trainsize", type = float, default = [None], nargs = '*')
	parser.add_argument("-vls", "--valsize", type = float, default = [0.9], nargs = '*')
	parser.add_argument("-tss", "--testsize", type = float, default = [0.1], nargs = '*')
	parser.add_argument("-lm1", "--lamda1", type = float, default = [10], nargs = '*')
	parser.add_argument("-lm2", "--lamda2", type = float, default = [100], nargs = '*')
	parser.add_argument("-mit", "--maxiter", type = float, default = [100], nargs = '*')
	parser.add_argument("-tol", "--tol", type = float, default = [0.001], nargs = '*')
	parser.add_argument("-see", "--seed", type = int, default = [None], nargs = '*')
	args = parser.parse_args()

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	magnitudes = ['0', '0.01', '0.001', '0.0014', '0.002', '0.005', '0.0005']
	rce = args.traintype[0] == 'rce'
	parameters = [values[0] for name, values in vars(args).items()]
	experiment(*parameters)