from utils6 import *
import numpy as np, re
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier

class Detector :
	def __init__(self, name, input_size, maha_mag = None):
		if 'Maha' in name and maha_mag is None:
			raise ValueError('Maha detector requires magnitude')
		self.name = name
		self.input_size = input_size
		self.maha_mag = maha_mag
		self.type = 'Maha' if 'Maha' in name else ('LID' if 'LID' in name else ('transport' if 'transport' in name else ('norms and cosines' if 'norms cosines' in name else ('norms' if 'norms' in name else 'cosines'))))
		self.reset()
	def reset(self):
		self.n_clean_inputs, self.n_clean_detected, self.clean_detect_rate = 0, 0, 0
		self.n_adver_inputs, self.n_adver_correct, self.n_adver_detected, self.n_adver_wrong_and_detected, self.n_adver_correct_or_detected = 0, 0, 0, 0, 0
		self.adver_detect_rate, self.detect_acc, self.adver_correct_or_detected_rate, self.successful_adver_detect_rate = 0, 0, 0, 0
		self.n_clean_correct_adver_wrong, self.n_clean_correct_adver_wrong_adver_detected, self.successful_adver_on_correct_clean_detect_rate = 0, 0, 0
		self.n_clean_and_noisy_correct_adver_wrong, self.n_clean_and_noisy_correct_adver_wrong_and_detected = 0, 0
	def update_counters(self, clean_correct, clean_detected, adver_correct, adver_detected, noisy_correct = None):
		self.n_clean_inputs += 1
		self.n_clean_detected += clean_detected
		self.n_adver_inputs += 1
		self.n_adver_correct += adver_correct
		self.n_adver_detected += adver_detected
		self.n_adver_wrong_and_detected += not adver_correct and adver_detected
		self.n_adver_correct_or_detected += adver_correct or adver_detected
		self.n_clean_correct_adver_wrong += clean_correct and not adver_correct
		self.n_clean_correct_adver_wrong_adver_detected += clean_correct and not adver_correct and adver_detected
		if noisy_correct is not None:
			self.n_clean_and_noisy_correct_adver_wrong += clean_correct and noisy_correct and not adver_correct
			self.n_clean_and_noisy_correct_adver_wrong_and_detected += clean_correct and noisy_correct and adver_detected and not adver_correct
	def stats_(self): 
		self.FP = self.n_clean_detected / self.n_clean_inputs
		self.TP = self.n_adver_detected / self.n_adver_inputs
		self.acc = (self.n_adver_detected + self.n_clean_inputs - self.n_clean_detected) / (self.n_adver_inputs + self.n_clean_inputs)
		self.adver_correct_or_detected_rate = self.n_adver_correct_or_detected / self.n_adver_inputs
		self.success_adver_rate = self.n_adver_wrong_and_detected / (self.n_adver_inputs - self.n_adver_correct) if self.n_adver_inputs - self.n_adver_correct > 0 else -1
		self.success_adver_correct_clean_rate = self.n_clean_correct_adver_wrong_adver_detected / self.n_clean_correct_adver_wrong if self.n_clean_correct_adver_wrong > 0 else -1
		self.success_adver_correct_clean_noisy_rate = self.n_clean_and_noisy_correct_adver_wrong_and_detected / self.n_clean_and_noisy_correct_adver_wrong if self.n_clean_and_noisy_correct_adver_wrong > 0 else -1
	def stats(self):
		self.stats_()
		return {'FP': self.FP, 'TP': self.TP, 'Acc': self.acc, 'succes attacks detection rate': self.success_adver_rate, 
				'succes attacks on clean correct detection rate': self.success_adver_correct_clean_rate, 
				'succes attacks on clean and noisy correct detection rate': self.success_adver_correct_clean_noisy_rate}



class TransportDetector(Detector):
	def __init__(self, transport_interval) :
		Detector.__init__(self, 'transport attack detector', 1)
		self.transport_interval = transport_interval
	def detect(self, transport, norms = None, cosines = None, pred = None, M = None, lid = None):
		if not (type(transport) is float or type(transport) is int):
			raise ValueError('Transport detector takes only one value (transport cost)')
		return transport < self.transport_interval[0] or transport > self.transport_interval[1]

class IntervalsDetector(Detector):
	def __init__(self, name, intervals) :
		Detector.__init__(self, name, len(intervals))
		self.intervals = intervals
	def detect(self, transport = None, norms = None, cosines = None, pred = None, M = None, lid = None):
		x = norms if self.type == 'norms' else (cosines if self.type == 'cosines' else norms + cosines)
		if len(x) != self.input_size:
			raise ValueError('Not the same number of x and intervals')
		return sum([x[i] < self.intervals[i][0] or x[i] > self.intervals[i][1] for i in range(len(x))]) > 0

class TrainedDetector(Detector):
	def __init__(self, name, trained_detector, maha_mag = None) :
		Detector.__init__(self, name, trained_detector.n_features_in_, maha_mag)
		self.trained_detector = trained_detector
	def detect(self, transport = None, norms = None, cosines = None, pred = None, M = None, lid = None):
		x = M if self.type == 'Maha' else (lid if self.type == 'LID' else (norms if self.type == 'norms' else (cosines if self.type == 'cosines' else norms + cosines)))
		if len(x) != self.input_size:
			raise ValueError('Vector of input to detector has wrong dimension, input was', x, 'expected input size')
		return self.trained_detector.predict(np.array([x]))[0]

class ClassConditionalDetector(Detector):
	def __init__(self, name, class_detectors, maha_mag = None) :
		Detector.__init__(self, name, [detector.input_size for detector in class_detectors], maha_mag)
		self.class_detectors = class_detectors
	def detect(self, transport = None, norms = None, cosines = None, pred = None, M = None, lid = None):
		return self.class_detectors[pred.item()].detect(transport, norms, cosines, pred, M, lid)

class EnsembleDetector(Detector):
	def __init__(self, name, detectors, maha_mag = None) :
		Detector.__init__(self, name, [detector.input_size for detector in detectors], maha_mag)
		self.detectors = detectors
	def detect(self, transport = None, norms = None, cosines = None, pred = None, M = None, lid = None):
		return sum([detector.detect(transport, norms, cosines, pred, M, lid) for detector in self.detectors]) > 0

class EnsembleVoteDetector(Detector):
	def __init__(self, name, detectors, maha_mag = None) :
		Detector.__init__(self, name, [detector.input_size for detector in detectors], maha_mag)
		self.detectors = detectors
	def detect(self, transport = None, norms = None, cosines = None, pred = None, M = None, lid = None):
		return sum([detector.detect(transport, norms, cosines, pred, M, lid) for detector in self.detectors]) > len(self.detectors) / 2

def get_trained_detector(name, X, Y, timeit = False, maha_mag = None):
	if 'Maha' in name and maha_mag is None:
		raise ValueError('Maha detector requires magnitude')
	t0 = time.time()
	if 'LR' in name:
		detector = TrainedDetector(name, LogisticRegressionCV().fit(X, Y), maha_mag)
	elif 'RF' in name:
		detector = TrainedDetector(name, RandomForestClassifier().fit(X, Y), maha_mag)
	if timeit:
		print('Training', name, 'took', time.time() - t0, 'seconds')
	return detector

