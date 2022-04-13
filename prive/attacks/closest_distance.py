"""Closest-distance attacks """

import numpy as np
from sklearn.metrics import roc_curve

from .base_classes import Attack
from ..threat_models.mia import TargetedMIA


class ClosestDistanceAttack(Attack):  # or, honestly, just Attack.
	"""Attack that looks for the closest record to a given target in the
	    synthetic data to determine whether the target was in the
	    training dataset.

	   This attack predicts that a target record is in the training dataset
	    iff:  min_{x in synth_data} distance(x, target) <= threshold.
	   The threshold can either be specified, or selected automatically.
	"""

	def __init__(self, threat_model, distance_function=None, threshold=None, fpr=None, tpr=None):
		"""Create the attack with chosen parameters.

		   INPUT:
		    - threat_model: 
		    - distance_function (callable): maps (record1, record2) to a positive
		       float. If left None (default), this is self._default_distance.
		    - threshold: decision threshold for the classifier. If lest None (default),
		       the threshold is learned from training data.
		    - fpr/tpr: the target false/true positive rate for the threshold selection.

		    Exactly one of threshold, fpr or tpr must be not None.
		"""
		assert isinstance(threat_model, TargetedMIA), \
			 "Incompatible attack model: needs targeted MIA."
		MIAttack.__init__(self, threat_model)
		self.target_record = self.threat_model.target_record
		self.distance_function = distance_function or self._default_distance
		self.threshold = threshold
		self.fpr, self.tpr = fpr, tpr
		assert ((fpr is None) + (tpr is None) + (threshold is None)) == 1,\
			'Exactly one of threshold, fpr or tpr must be specified.'


	def train(self, num_samples = 100):
		"""If needed, automatically select a threshold."""
		self.threat_model = threat_model
		if not (self.threshold is None):
			return  # No training required.
		# If the threshold is not specified, train to get the desired tpr or fpr.
		synthetic_datasets, labels = self.threat_model.generate_training_samples(num_samples)
		# Compute the roc curve with - threshold, since the decision we use is
		#  score <= threshold, but roc_curve uses score >= threshold.
		fpr_all, tpr_all, thresholds = roc_curve(labels, - self.score(synthetic_datasets))
		# Select the threshold such that the fpr/tpr is closest to target.
		if fpr is not None:
			index = np.argmin(np.abs(fpr_all - fpr))
		else:
			index = np.argmin(np.abs(tpr_all - tpr))
		self.threshold = - thresholds[index]


	def attack(self, datasets):
		if self.threshold is None:
			raise Exception('Please train this attack.')
		return [score <= self.threshold for score in self.score(datasets)]


	def score(self, datasets):
		scores = []
		for ds in datasets:
			scores.append(min([self.distance_function(record, self.target_record) \
				for record in ds.iter_records()]))                                   # TODO: see how we define dataset.
		return np.array(scores)


	def _default_distance(self, x, y):
		""""""
		return 0
