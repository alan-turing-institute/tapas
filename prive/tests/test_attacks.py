"""A test for some attack classes."""

import unittest
from unittest import TestCase

import pandas as pd

import sys
sys.path.append('../..')
from prive.datasets import TabularDataset, TabularRecord
from prive.datasets.data_description import DataDescription
from prive.threat_models import TargetedAuxiliaryDataMIA

# The classes being tested.
from prive.attacks import ClosestDistanceAttack


dummy_data_description = DataDescription(
	[{'name': 'a', 'type': 'countable', 'description': 'integer'},
	 {'name': 'b', 'type': 'countable', 'description': 'integer'}])

dummy_data = pd.DataFrame([(0, 1), (0, 2), (3, 4), (3, 5)], columns=['a', 'b'])


class TestClosestDistance(TestCase):
	"""Test the closest-distance attack."""

	def setUp(self):
		self.dataset = TabularDataset(
			dummy_data,
			dummy_data_description
		)

	def _make_mia(self, a, b):
		"""Helper function to generate a MIA threat model."""
		return TargetedAuxiliaryDataMIA(
			target_record = self._make_target(a,b),
			dataset = self.dataset,
			generator = lambda x: x,  # Dummy generator
			sample_real_frac = 0.5)

	def _make_target(self, a, b):
		"""Helper function to generate a target record."""
		return TabularDataset(
			pd.DataFrame([(a,b)], columns=['a', 'b']),
			dummy_data_description)

	def test_dummy(self):
		# Take a record that is not in the dataset (distance 1/2).
		# Recall that by default, this uses the Hamming distance.
		mia = self._make_mia(0, 0)
		attack = ClosestDistanceAttack(threshold = 0.3)
		attack.train(mia)
		# Check that the training worked as intended.
		self.assertEqual(attack.trained, True)

		# Check that the score is working as intended.
		scores = attack.attack_score([rec for rec in self.dataset])
		self.assertEqual(len(scores), len(self.dataset))
		for score, distance in zip(scores, [0.5, 0.5, 1, 1]):
			self.assertEqual(score, distance)
		# Assert that the total score and decisions are ok.
		self.assertEqual(attack.attack_score([self.dataset])[0], 0.5)
		self.assertEqual(attack.attack([self.dataset])[0], False)

		# Perform the attack for a user *in* the dataset.
		attack = ClosestDistanceAttack(threshold = 0.3)
		attack.train(self._make_mia(0, 1))
		print(attack.attack_score([rec for rec in self.dataset]))
		self.assertEqual(attack.attack([self.dataset])[0], True)

	def test_training(self):
		# Check that the threshold selection works.
		pass


if __name__ == '__main__':
	unittest.main()