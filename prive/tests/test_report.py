import unittest
import numpy as np
from prive.report import MIAttackSummary


class MiAttackSummaryTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.predictions = (np.random.randint(2, size=100))
        self.labels = (np.random.randint(2, size=100))
        self.attack_summary = MIAttackSummary(self.labels, self.predictions, 'Random', 'Groundhog')

    def test_accuracy(self):
        self.assertEqual(self.attack_summary.accuracy, np.mean(self.predictions == self.labels))

    def test_fp(self):
        self.assertEqual(self.attack_summary.fp, np.sum(self.predictions[np.where(self.labels == 0)[0]] == 1)/len(np.where(self.labels == 0)[0]))

    def test_tp(self):
        self.assertEqual(self.attack_summary.tp, np.sum(self.predictions[np.where(self.labels == 1)[0]] == 1)/len(np.where(self.labels == 1)[0]))
