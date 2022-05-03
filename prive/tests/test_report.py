import unittest
import numpy as np
from prive.report import MiAttackSummary


class MiAttackSummaryTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.predictions = list(np.random.randint(2, size=100))
        self.labels = list(np.random.randint(2, size=100))
        self.attack_summary = MiAttackSummary(self.labels, self.predictions, 'Random', 'Groundhog')

    def test_accuracy(self):
        self.assertEqual(self.attack_summary.accuracy, 0.49)
