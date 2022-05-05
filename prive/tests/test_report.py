import os
import unittest
import numpy as np
from prive.report import MIAttackSummary


class MiAttackSummaryTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.predictions = (np.random.randint(2, size=100))
        self.labels = (np.random.randint(2, size=100))
        self.attack_summary = MIAttackSummary(self.labels, self.predictions, 'Random', 'Groundhog', 1)

    def test_accuracy(self):
        self.assertEqual(self.attack_summary.accuracy, np.mean(self.predictions == self.labels))

    def test_fp(self):
        self.assertEqual(self.attack_summary.fp, np.sum(self.predictions[np.where(self.labels == 0)[0]] == 1) / len(
            np.where(self.labels == 0)[0]))

    def test_tp(self):
        self.assertEqual(self.attack_summary.tp, np.sum(self.predictions[np.where(self.labels == 1)[0]] == 1) / len(
            np.where(self.labels == 1)[0]))

    def test_mia_advantage(self):
        self.assertEqual(round(self.attack_summary.mia_advantage - 0, 1), 0)

    def test_privacy_gain(self):
        self.assertEqual(round(self.attack_summary.privacy_gain - 1, 1), 0)

    def test_get_metrics(self):
        df = self.attack_summary.get_metrics()
        self.assertFalse(df.empty)

    def test_write_metrics(self):
        self.attack_summary.write_metrics(os.path.dirname(__file__), 10)

        file_name = f'result_Groundhog_Random_target1_10.csv'

        self.assertTrue(os.path.exists(file_name))
