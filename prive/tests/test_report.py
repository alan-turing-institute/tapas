import os
import unittest
import numpy as np
from prive.report import MIAttackReport


class MIAReport(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)

        attacks = []
        for i in range(10):
            attack_dict = {}
            attack_dict['labels'] = np.random.randint(2, size=100)
            attack_dict['predictions'] = np.random.randint(2, size=100)
            attack_dict['generator_info'] = 'Random'
            attack_dict['dataset_info'] = 'Test'
            attack_dict['target_id'] = 1
            attack_dict['attack_info'] = 'RandomGuess'

            attacks.append(attack_dict)

        self.report = MIAttackReport.load_summary_statistics(attacks)

    def test_setup(self):
        self.assertTrue(self.report.attacks_data.shape, [9, 10])
