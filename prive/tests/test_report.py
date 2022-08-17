import os
import unittest
import numpy as np
import datetime

from prive.report import MIAttackReport


class MIAReport(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

        # generating attack data in the format that is expected for the MIAttackReport class.
        attacks = []
        for i in range(1000):
            attack_dict = {
                "labels": np.random.randint(2, size=100),
                "predictions": np.random.randint(2, size=100),
                "scores": np.random.random(size=100),
                "generator": np.random.choice(
                    ["RandomGenerator1", "RandomGenerator2", "RandomGenerator3"]
                ),
                "dataset": np.random.choice(["TestData1"]),
                "target_id": str(np.random.randint(1, 5, size=1)[0]),
                "attack": np.random.choice(["RandomGuess1", "RandomGuess2"]),
            }

            attacks.append(attack_dict)

        self.report = MIAttackReport.load_summary_statistics(attacks)

    def test_setup(self):
        self.assertTrue(self.report.attacks_data.shape, [9, 10])

    def test_create_report(self):
        filepath_timestamp = os.path.join(
            os.path.dirname(__file__),
            f'outputs/prive_report_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}',
        )

        self.report.create_report(filepath_timestamp)

        self.assertTrue(os.path.exists(filepath_timestamp))

        n_figures = len(
            [
                f
                for f in os.listdir(filepath_timestamp)
                if os.path.isfile(os.path.join(filepath_timestamp, f))
            ]
        )

        self.assertEqual(n_figures, 16)
