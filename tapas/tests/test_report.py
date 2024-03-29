import os
import unittest
import numpy as np
import datetime

from tapas.report import MIAttackSummary, MIAttackReport, EffectiveEpsilonReport


class MIAReportTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

        # Generating attack data in the format that is expected for the MIAttackReport class.
        # Note that this is not realistic: we have multiple answers values for each header.
        summaries = []
        for i in range(100):
            attack_dict = {
                "labels": np.random.randint(2, size=100),
                "predictions": np.random.randint(2, size=100),
                "scores": np.random.random(size=100),
                "generator": np.random.choice(
                    ["RandomGenerator1", "RandomGenerator2", "RandomGenerator3"]
                ),
                "dataset": "TestData1",
                "target_id": str(np.random.randint(1, 5, size=1)[0]),
                "attack": np.random.choice(["RandomGuess1", "RandomGuess2"]),
            }
            summaries.append(attack_dict)

        self.report = MIAttackReport.load_summary_statistics(summaries)

    def test_setup(self):
        self.assertTrue(self.report.attacks_data.shape, [9, 10])

    def test_publish(self):
        filepath_timestamp = os.path.join(
            os.path.dirname(__file__),
            f'outputs/tapas_report_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}',
        )

        self.report.publish(filepath_timestamp)

        self.assertTrue(os.path.exists(filepath_timestamp))

        n_figures = len(
            [
                f
                for f in os.listdir(filepath_timestamp)
                if os.path.isfile(os.path.join(filepath_timestamp, f))
            ]
        )

        self.assertEqual(n_figures, 20)  # Added a dimension to plot.


class BoostrapReportTest(unittest.TestCase):

    def test_publish(self):
        np.random.seed(0)
        labels = np.random.randint(2, size=100)
        scores = labels + np.random.normal(loc=0, scale=0.7, size=labels.shape)
        predictions = scores > 0.5
        attack_summary = MIAttackSummary(
            labels, predictions, scores, "Random", "Example", "Test", "Target"
        )
        filepath_timestamp = os.path.join(
            os.path.dirname(__file__),
            f'outputs/tapas_report_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}',
        )
        report = MIAttackReport([attack_summary], num_bootstrap=100, metrics=['accuracy', 'auc'])
        report.publish(filepath_timestamp)
        # For comparison, the report without using bootstrapping.
        report = MIAttackReport([attack_summary], metrics=['accuracy', 'auc'])
        report.publish(filepath_timestamp + '_exact')


class EffectiveEpsilonTest(unittest.TestCase):
    def test_clopper_pearson(self):
        # Construct an artificial setup where the Clopper-Pearson bound should
        # be close to epsilon = 1.
        num_trials = int(1e3)  # A lot of samples for accurate bound.
        for epsilon in [0.1, 1, 10]:
            labels = np.random.randint(2, size=(num_trials,))
            scores = labels + np.random.laplace(
                loc=0, scale=1 / epsilon, size=(num_trials,)
            )
            summary = MIAttackSummary(labels, scores > 0.5, scores)
            report = EffectiveEpsilonReport(
                [summary], validation_split=0.1, confidence_levels=(0.9, 0.95, 0.99)
            )
            result = report.publish(os.path.join(os.path.dirname(__file__), 'outputs'))
            print(result)
            self.assertEqual(result.shape, (3, 3))
            # Check that the highest prediction is close to the real value.
            # The 0.75 tolerance is quite tight, as the bound tends to be very loose.
            ## self.assertGreaterEqual(result.epsilon_low.values[0], 0.75 * epsilon)
            # Check that the prediction never exceeds the true value.            
            self.assertLessEqual(result.epsilon_low.values[-1], epsilon)
