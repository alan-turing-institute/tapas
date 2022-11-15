"""A test for some attack classes."""

import unittest
from unittest import TestCase

import numpy as np
import pandas as pd

from tapas.datasets import TabularDataset, TabularRecord
from tapas.datasets.data_description import DataDescription
from tapas.threat_models import (
    TargetedMIA,
    TargetedAIA,
    AuxiliaryDataKnowledge,
    BlackBoxKnowledge,
)
from tapas.generators import Raw

# The classes being tested.
from tapas.attacks import (
    ClosestDistanceMIA,
    GroundhogAttack,
    NaiveSetFeature,
    HistSetFeature,
    CorrSetFeature,
    FeatureBasedSetClassifier,
    HammingDistance,
    LpDistance,
)

from sklearn.linear_model import LogisticRegression

## Test for closest-distance.

dummy_data_description = DataDescription(
    [
        {"name": "a", "type": "countable", "representation": "integer"},
        {"name": "b", "type": "countable", "representation": "integer"},
    ]
)

dummy_data = pd.DataFrame([(0, 1), (0, 2), (3, 4), (3, 5)], columns=["a", "b"])


## Test for closest-distance attack.


class TestClosestDistance(TestCase):
    """Test the closest-distance attack."""

    def setUp(self):
        self.dataset = TabularDataset(dummy_data, dummy_data_description)

    def _make_mia(self, a, b):
        """Helper function to generate a MIA threat model."""
        return TargetedMIA(
            AuxiliaryDataKnowledge(self.dataset, auxiliary_split=0.5, num_training_records=2),
            self._make_target(a, b),
            BlackBoxKnowledge(Raw(), num_synthetic_records=None),
        )

    def _make_target(self, a, b):
        """Helper function to generate a target record."""
        return TabularDataset(
            pd.DataFrame([(a, b)], columns=["a", "b"]), dummy_data_description
        )

    def test_dummy(self):
        # Check whether the attack works on a dummy dataset,
        #  with a specified threshold.

        # Take a record that is not in the dataset (distance 1/2).
        mia = self._make_mia(0, 0)
        print(mia.generate_training_samples(100))
        attack = ClosestDistanceMIA(criterion=("threshold", -0.3))
        attack.train(mia)
        # Check that the training worked as intended.
        self.assertEqual(attack._threshold, -0.3)
        # Check that the score is working as intended.
        scores = attack.attack_score([rec for rec in self.dataset])
        self.assertEqual(len(scores), len(self.dataset))
        for score, distance in zip(scores, [1, 1, 2, 2]):
            self.assertEqual(score, -distance)
        # Assert that the total score and decisions are ok.
        self.assertEqual(attack.attack_score([self.dataset])[0], -1)
        self.assertEqual(attack.attack([self.dataset])[0], False)

        # Perform the attack for a user *in* the dataset.
        attack = ClosestDistanceMIA(criterion=("threshold", -0.3))
        attack.train(self._make_mia(0, 1))
        print("attack")
        self.assertEqual(attack.attack([self.dataset])[0], True)

    def test_training(self):
        # Check that the threshold selection works.
        # This merely checks that the code runs, not that it is correct.
        mia = TargetedMIA(
            AuxiliaryDataKnowledge(
                self.dataset, auxiliary_split=0.5, num_training_records=2
            ),
            self._make_target(0, 4),
            BlackBoxKnowledge(generator=Raw(), num_synthetic_records=2),
            replace_target=True,
        )
        attack_tpr = ClosestDistanceMIA(criterion=("tpr", 0.1))
        attack_tpr.train(mia)
        attack_fpr = ClosestDistanceMIA(criterion=("fpr", 0.1))
        attack_fpr.train(mia)

    def test_distances(self):
        # Check that the other distances run and have zero.
        num_cat = 10
        num_records = 21
        full_dataset = TabularDataset(
            pd.DataFrame(
                zip(
                    np.random.randint(num_cat, size=(num_records,)),
                    np.random.random(size=(num_records,)),
                ),
                columns=["a", "b"],
            ),
            DataDescription(
                [
                    {"name": "a", "type": "finite", "representation": num_cat},
                    {"name": "b", "type": "countable", "representation": "integer"},
                ]
            ),
        )
        # Also select a subset of smaller size.
        num_records_small = 5
        small_dataset = full_dataset.create_subsets(1, num_records_small)[0]
        # Check a few distances.
        distances = [
            HammingDistance(),
            HammingDistance(columns=['a']),
            LpDistance(2),
            LpDistance(4),
            LpDistance(2, weights=np.random.random(size=(num_cat + 1))),
            0.5 * LpDistance(2) + HammingDistance() * 0.5
        ]
        for d in distances:
            array_of_dists = d(full_dataset, full_dataset)
            # Check that the size of the array is correct.
            self.assertEqual(array_of_dists.shape, (num_records, num_records))
            # Check that distance to self is 0.
            for i in range(num_records):
                self.assertEqual(array_of_dists[i, i], 0)
            # Check that the distance is symmetrical.
            for i in range(num_records):
                for j in range(i + 1, num_records):
                    self.assertEqual(array_of_dists[i, j], array_of_dists[j, i])
            # Check that the size is correct for smaller dataset.
            self.assertEqual(
                d(small_dataset, full_dataset).shape, (num_records_small, num_records)
            )


## Test for features.


class TestSetFeatures(TestCase):
    """Test whether the set features defined for Groundhog are implemented correctly."""

    def test_naive(self):
        """Test that the naive features work properly."""
        num_records = 20
        num_datasets = 10
        num_finite = 2
        data_description = DataDescription(
            [
                {"name": "a", "type": "real", "representation": "number"},
                {"name": "b", "type": "real", "representation": "number"},
                {"name": "c", "type": "finite", "representation": num_finite},
            ]
        )
        real_data = [
            np.concatenate(
                (
                    np.random.random(size=(num_records, 2)),
                    np.random.randint(num_finite, size=(num_records, 1)),
                ),
                axis=1,
            )
            for _ in range(num_datasets)
        ]
        datasets = [
            TabularDataset(
                pd.DataFrame(data, columns=["a", "b", "c"]), data_description
            )
            for data in real_data
        ]
        feature = NaiveSetFeature()
        values = feature(datasets)
        # Check that it has the proper shape.
        self.assertEqual(values.shape, (num_datasets, 3 * (2 + num_finite)))
        # Check that it is correct (for continuous variables only).
        # This feature set starts with means for all variables (finite vars are
        # one-hot encoded), then medians and finally variances.
        for data, val in zip(real_data, values):
            print(val)
            self.assertAlmostEqual(data[:, 0].mean(axis=0), val[0])
            self.assertAlmostEqual(data[:, 1].mean(axis=0), val[1])
            self.assertAlmostEqual(np.median(data[:, 0], axis=0), val[2 + num_finite])
            self.assertAlmostEqual(
                np.median(data[:, 1], axis=0), val[2 + num_finite + 1]
            )
            self.assertAlmostEqual(data[:, 0].var(axis=0), val[2 * (2 + num_finite)])
            self.assertAlmostEqual(
                data[:, 1].var(axis=0), val[2 * (2 + num_finite) + 1]
            )

    def test_histogram(self):
        """Test that the histogram features work properly."""
        data_description = DataDescription(
            [
                {"name": "a", "type": "real", "representation": "number"},
                {"name": "b", "type": "finite", "representation": ["x", "y", "z"]},
            ]
        )
        data1 = pd.DataFrame(
            [(0.1, "x"), (0.9, "y"), (0.7, "x"), (0.9, "z")], columns=["a", "b"]
        )
        data2 = pd.DataFrame([(0.5, "z")], columns=["a", "b"])
        feature = HistSetFeature(num_bins=5, bounds=(0, 1))
        histograms = feature(
            [
                TabularDataset(data1, data_description),
                TabularDataset(data2, data_description),
            ]
        )
        self.assertEqual(histograms.shape, (2, 8))
        # Bins (0,.2), (.2, .4), (.4, .6), (.6, .8), (.8, 1)
        expected_answers = np.array(
            [
                [1 / 4, 0, 0, 1 / 4, 2 / 4, 2 / 4, 1 / 4, 1 / 4],
                [0, 0, 1, 0, 0, 0, 0, 1],
            ]
        )
        # Check that the features are the proper answer.
        for computed, expected in zip(histograms.flatten(), expected_answers.flatten()):
            self.assertEqual(computed, expected)

    def test_combination(self):
        """Test whether combining feature maps works."""
        data_description = DataDescription(
            [
                {"name": "a", "type": "real", "representation": "number"},
                {"name": "b", "type": "finite", "representation": ["x", "y", "z"]},
            ]
        )
        num_records = 100
        dataset = TabularDataset(
            pd.DataFrame(
                zip(
                    np.random.random(size=(num_records,)),
                    np.random.choice(
                        ["x", "y", "z"], size=(num_records,), replace=True
                    ),
                ),
                columns=["a", "b"],
            ),
            data_description,
        )
        num_bins = 10
        feature = (
            NaiveSetFeature()
            + HistSetFeature(num_bins=num_bins, bounds=(0, 1))
            + CorrSetFeature()
        )
        result = feature([dataset])
        # We only test whether the size of the output is correct.
        # We assume the content is correct, from other tests.
        num_continuous = 1
        discrete_1hot = 3
        num_columns = num_continuous + discrete_1hot
        self.assertEqual(
            result.shape,
            (
                1,
                3 * num_columns  # Naive
                + num_bins * num_continuous  # Hist
                + discrete_1hot  # Hist
                + num_columns * (num_columns - 1) / 2,  # Corr
            ),
        )


## Test for the Groundhog attack.


class TestGroundHog:
    """Test whether the groundhog attack (Stadler et al.) works."""

    def test_groundhog_runs(self):
        """Test whether the Groundhog attack runs."""
        values = ["x", "y", "z"]
        num_records = 1000
        total_dataset = TabularDataset(
            pd.DataFrame(
                zip(
                    np.random.random(size=num_records),
                    np.random.choice(values, size=num_records, replace=True),
                ),
                columns=["a", "b"],
            ),
            DataDescription(
                [
                    {"name": "a", "type": "real", "representation": "number"},
                    {"name": "b", "type": "finite", "representation": values},
                ]
            ),
        )
        mia = TargetedMIA(
            AuxiliaryDataKnowledge(
                total_dataset, auxiliary_split=0.5, num_training_records=200
            ),
            total_dataset.sample(1),  # Random target.
            BlackBoxKnowledge(Raw(), num_synthetic_records=200),
        )
        attack = GroundhogAttack()
        attack.train(mia)


if __name__ == "__main__":
    unittest.main()
