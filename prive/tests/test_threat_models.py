"""A test for threat models."""

from unittest import TestCase

import pandas as pd

import sys

sys.path.append("../..")
from prive.datasets import TabularDataset, TabularRecord
from prive.datasets.data_description import DataDescription
from prive.threat_models import (
    TargetedMIA,
    TargetedAIA,
    AuxiliaryDataKnowledge,
    BlackBoxKnowledge,
)
from prive.generators import Raw

dummy_data_description = DataDescription(
    [
        {"name": "a", "type": "countable", "description": "integer"},
        {"name": "b", "type": "countable", "description": "integer"},
        {"name": "c", "type": "countable", "description": "integer"},
    ]
)

dummy_data = pd.DataFrame(
    [(0, 1, 0), (0, 2, 1), (3, 4, 0), (3, 5, 1), (6, 6, 1)], columns=["a", "b", "c"]
)

dataset = TabularDataset(dummy_data, dummy_data_description)

# Choose the target record (4), and remove it from the dataset.
target_record = dataset.get_records([4])
dataset = dataset.drop_records([4])

knowledge_on_data = AuxiliaryDataKnowledge(
    dataset, sample_real_frac=0.5, num_training_records=2
)
knowledge_on_sdg = BlackBoxKnowledge(Raw(), num_synthetic_records=None)


class TestMIA(TestCase):
    """Test the membership-inference attack."""

    def _test_labelling_helper(self, generate_pairs, replace_target):
        """Test whether the datasets are correctly labelled."""
        mia = TargetedMIA(
            knowledge_on_data,
            target_record,
            knowledge_on_sdg,
            generate_pairs=generate_pairs,
            replace_target=replace_target,
        )
        # Check that we generate the correct number of samples.
        num_samples = 100
        datasets, labels = mia.generate_training_samples(num_samples)
        self.assertEqual(len(datasets), num_samples * (1 + generate_pairs))
        self.assertEqual(len(datasets), len(labels))
        # We here use RAW as a generator, so the datasets generated are the
        # training datasets directly. We can thus check target membership on
        # the dataset and that the labels are correct.
        for ds, target_in in zip(datasets, labels):
            self.assertEqual(len(ds), 2 if (replace_target or not target_in) else 3)
            self.assertEqual(target_record in ds, target_in)

    def test_labelling_basic(self):
        self._test_labelling_helper(False, False)

    def test_labelling_pairs(self):
        self._test_labelling_helper(False, True)

    def test_labelling_replace(self):
        self._test_labelling_helper(True, False)


class TestAIA(TestCase):
    """Test the attribute-inference attack."""

    def test_labelling(self):
        """Test whether the datasets are correctly labelled."""
        aia = TargetedAIA(
            knowledge_on_data, target_record, "c", [0, 1], knowledge_on_sdg
        )
        num_samples = 100
        datasets, labels = aia.generate_training_samples(num_samples)
        self.assertEqual(len(datasets), num_samples)
        self.assertEqual(len(datasets), len(labels))
        for ds, target_value in zip(datasets, labels):
            record = target_record.copy()
            record.set_value("c", target_value)
            print(ds.data, '\n', target_record.data, target_value)
            self.assertEqual(record in ds, True)
            pass
