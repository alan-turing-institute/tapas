"""A test for threat models."""

import os
from unittest import TestCase

import numpy as np
import pandas as pd

from prive.datasets import TabularDataset, TabularRecord
from prive.datasets.data_description import DataDescription
from prive.threat_models import (
    ThreatModel,
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
    dataset, auxiliary_split=0.5, num_training_records=2
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
            print(ds.data, "\n", target_record.data, target_value)
            self.assertEqual(record in ds, True)


class TestAttackerKnowledge(TestCase):
    """Test the attacker knowledge."""

    def test_auxiliary_dataset(self):
        gen_data = lambda size: TabularDataset(
            pd.DataFrame(
                np.random.randint(10, size=(size, 3)), columns=["a", "b", "c"]
            ),
            dummy_data_description,
        )
        # Check that the auxiliary and test datasets have appropriate size.
        for aux_size, test_size, split, full_size in [
            (20, 20, 0.5, 1000),
            (0, 0, 0.1, 100),
            (117, 39, 0.8, None),
        ]:
            dataset = gen_data(full_size) if full_size is not None else None
            threat_model = AuxiliaryDataKnowledge(
                dataset=dataset,
                auxiliary_split=split,
                aux_data=gen_data(aux_size) if aux_size > 0 else None,
                test_data=gen_data(test_size) if test_size > 0 else None,
            )
            # Compute the contribution of the full dataset to auxiliary and test data.
            aux_split_size = int(split * full_size) if full_size is not None else 0
            test_split_size = full_size - aux_split_size if full_size is not None else 0
            # Check that sizes are as expected.
            self.assertEqual(len(threat_model.aux_data), aux_size + aux_split_size)
            self.assertEqual(len(threat_model.test_data), test_size + test_split_size)


class TestSaveLoad(TestCase):
    """Check whether saving/loading threat models works."""

    def test_save_then_load(self):
        name = os.path.join(os.path.dirname(__file__), "outputs/threat_model_test")
        threat_model = TargetedMIA(
            knowledge_on_data,
            target_record,
            knowledge_on_sdg,
            generate_pairs=False,
            replace_target=False,
        )
        training_samples = 103
        testing_samples = 42
        threat_model.generate_training_samples(training_samples)
        threat_model._generate_samples(testing_samples, False)
        threat_model.save(name)
        threat_model_2 = ThreatModel.load(name)
        # Check that the models are identical:
        self.assertIsInstance(threat_model_2, TargetedMIA)
        # Check that the target records are the same.
        for x, y in zip(
            threat_model.target_record.data, threat_model_2.target_record.data
        ):
            self.assertEqual(x, y)
        # The following is specific to TargetedMIA, and checks that the internal
        # memory of the object is properly set. This is the most important
        # feature of .save and .load: to not have to recompute the datasets.
        self.assertEqual(len(threat_model_2._memory[True][0]), training_samples)
        self.assertEqual(len(threat_model_2._memory[False][0]), testing_samples)
        # Check that the samples are identical (from memory:).
        datasets1, labels1 = threat_model._memory[True]
        datasets2, labels2 = threat_model_2._memory[True]
        for l1, l2 in zip(labels1, labels2):
            self.assertEqual(l1, l2)
        for d1, d2 in zip(datasets1, datasets2):
            self.assertEqual(len(d1), len(d2))
            for x, y in zip(d1, d2):
                for v1, v2 in zip(x.data.values[0], y.data.values[0]):
                    self.assertEqual(v1, v2)
        # Finally, check that the name is properly set.
        self.assertEqual(threat_model_2._name, name)
