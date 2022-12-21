"""A test for threat models."""

import os
from unittest import TestCase

import itertools
import numpy as np
import pandas as pd
import pytest

from tapas.datasets import TabularDataset, TabularRecord
from tapas.datasets.data_description import DataDescription
from tapas.threat_models import (
    ThreatModel,
    TargetedMIA,
    TargetedAIA,
    AuxiliaryDataKnowledge,
    BlackBoxKnowledge,
    NoBoxKnowledge,
    UncertainBoxKnowledge,
)
from tapas.generators import Raw, Generator

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
        self.assertEqual(mia.multiple_label_mode, False)
        # Check that we generate the correct number of samples.
        num_samples = 100
        datasets, labels = mia.generate_training_samples(num_samples)
        self.assertEqual(len(datasets), num_samples)
        self.assertEqual(len(datasets), len(labels))
        # We here use RAW as a generator, so the datasets generated are the
        # training datasets directly. We can thus check target membership on
        # the dataset and that the labels are correct.
        for ds, target_in in zip(datasets, labels):
            self.assertEqual(len(ds), 2 if (replace_target or not target_in) else 3)
            self.assertEqual(target_record in ds, target_in)

    def test_labelling_default(self):
        self._test_labelling_helper(False, False)

    def test_labelling_pairs(self):
        self._test_labelling_helper(True, False)

    def test_labelling_replace(self):
        self._test_labelling_helper(False, True)

    def test_labelling_replace_pairs(self):
        self._test_labelling_helper(True, True)


class TestMIAMultipleTargets(TestCase):
    """Test the membership-inference attack with multiple targets."""

    def _test_multiple_targets(self, generate_pairs, replace_target):
        # Some parameters.
        num_training_records = 100
        num_targets = 10
        # Generate all combinations (so that records are unique!).
        large_dummy_data = TabularDataset(
            pd.DataFrame(
                list(itertools.product(range(10), range(10), range(10))),
                columns=["a", "b", "c"],
            ),
            dummy_data_description,
        )
        # Select a large number of targets.
        target_idxs = np.random.choice(
            len(large_dummy_data), size=(num_targets,), replace=False
        )
        target_records = large_dummy_data.get_records(target_idxs)
        large_dummy_data.drop_records(target_idxs, in_place=True)
        # Create the threat model with multiple targets.
        mia = TargetedMIA(
            AuxiliaryDataKnowledge(
                large_dummy_data,
                auxiliary_split=0.5,
                num_training_records=num_training_records,
            ),
            target_records,
            knowledge_on_sdg,
            replace_target=replace_target,
            generate_pairs=generate_pairs,
        )
        self.assertEqual(mia.multiple_label_mode, True)
        # Generate datasets and check the labelling.
        for r, threat_model_targeted in zip(target_records, mia):
            # Check that the target record is properly set.
            self.assertEqual(len(threat_model_targeted.target_record), 1)
            for x, y in zip(
                threat_model_targeted.target_record.data.values[0], r.data.values[0]
            ):
                self.assertEqual(x, y)
            # Generate some datasets (unchanged through raw).
            num_generated_samples = 20
            datasets, labels = threat_model_targeted.generate_training_samples(
                num_generated_samples
            )
            self.assertEqual(len(datasets), num_generated_samples)
            self.assertEqual(len(labels), num_generated_samples)
            # Check that the datasets are properly labelled.
            for ds, target_in in zip(datasets, labels):
                # If targets are replaced, the dataset should always have the
                # same numbers of records.
                if replace_target:
                    self.assertEqual(len(ds), num_training_records)
                elif target_in:
                    # If not replacing, and this record has been *added* to the
                    # dataset, the size of ds is larger than the number of records.
                    # Note that we can't know the expected size without having
                    # access to all labels.
                    self.assertGreater(len(ds), num_training_records)
                self.assertEqual(r in ds, target_in)

    def test_labelling_default(self):
        self._test_multiple_targets(False, False)

    def test_labelling_pairs(self):
        self._test_multiple_targets(True, False)

    def test_labelling_replace(self):
        self._test_multiple_targets(False, True)

    def test_labelling_replace_pairs(self):
        self._test_multiple_targets(True, True)


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
            self.assertEqual(record in ds, True)

    def test_multiple_targets(self):
        """Test whether the datasets are correctly labelled, for multiple targets."""
        num_training_records = 100
        num_targets = 10
        # Generate all combinations (so that records are unique!), but with many more
        # values for (a,b) --> 900 records, and "c" being binary.
        large_dummy_data = TabularDataset(
            pd.DataFrame(
                list(itertools.product(range(30), range(30), (0, 1))),
                columns=["a", "b", "c"],
            ),
            dummy_data_description,
        )
        # Select a large number of targets.
        target_idxs = np.random.choice(
            len(large_dummy_data), size=(num_targets,), replace=False
        )
        target_records = large_dummy_data.get_records(target_idxs)
        large_dummy_data.drop_records(target_idxs, in_place=True)
        # Create the threat model.
        aia = TargetedAIA(
            AuxiliaryDataKnowledge(
                large_dummy_data,
                auxiliary_split=0.5,
                num_training_records=num_training_records,
            ),
            target_records,
            "c",
            [0, 1],
            knowledge_on_sdg,
        )
        # Generate datasets and check the labelling.
        for r, threat_model_targeted in zip(target_records, aia):
            # Check that the target record is found in the dataset.
            self.assertEqual(len(threat_model_targeted.target_record), 1)
            for x, y, col in zip(
                threat_model_targeted.target_record.data.values[0],
                r.data.values[0],
                ["a", "b", "c"],
            ):
                # Check equality for non-sensitive attributes.
                if col != "c":
                    self.assertEqual(x, y)
            # Generate some datasets (unchanged through raw).
            num_generated_samples = 20
            datasets, labels = threat_model_targeted.generate_training_samples(
                num_generated_samples
            )
            self.assertEqual(len(datasets), num_generated_samples)
            self.assertEqual(len(labels), num_generated_samples)
            # Check that the record with the correct value is found in the dataset.
            for ds, value in zip(datasets, labels):
                record = r.copy()
                record.set_value("c", value)
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

    def test_no_box(self):
        gen = NoBoxKnowledge(Raw(), 2)
        with pytest.raises(Exception) as err:
            gen(dataset, training_mode=True)
        gen(dataset, training_mode=False)


    def test_uncertain_box(self):
        # First, define a silly 1-dimensional generator.
        class Replicator(Generator):
            def __call__(self, dataset, num_samples, mean=0):
                return np.full((num_samples,), mean)
            def fit(self, *args): pass
            def generate(self, *args): pass

        # Then, define a threat model using this, and test it.
        gen = UncertainBoxKnowledge(
            Replicator(), 1, lambda: {"mean": np.random.normal()}, {"mean": 117}
        )
        records_train = [gen(None, training_mode = True) for _ in range(1000)]
        records_test = [gen(None, training_mode = False) for _ in range(1000)]
        self.assertTrue(np.mean(records_train) < 4)  # Unlikely to fail.
        self.assertTrue(np.std(records_train) < 2)
        for x in records_test:
            self.assertEqual(x[0], 117)


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
