"""Threat models for Membership Inference Attacks."""

import numpy as np
import pandas as pd

from .base_classes import ThreatModel


class TargetedMembershipInference(ThreatModel):
    """
    Abstract base class for a targeted MIA.

    A MIA (membership inference attack) aims at identifying whether a given
    target record is in the training dataset.
    """

    def __init__(self, target_record):
        self.target_record = target_record

    # Generating training and testing samples depends on the assumptions!



class AuxiliaryDataMIA(TargetedMembershipInference):
    """
    This threat model assumes access to some data and some knowledge of
    the algorithm that will be used as generator, specified by passing a
    shadow model. If no shadow model is passed, full access is assumed.
    """

    def __init__(self, target_record, dataset=None, dataset_sampler=None,
        generator=None, num_training_records=1000, num_synthetic_records=1000,
        auxiliary_test_split=0.9, memoize_datasets=True):
        """Create a MIA threat model with black-box access to the generator,
            and where the attacker has access to an auxiliary dataset.

        Args:
            target_record (Dataset.Record): the target of the MIA.
            dataset (Dataset): the complete dataset, from which the auxiliary
                and testing datasets will be sampled.
            dataset_sampler (callable): alternatively, a sampler for the datasets.
                (not implemented yet).
            generator (callable): the generating model, as a black-box.
            num_training_records (int): number of training samples in the
                private training dataset.
            num_synthetic_records (int): number of synthetic samples to generate.
            auxiliary_test_split (float in [0,1]): fraction of the dataset to
                use as auxiliary information available to attacker.
            memoize_datasets (bool): whether to remember datasets
        """
        assert (aux_data is None) or (sample_real_fraction is None), \
            'At least one of data or sample_real_fraction must be given'
        assert (0 <= auxiliary_test_split <= 1), \
            'Auxiliary dataset size must be in [0, 1].'
        TargetedMembershipInference.__init__(self, target_record)
        # Split the dataset between auxiliary and testing dataset.
        self.auxiliary_test_split = auxiliary_test_split
        self.num_training_records = num_training_records
        self.num_synthetic_records = num_synthetic_records
        self.datasets = self._split_auxiliary_testing(dataset, auxiliary_test_split)
        self.generator = generator


    def _split_auxiliary_testing(self, dataset, auxiliary_test_split):
        """Split a dataset between training and testing dataset."""
        self.dataset = dataset.copy()               
        # Remove the target from the dataset, if present.
        self.dataset.drop(self.target_record)                          # TODO: define dataset.
        # Split the dataset in parts.
        auxiliary, testing = self.dataset.split(auxiliary_test_split)  # TODO: define dataset.
        return {'auxiliary': auxiliary, 'testing': testing}


    def _sample_datasets(self, num_samples, num_synthetic_records=None,
        replace_target=False, training=True):
        """Generate synthetic datasets and labels using self.shadow_model.

        Args:
            num_samples (int): Number of training dataset *pairs* to generate
                per target. The pairs are composed of a dataset D sampled without
                replacement the auxiliary or testing data, and D u {target}.
            num_synthetic_records (int): Size of synthetic datasets to generate.
                If None (default), use self.num_synthetic_records.
            targets (List[pd.DataFrame]): List of targets to generate datasets
                for.
            replace_target (bool): Indicates whether target is included by
                replacing a row or by extending the training data. If True, a
                data row will be removed before adding target. Default False.
            training (bool): Whether to sample datasets from the auxiliary 
                (True) or test (False) dataset.

        Returns:
            datasets (List[pd.DataFrame]): List of generated datasets
            labels (List[np.array]): Numpy array of shape (N, len(targets))
                where the (i,j)-th entry indicates whether or not target j
                was in the training data that was used to generate dataset i.
        """
        synthetic_datasets = []
        labels = np.zeros((2*num_samples, 1))

        # If training, sample datasets from the auxiliary dataset. Otherwise,
        #  sample datasets from the testing dataset.
        dataset = self.datasets['auxiliary' if training else 'testing'].copy()

        # TODO: the interface should include some number of synthetic records to produce.
        num_synthetic_records = num_synthetic_records or self.num_synthetic_records

        for i in range(num_samples):
            # Compute a sample dataset D and generator(D).
            training_dataset = dataset.subsample(size=self.num_training_records)
            synthetic_dataset = self.generator(training_dataset)
            synthetic_datasets.append(synthetic_dataset)
            # Then, add  - or replace a record by - the target the record.
            if replace_target:
                training_dataset = training_dataset.drop_random_record()  # TODO: define dataset.
            training_dataset = training_dataset.add(self.target_record)   # TODO: define dataset.
            synthetic_dataset = self.generator(training_dataset)
            synthetic_datasets.append(synthetic_dataset)
            # Replace the label for the dataset containing the target.
            labels[2*i+1] = 1
        
        return synthetic_datasets, labels


    def generate_training_samples(self, num_samples, num_synthetic_records=None,
                                  replace_targets=False):
        """Generate samples according to the attacker's known information.

            (see _sample_datasets for the specific arguments)."""
        return self._sample_datasets(num_samples, num_synthetic_records,
            training=True, replace_target=replace_targets)


    def generate_testing_samples(self, num_samples, num_synthetic_records=None,
                                replace_targets=False):
        """Generate testing samples (depending on known information).

            (see _sample_datasets for the specific arguments)."""
        return self._sample_datasets(num_samples, num_synthetic_records,
            training=False, replace_target=replace_targets)
