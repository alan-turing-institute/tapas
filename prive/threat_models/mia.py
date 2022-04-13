"""Threat models for Membership Inference Attacks."""

import numpy as np
import pandas as pd

from .base_classes import ThreatModel, StaticDataThreatModel
from ..attacks import Attack # for typing
from ..datasets import Dataset # for typing
from ..generators import Generator # for typing


class TargetedMIA(ThreatModel):
    """
    Abstract base class for a targeted MIA.

    A MIA (membership inference attack) aims at identifying whether a given
    target record is in the training dataset.
    """

    def __init__(self, target_record):
        self.target_record = target_record

    # Generating training and testing samples depends on the assumptions!


class TargetedAuxiliaryDataMIA(StaticDataThreatModel):
    """
    This threat model assumes access to some data and some knowledge of
    the algorithm that will be used as generator, specified by passing a
    shadow model. If no shadow model is passed, full access is assumed.
    """

    def __init__(self,
                 target_record: Dataset,
                 dataset: Dataset,
                 # dataset_sampler=None,
                 generator: Generator,
                 aux_data: Dataset = None,
                 sample_real_frac: float = 0.,
                 shadow_model: Generator = None,
                 num_training_records: int = 1000,
                 num_synthetic_records: int = 1000,
                 memorise_datasets: bool = True):
        """
        Initialise threat model with ground truth target record, dataset and
        generator. Additionally, either aux_data or sample_real_frac must be
        provided in order to initialise the adverary's data knowledge. Optionally
        a shadow_model can be provided to indicate that the adversary does not
        have complete knowledge of the generator and chooses to model it as
        shadow_model.

        Parameters
        ----------
        target_record : Dataset
            Record to be targeted by the attacks.
        dataset : Dataset
            Real dataset to use to generate test synthetic datasets from.
        generator : Generator
            Generator to use to generate test datasets.
        aux_data : Dataset, optional
            Dataset that the adversary is assumed to have access to. This or
            sample_real_frac must be provided. The default is None.
        sample_real_frac : float, optional
            Fraction of real data to sample and assume adversary has access to.
            Must be in [0, 1]. This must be > 0 or aux_data provided.
            The default is 0.
        shadow_model : Generator, optional
            Adversary's model of the generator, if not provided, the adversary
            is assumed to have full access to the generator. The default is None.
        num_training_records : int, optional
            Number of training records to use to train each copy of shadow_model,
            when generating synthetic training datasets for the attack.
            The default is 1000.
        num_synthetic_records : int, optional
            Number of synthetic records to generate in each synthetic dataset.
            The default is 1000.
        memorise_datasets : bool, optional
            Whether to save generated datasets. The default is True.

        Returns
        -------
        None.

        """
        assert (aux_data is not None) or (sample_real_fraction != 0.), \
            'At least one of aux_data or sample_real_fraction must be given'
        assert (0 <= sample_real_frac <= 1), \
            f'sample_real_frac must be in [0, 1], got {sample_real_frac}'
        if aux_data:
            assert aux_data.description == dataset.description, \
                'aux_data does not match the description of dataset'

        ## Set up ground truth
        self.target_record = target_record
        self.dataset = dataset
        self.generator = generator

        ## Set up adversary's knowledge
        self.adv_data_ = {'aux': aux_data or self.dataset.empty(), # TODO: Implement empty method on Dataset
                          'real': self.dataset.sample(frac=sample_real_frac)} # TODO: Add frac kwarg to dataset.sample
        # If no shadow model provided, assume full access to generator
        self.shadow_model = shadow_model or generator

        ## Set up hyperparameters for how the adversary will create shadow models
        self.num_training_records = num_training_records
        self.num_synthetic_records = num_synthetic_records
        self.memorise_datasets = memorise_datasets


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


    # JJ: I think this can be removed in favour of test
    def generate_testing_samples(self, num_samples, num_synthetic_records=None,
                                replace_targets=False):
        """Generate testing samples (depending on known information).

            (see _sample_datasets for the specific arguments)."""
        return self._sample_datasets(num_samples, num_synthetic_records,
            training=False, replace_target=replace_targets)

    # TODO: Better docstring description
    def test(self,
             attack: Attack,
             num_samples: int,
             num_synthetic_records: int = None,
             replace_targets: bool = False) -> tuple(list(int), list(int)):
        """
        Test an attack against this threat model.

        Parameters
        ----------
        attack : Attack
            Attack to test.
        num_samples : int
            Number of test datasets to generate and test against.
        num_synthetic_records : int, optional
            Number of synthetic records to generate per synthetic dataset.
            The default is None.
        replace_targets : bool, optional
            Whether or not to remove a row before adding the target in each dataset.
            The default is False.

        Returns
        -------
        tuple(list(int), list(int))
            Tuple of (ground_truth, guesses), where ground_truth indicates
            what the attack needed to guess, and guesses are the attack's actual
            guesses.

        """
        # Generate test samples
        test_datasets, test_labels = self._sample_datasets(
            num_samples, num_synthetic_records, training=False, replace_targets=replace_targets)

        # Attack makes guesses about test samples
        guesses = attack.attack(test_datasets)

        return test_labels, guesses