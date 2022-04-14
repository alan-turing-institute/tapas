"""
Threat models for Membership Inference Attacks.
"""

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
        self.dataset.drop_records([target_record.id], in_place=True) # Remove target
        self.generator = generator

        ## Set up adversary's knowledge
        self._adv_data = {
            'aux': aux_data.drop_records([target_record.id] or self.dataset.empty(), # TODO: Implement empty method on Dataset
            'real': self.dataset.sample(frac=sample_real_frac), # TODO: Add frac kwarg to dataset.sample
            'target': self.target_record}
        # If no shadow model provided, assume full access to generator
        self.shadow_model = shadow_model or generator

        ## Set up hyperparameters for how the adversary will create shadow models
        self.num_training_records = num_training_records
        self.num_synthetic_records = num_synthetic_records
        self.memorise_datasets = memorise_datasets # JJ: I think this can be done at run-time

        self.train_sets = None
        self.test_sets = None


    @property
    def adv_data(self):
        """
        Dataset: The data the adversary has access to.

        """
        return self._adv_data['aux'] + self._adv_data['real']

    def _generate_datasets(self,
                           num_samples: int,
                           num_synthetic_records: int = None,
                           replace_target: bool = False,
                           training: bool = True) -> tuple(list[Dataset], list(int)]):
        """
        Generate synthetic datasets and labels using self.shadow_model. Synthetic
        datasets are generated in pairs, one from D, and one from D u {target},
        where D is sampled without replacement from either self.adv_data or
        self.dataset depending on whether or not training=True.

        Parameters
        ----------
        num_samples : int
            Number of training dataset *pairs* to generate.
        num_synthetic_records : int, optional
            Size of synthetic datasets to generate. If None, use
            self.num_synthetic_records. The default is None.
        replace_target : bool, optional
            Indicates whether target is included by replacing a record or by
            just adding it. If True, a random record will be removed before
            adding target. The default is False.
        training : bool, optional
            If True, D's will be sampled from the adversary's data (self.adv_data).
            Otherwise, D's will be sampled from the real data (self.datasets).
            The default is True.

        Returns
        -------
        tuple(list[Dataset], np.ndarray)
            List of generated synthetic datasets. List of labels.

        """
        # If training, sample datasets from the adversary's data. Otherwise,
        # sample datasets from the real dataset.
        dataset = (self.adv_data if training else self.dataset).copy()

        # TODO: the interface should include some number of synthetic records to produce.
        num_synthetic_records = num_synthetic_records or self.num_synthetic_records

        # Split the data into subsets
        datasets = dataset.create_subsets(num_samples, num_synthetic_records)

        synthetic_without_target = []
        synthetic_with_target = []

        for training_dataset in datasets:
            # Compute generator(D)
            synthetic_without_target.append(self.generator(training_dataset))

            # Then, add  - or replace a record by - the target the record.
            if replace_target:
                training_dataset = training_dataset.replace(self.target_record)  # TODO: define dataset.
            else:
                training_dataset = training_dataset.add(self.target_record)
            synthetic_with_target.append(self.generator(training_dataset))

        synthetic_datasets = synthetic_without_target + synthetic_with_target
        labels = ([0] * num_samples) + ([1] * num_samples)

        return synthetic_datasets, labels # TODO: Think about label return type

    def generate_training_samples(self,
                                  num_samples: int,
                                  num_synthetic_records: int = None,
                                  replace_targets: bool = False) -> tuple(list[Dataset], list[int]):
        """
        Generate samples according to the attacker's known information.
        (See _generate_datasets for the specific arguments.) This is just
        short-hand for calling _generate_datasets with training=True.

        """
        return self._generate_datasets(num_samples, num_synthetic_records,
            training=True, replace_target=replace_targets)

    # TODO: Better docstring description
    def test(self,
             attack: Attack,
             num_samples: int,
             num_synthetic_records: int = None,
             replace_targets: bool = False,
             save_datasets: bool = False) -> tuple(list(int), list(int)):
        """
        Test an attack against this threat model. First, random subsets of size
        self.num_training_records are sampled from self.dataset (the real dataset)
        and then self.target_record is added to each of them. The attack is run
        on both the datasets with and without the target and their guesses
        are returned alongside the ground-truth for each dataset. The generated
        test datasets can be saved by setting save_datasets=True.

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
        save_datasets : bool, optional
            Whether or not to save the generated test datasets into the threat
            model to be used for other attacks.

        Returns
        -------
        tuple(list(int), list(int))
            Tuple of (ground_truth, guesses), where ground_truth indicates
            what the attack needed to guess, and guesses are the attack's actual
            guesses.

        """
        # Check for existing datasets
        if self.test_sets:
            num_extra_samples = max(0, num_samples - len(self.test_sets['datasets']))

            # If we don't need any more samples, return the existing ones
            if num_extra_samples = 0:
                return self.test_sets['datasets'][:num_samples], self.test_sets['labels'][:num_samples]

        else:
            num_extra_samples = num_samples

        # Generate test samples
        test_datasets, test_labels = self._generate_datasets(
            num_extra_samples, num_synthetic_records, replace_targets=replace_targets, training=False)

        # Save datasets if required
        if save_datasets:
            if not self.test_sets:
                self.test_sets = {'datasets': test_datasets, 'labels': test_labels}
            else:
                self.test_sets['datasets'] += test_datasets
                self.test_sets['labels'] += test_labels

        # Attack makes guesses about test samples
        guesses = attack.attack(test_datasets)

        return test_labels, guesses