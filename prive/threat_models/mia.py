"""
Threat models for Membership Inference Attacks (MIA).

Membership inference attacks aim at detecting the presence of a specific
record in the training dataset from the synthetic dataset observed.

"""

# Type checking stuff
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..attacks import Attack  # for typing
    from ..datasets import Dataset  # for typing
    from ..generators import Generator  # for typing

from .base_classes import ThreatModel, TrainableThreatModel
from .attacker_knowledge import AttackerKnowledgeOnData, AttackerKnowledgeOnGenerator

import numpy as np


class AppendTarget(AttackerKnowledgeOnData):
    """
    Randomly add a given target to the datasets samples from auxiliary data.
    This class can be used to augment AttackerKnowledgeOnData objects that
    represent "generic" knowledge of the private dataset in order to use
    them for membership inference attacks. This is more of a wrapper than
    a standalone object.

    By default, this operatees as an AttackerKnowledgeOnData object, but the
    generate_datasets method can also output a list of labels describing
    target membership to the dataset.

    """

    def __init__(
        self,
        attacker_knowledge: AttackerKnowledgeOnData,
        target_record: Dataset,
        generate_pairs=True,
        replace_target=False,
    ):
        """
        Wrap an AttackerKnowledgeOnData object by appending a record.

        Parameters
        -----
        attacker_knowledge: AttackerKnowledgeOnData
            The data knowledge from which datasets are generated.
        target_record: Dataset
            The target record to append half of the time.
        generate_pairs: bool, default True
            Whether to output pairs of datasets (positive and negative) or
            randomly choose for each dataset.
        replace_target: bool, default False
            Whether to replace a record, instead of appending.

        """
        self.attacker_knowledge = attacker_knowledge
        self.target_record = target_record
        self.generate_pairs = generate_pairs
        self.replace_target = replace_target

    def generate_datasets(
        self, num_samples: int, training: bool = True, with_labels: bool = False
    ) -> list[Dataset]:
        """
        Generates datasets according to the attacker's knowledge, and randomly
        appends the target record to half of them.

        If `with_labels` is True, this also returns the labels (whether the
        target was in the private dataset).

        """
        # Generate the datasets from the attacker knowledge.
        datasets = self.attacker_knowledge.generate_datasets(num_samples, training)
        if self.generate_pairs:
            # If pairs are required, duplicate the list and assign labels 0 and 1.
            datasets = datasets + datasets
            labels = [0] * num_samples + [1] * num_samples
        else:
            # Pick random labels for each dataset.
            labels = list(np.random.random() <= 0.5)
        # Produce a list of datasets with appended target where label = 1.
        app_datasets = [
            (
                ds.replace(self.target_record)
                if self.replace_target
                else ds.add_records(self.target_record)
            )
            if l
            else ds
            for ds, l in zip(datasets, labels)
        ]
        if with_labels:
            return app_datasets, labels
        return app_datasets


class TargetedMIA(TrainableThreatModel):
    """
    This threat model implements a MIA with arbitrary attacker knowledfe on
    data and generator.

    """

    def __init__(
        self,
        attacker_knowledge_data: AppendTarget,
        attacker_knowledge_generator: AttackerKnowledgeOnGenerator,
        memorise_datasets=True,
    ):
        self.target_record = attacker_knowledge_data.target_record
        self.atk_know_data = attacker_knowledge_data
        self.atk_know_gen = attacker_knowledge_generator
        # Also, handle the memoisation to prevent recomputing datasets.
        self.memorise_datasets = memorise_datasets
        # maps training = True/False -> list of datasets, list of labels.
        self._memory = {True: ([], []), False: ([], [])}

    def _generate_samples(
        self, num_samples: int, training: bool = True, ignore_memory: bool = False
    ) -> tuple[list[Dataset], list[bool]]:
        """
        Internal method to generate samples for training or testing. This outputs 
        two lists, the first of synthetic datasets and the second of labels (1 if
        the target is in the training dataset used to produce the corresponding
        dataset, and 0 otherwise).

        Parameters
        ----------
        num_samples: int
            The number of synthetic datasets to generate.
        training: bool (default, True)
            whether to generate samples from the training or test distribution.
        ignore_memory: bool, default False
            Whether to use the memoized datasets, or ignore them.
        """
        # Retrieve memoized samples (if needed).
        if not ignore_memory:
            mem_datasets = []
            mem_labels = []
        else:
            mem_datasets, mem_labels = self._memory[training]
            num_samples -= len(mem_datasets)
            # No samples are needed! Return what is in memory:
            if num_samples <= 0:
                return mem_datasets[:num_samples], mem_labels[:num_samples]
        # Generate sample.
        training_datasets, gen_labels = self.atk_know_data.generate_datasets(
            num_samples, training=True, with_labels=True
        )
        gen_datasets = [self.atk_know_gen(ds) for ds in training_datasets]
        # Add the datasets generated to the memory.
        if not ignore_memory:
            self._memory[training] = (
                mem_datasets + gen_datasets,
                mem_labels + gen_labels,
            )
        # Combine results from the memory with generated results.
        return mem_datasets + gen_datasets, mem_labels + gen_labels

    def generate_training_samples(
        self, num_samples: int, ignore_memory: bool = False
    ) -> tuple[list[Dataset], list[bool]]:
        """
        Generate samples to train an attack.

        Parameters
        ----------
        num_samples: int
            The number of synthetic datasets to generate.
        ignore_memory: bool, default False
            Whether to use the memoized datasets, or ignore them.
        """
        return self._generate_samples(num_samples, True, ignore_memory)

    def test(
        self, attack: Attack, num_samples: int = 100, ignore_memory: bool = False
    ) -> tuple[list[int], list[int]]:
        """
        Test an attack against this threat model. This samples `num_samples`
        testing synthetic datasets along with labels revealing whether the
        target was part of the original dataset. It then runs the attack on
        each synthetic dataset. The labels and predicted labels are returned.

        Parameters
        ----------
        attack : Attack
            Attack to test.
        num_samples : int
            Number of test datasets to generate and test against.
        ignore_memory: bool, default False
            Whether to use the memoized datasets, or ignore them.

        Returns
        -------
        tuple(list(int), list(int))
            Tuple of (ground_truth, guesses), where ground_truth indicates
            what the attack needed to guess, and guesses are the attack's actual
            guesses.
        """
        test_datasets, truth_labels = self._generate_samples(
            num_samples, False, ignore_memory
        )
        pred_labels = attack.attack(test_datasets)
        return pred_labels, truth_labels
