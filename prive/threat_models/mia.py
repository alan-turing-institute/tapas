"""
Threat models for Membership Inference Attacks (MIA).

Membership inference attacks aim at detecting the presence of a specific
record in the training dataset from the synthetic dataset observed.

"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..attacks import Attack
    from ..datasets import Dataset
    from ..generators import Generator

from .base_classes import ThreatModel, TrainableThreatModel
from .attacker_knowledge import (
    AttackerKnowledgeOnData,
    AttackerKnowledgeOnGenerator,
    AttackerKnowledgeWithLabel,
    LabelInferenceThreatModel,
)

from ..report import MIAttackSummary

import numpy as np


class MIALabeller(AttackerKnowledgeWithLabel):
    """
    Randomly add a given target to the datasets sampled from auxiliary data.
    This class can be used to augment AttackerKnowledgeOnData objects that
    represent "generic" knowledge of the private dataset in order to use
    them for membership inference attacks.

    You may use this explicitly to feed into a LabelInferenceThreatModel, but
    this is meant mostly as an internal method to make MIAs.

    """

    def __init__(
        self,
        attacker_knowledge: AttackerKnowledgeOnData,
        target_records: Dataset,
        generate_pairs=True,
        replace_target=False,
    ):
        """
        Wrap an AttackerKnowledgeOnData object by appending a record.

        Parameters
        -----
        attacker_knowledge: AttackerKnowledgeOnData
            The data knowledge from which datasets are generated.
        target_records: Dataset
            The target records to append to the dataset. If several records
            are provided, these records are randomly added to the dataset
            independently from each other.
        generate_pairs: bool, default True
            Whether to output pairs of datasets differing only by the presence
            of the target record, or randomly choose for each dataset.
            If multiple targets are provided, then the pairs of datasets differ
            by exactly all of the multiple targets (as in, if a record x from
            the targets is in D, it is not in D', but the membership of each
            target is independent from the other targets).
        replace_target: bool, default False
            Whether to replace a record, instead of appending.

        """
        self.attacker_knowledge = attacker_knowledge
        self.target_records = target_records
        self.generate_pairs = generate_pairs
        self.replace_target = replace_target

    def generate_datasets_with_label(
        self, num_samples: int, training: bool = True
    ) -> tuple[list[Dataset], list[int]]:
        """
        Generate `num_samples` training or testing datasets with corresponding
        labels (arbitrary ints or bools).

        """
        # If generating pairs, make num_samples dividable by 2.
        if self.generate_pairs and num_samples // 2:
            num_samples += 1

        # Generate the datasets from the attacker knowledge.
        datasets = self.attacker_knowledge.generate_datasets(
            num_samples // 2 if self.generate_pairs else num_samples, training
        )

        # Compute modified datasets and corresponding labels by adding records
        # according to the labels. If self.generate_pairs, each iteration of
        # the loop creates two paired datasets.
        mod_datasets = []
        mod_labels = []
        for i_ds, dataset in enumerate(datasets):
            # Copy the datasets to be modified in place.
            dataset = dataset.copy()
            if self.generate_pairs:
                dataset2 = dataset.copy()
            # For each target, assign a random label.
            labels = np.random.randint(2, size=len(self.target_records)) == 1
            # If replace_target, then we first remove entries for
            if self.replace_target:
                # We first choose an entry e for each target record such that if
                # the target x is in the data, then e is removed (and vice versa).
                # This is to avoid replacing other target records. The reason we
                # do not use .replace_records is to be able to generate pairs.
                replace_indices = np.random.choice(
                    len(dataset), size=len(self.target_records), replace=False
                )
                # Remove the indices where label=1.
                dataset.drop_records(
                    [idx for idx, l in zip(replace_indices, labels) if l],
                    in_place=True,
                    n=0,  # Ensures that no records are dropped if the list is empty.
                )
                if self.generate_pairs:
                    # If generating pairs, remove indices where label=0 in dataset2.
                    dataset2.drop_records(
                        [idx for idx, l in zip(replace_indices, labels) if not l],
                        in_place=True,
                        n=0,  # Same as above.
                    )
            #
            for record, label in zip(self.target_records, labels):
                # If the label is 1, modify dataset.
                if label:
                    dataset.add_records(record, in_place=True)
                # If generating pairs and the label is 0, the label is 1 for
                # the other dataset in the pair. Modify dataset2
                elif self.generate_pairs:
                    dataset2.add_records(record, in_place=True)
            # Labels need to be converted, either as lists or int/float (if only one).
            _convert = list if len(self.target_records) > 1 else lambda x: x[0]
            mod_datasets.append(dataset)
            mod_labels.append(_convert(labels))
            if self.generate_pairs:
                mod_datasets.append(dataset2)
                mod_labels.append(_convert(labels == False))  # Negation.

        return mod_datasets, mod_labels

    @property
    def label(self):
        return self.attacker_knowledge.label


class TargetedMIA(LabelInferenceThreatModel):
    """
    This threat model implements a MIA with arbitrary attacker knowledge on
    data and generator.

    """

    def __init__(
        self,
        attacker_knowledge_data: AttackerKnowledgeOnData,
        target_records: Dataset,
        attacker_knowledge_generator: AttackerKnowledgeOnGenerator,
        generate_pairs: bool = True,
        replace_target: bool = False,
        memorise_datasets: bool = True,
        iterator_tracker: Callable[[list], Iterable] = None,
    ):
        LabelInferenceThreatModel.__init__(
            self,
            MIALabeller(
                attacker_knowledge_data, target_records, generate_pairs, replace_target
            ),
            attacker_knowledge_generator,
            memorise_datasets,
            iterator_tracker=iterator_tracker,
            num_labels=len(target_records),
        )
        # Save the target recordS, and the current record (0).
        if self.multiple_label_mode:
            # Since calling .get_records creates a new Dataset object every
            # time, and involves indices, we instead compute the records once
            # and for all.
            self._target_records = [r for r in target_records]
            # This sets self.target_record.
            self.set_label(0)
        else:
            self.target_record = target_records

    # Wrap the test method to output a MIAttackSummary.
    def test(
        self, attack: Attack, num_samples: int = 100, ignore_memory: bool = False,
    ) -> MIAttackSummary:
        """
        see prive.threat_models.LabelInferenceThreatModel.test for more information.
        """
        # Run the test method from LabelInferenceThreatModel, unchanged.
        pred_labels, truth_labels = LabelInferenceThreatModel.test(
            self, attack, num_samples, ignore_memory
        )
        # Post-process this as a MIAttackSummary.
        return MIAttackSummary(
            truth_labels,
            pred_labels,
            generator_info = self.atk_know_gen.label,
            attack_info = attack.label,
            dataset_info = self.atk_know_data.label,
            target_id = self.target_record.label,
        )

    def set_label(self, label):
        """
        If the attack is performed against multiple targets, this sets the
        target record to use when outputting labels.

        """
        # Use the parent class's set_label. The main reason we override this
        # method is to also modify self.target_record.
        LabelInferenceThreatModel.set_label(self, label)
        # We also set self.target_record, to be used by .
        self.target_record = self._target_records[label]
