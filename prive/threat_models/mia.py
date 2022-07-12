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
from .attacker_knowledge import (
    AttackerKnowledgeOnData,
    AttackerKnowledgeOnGenerator,
    AttackerKnowledgeWithLabel,
    LabelInferenceThreatModel,
)

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

    def generate_datasets_with_label(
        self, num_samples: int, training: bool = True
    ) -> tuple[list[Dataset], list[int]]:
        """
        Generate `num_samples` training or testing datasets with corresponding
        labels (arbitrary ints or bools).

        """
        # Generate the datasets from the attacker knowledge.
        datasets = self.attacker_knowledge.generate_datasets(num_samples, training)
        if self.generate_pairs:
            # If pairs are required, duplicate the list and assign labels 0 and 1.
            datasets = datasets + datasets
            labels = [0] * num_samples + [1] * num_samples
        else:
            # Pick random labels for each dataset.
            labels = list(np.random.random(size=(num_samples,)) <= 0.5)
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
        return app_datasets, labels


class TargetedMIA(LabelInferenceThreatModel):
    """
    This threat model implements a MIA with arbitrary attacker knowledge on
    data and generator.

    """

    def __init__(
        self,
        attacker_knowledge_data: AttackerKnowledgeOnData,
        target_record: Dataset,
        attacker_knowledge_generator: AttackerKnowledgeOnGenerator,
        generate_pairs: bool = True,
        replace_target: bool = False,
        memorise_datasets: bool = True,
        iterator_tracker: Callable[[list], Iterable] = None,
    ):
        LabelInferenceThreatModel.__init__(
            self,
            MIALabeller(
                attacker_knowledge_data, target_record, generate_pairs, replace_target
            ),
            attacker_knowledge_generator,
            memorise_datasets,
            iterator_tracker=iterator_tracker,
        )
        self.target_record = target_record
