"""
Threat Models for Attribute Inference Attacks.

Attribute Inference Attacks aim at inferring the value of a sensitive attribute
for a target user, given some known attributes and the synthetic data.

"""

# Type checking stuff
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..attacks import Attack  # for typing
    from ..datasets import TabularRecord
    from ..generators import Generator  # for typing

from .base_classes import ThreatModel, TrainableThreatModel
from .attacker_knowledge import (
    AttackerKnowledgeOnData,
    AttackerKnowledgeOnGenerator,
    AttackerKnowledgeWithLabel,
    LabelInferenceThreatModel,
)

import numpy as np


class AIALabeller(AttackerKnowledgeWithLabel):
    """
    Replace a record in the private dataset with a given target record,
    and randomly set the value of a given sensitive attribute in that record.

    """

    def __init__(
        self,
        attacker_knowledge: AttackerKnowledgeOnData,
        target_record: TabularRecord,
        sensitive_attribute: str,
        attribute_values: list,
        distribution: list = None,
    ):
        """
        Wrap an AttackerKnowledgeOnData object by appending a record with
        randomized sensitive attribute

        Parameters
        ----------
        attacker_knowledge: AttackerKnowledgeOnData
            The data knowledge from which datasets are generated.
        target_record: Dataset
            The target record to add to the dataset.
        sensitive_attribute: str
            The name of the attribute to randomise.
        attribute_values: list
            All values that the attribute can take.
        distribution: list (None as default)
            Distribution from which to sample attribute values, a list of real
            numbers in [0,1] which sums to 1. By default (None), the uniform
            distribution is used.
        """
        self.attacker_knowledge = attacker_knowledge
        self.target_record = target_record
        self.sensitive_attribute = sensitive_attribute
        self.attribute_values = attribute_values
        self.distribution = distribution

    def generate_datasets_with_label(
        self, num_samples: int, training: bool = True
    ) -> tuple[list[Dataset], list[int]]:
        """
        Generate `num_samples` training or testing datasets with corresponding
        labels (arbitrary ints or bools).

        """
        # Generate the datasets from the attacker knowledge.
        datasets = self.attacker_knowledge.generate_datasets(num_samples, training)
        # Sample target attributes iid.
        labels = list(
            np.random.choice(
                self.attribute_values,
                size=(num_samples,),
                replace=True,
                p=self.distribution,
            )
        )
        # Modify the record with all possible values, and save.
        modified_records = {}
        for value in self.attribute_values:
            r = self.target_record.copy()
            r.set_value(self.sensitive_attribute, value)
            modified_records[value] = r
        # Replace the records in each dataset, and return the labels.
        return (
            [
                ds.replace(modified_records[v], in_place=False)
                for ds, v in zip(datasets, labels)
            ],
            labels,
        )


class TargetedAIA(LabelInferenceThreatModel):
    """
    This threat model implements a MIA with arbitrary attacker knowledge on
    data and generator.

    """

    def __init__(
        self,
        attacker_knowledge_data: AttackerKnowledgeOnData,
        target_record: TabularRecord,
        sensitive_attribute: str,
        attribute_values: list,
        attacker_knowledge_generator: AttackerKnowledgeOnGenerator,
        distribution: list = None,
        memorise_datasets: bool = True,
        iterator_tracker: Callable[[list], Iterable] = None,
    ):
        LabelInferenceThreatModel.__init__(
            self,
            AIALabeller(
                attacker_knowledge_data,
                target_record,
                sensitive_attribute,
                attribute_values,
                distribution,
            ),
            attacker_knowledge_generator,
            memorise_datasets,
            iterator_tracker=iterator_tracker,
        )
        self.target_record = target_record
        self.sensitive_attribute = sensitive_attribute
        self.attribute_values = attribute_values
        self.distribution = distribution
