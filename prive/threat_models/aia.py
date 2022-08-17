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
from ..report import AIAttackSummary, BinaryAIAttackSummary

import numpy as np


class AIALabeller(AttackerKnowledgeWithLabel):
    """
    Replace a record in the private dataset with a given target record,
    and randomly set the value of a given sensitive attribute in that record.

    """

    def __init__(
        self,
        attacker_knowledge: AttackerKnowledgeOnData,
        target_records: TabularRecord,
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
        target_records: Dataset
            The target records to add to the dataset with different sensitive
            attribute values. If this contains more than one record, the values
            for each record is sampled independently from all others.
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
        self.target_records = target_records
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
        # Sample target attributes i.i.d. for each record and dataset.
        all_labels = list(
            np.random.choice(
                self.attribute_values,
                size=(num_samples, len(self.target_records)),
                replace=True,
                p=self.distribution,
            )
        )
        # Modify the records with all possible values, and save the resulting
        # records for efficiency purposes.
        modified_records = []
        for r in self.target_records:
            L = {}
            for value in self.attribute_values:
                r = r.copy()
                r.set_value(self.sensitive_attribute, value)
                L[value] = r
            modified_records.append(L)
        # For each dataset, remove random records and add the modified records.
        mod_datasets = []
        for ds, labels in zip(datasets, all_labels):
            # Remove random entries from the dataset, all at once (so as to avoid
            # removing records added from target_records).
            ds = ds.drop_records(
                np.random.choice(len(ds), size=len(self.target_records), replace=False)
            )
            # Add records one by one, with corresponding label.
            for r, v, mod_r in zip(self.target_records, labels, modified_records):
                ds.add_records(mod_r[v], in_place=True)
            mod_datasets.append(ds)
        # Convert labels to a 1-dimensional list if only one target record is given.
        if len(self.target_records) == 1:
            all_labels = [l[0] for l in all_labels]
        # Replace the records in each dataset, and return the labels.
        return mod_datasets, all_labels

    @property
    def label(self):
        return self.attacker_knowledge.label


class TargetedAIA(LabelInferenceThreatModel):
    """
    This threat model implements a MIA with arbitrary attacker knowledge on
    data and generator.

    """

    def __init__(
        self,
        attacker_knowledge_data: AttackerKnowledgeOnData,
        target_record: TabularDataset,
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
            num_labels=len(target_record),
        )
        self.sensitive_attribute = sensitive_attribute
        self.attribute_values = attribute_values
        self.distribution = distribution
        # See mia.py for the following bit of code.
        if self.multiple_label_mode:
            self._target_records = [r for r in target_record]
            self.set_label(0)
        else:
            self.target_record = target_record

    # Wrap the test method to output a AIAttackSummary.
    def _wrap_output(self, truth_labels, pred_labels, scores, attack):
        # If only two values are possible, use the binary valued report.
        # The second value is treated as the positive label.
        if len(self.attribute_values) == 2:
            ReportClass = BinaryAIAttackSummary
            kwargs = {"positive_value": self.attribute_values[1]}
        # Otherwise, we use the more general class.
        else:
            ReportClass = AIAttackSummary
            kwargs = {}
        return ReportClass(
            truth_labels,
            pred_labels,
            scores,
            generator_info=self.atk_know_gen.label,
            attack_info=attack.label,
            dataset_info=self.atk_know_data.label,
            target_id=self.target_record.label,
            sensitive_attribute=self.sensitive_attribute,
            **kwargs
        )

    def set_label(self, label):
        """
        If the attack is performed against multiple targets, this sets the
        target record to use when outputting labels.

        """
        # See mia.py for the following bit of code.
        LabelInferenceThreatModel.set_label(self, label)
        self.target_record = self._target_records[label]
