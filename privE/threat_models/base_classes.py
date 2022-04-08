"""Threat Models describe the assumptions under which an attack takes place.

Threat Models are composed of three elements:
 1. What the attacker aims to infer (e.g., membership, attribute).
 2. What the attacker knows about the generator (no-, black-, white-box).
 3. What the attacker knows about the training dataset.

A threat model thus describes what an attack wants to predict, and how it
can do that. It also describes how to _evaluate_ the success of an attack.
For instance, for a black-box membership inference attack with auxiliary data,
the attacker is able to run the generator on datasets samples from the
auxiliary data, which may or may not contain a target record. The evaluation
of the attack is performed on datasets from a disjoint dataset (test set),
from which training datasets are sampled, with or without the target record.
"""

from abc import ABC, abstractmethod

class ThreatModel(ABC):
    """Abstract base class for a threat model.

    A threat model describes the conditions under which an attack takes place:
     * What the attacker is trying to learn.
     * What the attacker knows about the method.
     * What the attacker knows about the training dataset.
    """

    @abstractmethod
    def generate_training_samples(self, num_samples):
        pass

    @abstractmethod
    def generate_testing_samples(self, num_samples):
        pass