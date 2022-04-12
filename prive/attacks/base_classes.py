"""Abstract base classes for various privacy attacks."""

from abc import ABC, abstractmethod
from ..threat_models.base_classes import ThreatModel


class Attack(ABC):
    """
    Abstract base class for all privacy attacks.

    This class defines (only) three common elements of attacks:
     - a threat model: all attacks make some assumptions on the attacker,
         that are captured by the threat model.
     - a .train method (that can be left empty), that selects parameters for
         the attack to make decisions.
     - a .confidence method, that makes a decision for a (list of) dataset(s).
         This also has a companion .score method that can be ignored if not
         meaningful, but can be useful for deeper analysis of attacks.
    """

    def __init__(self, threat_model: ThreatModel):
        self.threat_model = threat_model

    @abstractmethod
    def train(self):
        """Train parameters of the attack."""
        pass

    @abstractmethod
    def attack(self, datasets):
        """Perform the attack on each dataset in a list and return a
           (discrete) decision."""
        pass

    @abstractmethod
    def attack_score(self, datasets):
        """Perform the attack on each dataset in a list, but return
           a confidence score (specifically for classification tasks)."""
        pass


# TODO(design)
# For later discussion: not sure if we want to keep classes for different types
#  of attacks. For instance, we can implement the Groundhog attack for MIAs and
#  AIAs with the exact same code (since the sampling logic of train/test is in
#  the threat model).

# class MIAttack(Attack):
#     """
#     Abstract base class for membership inference attacks
#     """
#     @abstractmethod
#     def attack(self, target, priv_output, *args, **kwargs):
#         """Infer presence of target in (training data that generated) priv_output"""
#         pass


# class AIAttack(Attack):
#     """
#     Abstract base class for attribute inference attacks
#     """
#     @abstractmethod
#     def attack(self, target, priv_output, target_cols, *args, **kwargs):
#         pass