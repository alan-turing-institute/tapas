"""
Abstract base classes for various privacy attacks.

"""

from abc import ABC, abstractmethod


class Attack(ABC):
    """
    Abstract base class for all privacy attacks.

    This class defines (only) three common elements of attacks:

    * a .train method (that can be left empty), that selects parameters for
      the attack to make decisions.
    * a .attack method, that makes a binary decision for a (list of) dataset(s).
    * a .attack_score method that can be ignored if not meaningful, but can be
      useful for deeper analysis of attacks.

    """

    @abstractmethod
    def train(self, threat_model):
        """
        Train parameters of the attack.

        """
        pass

    @abstractmethod
    def attack(self, datasets):
        """
        Perform the attack on each dataset in a list and return a (discrete) decision.

        """
        pass

    @abstractmethod
    def attack_score(self, datasets):
        """
        Perform the attack on each dataset in a list, but return a confidence
        score (specifically for classification tasks).

        """
        pass

    @property
    def label(self):
        """
        A label to describe this attack in reports.

        """
        return "Unnamed Attack"

    def __str__(self):
        return self.label
