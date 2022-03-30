"""Abstract base classes for various privacy attacks"""
from abc import ABC, abstractmethod

class Attack(ABC):
    """
    Abstract base class for all privacy attacks
    """
    @abstractmethod
    def train(self, *args):
        """Train attack model"""
        pass

    @abstractmethod
    def attack(self, *args):
        """Perform attack"""
        pass


class MIAttack(Attack):
    """
    Abstract base class for membership inference attacks
    """
    @abstractmethod
    def attack(self, target, priv_output, *args, **kwargs):
        """Infer presence of target in (training data that generated) priv_output"""
        pass


class AIAttack(Attack):
    """
    Abstract base class for attribute inference attacks
    """
    @abstractmethod
    def attack(self, target, priv_output, target_cols, *args, **kwargs):
        pass


class Classifier(ABC):
    """
    Abstract base class for a classifier
    """
    @abstractmethod
    def fit(X, y, *args, **kwargs):
        """Fit classifier to data"""
        pass

    @abstractmethod
    def predict(X, *args, **kwargs):
        """Predict classes from input data"""
        pass


class GroundhogClassifier(Classifier):
    """
    Abstract base class for a classifier that can be used in the groundhog attack
    """
    pass