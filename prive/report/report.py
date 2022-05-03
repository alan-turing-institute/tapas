"""Classes to summarise an attacks"""
from abc import ABC, abstractmethod
import numpy as np


class AttackSummary(ABC):

    @abstractmethod
    def calculate_metrics(self):
        """
        Calculate metrics relevant for an attack

        """
        pass

    @abstractmethod
    def write(self, output_path):
        """
        Write metrics to file.

        """
        pass


class MiAttackSummary(AttackSummary):

    def __init__(self, labels, predictions, generator_info, attack_info):
        """

        Parameters
        ----------
        labels
        predictions
        generator_info
        attack_info
        """

        self.labels = labels
        self.predictions = predictions
        self.generator_info = generator_info
        self.attack_info = attack_info

    @property
    def accuracy(self):
        return np.mean(np.array(self.predictions) == np.array(self.labels))

    def calculate_metrics(self, labels, predictions, generator_info, attack_info):
        pass

    def write(self):
        pass
