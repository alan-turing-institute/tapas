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


class MIAttackSummary(AttackSummary):

    def __init__(self, labels, predictions, generator_info, attack_info):
        """

        Parameters
        ----------
        labels
        predictions
        generator_info
        attack_info
        """

        self.labels = np.array(labels)
        self.predictions = np.array(predictions)
        self.generator_info = generator_info
        self.attack_info = attack_info

    @property
    def accuracy(self):
        """
        Accuracy of the attacks based on the rate of correct predictions.

        Returns
        -------
        float

        """
        return np.mean(self.predictions == self.labels)

    @property
    def tp(self):
        """
        True positives based on rate of attacks where the target is correctly inferred
        as being in the sample.

        Returns
        -------
        float

        """
        targetin = np.where(self.labels == 1)[0]
        return np.sum(self.predictions[targetin] == 1)/len(targetin)

    @property
    def fp(self):
        """
        False positives based on rate of attacks where the target is incorrectly inferred
        as being in the sample.

        Returns
        -------
        float

        """
        targetout = np.where(self.labels == 0)[0]
        return np.sum(self.predictions[targetout] == 1) / len(targetout)

    def calculate_metrics(self, labels, predictions):
        pass

    def write(self):
        pass
