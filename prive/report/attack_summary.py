"""Classes to summarise an attack"""
import os
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class AttackSummary(ABC):
    @abstractmethod
    def get_metrics(self):
        """
        Calculate metrics relevant for an attack

        """
        pass

    @abstractmethod
    def write_metrics(self, output_path):
        """
        Write metrics to file.

        """
        pass


class MIAttackSummary(AttackSummary):
    def __init__(
        self, labels, predictions, generator_info, attack_info, dataset_info, target_id
    ):
        """

        Initialise the MIAttackSummary Class.

        Parameters
        ----------
        labels
        predictions
        generator_info
        attack_info
        dataset_info
        target_id
        """

        self.labels = np.array(labels)
        self.predictions = np.array(predictions)
        self.generator_info = generator_info
        self.attack_info = attack_info
        self.dataset_info = dataset_info
        self.target_id = target_id

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
        return np.sum(self.predictions[targetin] == 1) / len(targetin)

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

    @property
    def mia_advantage(self):
        """
        MIA attack advantage as defined by Stadler et al.

        Returns
        -------
        float

        """
        return self.tp - self.fp

    @property
    def privacy_gain(self):
        """
        Privacy gain as defined by Stadler et al.

        Returns
        -------
        float

        """
        return 1 - self.mia_advantage

    def get_metrics(self):
        """
        Calculates all MIA relevant metrics and returns it as a dataframe.

        Returns
        -------
        A dataframe
            A dataframe with attack info and metrics

        """

        return pd.DataFrame(
            [
                [
                    self.dataset_info,
                    self.target_id,
                    self.generator_info,
                    self.attack_info,
                    self.accuracy,
                    self.tp,
                    self.fp,
                    self.mia_advantage,
                    self.privacy_gain,
                ]
            ],
            columns=[
                "dataset",
                "target_id",
                "generator_info",
                "attack_info",
                "accuracy",
                "true_positive_rate",
                "false_positive_rate",
                "mia_advantage",
                "privacy_gain",
            ],
        )

    def write_metrics(self, filepath, attack_iter):
        """
        Write metrics to a CSV file

        Parameters
        ----------
        filepath: str
            Path where the CSV is to be saved.
        attack_iter: int
            id of the iteration of the attack (to distinguish file from others).

        Returns
        -------
        None

        """

        file_name = f"result_{self.dataset_info}_{self.attack_info}_{self.generator_info}_Target{self.target_id}_{attack_iter}.csv"

        self.get_metrics().to_csv(os.path.join(filepath, file_name), index=False)
