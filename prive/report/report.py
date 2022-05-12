"""Classes to report summary of attacks"""
from abc import ABC, abstractmethod
from prive.report import MIAttackSummary
import os
from prive.utils.plots import metric_comparison_plots
import pandas as pd
import datetime

class Report(ABC):
    @classmethod
    def get_summary_statistics(self):
        """
        Load attacks data, calculate summary statistics and initialise object.

        """
        pass

    @abstractmethod
    def compare_generators(self, output_path):
        """
        Plot different generators for same attack-data.

        """
        pass

    @abstractmethod
    def compare_attacks(self, output_path):
        """
        Plot different attacks for same generator-data.

        """
        pass

    @abstractmethod
    def compare_datasets(self, output_path):
        """
        Plot different datasets for same generator-attacks.

        """
        pass
    @abstractmethod
    def compare_datasets(self, output_path):
        """
        Plot different datasets for same generator-attacks.

        """
        pass


class MIAttackReport(Report):
    def __init__(self, df, metrics):
        """
        Initialise class

        Parameters
        ----------
        df: dataframe
            Dataframe where each row is the result of a given attack.


        """
        self.attacks_data = df
        self.metrics = metrics

    @classmethod
    def load_summary_statistics(cls, attacks,
                                metrics=None):
        """
        Load attacks data, calculate summary statistics, merge into a single dataframe and initialise object.

        Parameters
        ----------
        attacks: list[dicts]
            List of dictionaries with results of attacks
        metrics: list[str]

        Returns
        -------
        MIAReport class

        """

        if metrics is None:
            metrics = ["accuracy", "true_positive_rate", "false_positive_rate", "mia_advantage",
                       "privacy_gain"]
        df_list = []

        for attack in attacks:
            mia_summary = MIAttackSummary(
                attack["labels"],
                attack["predictions"],
                attack["generator"],
                attack["attack"],
                attack["dataset"],
                attack["target_id"],
            ).get_metrics()

            df_list.append(mia_summary)

        return cls(pd.concat(df_list), metrics)

    def compare_generators(self, filepath):
        """
        For each pair of datasets-attacks available in the data make a figure comparing performance between
        different generators and metrics. Figures are saved to disk.

        Parameters
        ----------
        filepath: str
            Path where the figure is to be saved.

        Returns
        -------
        None
        """

        metric_comparison_plots(data=self.attacks_data, comparison_label='generator', pairs_label=['dataset', 'attack'],
                                metrics=self.metrics, targets_label='target_id',
                                output_path=filepath)

        return None

    def compare_attacks(self, filepath):

        """

        For each pair of datasets-generators available in the data make a figure comparing performance between
        different attacks and metrics. Figures are saved to disk.

       Parameters
        ----------
        filepath: str
            Path where the figure is to be saved.

        Returns
        -------
        None

        """

        metric_comparison_plots(data=self.attacks_data, comparison_label='attack', pairs_label=['dataset', 'generator'],
                                metrics=self.metrics, targets_label='target_id',
                                output_path=filepath)

        return None

    def compare_datasets(self, filepath):

        """

        For each pair of attacks-generators available in the data make a figure comparing performance between
        different datasets and metrics. Figures are saved to disk.

       Parameters
        ----------
        filepath: str
            Path where the figure is to be saved.

        Returns
        -------
        None

        """

        metric_comparison_plots(data=self.attacks_data, comparison_label='dataset', pairs_label=['attack', 'generator'],
                                metrics=self.metrics, targets_label='target_id',
                                output_path=filepath)

        return None

    def create_report(self,filepath):
        """
        Make all comparison plots and save them to disk.

        Parameters
        ----------
        filepath: str
            Path where the figure is to be saved.

        Returns
        -------
        None

        """

        self.compare_generators(filepath)
        self.compare_attacks(filepath)
        self.compare_datasets(filepath)

        print (f'All figures saved to directory {filepath}')
