"""Classes to report summary of attacks"""
from abc import ABC, abstractmethod
from prive.report import MIAttackSummary
import pandas as pd


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


class MIAttackReport(Report):
    def __init__(self, df):
        """
        Initialise class

        Parameters
        ----------
        df: dataframe
            Dataframe where each row is the result of a given attack.


        """
        self.attacks_data = df

    @classmethod
    def load_summary_statistics(cls, attacks):
        """
        Load attacks data, calculate summary statistics, merge into a single dataframe and initialise object.

        Parameters
        ----------
        attacks: list[dicts]
            List of dictionaries with results of attacks

        Returns
        -------
        MIAReport class

        """

        df_list = []

        for attack in attacks:

            mia_summary = MIAttackSummary(
                attack["labels"],
                attack["predictions"],
                attack["generator_info"],
                attack["attack_info"],
                attack["dataset_info"],
                attack["target_id"],
            ).get_metrics()

            df_list.append(mia_summary)

        return cls(pd.concat(df_list))

    def compare_generators(self, output_path):

        pass

    def compare_attacks(self, output_path):

        pass

    def compare_datasets(self, output_path):

        pass
