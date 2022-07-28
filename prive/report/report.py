"""
Classes to produce reports summarising the success of attacks.

Reports combine several attack summaries together to produce a more
comprehensive picture of the robustness of a generator against attacks.

"""

from abc import ABC, abstractmethod
import pandas as pd

from prive.report import MIAttackSummary
from .utils import metric_comparison_plots


class Report(ABC):
    @abstractmethod
    def compare(self, output_path):
        """
        Plot different generators for same attack-data.

        """
        pass


class MIAttackReport(Report):
    """
    Report and visualise performance of a series of Membership inference attacks.

    """

    # List of all metrics that can be used in a report.
    ALL_METRICS = [
        "accuracy",
        "true_positive_rate",
        "false_positive_rate",
        "mia_advantage",
        "privacy_gain",
    ]

    def __init__(self, summaries, metrics=None):
        """
        Initialise MIAttackReport class.

        Parameters
        ----------
        summaries: dataframe Dataframe where each row is the result of a given attack as obtained
        from the MIASummaryAttack class. The dataframe must have the following structure.
            Index:
                RangeIndex
            Columns:
                dataset: str
                target_id: str
                generator: str
                attack: str
                accuracy: float
                true_positive_rate: float
                false_positive_rate: float
                mia_advantage: float
                privacy_gain: float
            Alternatively, this can be passed as an iterable of MIAttackSummary objects, in
            which case .get_metrics() is called on each object, and the results are concatenated.

        metrics: list[str]
            List of metrics to be used in the report, these can be any of the following:
            "accuracy", "true_positive_rate", "false_positive_rate", "mia_advantage", "privacy_gain".
            If left as None, all metrics are used.


        """
        if not isinstance(summaries, pd.DataFrame):
            summaries = pd.concat([s.get_metrics() for s in summaries])
        self.attacks_data = summaries
        self.metrics = metrics or MIAttackReport.ALL_METRICS

    @classmethod
    def load_summary_statistics(cls, attacks, metrics=None):
        """
        Load attacks data, calculate summary statistics, merge into a single dataframe and initialise object.

        Parameters
        ----------
        attacks: list[dict]
            List of dictionaries with results of attacks. Each dictionary should contain the following keys:

            dict:
            labels: list[int]
                List with true labels of the target membership in the dataset.
            predictions: list[int]
                List with the predicted labels of the target membership in the dataset.
            generator_info: str
                Metadata with information about the method used to generate the dataset.
            attack_info: str
                Metadata with information about the attacked used to infer membership of the target on the dataset.
            dataset_info: str
                Metadata with information about the original raw dataset.
            target_id: str
                Metadata with information about the target record used on the attack.

        metrics: list[str] List of metrics to be included in the report, these can be any of the following:
        "accuracy", "true_positive_rate", "false_positive_rate", "mia_advantage", "privacy_gain".

        Returns
        -------
        MIAReport class

        """

        if metrics is None:
            metrics = [
                "accuracy",
                "true_positive_rate",
                "false_positive_rate",
                "mia_advantage",
                "privacy_gain",
            ]
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

    def compare(self, comparison_column, fixed_pair_columns, marker_column, filepath):
        """
        For a fixed pair of datasets-attacks-generators-target available in the data make a figure comparing
        performance between metrics. Options configure which dimension to compare against. Figures are saved to disk.

        Parameters
        ----------
        comparison_column: str
            Column in dataframe that be used to make point plot comparison in the x axis. It can be either: 'generator',
        'dataset', 'attack' or 'target_id'.
        fixed_pair_columns: list[str]
             Columns in dataframe to fix for a given figure in order to make meaningful comparisons. It can be any pair
        of the following:'generator', 'dataset', 'attack' or 'target_id'.
        marker_column: str
            Column in dataframe that be used to as marker in a point plot comparison. It can be either: 'generator',
        'attack' or 'target_id'.
        filepath: str
            Path where the figure is to be saved.

        Returns
        -------
        None

        """

        metric_comparison_plots(
            data=self.attacks_data,
            comparison_label=comparison_column,
            fixed_pair_label=fixed_pair_columns,
            metrics=self.metrics,
            marker_label=marker_column,
            output_path=filepath,
        )

        return None

    def create_report(self, filepath):
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

        # compare generators and target ids for fixed dataset-atacks
        self.compare("generator", ["dataset", "attack"], "target_id", filepath)

        # compare attacks and target ids for fixed dataset-generators
        self.compare("attack", ["dataset", "generator"], "target_id", filepath)

        # compare datasets and target ids for fixed attacks-generators
        self.compare("dataset", ["attack", "generator"], "target_id", filepath)

        # compare targets and generators ids for fixed attacks-dataset
        self.compare("target_id", ["dataset", "attack"], "generator", filepath)

        # compare targets and attacks ids for fixed dataset-generators
        self.compare("target_id", ["dataset", "generator"], "attack", filepath)

        print(f"All figures saved to directory {filepath}")
