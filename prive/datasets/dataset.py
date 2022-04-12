"""Abstract base classes to represent the data object"""
from abc import ABC, abstractmethod
import json
import pandas as pd


class Dataset(ABC):
    """Base class for the dataset object
    """

    def __init__(self, dataset, description):
        self.dataset = dataset
        self.description = description

    @abstractmethod
    def read(self, input_path):
        """
        Read dataset and description file
        """
        pass

    @abstractmethod
    def write(self, output_path):
        """
        Write dataset and description to file
        """
        pass

    @abstractmethod
    def sample(self, n_samples):
        """
        Sample from dataset a set of records.
        """
        pass

    @abstractmethod
    def get_record(self, record_ids):
        """
        Select and return a record(s).
        """
        pass

    @abstractmethod
    def drop_record(self, record_ids):
        """
        Drop a record(s) and return modified dataset.
        """
        pass

    @abstractmethod
    def add_record(self, record):
        """
        Add record(s) to dataset and return modified dataset.
        """
        pass

    @abstractmethod
    def replace(self, record_in, record_out):
        """
        Replace a row with a given row.
        """
        pass

    @abstractmethod
    def create_subsets(self, n, sample_size, record_drop=None):
        """ Create a number of training datasets (sub-samples from main dataset)
         of a given sample size and with the option to remove a record"""
        pass

    def __add__(self, other):
        """
        Adding two Dataset objects together
        """
        pass


class TabularDataset(Dataset):
    """
    Class for tabular dataset object. The tabular data is a Pandas Dataframe
    and the data description is a dictionary.
    """

    def __init__(self, dataset, description):
        super().__init__(dataset, description)

    @classmethod
    def read(cls, filepath):
        """
        Read csv and json files for dataframe and description dictionary respectively.

        Parameters
        ----------
        filepath (str): Path where the csv and json file are located

        Returns
        -------
        A TabularDataset instantiated object.

        """
        with open(f'{filepath}.json') as f:
            description = json.load(f)

        # TODO: something like this to determine types and columns to be read in the dataframe
        # dtypes = {cd['name']: _get_dtype(cd) for cd in self.description['columns']}
        # columns = self.description['columns']

        dataset = pd.read_csv(f'{filepath}.csv')

        return cls(dataset, description)

    def write(self, filepath):
        """
        Write dataset and description to file

        Parameters
        ----------
        filepath (str): Path where the csv and json file are saved
        """

        with open(f'{filepath}.csv', 'w') as fp:
            json.dump(dict, fp)

        self.dataset.to_csv(filepath)

    def sample(self, n_samples):
        """
        Sample from a TabularDataset object a set of records.

        Parameters
        ----------
        n_samples (int): Number of records to sample

        Returns
        -------

        A TabularDataset object with a sample of the records of the original object.


        """
        return TabularDataset(dataset=self.dataset.sample(n_samples), description=self.description)

    def get_record(self, record_ids):
        """
        Select and return a record(s).
        """
        pass

    def drop_record(self, record_ids):
        """
        Drop a record(s) and return modified dataset.
        """
        pass

    def add_record(self, record):
        """
        Add record(s) to dataset and return modified dataset.
        """
        pass

    def replace(self, record_in, record_out):
        """
        Replace a row with a given row.
        """
        pass

    def create_subsets(self, n, sample_size, record_drop=None):
        """ Create a number of training datasets (sub-samples from main dataset)
         of a given sample size and with the option to remove a recotd"""
        pass

    def __add__(self, other):
        """
        Adding two TabularDataset objects with the same data description together

        Parameters
        ----------
        other (TabularDataset): A TabularDataset object .

        Returns
        -------

        A TabularDataset object with the addition of two initial objects.


        """

        assert self.description == other.description, "Both datasets must have the same data description"

        return TabularDataset(pd.concat([self.dataset, other.dataset]), self.description)
