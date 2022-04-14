"""Classes to represent the data object"""
from prive.utils.data import index_split
from abc import ABC, abstractmethod
import json
import pandas as pd
import numpy as np


class Dataset(ABC):
    """
    Base class for the dataset object.

    """

    @abstractmethod
    def read(self, input_path):
        """
        Read dataset and description file.

        """
        pass

    @abstractmethod
    def write(self, output_path):
        """
        Write dataset and description to file.

        """
        pass

    @abstractmethod
    def sample(self, n_samples):
        """
        Sample from dataset a set of records.

        """
        pass

    @abstractmethod
    def get_records(self, record_ids):
        """
        Select and return a record(s).

        """
        pass

    @abstractmethod
    def drop_records(self, record_ids):
        """
        Drop a record(s) and return modified dataset.

        """
        pass

    @abstractmethod
    def add_records(self, records):
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
    def create_subsets(self, n, sample_size, drop_records=None):
        """
        Create a number of training datasets (sub-samples from main dataset)
        of a given sample size and with the option to remove some records.

        """
        pass

    def __add__(self, other):
        """
        Adding two Dataset objects together.

        """
        pass


class TabularDataset(Dataset):
    """
    Class for tabular dataset object. The tabular data is a Pandas Dataframe
    and the data description is a dictionary.

    """
    def __init__(self, dataset, description):
        self.dataset = dataset
        self.description = description

    @classmethod
    def read(cls, filepath):
        """
        Read csv and json files for dataframe and description dictionary respectively.

        Parameters
        ----------
        filepath: str
            Path where the csv and json file are located.

        Returns
        -------
        TabularDataset
            A TabularDataset instantiated object.

        """
        with open(f'{filepath}.json') as f:
            description = json.load(f)

        # TODO: something like this to determine types
        #  and columns to be read in the dataframe once we ha a defined json
        # dtypes = {cd['name']: _get_dtype(cd) for cd in self.description['columns']}
        # columns = self.description['columns']

        dataset = pd.read_csv(f'{filepath}.csv')

        return cls(dataset, description)

    def write(self, filepath):
        """
        Write dataset and description to file

        Parameters
        ----------
        filepath: str
            Path where the csv and json file are saved.

        """

        with open(f'{filepath}.csv', 'w') as fp:
            json.dump(dict, fp)

        self.dataset.to_csv(filepath)

    def sample(self, n_samples):
        """
        Sample from a TabularDataset object a set of records.

        Parameters
        ----------
        n_samples: int
            Number of records to sample.

        Returns
        -------
        TabularDataset
            A TabularDataset object with a sample of the records of the original object.

        """
        return TabularDataset(dataset=self.dataset.sample(n_samples), description=self.description)

    def get_records(self, record_ids):
        """
        Get a record from the TabularDataset object

        Parameters
        ----------
        record_ids: list[int]
            List of indexes of records to retrieve.

        Returns
        -------
        TabularDataset
            A TabularDataset object with the record(s).

        """

        # TODO: what if the index is supposed to be a column? an identifier?
        return TabularDataset(self.dataset.iloc[record_ids], self.description)

    def drop_records(self, record_ids=[]):
        """
        Drop records from the TabularDataset object, if record_ids is empty it will drop a random record.

        Parameters
        ----------
        record_ids: list[int]
            List of indexes of records to drop.

        Returns
        -------
        TabularDataset
            A TabularDataset object without the record(s).

        """
        if len(record_ids)==0:
            # drop a random record
            return TabularDataset(self.dataset.drop(np.random.randint(self.dataset.shape[0], size=1)), self.description)


        return TabularDataset(self.dataset.drop(record_ids), self.description)

    def add_records(self, records):
        """
        Add record(s) to dataset and return modified dataset.

        Parameters
        ----------
        records: TabularDataset
            A TabularDataset object with the record(s) to add.

        Returns
        -------
        TabularDataset
            A new TabularDataset object with the record(s).

        """

        # this does the same as the __add__
        return self.__add__(records)

    def replace(self, records_in, records_out=[]):
        """

        Replace a record with another one in the dataset, if records_out is empty it will remove a random record.

        Parameters
        ----------
        records_in: TabularDataset
            A TabularDataset object with the record(s) to add.
        records_out: list(int)
            List of indexes of records to drop.

        Returns
        -------
        TabularDataset
            A modified TabularDataset object with the replaced record(s).

        """

        reduced_dataset = self.drop_records(records_out)

        return reduced_dataset.add_records(records_in)

        pass

    def create_subsets(self, n, sample_size):
        """
        Create a number of training datasets (sub-samples from main dataset)
        of a given sample size  with and without target records.

        Parameters
        ----------
        n: int
            Number of datasets to create.
        sample_size: int
            Size of the subset datasets to be created.

        Returns
        -------
        list(TabularDataset)
            A lists containing subsets of the data with and without the target record(s).

        """

        # create splits
        splits = index_split(self.dataset.shape[0],sample_size,n)

        # list of TabularDataset without target record(s)
        subsamples = [self.get_records(train_index) for train_index in splits]

        return subsamples

    def __add__(self, other):
        """
        Adding two TabularDataset objects with the same data description together

        Parameters
        ----------
        other: (TabularDataset)
            A TabularDataset object .

        Returns
        -------
        TabularDataset
            A TabularDataset object with the addition of two initial objects.


        """

        assert self.description == other.description, "Both datasets must have the same data description"

        return TabularDataset(pd.concat([self.dataset, other.dataset]), self.description)
