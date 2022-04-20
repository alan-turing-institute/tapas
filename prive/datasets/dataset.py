"""Classes to represent the data object"""
from prive.utils.data import index_split, get_dtype
from abc import ABC, abstractmethod
import json

import numpy as np
import pandas as pd

from prive.utils.data import index_split

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

    @abstractmethod
    def __add__(self, other):
        """
        Adding two Dataset objects together.

        """
        pass

    @abstractmethod
    def __iter__(self):
        """
        Returns an iterator over records in the dataset.
        """
        pass


class TabularDataset(Dataset):
    """
    Class for tabular dataset object. The tabular data is a Pandas Dataframe
    and the data description is a dictionary.

    """

    def __init__(self, data, description):
        self.data = data
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

        dtypes = {i: get_dtype(cd['representation'], cd['type']) for i, cd in enumerate(description)}

        data = pd.read_csv(f'{filepath}.csv', header=None, dtype=dtypes, index_col=None)

        return cls(data, description)

    def write(self, filepath):
        """
        Write data and description to file

        Parameters
        ----------
        filepath: str
            Path where the csv and json file are saved.

        """

        with open(f'{filepath}.json', 'w') as fp:
            json.dump(self.description, fp)

        self.data.to_csv(filepath+'.csv')

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
        return TabularDataset(data=self.data.sample(n_samples), description=self.description)

    def get_records(self, record_ids):
        """
        Get a record from the TabularDataset object

        Parameters
        ----------
        record_ids : list[int]
            List of indexes of records to retrieve.

        Returns
        -------
        TabularDataset
            A TabularDataset object with the record(s).

        """

        # TODO: what if the index is supposed to be a column? an identifier?
        return TabularDataset(self.data.iloc[record_ids], self.description)

    def drop_records(self, record_ids=[], n=1, in_place=False):
        """
        Drop records from the TabularDataset object, if record_ids is empty it will drop a random record.

        Parameters
        ----------
        record_ids : list[int]
            List of indexes of records to drop.
        in_place : bool
            Bool indicating whether or not to change the dataset in-place or return
            a copy. If True, the dataset is changed in-place. The default is False.

        Returns
        -------
        TabularDataset
            A TabularDataset object without the record(s).

        """
        if len(record_ids) == 0:
            # drop n random records if none provided
            record_ids = np.random.randint(self.data.shape[0], size=n).tolist()

        new_data = self.data.drop(record_ids)

        if in_place:
            self.data = new_data
            return

        return TabularDataset(new_data, self.description)

    def add_records(self, records, in_place=False):
        """
        Add record(s) to dataset and return modified dataset.

        Parameters
        ----------
        records : TabularDataset
            A TabularDataset object with the record(s) to add.
        in_place : bool
            Bool indicating whether or not to change the dataset in-place or return
            a copy. If True, the dataset is changed in-place. The default is False.

        Returns
        -------
        TabularDataset
            A new TabularDataset object with the record(s).

        """

        if in_place:
            assert self.description == records.description, "Both datasets must have the same data description"

            self.data = pd.concat([self.data, records.data])
            return

        # if not in_place this does the same as the __add__
        return self.__add__(records)

    def replace(self, records_in, records_out=[], in_place=False):
        """
        Replace a record with another one in the dataset, if records_out is empty it will remove a random record.

        Parameters
        ----------
        records_in: TabularDataset
            A TabularDataset object with the record(s) to add.
        records_out: list(int)
            List of indexes of records to drop.
        in_place : bool
            Bool indicating whether or not to change the dataset in-place or return
            a copy. If True, the dataset is changed in-place. The default is False.

        Returns
        -------
        TabularDataset
            A modified TabularDataset object with the replaced record(s).

        """
        if len(records_out) > 0:
            assert len(records_out) == len(records_in), \
                f'Number of records out must equal number of records in, got {len(records_out)}, {len(records_in)}'

        # TODO: Should multiple records_in with no records_out be supported?
        if in_place:
            self.drop_records(records_out, in_place=in_place)
            self.add_records(records_in, in_place=in_place)
            return

        # pass n as a back-up in case records_out=[]
        reduced_dataset = self.drop_records(records_out, n=len(records_in))

        return reduced_dataset.add_records(records_in)

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
        splits = index_split(self.data.shape[0], sample_size, n)

        # list of TabularDataset without target record(s)
        subsamples = [self.get_records(train_index) for train_index in splits]

        return subsamples

    def __add__(self, other):
        """
        Adding two TabularDataset objects with the same data description together

        Parameters
        ----------
        other: (TabularDataset)
            A TabularDataset object.

        Returns
        -------
        TabularDataset
            A TabularDataset object with the addition of two initial objects.

        """

        assert self.description == other.description, "Both datasets must have the same data description"

        return TabularDataset(pd.concat([self.data, other.data]), self.description)

    def __iter__(self):
        """
        Returns an iterator over records in this dataset,

        Returns
        -------
        iterator
            An iterator object that iterates over individual records, as TabularDatasets.

        """
        # iterrows() returns tuples (index, record), and map applies a 1-argument
        # function to the iterable it is given, hence why we have idx_and_rec
        # instead of the cleaner (idx, rec).
        convert_record = lambda idx_and_rec: TabularDataset(
            # iterrows() outputs pd.Series rather than .DataFrame, so we convert here:
            data=idx_and_rec[1].to_frame().T, description=self.description)
        return map(convert_record, self.data.iterrows())

    def __len__(self):
        """
        Returns the number of records in this dataset.

        Returns
        -------
        integer
            length: number of records in this dataset.

        """
        return self.data.shape[0]

    def __contains__(self, item):
        """
        Determines the truth value of `item in self`. The only items considered
        to be in a TabularDataset are the rows, treated as 1-row TabularDatasets.

        Parameters
        ----------
        item : Object
            Object to check membership of.

        Returns
        -------
        bool
            Whether or not item is considered to be contained in self.

        """
        if not isinstance(item, TabularDataset):
            raise ValueError(f'Only TabularDatasets can be checked for containment, not {type(item)}')
        if len(item) != 1:
            raise ValueError(f'Only length-1 TabularDatasets can be checked for containment, got length {len(item)})')

        return (self.data == item.data.iloc[0]).all(axis=1).any()