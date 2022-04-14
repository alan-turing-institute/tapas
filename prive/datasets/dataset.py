"""Classes to represent the data object"""
from sklearn.model_selection import ShuffleSplit
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
    def add_records(self, record):
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
         of a given sample size and with the option to remove some records"""
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

    def get_records(self, record_ids):
        """
        Get a record from the TabularDataset object

        Parameters
        ----------
        record_ids (List[int]): List of indexes of records to retrieve

        Returns
        -------

        A TabularDataset object with the record(s).

        """

        # TODO: what if the index is supposed to be a column? an identifier?
        return TabularDataset(self.dataset.iloc[record_ids], self.description)

    def drop_records(self, record_ids):
        """
        Get a record from the TabularDataset object

        Parameters
        ----------
        record_ids (List[int]): List of indexes of records to drop

        Returns
        -------

        A TabularDataset object without the record(s).

        """
        # TODO: what if the index is supposed to be a column? an identifier?
        return TabularDataset(self.dataset.drop(record_ids), self.description)

    def add_records(self, records):
        """
        Add record(s) to dataset and return modified dataset.

        Parameters
        ----------
        records (TabularDataset): A TabularDataset object with the record(s) to add.

        Returns
        -------

        A new TabularDataset object with the record(s).
        """

        # this does the same as the __add__
        return self.__add__(records)

    def replace(self, records_in, records_out):
        """

        Replace a record with another one in the dataset.

        Parameters
        ----------
        records_in (TabularDataset): A TabularDataset object with the record(s) to add.
        records_out (List(int)): List of indexes of records to drop

        Returns
        -------

        A modified TabularDataset object with the replaced record(s).


        """

        reduced_dataset = self.drop_records(records_out)

        return reduced_dataset.add_records(records_in)

        pass

    def create_subsets(self, n, sample_size, target_records):
        """
        Create a number of training datasets (sub-samples from main dataset)
         of a given sample size  with and without target records.

        Parameters
        ----------
        n (int): Number of datasets to create.
        sample_size (int) : Size of the subset datasets to be created
        target_records (List(int)): List of indexes of the target records.

        Returns
        -------

        Two List(TabularDataset) containing subsets of the data with and without the target record(s).

        """
        # remove target records
        dataset_notarget = self.drop_records(target_records)

        # create splits
        kf = ShuffleSplit(n_splits=n, train_size=sample_size)

        # list of TabularDataset without target record(s)
        without_target = []
        for train_index, _ in kf.split(dataset_notarget.dataset):
            without_target.append(self.get_records(train_index))

        # list of TabularDataset with target record(s), the size of these subsets will be sample_size +1
        # TODO: Do we want to replace instead of add?
        with_target = []
        for train_index, _ in kf.split(dataset_notarget.dataset):
            with_target.append(self.get_records(train_index) + self.get_records(target_records))

        return with_target, without_target

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
