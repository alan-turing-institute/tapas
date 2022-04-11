"""Abstract base classes to represent the data object"""
from abc import ABC, abstractmethod
import json
import pandas as pd

class Dataset(ABC):

    """Base class for the dataset object """

    def __init__(self, dataset=None, description = None):
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
         of a given sample size and with the option to remove a recotd"""
        pass

    def __add__(self, other):
        """
        Adding two Dataset objects toguether
        """
        pass



class TabularDataset(Dataset):

    """Base class for data objects """
    def __init__(self, input_path=None,dataset=None, description=None):
        if input_path:
            self.read(input_path)
        else:
            self.dataset = dataset
            self.description = description


    def read(self, input_path):
        """
        Read csv and json files dataframe and and dictionary respectively.

        :param input_path: A string with the path where the csv and json file are located
        :return: A pandas dataframe and a dictionary with data description.
        """
        with open(f'{input_path}.json') as f:
            self.description = json.load(f)


        #TODO: something like this to determine types and columns to be read in the dataframe
        #dtypes = {cd['name']: _get_dtype(cd) for cd in self.description['columns']}
        #columns = self.description['columns']

        self.dataset = pd.read_csv(f'{input_path}.csv')


    def write(self, output_path):
        """
        Write dataset and description to file

        :param output_path: A string with the path where the csv and json file are going to be saved.
        :return: None
        """
        with open(f'{output_path}.csv', 'w') as fp:
            json.dump(dict, fp)

        self.dataset.to_csv(output_path)


    def sample(self, n_samples):
        """
        Sample from dataframe a set of records.
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

        :param other: A TabularDataset object.
        :return: A new TabularDataset object that contains both objects.
        
        """
        assert self.description == other.description, "Both datasets must have the same data description"

        return TabularDataset(dataset=pd.concat([self.dataset,other.dataset]),description=self.description)


