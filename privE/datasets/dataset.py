"""Abstract base classes to represent the data object"""
from abc import ABC, abstractmethod
import json
import pandas as pd

class Dataset(ABC):

    """Base class for the dataset object """

    def __init__(self):
        self.description = None
        self.dataset = None

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
    # For convenience, we also map the __call__ function to .read.
    def __call__(self, input_path):
        return self.read(input_path)

