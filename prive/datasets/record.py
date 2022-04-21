from prive.datasets import TabularDataset


class TabularRecord(TabularDataset):
    """
     Class for tabular record object. The tabular data is a Pandas Dataframe with 1 row
     and the data description is a dictionary.

     """

    def __init__(self, data, description):
        super().__init__(data, description)
