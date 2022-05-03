import json
from itertools import combinations

import numpy as np
import pandas as pd

from .constants import *


def get_num_features(meta_dict): # TODO: Write tests and refactor
    """
    Infers dimension of encoded data based on data's metadata dictionary.
    Continuous variables will require one dimension, whereas categorical
    and ordinal variables require one dimension per category.

    Args:
        meta_dict (dict) : Metadata dictionary containing information about
            a dataset: type of each column, range of continuous
            variables, range of categories for cateogrical variables.
            Formatted as the output of read_meta(metadata) for a metadata
            object.

    Returns:
        nfeatures (int) : Number of dimensions inferred from metadata.
    """
    nfeatures = 0

    for cname, cdict in meta_dict.items():
        data_type = cdict['type']
        if data_type == FLOAT or data_type == INTEGER:
            nfeatures += 1

        elif data_type == CATEGORICAL or data_type == ORDINAL:
            nfeatures += len(cdict['categories'])

        else:
            raise ValueError(f'Unkown data type {data_type} for attribute {cname}')

    return nfeatures


def encode_data(dataset, infer_ranges=False): # TODO: Write tests
    """
    Convert raw data to an np.ndarray with continuous features normalised and
    categorical features one-hot encoded.

    Parameters
    ----------
        dataset : TabularDataset
            Tabular dataset to encode.
        infer_ranges : bool
            If false, will use ranges provided in metadata,
            otherwise, will use input data to infer ranges of the continuous
            variables and will update metadata in-place with the new ranges.

    Returns
    -------
        encoded_data : np.ndarray
            Encoded data (normalised and one-hot encoded).

    """
    n_samples = len(dataset)
    nfeatures = dataset.description.encoded_dim
    encoded_data = np.empty((n_samples, nfeatures))
    cidx = 0

    for i, cdict in enumerate(dataset.description):
        name = cdict['name']
        d_type = cdict['type']
        d_repr = cdict['representation']
        col_data = dataset.data[i]

        if d_type == 'finite':
            if isinstance(d_repr, int):
                col_cats = list(range(d_repr))
            else:
                col_cats = d_repr

            col_data_onehot = one_hot(col_data, col_cats)
            encoded_data[:, cidx : cidx + len(col_cats)] = col_data_onehot

            cidx += len(col_cats)

        elif d_type == 'finite/ordered' and not isinstance(d_repr, int):
            def f(val):
                return d_repr.index(val)
            col_data_numeric = col_data.apply(f)
            encoded_data[:, cidx] = col_data_numeric

            cidx += 1

        else:
            encoded_data[:, cidx] = col_data
            cidx += 1

    return encoded_data


def one_hot(col_data, categories):
    col_data_onehot = np.zeros((len(col_data), len(categories)))
    cidx = [categories.index(c) for c in col_data]
    col_data_onehot[np.arange(len(col_data)), cidx] = 1

    return col_data_onehot

def index_split(max_index, split_size, num_splits):
    """
    Generate training indices without replacement. If max_index is smaller than
    is necessary to generate all splits without replacement, then multiple sets
    of indices will be generated, each without replacement. Logic is not currently
    implemented to ensure each index is in maximally many index splits.
    Args:
        max_index (int): Max index (size of dataset to split)
        split_size (int): Number of indices per split
        num_splits (int): Number of splits to make
    Returns:
        indices (List[np.ndarray]): List of numpy arrays of indices.
    """
    splits_per_repeat = max_index // split_size
    num_repeats = num_splits // splits_per_repeat
    remainder_splits = num_splits % splits_per_repeat
    indices = []
    for _ in range(num_repeats):
        index_array = np.arange(max_index)
        np.random.shuffle(index_array)
        indices += [index_array[split_size*i : split_size*(i+1)] for i in range(splits_per_repeat)]
    index_array = np.arange(max_index)
    np.random.shuffle(index_array)
    indices += [index_array[split_size*i : split_size*(i+1)] for i in range(remainder_splits)]

    return indices

def get_dtype(col_type, col_repr):
    """

    Return the pandas type of a column based on the json schema for the dataset.

    Parameters
    ----------
    col_type: str
        The abstract type of the data column (e.g. ``"finite"``,
        ``"countable/ordered"``, etc).
    col_repr: object
        Either a string, an integer, or a list of string.
        The interpretation depends upon ``col_type``.


    Returns
    -------
    dtype
        A type for the given column. Currently either int, float, or str.

    """
    ## All types that have some subset of the integers or the reals as their representation
    ## are stored as int or float. Applications should look at the schema to determine whether
    ## what subset it is and whether the type is ordered
    if col_repr == 'integer':
        return int
    elif col_repr == 'number':
        return float
    elif col_repr == 'string':
        return str
    elif col_repr == 'date' or col_repr == 'datetime':
        return str         
    elif col_type == 'finite' or col_type == 'finite/ordered':
        if isinstance(col_repr, list):
            return str
        elif isinstance(col_repr, int):
            return int
    else:
        raise RuntimeError("Unknown type/representation when parsing data")
