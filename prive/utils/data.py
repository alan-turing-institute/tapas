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


def encode_data(data, meta_dict, infer_ranges=False): # TODO: Write tests
    """
    Convert raw data to an np.ndarray with continuous features normalised and
    categorical features one-hot encoded.

    Args:
        data (pd.DataFrame) : Raw data to convert.
        meta_dict (dict) : Metadata dictionary containing information about
            each column of the data: type of each column, range of continuous
            variables, range of categories for cateogrical variables.
        infer_ranges (bool) : If false, will use ranges provided in metadata,
            otherwise, will use input data to infer ranges of the continuous
            variables and will update metadata in-place with the new ranges.

    Returns:
        encoded_data (np.ndarray) : Encoded data (normalised and one-hot encoded).
    """
    n_samples = len(data)
    nfeatures = get_num_features(meta_dict)
    encoded_data = np.empty((n_samples, nfeatures))
    cidx = 0

    for attr_name, cdict in meta_dict.items():
        data_type = cdict['type']
        col_data = data[attr_name]

        if data_type == FLOAT or data_type == INTEGER:
            # Normalise continuous data
            if infer_ranges:
                col_max = max(col_data)
                col_min = min(col_data)

                meta_dict[attr_name]['max'] = col_max
                meta_dict[attr_name]['min'] = col_min

            else:
                col_max = cdict['max']
                col_min = cdict['min']

            encoded_data[:, cidx] = np.array(
                np.true_divide(col_data - col_min, col_max + ZERO_TOL)
            )

            cidx += 1

        elif data_type == CATEGORICAL or data_type == ORDINAL:
            # One-hot encoded categorical columns
            col_cats = cdict['categories']
            col_data_onehot = one_hot(col_data, col_cats)
            encoded_data[:, cidx : cidx + len(col_cats)] = col_data_onehot

            cidx += len(col_cats)

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

def get_dtype(representation, column_type):
    """

    Return the pandas type of a column based on the json schema for the dataset.

    Parameters
    ----------
    representation: object
        Either a string, an integer, or a list of string.
        The meaning depends upon ``column_type``.
    column_type: str
        The abstract type of the data column (e.g. ``"finite"``,
        ``"countable/ordered"``, etc).

    Returns
    -------
    dtype
        A type for the given column.

    """
    if representation == 'integer':
        return np.int
    elif representation == 'number':
        return np.float
    elif representation == 'date':
        return pd.datetime
    elif isinstance(representation, list):
        ordered = True if 'ordered' in column_type else False
        return pd.CategoricalDtype(categories=representation, ordered=ordered)
    elif isinstance(representation, (int)):
        ordered = True if 'ordered' in column_type else False
        return pd.CategoricalDtype(categories=range(representation), ordered=ordered)
    else:
        np.object
