"""A set of utilities to do target selection and canaries."""
from .dataset import TabularDataset, TabularRecord, DataDescription


def create_canary(dataset: TabularDataset):
    """
    Create a "canary", a record which should stand out in the dataset and
    therefore is more likely to be identified in MIAs.

    This works by adding new categories to the description that are unique to
    this record, and setting continuous attributes as the current maximum plus
    some margin.

    This also creates a modified dataset with an updated description. Importantly,
    the canary is *not* part of the modified dataset, so it does not need to be
    removed before passing the dataset to auxiliary knowledge.

    Parameters
    ----------
    dataset: TabularDataset
        The dataset for which to generate a canary.

    Returns
    -------
    new_dataset: TabularDataset
        The input dataset with an updated description that takes the canary
        into account. The content is unchanged, dataset.data = new_dataset.data.
    record: TabularDataset
        The canary. Use this record as target in MIAs.

    """
    # First, modify the description to allow the canary. This also computes the
    # values to assign to the canary.
    canary_values = []
    new_schema = []
    for column in dataset.description:
        new_column = dict(column)
        if column["type"].startswith("finite"):
            # Integer: add one more value and increase the maximum.
            if isinstance(column["representation"], int):
                N = column["representation"]
                new_column["representation"] = N + 1
                canary_values.append(N)
            # List of strings: add the string CANARY.
            elif isinstance(column["representation"], list):
                # TODO: check that this is not already in the acceptable keys.
                new_key = "CANARY"
                new_column["representation"] = column["representation"] + [new_key]
                canary_values.append(new_key)
            # Otherwise: this is weird.
            else:
                raise Exception(
                    f"Unrecognised representation for type={column['type']}: {column['representation']}"
                )
        elif column["type"].startswith("real") or column["type"].startswith(
            "countable"
        ):
            # Add a number larger than the current max.
            new_max = dataset.data[column["name"]].max() + 1
            canary_values.append(new_max)
            # TODO: add string/datetime support.
        elif column["type"] == "interval":
            # Add the extremity of the interval.
            canary_values.append(1)
        else:
            raise Exception(f"Unrecognised data type: {column['type']}.")
        # Add the (potentially modified) new column.
        new_schema.append(new_column)
    # Second, create a dataset with the new description, and the canary from these values.
    new_description = DataDescription(new_schema, dataset.description.label)
    new_dataset = TabularDataset(dataset.data, new_description)
    canary = TabularRecord([canary_values], new_description, "canary")
    return new_dataset, canary
