"""A test for the Generator class"""

from unittest import TestCase
from warnings import filterwarnings

import numpy as np
import pandas as pd
import pandas.testing as pdt

import tapas.datasets as datasets
import tapas.generators as generators


class TestGenerator(TestCase):
    def setUp(self):
        ## An example dataset of 998 rows
        self.dataset = datasets.TabularDataset.read("tests/data/test_texas")

    def test_generate_raw(self):
        raw = generators.Raw()
        raw.fit(self.dataset)
        ds = raw.generate(5, random_state=0)

        ## A previously-saved dataset generated as above
        baseline_dataset = datasets.TabularDataset.read("tests/data/test_texas_sample0")

        pdt.assert_frame_equal(ds.data.reset_index(drop=True), baseline_dataset.data)

    def test_generator_from_exe(self):
        exe = generators.GeneratorFromExecutable("tests/bin/raw")

        ## The following is identical to test_generate_raw
        exe.fit(self.dataset)
        ds = exe.generate(5)  ## TODO: Figure out how to make reproducible call
        print(ds.data)

        ## A previously-saved dataset generated as above
        baseline_dataset = datasets.TabularDataset.read("tests/data/test_texas_sample0")

        # Check all generated rows are in the original data
        for row in ds:
            self.assertIn(row, self.dataset)

        # TODO: reproducible call should uncomment this
        # pdt.assert_frame_equal(ds.data.reset_index(drop = True), baseline_dataset.data)


if __name__ == "__main__":
    unittest.main()
