"""A test for the Generator class"""

from unittest import TestCase
from warnings import filterwarnings

import numpy as np
import pandas as pd
import pandas.testing as pdt 

# filterwarnings('ignore')

import prive.datasets as datasets 
import prive.generators as generators


class TestGenerator(TestCase):
    def setUp(self):
        ## An example dataset of 998 rows
        self.dataset = datasets.TabularDataset.read("tests/data/test_texas")

    def test_generate_raw(self):
        raw = generators.Raw()
        raw.fit(self.dataset)
        ds = raw.generate(5, random_state = 0)

        ## A previously-saved dataset generated as above
        baseline_dataset = datasets.TabularDataset.read("tests/data/test_texas_sample0")

        print("Baseline")
        print(baseline_dataset.data)
        print("Sample")
        print(ds.data)

        pdt.assert_frame_equal(ds.data.reset_index(drop = True), baseline_dataset.data)



if __name__ == "__main__":
    unittest.main()
