# Unit Tests

The recommended way to run these tests locally and check your environment is using `pytest`. If you installed the dependencies using `poetry install` then this will be present. If you installed without dev-dependencies (i.e. by using `poetry install --no-dev`) then you can just run `poetry install` and `pytest` will be installed.

Once `pytest` is installed, the unit tests can be run from the root directory of the project via
```
pytest
```

For simplicity the tests are based on a small sample of the texas census dataset (the first 999 entries).