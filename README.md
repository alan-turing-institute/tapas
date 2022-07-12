![tests](https://github.com/alan-turing-institute/privacy-sdg-toolbox/actions/workflows/ci.yml/badge.svg) [![Documentation Status](https://readthedocs.org/projects/privacy-sdg-toolbox/badge/?version=latest)](https://privacy-sdg-toolbox.readthedocs.io/en/latest/?badge=latest)

# privacy-sdg-toolbox

Evaluating the privacy of synthetic data with an adversarial toolbox. [Documentation.](https://privacy-sdg-toolbox.readthedocs.io/en/latest/index.html#)

## Direct Installation

### Requirements
The framework and its building blocks have been developed and tested under Python 3.9.


#### Poetry installation
To mimic our environment exactly, we recommend using `poetry`. To install poetry (system-wide), follow the instructions [here](https://python-poetry.org/docs/).

Then run
```
poetry install
```
from inside the project directory. This will create a virtual environment (default `.venv`), that can be accessed by running `poetry shell`, or in the usual way (with `source .venv/bin/activate`).

#### Pip installation (includes command-line tool)

It is also possible to install from pip:
```
pip install git+https://github.com/alan-turing-institute/privacy-sdg-toolbox
```

Doing so installs a command-line tool, `prive`, somewhere in your path. (Eg, on
a MacOS system with pip installed via homebrew, the tool ends up in a homebrew
bin director.) 
